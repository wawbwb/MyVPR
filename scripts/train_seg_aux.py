"""Matched shared-backbone semantic auxiliary screen; no CLIP or new gating."""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from scripts.eval_condition_robustness import load_inference_model_from_ckpt
from src.core.vpr_datamodule import VPRDataModule
from src.losses.vpr_losses import VPRLossFunction
from src.models.query_semantic import (
    _file_sha256, _validate_ru_checkpoint_config,
    verify_query_semantic_cache_hashes,
)
from src.models.seg_aux import SegAuxVPR


class SegAuxDataModule(VPRDataModule):
    def setup(self, stage=None):
        if stage == 'validate':
            self.val_datasets = [self._get_val_dataset(name) for name in self.val_set_names]
        else:
            super().setup(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/boq_dinov2_seg_aux.yaml')
    parser.add_argument('--mode', required=True, choices=['vpr_only', 'aligned', 'shuffled'])
    parser.add_argument('--init-checkpoint', required=True)
    parser.add_argument('--device', type=int, default=1, help='Visible CUDA index')
    parser.add_argument('--output', required=True, help='New directory, must not exist')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite run: {output}')
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    s = cfg['seg_aux']
    s['mode'] = args.mode
    for key in ('lr', 'head_lr', 'lambda_seg'):
        if not math.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not 1 <= s['unfrozen_blocks'] <= 12:
        raise ValueError('unfrozen_blocks must be in [1,12]')
    if s['epochs'] < 1 or s['diagnostic_interval'] < 1:
        raise ValueError('epochs and diagnostic_interval must be positive')
    checkpoint = Path(args.init_checkpoint)
    if not checkpoint.is_file() or _file_sha256(checkpoint) != s['init_sha256']:
        raise ValueError('Missing checkpoint or RU SHA256 mismatch')
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    _validate_ru_checkpoint_config(source)
    config = copy.deepcopy(source['hyper_parameters'])
    del source
    cache = Path(s['cache_dir'])
    manifest = json.loads((cache / 'manifest.json').read_text())
    if (not manifest.get('complete') or manifest.get('grid_size') != [20, 20]
            or manifest.get('num_classes') != 150
            or manifest.get('target_image_size') != [280, 280]):
        raise ValueError('Requires complete ADE20K-150 grid20 cache for 280x280')
    if (manifest.get('model_name') != 'nvidia/segformer-b0-finetuned-ade-512-512'
            or manifest.get('resolved_commit') != '489d5cd81a0b59fab9b7ea758d3548ebe99677da'):
        raise ValueError('Semantic teacher identity differs from the matched cached teacher')
    hashes = verify_query_semantic_cache_hashes(cache, manifest)
    L.seed_everything(cfg['seed'], workers=True)
    visual = load_inference_model_from_ckpt(checkpoint, 'cpu')
    if visual.spatial_attn_head is not None or visual.semantic_region_gate is None:
        raise ValueError('Expected plain RU-BoQ without extra spatial attention')
    config['seed'] = cfg['seed']
    config['seg_aux'] = s
    config['backbone']['params']['num_unfrozen_blocks'] = s['unfrozen_blocks']
    config['trainer'] = dict(max_epochs=s['epochs'], precision='32-true',
                             lr=s['lr'], optimizer='adamw', max_steps=-1)
    # Preserve checkpoint reconstruction keys, but disable historical supervision.
    config['distillation'] = {'enabled': True, 'semantic_region': {
        'enabled': True, 'mode': 'repeatability_uniqueness_only',
        'alpha': 0.2, 'lambda_target': 0.0, 'apply_pretrained_gate': True}}
    config['datamodule'] = dict(
        train_set_name='gsv-cities', cities='all', batch_size=40, img_per_place=4,
        train_image_size=[280, 280], val_image_size=[280, 280],
        val_set_names=['msls-val'], augmentation_mode='photometric',
        num_workers=s['num_workers'], query_semantic_cache_dir=str(cache),
        query_semantic_selection='shuffled' if args.mode == 'shuffled' else 'aligned',
    )
    dm = SegAuxDataModule(**config['datamodule'])
    model = SegAuxVPR(visual.backbone, visual.aggregator,
                      visual.semantic_region_gate, VPRLossFunction(), config)
    device = torch.device(f'cuda:{args.device}')
    model.to(device).eval()
    # Both paths share loaded modules, but independently compose the forward;
    # this catches an accidentally omitted/doubled RU gate before any update.
    with torch.inference_mode():
        sample = torch.zeros(2, 3, 280, 280, device=device)
        difference = (model(sample) - visual(sample)).abs().max().item()
    if difference > 1e-6:
        raise RuntimeError(f'Initial RU descriptor mismatch: {difference}')
    print(f'PASS initial RU descriptor equality: {difference:.3e}', flush=True)
    del visual
    # Identical data/augmentation RNG after all initialization and audit work.
    L.seed_everything(cfg['seed'], workers=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    contract = dict(config=config, cache_sha256=hashes,
                    manifest_sha256=_file_sha256(cache / 'manifest.json'),
                    init_descriptor_error=difference, smoke_test=args.smoke_test,
                    trainable=[n for n, p in model.named_parameters() if p.requires_grad])
    (output / 'contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    callbacks = [TQDMProgressBar(refresh_rate=1)]
    if not args.smoke_test:
        callbacks.append(ModelCheckpoint(
            dirpath=output / 'checkpoints', filename='epoch{epoch:02d}-step{step}',
            auto_insert_metric_name=False, save_top_k=-1,
            every_n_epochs=1, save_last=True,
        ))
    trainer = L.Trainer(
        accelerator='gpu', devices=[args.device], precision='32-true',
        max_epochs=1 if args.smoke_test else s['epochs'],
        max_steps=-1, limit_train_batches=2 if args.smoke_test else 1.0,
        limit_val_batches=0 if args.smoke_test else 1.0,
        num_sanity_val_steps=0, gradient_clip_val=1.0,
        callbacks=callbacks, enable_checkpointing=not args.smoke_test,
        logger=[CSVLogger(str(output), name='csv'), TensorBoardLogger(str(output), name='tb')],
        log_every_n_steps=1 if args.smoke_test else 20,
    )
    if not args.smoke_test:
        # Reproduce full MSLS before training in each matched arm.
        trainer.validate(model, datamodule=dm)
        baseline = float(trainer.callback_metrics['msls-val/R1'])
        if abs(baseline - 675 / 740) > 1e-6:
            raise RuntimeError(f'RU baseline mismatch: {baseline}, expected 675/740')
        L.seed_everything(cfg['seed'], workers=True)
    trainer.fit(model, datamodule=dm)
    expected_steps = (2 if args.smoke_test else
                      int(trainer.num_training_batches) * s['epochs'])
    if trainer.interrupted or trainer.global_step != expected_steps:
        raise RuntimeError(f'Incomplete training: {trainer.global_step}/{expected_steps} steps')
    (output / 'completed.json').write_text(json.dumps({
        'smoke_test': args.smoke_test, 'global_step': trainer.global_step,
        'epochs_requested': 1 if args.smoke_test else s['epochs'],
        'status': 'SMOKE PASS' if args.smoke_test else 'TRAINING COMPLETE (not an efficacy verdict)',
    }, indent=2), encoding='utf-8')
    print('SMOKE PASS' if args.smoke_test else 'TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
