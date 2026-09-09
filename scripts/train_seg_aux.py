"""Matched shared-backbone semantic auxiliary screen; no CLIP or new gating."""

import argparse
import copy
import json
import math
import sys
import uuid
from pathlib import Path
import csv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightning as L
import torch
import yaml
from tqdm import tqdm
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


def prewarm_head(model, dm, config, output, device, smoke=False):
    """Separate head-only optimizer; retrieval parameters and state stay untouched."""
    steps = int(config['seg_aux'].get('head_warmup_steps', 0))
    if not steps or model.mode == 'vpr_only':
        return
    steps = min(steps, 2) if smoke else steps
    model.eval()
    model.seg_head.train()
    optimizer = torch.optim.AdamW(model.seg_head.parameters(),
                                 lr=config['seg_aux']['head_lr'],
                                 weight_decay=config['seg_aux']['weight_decay'])
    # Detect in-place modification of any retrieval parameter during prewarm.
    retrieval = list(model.backbone.parameters()) + list(model.aggregator.parameters()) + list(model.semantic_region_gate.parameters())
    versions = [p._version for p in retrieval]
    dm.setup('fit')
    loader = dm.train_dataloader()
    iterator = iter(loader)
    with (output / 'head_warmup.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['step', 'seg_loss', 'accuracy', 'valid_fraction'])
        bar = tqdm(range(steps), desc='Head-only warmup (RU frozen)')
        for step in bar:
            try:
                images, _, meta = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, _, meta = next(iterator)
            with torch.no_grad():
                features = model.backbone(images.flatten(0, 1).to(device))
            logits = model.seg_head(features.detach())
            loss, stats = model.target(
                logits, meta['query_semantic_labels'].flatten(0, 1).to(device),
                meta['query_semantic_confidence'].flatten(0, 1).to(device),
                meta['query_semantic_cache_indices'].flatten().to(device))
            if not torch.isfinite(loss) or stats['query_semantic_valid_frac'].item() == 0:
                raise RuntimeError('Invalid head prewarm loss or empty supervision')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.seg_head.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            writer.writerow([step, loss.item(), stats['query_semantic_accuracy'].item(),
                             stats['query_semantic_valid_frac'].item()])
            handle.flush()
            bar.set_postfix(loss=f'{loss.item():.4f}')
    if versions != [p._version for p in retrieval] or any(p.grad is not None for p in retrieval):
        raise RuntimeError('Head-only prewarm unexpectedly modified retrieval parameters')
    optimizer.zero_grad(set_to_none=True)
    torch.save({'state_dict': model.seg_head.state_dict(), 'steps': steps}, output / 'prewarmed_head.pt')
    print('PASS head-only warmup; retrieval parameters unchanged. Joint optimizer starts fresh.', flush=True)
    # All matched arms start the joint phase with the same data/augmentation seed.
    L.seed_everything(config['seed'], workers=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/boq_dinov2_seg_aux.yaml')
    parser.add_argument('--mode', required=True, choices=['vpr_only', 'aligned', 'shuffled'])
    parser.add_argument('--init-checkpoint', required=True)
    parser.add_argument('--device', type=int, default=1, help='Visible CUDA index')
    parser.add_argument('--output', required=True, help='New run directory, or original directory with --resume')
    parser.add_argument('--resume', help='Original run checkpoints/last.ckpt; restores optimizer and epoch')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if args.resume and args.smoke_test:
        parser.error('--resume cannot be combined with --smoke-test')
    if args.resume:
        resume = Path(args.resume).resolve()
        if resume != (output / 'checkpoints' / 'last.ckpt').resolve():
            raise ValueError('Resume requires the original output directory and its checkpoints/last.ckpt')
        if not resume.is_file() or not (output / 'contract.json').is_file():
            raise FileNotFoundError('Resume checkpoint or original contract is missing')
        if (output / 'completed.json').exists():
            raise ValueError('Run already completed; refusing to resume')
    elif output.exists():
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
    for key in ('head_warmup_steps', 'seg_ramp_steps'):
        if not isinstance(s.get(key, 0), int) or s.get(key, 0) < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
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
    contract = dict(config=config, cache_sha256=hashes,
                    manifest_sha256=_file_sha256(cache / 'manifest.json'),
                    init_descriptor_error=difference, smoke_test=args.smoke_test,
                    trainable=[n for n, p in model.named_parameters() if p.requires_grad])
    if args.resume:
        original = json.loads((output / 'contract.json').read_text(encoding='utf-8'))
        # JSON normalization handles YAML tuples in historical configurations.
        normalized = json.loads(json.dumps(contract))
        for key in ('config', 'cache_sha256', 'manifest_sha256', 'trainable', 'smoke_test'):
            if original.get(key) != normalized[key]:
                raise ValueError(f'Resume contract mismatch: {key}; do not change experiment settings')
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if json.loads(json.dumps(state.get('hyper_parameters'))) != normalized['config']:
            raise ValueError('Resume checkpoint configuration does not match this run')
        if not state.get('optimizer_states') or not state.get('lr_schedulers'):
            raise ValueError('Resume needs full training state, not inference weights')
        if int(state.get('epoch', -1)) >= s['epochs'] - 1:
            raise ValueError('Checkpoint has already reached the requested final epoch')
        event = dict(checkpoint=str(resume), sha256=_file_sha256(resume),
                     saved_epoch=state['epoch'], saved_global_step=state['global_step'],
                     note='Epoch-boundary resume; data-worker RNG is not restored exactly')
        del state
        (output / f'resume_{uuid.uuid4().hex}.json').write_text(json.dumps(event, indent=2), encoding='utf-8')
        print(f"Resuming saved epoch {event['saved_epoch']} / step {event['saved_global_step']}; total budget remains {s['epochs']} epochs", flush=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        (output / 'contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    if not args.resume:
        prewarm_head(model, dm, config, output, device, args.smoke_test)
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
    if not args.smoke_test and not args.resume:
        # Reproduce full MSLS before training in each matched arm.
        trainer.validate(model, datamodule=dm)
        baseline = float(trainer.callback_metrics['msls-val/R1'])
        if abs(baseline - 675 / 740) > 1e-6:
            raise RuntimeError(f'RU baseline mismatch: {baseline}, expected 675/740')
        L.seed_everything(cfg['seed'], workers=True)
    trainer.fit(model, datamodule=dm, ckpt_path=str(resume) if args.resume else None)
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
