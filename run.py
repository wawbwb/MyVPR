# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import yaml
import importlib
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger
from src.core.vpr_datamodule import VPRDataModule
from src.core.vpr_framework import VPRFramework, VPRFrameworkDistill
from src.losses.vpr_losses import VPRLossFunction

from rich.traceback import install
install() # this is for better traceback formatting

# we mostly use mean and std of ImageNet dataset for normalization
# you can define your own mean and std values and use them
IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

# list of all cities to be used in "gsv-cities"
# if you want to use a subset cities, you edit the list
# and pass it to the VPRDataModule
ALL_CITIES = [
    'Bangkok', 
    'BuenosAires', 
    'LosAngeles', 
    'MexicoCity',
    'OSL', 
    'Rome', 
    'Barcelona', 
    'Chicago', 
    'Madrid', 
    'Miami',
    'Phoenix', 
    'TRT', 
    'Boston', 
    'Lisbon', 
    'Medellin', 
    'Minneapolis', 
    'PRG', 
    'WashingtonDC', 
    'Brussels',
    'London', 
    'Melbourne', 
    'Osaka', 
    'PRS',
]


def load_config(config_path='model_config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_instance(module_name, class_name, params):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(**params)


# This is called when the train mode is selected
def train(config):
    seed_everything(config["seed"], workers=True)
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True)
    torch.backends.cuda.enable_flash_sdp(True)

    # let's create the VPR DataModule
    # ── Distillation config ─────────────────────────────────────────
    distill_cfg = config.get('distillation', {})
    distill_enabled = distill_cfg.get('enabled', False)
    distill_mode = distill_cfg.get('mode', 'region_gate')
    spatial_cfg = distill_cfg.get('spatial_attn', {})
    spatial_attn_enabled = bool(spatial_cfg.get('enabled', False))
    lambda_attn = float(spatial_cfg.get('lambda_kl', 0.0))
    lambda_global = float(
        distill_cfg.get('lambda_global', 0.1 if distill_enabled else 0.0)
    )
    lambda_region = float(
        distill_cfg.get('lambda_region', 0.05 if distill_enabled else 0.0)
    )

    for name, value in (
        ('lambda_global', lambda_global),
        ('lambda_region', lambda_region),
        ('spatial_attn.lambda_kl', lambda_attn),
    ):
        if value < 0:
            raise ValueError(f"distillation.{name} must be non-negative")
    if lambda_attn > 0 and not spatial_attn_enabled:
        raise ValueError(
            "lambda_kl is non-zero but distillation.spatial_attn.enabled is false"
        )
    if lambda_attn > 0 and not distill_enabled:
        raise ValueError(
            "CLIP attention supervision requires distillation.enabled=true"
        )
    if not distill_enabled and (lambda_global > 0 or lambda_region > 0):
        raise ValueError(
            "Non-zero global/region weights require distillation.enabled=true"
        )

    teacher_required = distill_enabled and any(
        value > 0 for value in (lambda_global, lambda_region, lambda_attn)
    )
    return_augmented = (
        teacher_required and lambda_region > 0 and distill_mode == 'region_gate'
    )

    datamodule = VPRDataModule(
        train_set_name=config['datamodule']['train_set_name'],
        cities=config['datamodule']['cities'], # if None or "all" then we use all cities
        train_image_size=config['datamodule']['train_image_size'],
        batch_size=config['datamodule']['batch_size'],
        img_per_place=config['datamodule']['img_per_place'],
        random_sample_from_each_place=True,
        shuffle_all=False,
        num_workers=config['datamodule']['num_workers'],
        batch_sampler=None,
        mean_std=IMAGENET_MEAN_STD,
        val_set_names=config['datamodule']['val_set_names'],
        val_image_size=config['datamodule']['val_image_size'], # if None, the same as train_image_size
        return_augmented=return_augmented,
    )


    # Let's instantiate the backbone, aggregator and loss function. These are the main components of the VPRFramework
    # Make sure the model_config.yaml file is properly configured
    backbone = get_instance(config['backbone']['module'], config['backbone']['class'], config['backbone']['params'])
    out_channels = backbone.out_channels # all backbones should have an out_channels attribute
    
    # most of the time, the aggregator needs to know the number of output channels of the backbone
    # that arguments is passed to the aggregator as a parameter `in_channels` for some aggregators
    if 'in_channels' in config['aggregator']['params']:
        if config['aggregator']['params']['in_channels'] is None:
            config['aggregator']['params']['in_channels'] = out_channels
    
    aggregator = get_instance(config['aggregator']['module'], config['aggregator']['class'], config['aggregator']['params'])
    loss_function = get_instance(config['loss_function']['module'], config['loss_function']['class'], config['loss_function']['params'])

    # The phase-C student head is an inference-time component, so it is kept
    # separate from the teacher/distillation projections. This also permits a
    # lambda=0 architecture control without running CLIP.
    spatial_attn_head = None
    if spatial_attn_enabled:
        from src.models.distillation import SpatialAttentionHead

        spatial_attn_head = SpatialAttentionHead(
            in_channels=out_channels,
            num_heads=int(spatial_cfg.get('num_heads', 1)),
            gate_strength=float(spatial_cfg.get('gate_strength', 1.0)),
        )

    # ── Build distillation module (if enabled) ──────────────────────
    distill_module = None

    if teacher_required:
        from src.models.clip_teacher import CLIPTeacherEncoder
        from src.models.distillation import DistillationModule

        teacher = CLIPTeacherEncoder(
            model_name=distill_cfg.get('teacher', {}).get(
                'model_name', 'ViT-B-16'
            ),
            pretrained=distill_cfg.get('teacher', {}).get('pretrained', 'openai'),
            dynamic_categories=distill_cfg.get('dynamic_categories', None),
        )

        # Infer student global descriptor dimension via a dummy forward
        agg_params = config['aggregator']['params']
        with torch.no_grad():
            _h = agg_params.get('in_h', 20)
            _w = agg_params.get('in_w', 20)
            _dummy = torch.randn(2, out_channels, _h, _w)
            _outs = aggregator(_dummy)
            student_global_dim = _outs[0].shape[1] if isinstance(_outs, (tuple, list)) else _outs.shape[1]

        distill_module = DistillationModule(
            teacher=teacher,
            teacher_token_dim=teacher.token_dim,
            teacher_global_dim=teacher.global_dim,
            student_feat_channels=out_channels,
            student_global_dim=student_global_dim,
            proj_dim=distill_cfg.get('proj_dim', None),
            tau=distill_cfg.get('tau', 0.07),
            distill_mode=distill_mode,
        )

    if teacher_required or spatial_attn_enabled:
        vpr_model = VPRFrameworkDistill(
            backbone=backbone,
            aggregator=aggregator,
            loss_function=loss_function,
            optimizer=config['trainer']['optimizer'],
            lr=config['trainer']['lr'],
            weight_decay=config['trainer']['wd'],
            warmup_steps=config['trainer']['warmup'],
            milestones=config['trainer']['milestones'],
            lr_mult=config['trainer']['lr_mult'],
            verbose= not config["silent"],
            config_dict=config, # pass the config to the framework in order to save it
            distill_module=distill_module,
            spatial_attn_head=spatial_attn_head,
            lambda_global=lambda_global,
            lambda_region=lambda_region,
            lambda_attn=lambda_attn,
            distill_warmup_steps=distill_cfg.get('distill_warmup_steps', 1500),
        )
    else:
        vpr_model = VPRFramework(
            backbone=backbone,
            aggregator=aggregator,
            loss_function=loss_function,
            optimizer=config['trainer']['optimizer'],
            lr=config['trainer']['lr'],
            weight_decay=config['trainer']['wd'],
            warmup_steps=config['trainer']['warmup'],
            milestones=config['trainer']['milestones'],
            lr_mult=config['trainer']['lr_mult'],
            verbose= not config["silent"],
            config_dict=config, # pass the config to the framework in order to save it
        )

    if config["compile"]:
        vpr_model = torch.compile(vpr_model)


    # Let's define the TensorBoardLogger
    # We will save under the logs directory 
    # and use the backbone name as the subdirectory
    # e.g. a BoQ model with ResNet50 backbone will be saved under logs/ResNet50/BoQ
    # this makes it easy to compared different aggregators with the same backbone
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"./logs/{backbone.backbone_name}",
        name=f"{aggregator.__class__.__name__}",
        default_hp_metric=False
    )
    
    # Let's define the checkpointing.
    # We use a callback and give it to the trained
    # The ModelCheckpoint callback saves the best k models based on a validation metric
    # In this example we are using msls-val/R1 as the metric to monitor
    # The checkpoint files will be saved in the logs directory (which we defined in the TensorBoardLogger)
    checkpoint_cb = ModelCheckpoint(
        monitor="msls-val/R1",
        filename="epoch({epoch:02d})_step({step:04d})_R1[{msls-val/R1:.4f}]_R5[{msls-val/R5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=3,
        mode="max",
    )
    
    # Let's define the progress bar, model summary and data summary callbacks
    from src.utils.callbacks import CustomRichProgressBar, CustomRRichModelSummary, DatamoduleSummary
    # there are multiple themes you can choose from. They are defined in src.utils.callbacks
    # example: default, cool_modern, vibrant_high_contrast, green_burgundy, magenta
    progress_bar_cb = CustomRichProgressBar(config["display_theme"])    
    model_summary_cb = CustomRRichModelSummary(config["display_theme"])    
    data_summary_cb = DatamoduleSummary(config["display_theme"])

    # Teacher construction and the descriptor-dimension probe consume random
    # numbers only in supervised runs. Reset here so C0 and C2/C3 see the same
    # sampler/augmentation RNG stream for a given seed.
    seed_everything(config["seed"], workers=True)

    trainer = Trainer(
        accelerator=config['trainer'].get('accelerator', 'gpu'),
        devices=config['trainer'].get('devices', [1]),
        logger=tensorboard_logger,
        num_sanity_val_steps=0, # is -1 to run one pass on all validation sets before training starts
        precision=config['trainer'].get('precision', '16-mixed'),
        max_epochs=config['trainer']['max_epochs'],
        check_val_every_n_epoch=1,
        callbacks=[
            checkpoint_cb,
            data_summary_cb,    # this will print the data summary
            model_summary_cb,   # this will print the model summary
            progress_bar_cb,    # this will print the progress bar
            ],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=10,
        fast_dev_run=config["dev"], # dev mode (only runs one train iteration and one valid iteration, no checkpointing and no performance tracking).
        enable_model_summary=False, # we are using our own model summary
    )

    # save the config into logs directory
    # with open(f"{tensorboard_logger.log_dir}/custom_config.yaml", 'w') as file:
    #     yaml.dump(config, file)
    
    trainer.fit(model=vpr_model, datamodule=datamodule)

def evaluate(config):
    print("Evaluation mode selected.")
    # Your evaluation logic here

def main():
    from argparser import parse_args
    config = parse_args()
    if config["train"]:
        train(config)
    # elif args.test:
        # evaluate(args, config)
    # else:
        # parser.print_help()

if __name__ == "__main__":
    main()
