# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import math
import operator
import json
from pathlib import Path

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
from src.query_semantic_cache import (
    QUERY_SEMANTIC_CACHE_SCHEMA,
    QUERY_SEMANTIC_CACHE_VERSION,
    QUERY_SEMANTIC_SHUFFLE_ALGORITHM,
)

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
    accelerator = str(
        config.get("trainer", {}).get("accelerator", "gpu")
    ).lower()
    configured_devices = config.get("trainer", {}).get("devices", [1])
    # Use Tensor Cores for the fp32 local-matching matrices used by semantic
    # targets (and remove PyTorch's repeated 3090 performance warning).
    torch.set_float32_matmul_precision("high")
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
    attention_target = str(spatial_cfg.get('target', 'clip_attention'))
    reliability_cfg = spatial_cfg.get('semantic_reliability', {}) or {}
    semantic_region_cfg = distill_cfg.get('semantic_region', {}) or {}
    semantic_region_enabled = bool(
        semantic_region_cfg.get('enabled', False)
    )
    semantic_region_mode = str(
        semantic_region_cfg.get('mode', 'full')
    ).lower()
    lambda_semantic_region = float(
        semantic_region_cfg.get('lambda_target', 0.0)
    )
    semantic_region_apply_pretrained = bool(
        semantic_region_cfg.get('apply_pretrained_gate', False)
    )
    query_semantic_cfg = distill_cfg.get('query_semantic', {}) or {}
    query_semantic_enabled = bool(query_semantic_cfg.get('enabled', False))
    query_semantic_mode = str(
        query_semantic_cfg.get('mode', 'architecture_only')
    ).lower()
    lambda_query_semantic = float(
        query_semantic_cfg.get('lambda_target', 0.0)
    )
    query_semantic_num_classes = int(
        query_semantic_cfg.get('num_classes', 150)
    )
    query_semantic_teacher_model = str(
        query_semantic_cfg.get(
            'teacher_model',
            'nvidia/segformer-b0-finetuned-ade-512-512',
        )
    )
    query_semantic_teacher_revision = str(
        query_semantic_cfg.get(
            'teacher_revision',
            '489d5cd81a0b59fab9b7ea758d3548ebe99677da',
        )
    )
    query_semantic_transformers_version = str(
        query_semantic_cfg.get(
            'teacher_transformers_version', '4.44.2'
        )
    )
    query_semantic_min_confidence = float(
        query_semantic_cfg.get('min_confidence', 0.5)
    )
    query_semantic_ignore_index = int(
        query_semantic_cfg.get('ignore_index', 255)
    )
    query_semantic_random_seed = int(
        query_semantic_cfg.get('random_seed', config['seed'])
    )
    initial_checkpoint = config['trainer'].get('init_checkpoint')
    initial_checkpoint_sha256 = config['trainer'].get(
        'init_checkpoint_sha256'
    )
    freeze_base = bool(config['trainer'].get('freeze_base', False))
    detach_backbone_for_attn = bool(
        spatial_cfg.get('detach_backbone_for_kl', False)
    )
    lambda_global = float(
        distill_cfg.get('lambda_global', 0.1 if distill_enabled else 0.0)
    )
    lambda_region = float(
        distill_cfg.get('lambda_region', 0.05 if distill_enabled else 0.0)
    )
    semantic_alias_cfg = distill_cfg.get('semantic_alias', {}) or {}
    semantic_alias_enabled = bool(
        semantic_alias_cfg.get('enabled', False)
    )
    lambda_alias = float(semantic_alias_cfg.get('lambda', 0.0))
    semantic_alias_selection = str(
        semantic_alias_cfg.get('selection', 'clip')
    ).lower()
    semantic_alias_topk = int(
        semantic_alias_cfg.get('negative_topk', 1)
    )
    semantic_alias_min_distance = float(
        semantic_alias_cfg.get('min_geo_distance_m', 50.0)
    )
    semantic_alias_margin = float(
        semantic_alias_cfg.get('student_margin', 0.2)
    )
    semantic_alias_temperature = float(
        semantic_alias_cfg.get('loss_temperature', 0.05)
    )
    semantic_positive_cfg = distill_cfg.get('semantic_positive', {}) or {}
    semantic_positive_enabled = bool(
        semantic_positive_cfg.get('enabled', False)
    )
    raw_lambda_positive = semantic_positive_cfg.get('lambda', 0.0)
    if isinstance(raw_lambda_positive, bool):
        raise TypeError("distillation.semantic_positive.lambda must be numeric")
    lambda_positive = float(raw_lambda_positive)
    semantic_positive_selection = str(
        semantic_positive_cfg.get('selection', 'clip')
    ).lower()
    raw_positive_topk = semantic_positive_cfg.get('positive_topk', 1)
    if isinstance(raw_positive_topk, bool):
        raise TypeError(
            "distillation.semantic_positive.positive_topk must be an integer"
        )
    try:
        semantic_positive_topk = operator.index(raw_positive_topk)
    except TypeError as exc:
        raise TypeError(
            "distillation.semantic_positive.positive_topk must be an integer"
        ) from exc
    raw_teacher_chunk_size = semantic_positive_cfg.get(
        'teacher_chunk_size', 64
    )
    if isinstance(raw_teacher_chunk_size, bool):
        raise TypeError(
            "distillation.semantic_positive.teacher_chunk_size must be an integer"
        )
    try:
        semantic_positive_teacher_chunk_size = operator.index(
            raw_teacher_chunk_size
        )
    except TypeError as exc:
        raise TypeError(
            "distillation.semantic_positive.teacher_chunk_size must be an integer"
        ) from exc

    for name, value in (
        ('lambda_global', lambda_global),
        ('lambda_region', lambda_region),
        ('spatial_attn.lambda_kl', lambda_attn),
        ('semantic_alias.lambda', lambda_alias),
        ('semantic_positive.lambda', lambda_positive),
        ('semantic_region.lambda_target', lambda_semantic_region),
        ('query_semantic.lambda_target', lambda_query_semantic),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"distillation.{name} must be finite and non-negative"
            )
    if lambda_attn > 0 and not spatial_attn_enabled:
        raise ValueError(
            "lambda_kl is non-zero but distillation.spatial_attn.enabled is false"
        )
    if lambda_attn > 0 and not distill_enabled:
        raise ValueError(
            "CLIP attention supervision requires distillation.enabled=true"
        )
    valid_attention_targets = {
        'cls_attention',
        'clip_attention',
        'vpr_reliability',
        'vpr_semantic_reliability',
        'semantic_reliability',
    }
    if attention_target not in valid_attention_targets:
        raise ValueError(
            "distillation.spatial_attn.target must be clip_attention or "
            "semantic_reliability"
        )
    semantic_reliability_active = (
        lambda_attn > 0
        and attention_target
        in {
            'vpr_reliability',
            'vpr_semantic_reliability',
            'semantic_reliability',
        }
    )
    if semantic_reliability_active:
        if int(config['datamodule']['img_per_place']) < 2:
            raise ValueError(
                "semantic reliability requires datamodule.img_per_place >= 2"
            )
        if int(config['datamodule']['batch_size']) < 2:
            raise ValueError(
                "semantic reliability requires at least two places per batch"
            )
        if config.get('compile', False):
            raise ValueError(
                "semantic reliability uses dynamic pair mining and does not "
                "support --compile; run without that flag"
            )
    if detach_backbone_for_attn and not spatial_attn_enabled:
        raise ValueError(
            "detach_backbone_for_kl requires spatial_attn.enabled=true"
        )
    if not distill_enabled and (lambda_global > 0 or lambda_region > 0):
        raise ValueError(
            "Non-zero global/region weights require distillation.enabled=true"
        )
    if lambda_semantic_region > 0 and not semantic_region_enabled:
        raise ValueError(
            "semantic_region.lambda_target is non-zero but semantic_region.enabled is false"
        )
    if semantic_region_enabled and not distill_enabled:
        raise ValueError("semantic_region requires distillation.enabled=true")
    if semantic_region_mode not in {
        'repeatability_only',
        'repeatability_uniqueness_only',
        'semantic_only',
        'full',
        'shuffled',
    }:
        raise ValueError(
            "semantic_region.mode must be repeatability_only, "
            "repeatability_uniqueness_only, semantic_only, full, or shuffled"
        )
    semantic_region_target_active = (
        semantic_region_enabled and lambda_semantic_region > 0
    )
    semantic_region_gate_active = semantic_region_enabled and (
        semantic_region_target_active or semantic_region_apply_pretrained
    )
    if semantic_region_apply_pretrained and not semantic_region_enabled:
        raise ValueError(
            "semantic_region.apply_pretrained_gate requires enabled=true"
        )
    if semantic_region_target_active:
        if int(config['datamodule']['img_per_place']) < 2:
            raise ValueError("semantic_region requires img_per_place >= 2")
        if int(config['datamodule']['batch_size']) < 2:
            raise ValueError("semantic_region requires at least two places per batch")
        if config.get('compile', False):
            raise ValueError("semantic_region does not support --compile")
        uses_semantic_cache = semantic_region_mode in {
            'semantic_only', 'full', 'shuffled'
        }
        cache_dir = semantic_region_cfg.get('cache_dir')
        if uses_semantic_cache and not cache_dir:
            raise ValueError(
                f"semantic_region.cache_dir is required in {semantic_region_mode} mode"
            )
        if uses_semantic_cache and str(
            config['datamodule'].get('augmentation_mode', 'randaugment')
        ).lower() != 'photometric':
            raise ValueError(
                "cached semantic regions require datamodule.augmentation_mode="
                "photometric so image tokens stay spatially aligned"
            )
        if any(
            value > 0
            for value in (
                lambda_global,
                lambda_region,
                lambda_attn,
                lambda_alias,
                lambda_positive,
            )
        ) or spatial_attn_enabled:
            raise ValueError(
                "semantic_region experiments must disable the legacy global/"
                "region/attention/alias/positive paths so the ablation stays isolated"
            )

    from src.models.query_semantic import (
        QUERY_SEMANTIC_MODES,
        verify_query_semantic_cache_hashes,
    )

    if query_semantic_mode not in QUERY_SEMANTIC_MODES:
        raise ValueError(
            "distillation.query_semantic.mode must be architecture_only, "
            "aligned, shuffled, or random"
        )
    query_semantic_supervision_active = (
        query_semantic_enabled and lambda_query_semantic > 0
    )
    if lambda_query_semantic > 0 and not query_semantic_enabled:
        raise ValueError(
            "query_semantic.lambda_target is non-zero but enabled is false"
        )
    if query_semantic_enabled:
        if not distill_enabled:
            raise ValueError("query_semantic requires distillation.enabled=true")
        if config.get('compile', False):
            raise ValueError(
                "query_semantic zero-start compatibility does not support "
                "--compile"
            )
        if config['aggregator']['class'] != 'BoQ':
            raise ValueError("query_semantic currently requires the BoQ aggregator")
        if config['backbone']['class'] != 'DinoV2':
            raise ValueError(
                "query_semantic screen currently requires the DinoV2 backbone"
            )
        if not 2 <= query_semantic_num_classes <= 255:
            raise ValueError("query_semantic.num_classes must be in [2, 255]")
        if not 0.0 <= query_semantic_min_confidence <= 1.0:
            raise ValueError(
                "query_semantic.min_confidence must be in [0, 1]"
            )
        if not 0 <= query_semantic_ignore_index <= 255:
            raise ValueError("query_semantic.ignore_index must be in [0, 255]")
        if query_semantic_mode == 'architecture_only':
            if lambda_query_semantic != 0:
                raise ValueError(
                    "architecture_only requires query_semantic.lambda_target=0"
                )
        elif lambda_query_semantic <= 0:
            raise ValueError(
                f"query_semantic mode {query_semantic_mode} requires a positive "
                "lambda_target"
            )
        cache_dir = query_semantic_cfg.get('cache_dir')
        if query_semantic_supervision_active and not cache_dir:
            raise ValueError(
                f"query_semantic mode {query_semantic_mode} requires cache_dir"
            )
        try:
            train_height, train_width = (
                int(value)
                for value in config['datamodule']['train_image_size']
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "query_semantic requires train_image_size=[height,width]"
            ) from exc
        if train_height % 14 or train_width % 14:
            raise ValueError(
                "DINOv2 query semantics require train dimensions divisible by 14"
            )
        expected_query_grid = [train_height // 14, train_width // 14]
        if query_semantic_supervision_active:
            manifest_path = (
                Path(cache_dir).expanduser().resolve() / 'manifest.json'
            )
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"query semantic cache manifest not found: {manifest_path}"
                )
            with manifest_path.open('r', encoding='utf-8') as handle:
                query_manifest = json.load(handle)
            if (
                query_manifest.get('schema')
                != QUERY_SEMANTIC_CACHE_SCHEMA
                or query_manifest.get('version')
                != QUERY_SEMANTIC_CACHE_VERSION
                or not query_manifest.get('complete', False)
            ):
                raise ValueError(
                    "query semantic cache manifest is incomplete or unsupported"
                )
            if query_manifest.get('grid_size') != expected_query_grid:
                raise ValueError(
                    "query semantic cache grid does not match DINO features: "
                    f"expected {expected_query_grid}, found "
                    f"{query_manifest.get('grid_size')}"
                )
            if query_manifest.get('target_image_size') != [
                train_height,
                train_width,
            ]:
                raise ValueError(
                    "query semantic cache target_image_size does not match "
                    "datamodule.train_image_size"
                )
            if query_manifest.get('num_classes') != query_semantic_num_classes:
                raise ValueError(
                    "query semantic cache num_classes does not match the model"
                )
            if query_manifest.get('eligible_min_views') != int(
                config['datamodule']['img_per_place']
            ):
                raise ValueError(
                    "query semantic cache eligible_min_views does not match "
                    "datamodule.img_per_place"
                )
            if query_manifest.get('model_name') != query_semantic_teacher_model:
                raise ValueError(
                    "query semantic cache teacher model does not match the config"
                )
            if query_manifest.get('resolved_commit') != (
                query_semantic_teacher_revision
            ):
                raise ValueError(
                    "query semantic cache teacher commit does not match the "
                    "pinned config revision"
                )
            if query_manifest.get('transformers_version') != (
                query_semantic_transformers_version
            ):
                raise ValueError(
                    "query semantic cache transformers version does not "
                    "match the pinned config"
                )
            expected_cache_protocol = {
                'teacher_input': 'clean_rgb',
                'pooling': (
                    'bilinear_logits_to_target_then_softmax_then_'
                    'nonoverlap_avg_pool'
                ),
                'confidence_quantization': (
                    'round(clamp(top1_probability,0,1)*255)'
                ),
                'inference_precision': 'amp_float16',
                'shuffle_algorithm': QUERY_SEMANTIC_SHUFFLE_ALGORITHM,
            }
            for field, expected_value in expected_cache_protocol.items():
                if query_manifest.get(field) != expected_value:
                    raise ValueError(
                        f"query semantic cache {field} does not match the "
                        "preregistered clean-image patch protocol"
                    )
            verified_hashes = verify_query_semantic_cache_hashes(
                cache_dir, query_manifest
            )
            print(
                "Verified query-semantic cache SHA256: "
                + ", ".join(
                    f"{name}={digest[:12]}..."
                    for name, digest in verified_hashes.items()
                )
            )
        if query_semantic_supervision_active and str(
            config['datamodule'].get('augmentation_mode', 'randaugment')
        ).lower() != 'photometric':
            raise ValueError(
                "cached query semantics require augmentation_mode=photometric"
            )
        if not semantic_region_gate_active or semantic_region_mode != (
            'repeatability_uniqueness_only'
        ):
            raise ValueError(
                "query_semantic screen requires the pretrained "
                "repeatability_uniqueness_only gate"
            )
        if semantic_region_target_active:
            raise ValueError(
                "query_semantic frozen screen must set semantic_region."
                "lambda_target=0 to avoid recomputing the RU target"
            )
        if spatial_attn_enabled or any(
            value > 0
            for value in (
                lambda_global,
                lambda_region,
                lambda_attn,
                lambda_alias,
                lambda_positive,
            )
        ):
            raise ValueError(
                "query_semantic experiments must disable all legacy semantic "
                "and CLIP distillation paths"
            )
        if not initial_checkpoint:
            raise ValueError(
                "query_semantic requires --init-checkpoint (or trainer."
                "init_checkpoint) pointing to the trained RU checkpoint"
            )
        if not initial_checkpoint_sha256:
            raise ValueError(
                "query_semantic requires trainer.init_checkpoint_sha256 to "
                "pin the audited RU checkpoint"
            )
        if not freeze_base:
            raise ValueError(
                "the preregistered query_semantic screen requires "
                "trainer.freeze_base=true"
            )
    elif initial_checkpoint or initial_checkpoint_sha256 or freeze_base:
        raise ValueError(
            "trainer.init_checkpoint/freeze_base are reserved for the "
            "query-semantic screen"
        )

    semantic_alias_active = semantic_alias_enabled and lambda_alias > 0
    if lambda_alias > 0 and not semantic_alias_enabled:
        raise ValueError(
            "semantic_alias.lambda is non-zero but semantic_alias.enabled is false"
        )
    if semantic_alias_active and not distill_enabled:
        raise ValueError(
            "CLIP semantic-alias mining requires distillation.enabled=true"
        )
    if semantic_alias_selection not in {'clip', 'random', 'shuffled'}:
        raise ValueError(
            "distillation.semantic_alias.selection must be clip, random, "
            "or shuffled"
        )
    if semantic_alias_topk < 1:
        raise ValueError(
            "distillation.semantic_alias.negative_topk must be at least 1"
        )
    if semantic_alias_min_distance < 0:
        raise ValueError(
            "distillation.semantic_alias.min_geo_distance_m must be non-negative"
        )
    if not -1.0 <= semantic_alias_margin <= 1.0:
        raise ValueError(
            "distillation.semantic_alias.student_margin must be in [-1, 1]"
        )
    if semantic_alias_temperature <= 0:
        raise ValueError(
            "distillation.semantic_alias.loss_temperature must be positive"
        )
    if semantic_alias_active:
        if int(config['datamodule']['batch_size']) < 2:
            raise ValueError(
                "semantic-alias mining requires at least two places per batch"
            )
        if config.get('compile', False):
            raise ValueError(
                "semantic-alias mining uses dynamic pair selection and does not "
                "support --compile; run without that flag"
            )

    semantic_positive_active = (
        semantic_positive_enabled and lambda_positive > 0
    )
    if lambda_positive > 0 and not semantic_positive_enabled:
        raise ValueError(
            "semantic_positive.lambda is non-zero but "
            "semantic_positive.enabled is false"
        )
    if semantic_positive_active and not distill_enabled:
        raise ValueError(
            "CLIP semantic-positive mining requires distillation.enabled=true"
        )
    if semantic_positive_selection not in {
        'clip', 'random', 'shuffled', 'student'
    }:
        raise ValueError(
            "distillation.semantic_positive.selection must be clip, random, "
            "shuffled, or student"
        )
    if semantic_positive_topk < 1:
        raise ValueError(
            "distillation.semantic_positive.positive_topk must be at least 1"
        )
    if semantic_positive_teacher_chunk_size < 1:
        raise ValueError(
            "distillation.semantic_positive.teacher_chunk_size must be at least 1"
        )
    if semantic_positive_active:
        if int(config['datamodule']['img_per_place']) < 2:
            raise ValueError(
                "semantic-positive mining requires img_per_place >= 2"
            )
        if config.get('compile', False):
            raise ValueError(
                "semantic-positive mining uses dynamic pair selection and "
                "does not support --compile"
            )
        if spatial_attn_enabled or any(
            value > 0
            for value in (
                lambda_global,
                lambda_region,
                lambda_attn,
                lambda_alias,
            )
        ):
            raise ValueError(
                "semantic-positive experiments must disable global/region/"
                "attention distillation, spatial attention and semantic alias "
                "so the CLIP pair-selection effect remains isolated"
            )

    teacher_required = distill_enabled and any(
        value > 0
        for value in (
            lambda_global,
            lambda_region,
            lambda_attn,
            lambda_alias,
            lambda_positive,
        )
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
        return_metadata=(
            semantic_alias_active
            or semantic_positive_active
            or query_semantic_supervision_active
            or (
                semantic_region_target_active
                and semantic_region_mode in {'semantic_only', 'full', 'shuffled'}
            )
        ),
        return_teacher_view=semantic_positive_active,
        augmentation_mode=config['datamodule'].get(
            'augmentation_mode', 'randaugment'
        ),
        semantic_cache_dir=(
            semantic_region_cfg.get('cache_dir')
            if semantic_region_target_active
            and semantic_region_mode in {'semantic_only', 'full', 'shuffled'}
            else None
        ),
        query_semantic_cache_dir=(
            query_semantic_cfg.get('cache_dir')
            if query_semantic_supervision_active
            else None
        ),
        query_semantic_selection=(
            'aligned'
            if query_semantic_mode == 'architecture_only'
            else query_semantic_mode
        ),
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
    aggregator_semantic_classes = getattr(
        aggregator, 'semantic_num_classes', None
    )
    if query_semantic_enabled:
        if aggregator_semantic_classes != query_semantic_num_classes:
            raise ValueError(
                "aggregator.params.semantic_num_classes must equal "
                "distillation.query_semantic.num_classes"
            )
    elif aggregator_semantic_classes is not None:
        raise ValueError(
            "a semantic-conditioned aggregator requires "
            "distillation.query_semantic.enabled=true"
        )
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
            attention_target=attention_target,
            reliability_temperature=float(
                reliability_cfg.get('temperature', 0.1)
            ),
            reliability_negative_topk=int(
                reliability_cfg.get('negative_topk', 1)
            ),
            reliability_positive_weight=float(
                reliability_cfg.get('positive_weight', 1.0)
            ),
            reliability_negative_weight=float(
                reliability_cfg.get('negative_weight', 1.0)
            ),
            reliability_pair_chunk_size=int(
                reliability_cfg.get('pair_chunk_size', 32)
            ),
            semantic_alias_enabled=semantic_alias_active,
            semantic_alias_selection=semantic_alias_selection,
            semantic_alias_negative_topk=semantic_alias_topk,
            semantic_alias_min_geo_distance_m=semantic_alias_min_distance,
            semantic_alias_student_margin=semantic_alias_margin,
            semantic_alias_loss_temperature=semantic_alias_temperature,
            semantic_positive_enabled=semantic_positive_active,
            semantic_positive_selection=semantic_positive_selection,
            semantic_positive_topk=semantic_positive_topk,
            semantic_positive_teacher_chunk_size=(
                semantic_positive_teacher_chunk_size
            ),
        )
        # Alias/attention-only runs never execute the global projection.
        # Freeze it so DDP does not see an unused trainable parameter.
        if lambda_global <= 0:
            distill_module.student_global_proj.requires_grad_(False)

    semantic_region_gate = None
    semantic_region_target = None
    if semantic_region_gate_active:
        from src.models.semantic_region_gate import (
            SemanticRegionGate,
        )

        semantic_region_gate = SemanticRegionGate(
            in_channels=out_channels,
            alpha=float(semantic_region_cfg.get('alpha', 0.2)),
        )
    if semantic_region_target_active:
        from src.models.semantic_region_gate import (
            SemanticRegionReliabilityTarget,
        )

        semantic_region_target = SemanticRegionReliabilityTarget(
            mode=semantic_region_mode,
            match_grid=int(semantic_region_cfg.get('match_grid', 10)),
            target_scale=float(semantic_region_cfg.get('target_scale', 2.0)),
            place_chunk_size=int(
                semantic_region_cfg.get('place_chunk_size', 8)
            ),
            min_spatial_std=float(
                semantic_region_cfg.get('min_spatial_std', 1e-3)
            ),
        )

    query_semantic_target = None
    if query_semantic_supervision_active:
        from src.models.query_semantic import QuerySemanticTarget

        query_semantic_target = QuerySemanticTarget(
            mode=query_semantic_mode,
            num_classes=query_semantic_num_classes,
            min_confidence=query_semantic_min_confidence,
            ignore_index=query_semantic_ignore_index,
            random_seed=query_semantic_random_seed,
        )

    if (
        teacher_required
        or spatial_attn_enabled
        or semantic_region_gate_active
        or query_semantic_enabled
    ):
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
            lambda_alias=lambda_alias,
            lambda_positive=lambda_positive,
            distill_warmup_steps=distill_cfg.get('distill_warmup_steps', 1500),
            detach_backbone_for_attn=detach_backbone_for_attn,
            semantic_region_gate=semantic_region_gate,
            semantic_region_target=semantic_region_target,
            lambda_semantic_region=lambda_semantic_region,
            query_semantic_target=query_semantic_target,
            lambda_query_semantic=lambda_query_semantic,
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

    if query_semantic_enabled and initial_checkpoint:
        from src.models.query_semantic import warm_start_query_semantic_model

        warm_start_report = warm_start_query_semantic_model(
            vpr_model,
            initial_checkpoint,
            expected_sha256=initial_checkpoint_sha256,
        )
        print(
            "Query-semantic warm start: "
            f"{warm_start_report['checkpoint']} "
            f"sha256={warm_start_report['checkpoint_sha256'][:12]}...; "
            f"({warm_start_report['loaded_keys']} tensors loaded; "
            f"{len(warm_start_report['new_keys'])} new tensors initialized)"
        )
    if query_semantic_enabled and freeze_base:
        from src.models.query_semantic import freeze_for_query_semantic_screen

        trainable_names = freeze_for_query_semantic_screen(vpr_model)
        trainable_count = sum(
            parameter.numel()
            for parameter in vpr_model.parameters()
            if parameter.requires_grad
        )
        print(
            "Frozen RU base; query-semantic trainable tensors: "
            f"{len(trainable_names)} ({trainable_count:,} parameters)"
        )

    if config["compile"]:
        vpr_model = torch.compile(vpr_model)


    # Let's define the TensorBoardLogger
    # We will save under the logs directory 
    # and use the backbone name as the subdirectory
    # e.g. a BoQ model with ResNet50 backbone will be saved under logs/ResNet50/BoQ
    # this makes it easy to compared different aggregators with the same backbone
    experiment_name = aggregator.__class__.__name__
    if query_semantic_enabled:
        experiment_name += f"_query_semantic_{query_semantic_mode}"
    elif semantic_region_gate_active:
        experiment_name += f"_semantic_region_{semantic_region_mode}"
    elif str(
        config['datamodule'].get('augmentation_mode', 'randaugment')
    ).lower() == 'photometric':
        experiment_name += "_photometric"

    tensorboard_logger = TensorBoardLogger(
        save_dir=f"./logs/{backbone.backbone_name}",
        name=experiment_name,
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
        accelerator=accelerator,
        devices=configured_devices,
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
