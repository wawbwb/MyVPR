"""Pinned author-released BoQ, built offline with all checkpoint tensors required."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

COMMIT = '1a4965ea7dfd9bd0dd846adf7a0e430f68101d12'
WEIGHT_URL = 'https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/dinov2_12288.pth'
WEIGHT_SHA = 'd72ee0ce899e2790be6cf6d1d57f75dfad0c76475bd2198e6dc0c3f0221b3583'
SOURCE_SHA = {
    'backbones.py': 'c9c2924e462a207e35098caf627fd8cc9bdab6b701fe8a234a9993a540df05e4',
    'boq.py': '289343f5f7d41c47b6a3b604af61479b384f0edd223f85d3bef6ea3929c75c1c',
    'hubconf.py': '26d1cd9f1bd12b1f407b2441b7dae37223e7d0e59435ad88abebef813d7b1d2b',
    'LICENSE': 'ee735cd704d5231257d04ef6ae8202b74300c69687183cfaea5ebe4e3ba7a61d',
    'README.md': '8814026b5f465b3225e26b565f3bd15b0a413c8c5b3e65e6449b8e3a3cd5943d',
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_download(cache):
    cache = Path(cache)
    manifest = json.loads((cache / 'manifest.json').read_text())
    if manifest['commit'] != COMMIT or manifest['weight_url'] != WEIGHT_URL:
        raise ValueError('Official checkpoint provenance differs')
    if manifest['weight_sha256'] != WEIGHT_SHA or file_sha(cache / 'dinov2_12288.pth') != WEIGHT_SHA:
        raise ValueError('Official checkpoint SHA256 differs')
    if manifest['source_sha256'] != SOURCE_SHA:
        raise ValueError('Pinned official source inventory differs')
    for name, digest in SOURCE_SHA.items():
        if file_sha(cache / 'source' / name) != digest:
            raise ValueError('Official source changed: ' + name)
    return manifest


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialBoQ(nn.Module):
    """Same forward and registered names as the official hubconf VPRModel."""
    def __init__(self, backbone, aggregator):
        super().__init__()
        self.backbone, self.aggregator = backbone, aggregator
        self.semantic_region_gate = None

    def forward(self, images):
        return self.aggregator(self.backbone(images))[0]


def load_official(cache, dino_source, device='cpu'):
    cache, dino_source = Path(cache), Path(dino_source)
    verify_download(cache)
    if not (dino_source / 'hubconf.py').is_file():
        raise FileNotFoundError('Need the existing local DINOv2 source; no network fallback')
    backbones = load_source(cache / 'source/backbones.py', 'pinned_author_boq_backbones')
    aggregation = load_source(cache / 'source/boq.py', 'pinned_author_boq_aggregation')
    original_load = torch.hub.load

    def local_dino(repo, name, *args, **kwargs):
        if repo != 'facebookresearch/dinov2' or name != 'dinov2_vitb14' or args or kwargs:
            raise ValueError('Unexpected upstream backbone download request')
        return original_load(str(dino_source), name, source='local', pretrained=False)

    # Only replace base-weight discovery during official construction. The
    # complete author checkpoint subsequently replaces EVERY initialized tensor.
    with patch('torch.hub.load', side_effect=local_dino):
        backbone = backbones.DinoV2()
    aggregator = aggregation.BoQ(768, 384, 64, 2, 32)
    model = OfficialBoQ(backbone, aggregator)
    state = torch.load(cache / 'dinov2_12288.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    if any(not torch.equal(value, state[name]) for name, value in model.state_dict().items()):
        raise ValueError('Official checkpoint tensor not reproduced')
    return model.requires_grad_(False).eval().to(device)


def native_feature_layout(features):
    """Restore the author's post-CLS slice layout, including its batch stride.

    This preserves gradient propagation. A plain contiguous conversion can
    choose a different CUDA convolution path despite identical tensor values.
    """
    if features.ndim != 4:
        raise ValueError('Expected BCHW features')
    b, c, h, w = features.shape
    tokens = features.flatten(2).transpose(1, 2)
    padded = F.pad(tokens, (0, 0, 1, 0))
    return padded[:, 1:].transpose(1, 2).reshape(b, c, h, w)


def official_readout(self, features):
    return self.visual.aggregator(native_feature_layout(features))[0]


def build_adapter(visual, initialization_seed):
    from src.models.conv_residual_ablation import build_model
    model = build_model(visual, 'rms_tanh', initialization_seed)
    model.readout = MethodType(official_readout, model)
    return model
