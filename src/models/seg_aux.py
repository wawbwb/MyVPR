"""SegVPR-inspired shared-encoder auxiliary segmentation (not a reproduction)."""

import torch
from torch import nn

from src.core.vpr_framework import VPRFramework
from src.models.query_semantic import QuerySemanticTarget


class SegAuxVPR(VPRFramework):
    def __init__(self, backbone, aggregator, gate, loss_function, config):
        cfg = config['seg_aux']
        super().__init__(
            backbone, aggregator, loss_function, lr=cfg['lr'],
            weight_decay=cfg['weight_decay'], warmup_steps=0,
            milestones=[], config_dict=config,
        )
        self.semantic_region_gate = gate
        self.mode = cfg['mode']
        self.seg_weight = 0.0 if self.mode == 'vpr_only' else cfg['lambda_seg']
        self.diagnostic_interval = cfg['diagnostic_interval']
        # All arms instantiate the same head, preserving initialization RNG.
        self.seg_head = nn.Sequential(
            nn.Conv2d(backbone.out_channels, 128, 1), nn.GELU(),
            nn.Conv2d(128, 150, 1),
        )
        self.target = QuerySemanticTarget('aligned', 150, cfg['min_confidence'])
        self.backbone.requires_grad_(False)
        self.backbone.num_unfrozen_blocks = cfg['unfrozen_blocks']
        for block in self.backbone.dino.blocks[-cfg['unfrozen_blocks']:]:
            block.requires_grad_(True)
        # This repository's DINO forward returns pre-norm patch features;
        # keep its unused final norm frozen rather than listing dead parameters.
        self.aggregator.requires_grad_(True)
        self.semantic_region_gate.requires_grad_(False)
        self.seg_head.requires_grad_(self.seg_weight > 0)

    def _optimizer_param_groups(self):
        groups = super()._optimizer_param_groups()
        if self.seg_weight:
            groups.append({'params': list(self.seg_head.parameters()),
                           'lr': self.hparams['seg_aux']['head_lr'],
                           'weight_decay': self.weight_decay})
        return groups

    def features_and_descriptor(self, images):
        features = self.backbone(images)
        if not isinstance(features, torch.Tensor) or features.ndim != 4:
            raise ValueError('SegAux requires a DINO spatial tensor, no CLS tuple')
        gated, _, _ = self.semantic_region_gate(features)
        output = self.aggregator(gated)
        descriptor = output[0] if isinstance(output, (tuple, list)) else output
        return features, descriptor

    def forward(self, images):
        # Segmentation head is absent from the inference computation.
        return self.features_and_descriptor(images)[1]

    def semantic_weight_at_step(self, step):
        ramp = int(self.hparams['seg_aux'].get('seg_ramp_steps', 0))
        return self.seg_weight * (min(1.0, (step + 1) / ramp) if ramp else 1.0)

    def training_step(self, batch, batch_idx):
        images, place_labels, metadata = batch
        images = images.flatten(0, 1)
        features, descriptors = self.features_and_descriptor(images)
        vpr_loss, accuracy = self.compute_loss(descriptors, place_labels.flatten())
        self.log('vpr_loss', vpr_loss, prog_bar=True)
        self.log('batch_acc', accuracy, prog_bar=True)
        total = vpr_loss
        if self.seg_weight:
            effective_weight = self.semantic_weight_at_step(self.global_step)
            self.log('effective_seg_weight', effective_weight)
            # Deliberately NO detach: CE must update the shared DINO blocks.
            logits = self.seg_head(features)
            seg_loss, stats = self.target(
                logits,
                metadata['query_semantic_labels'].flatten(0, 1),
                metadata['query_semantic_confidence'].flatten(0, 1),
                metadata['query_semantic_cache_indices'].flatten(),
            )
            self.log('seg_loss', seg_loss, prog_bar=True)
            self.log_dict(stats)
            total = total + effective_weight * seg_loss
            if self.global_step % self.diagnostic_interval == 0:
                # Probe a shared parameter, not the semantic head. These are
                # unscaled gradients; autograd.grad does not populate .grad.
                probe = next(self.backbone.dino.blocks[-1].parameters())
                gv = torch.autograd.grad(vpr_loss, probe, retain_graph=True)[0].float()
                gs = torch.autograd.grad(seg_loss, probe, retain_graph=True)[0].float()
                nv, ns = gv.norm(), gs.norm()
                if not torch.isfinite(nv + ns):
                    raise RuntimeError('Nonfinite shared gradient; stop and inspect')
                if self.global_step == 0 and ns.item() == 0:
                    raise RuntimeError('Semantic loss does not reach shared backbone')
                self.log('seg_shared_grad_norm', ns)
                self.log('gradient_probe_step', float(self.global_step))
                self.log('vpr_shared_grad_norm', nv)
                self.log('weighted_seg_vpr_grad_ratio', effective_weight * ns / nv.clamp_min(1e-12))
                self.log('seg_vpr_grad_cosine', (gv * gs).sum() / (nv * ns).clamp_min(1e-12))
        if not torch.isfinite(total):
            raise RuntimeError('Nonfinite SegAux loss')
        self.log('loss', total, prog_bar=True)
        return total

    def on_before_optimizer_step(self, optimizer):
        # Runner uses FP32; a failed step must not silently become a valid run.
        for name, parameter in self.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f'Nonfinite gradient: {name}')
