"""V2 supervision only: SAME feature path/tensors, auxiliary after exchange.

Original V1 implementation is immutable. No centering, new projection, extra
encoder, rho increase, or new inference parameters are introduced here.
"""
import torch
from torch.nn import functional as F
from src.models.single_dino_interaction import SingleDinoInteraction, SingleDinoVPR, local_permutation
from src.models.hetero_interleave import interleave, uninterleave


class PostInteractionFusion(SingleDinoInteraction):
    def forward(self, anchor):
        if anchor.ndim != 4:
            raise ValueError('Need innovation-1 enhanced BCHW map')
        local = self.local_projection(anchor)
        local = local + .1*self.local_conv(local)
        global_map = self.global_projection(anchor)
        g = self.scan_tokens(global_map.flatten(2).transpose(1, 2), self.context_norm, self.context_scan)
        context = g.transpose(1, 2).reshape_as(global_map)
        l = local.flatten(2).transpose(1, 2)
        if self.arm == 'separate':
            g = self.scan_tokens(g, self.exchange_norm, self.exchange_scan)
            l = self.scan_tokens(l, self.exchange_norm, self.exchange_scan)
        else:
            if self.arm == 'permuted':
                permutation = local_permutation(*anchor.shape[-2:], anchor.device)
                l = l[:, permutation]
            g, l = uninterleave(self.scan_tokens(interleave(g, l), self.exchange_norm, self.exchange_scan))
            if self.arm == 'permuted':
                l = l[:, permutation.argsort()]
        # Only changed readout: both endpoints AFTER exchange, same normalized
        # GAP dimensions. Main descriptor path below remains exactly V1.
        auxiliary = (F.normalize(g.mean(1), dim=-1), F.normalize(l.mean(1), dim=-1))
        joint = torch.cat((g.transpose(1, 2).reshape_as(local), l.transpose(1, 2).reshape_as(local)), 1)
        gate = self.dynamic(torch.cat((context.mean((2, 3)), local.mean((2, 3))), 1)).sigmoid()
        mixed = joint + self.low_rank_up(F.silu(self.down(joint))*gate[:, :, None, None])
        fused, refine_gate = self.refine(self.fuse(mixed))
        residual = self.rho*anchor.square().mean(1, keepdim=True).sqrt().detach()*self.output(fused).tanh()
        stats = dict(relative_residual=residual.detach().flatten(1).norm(dim=1)/anchor.flatten(1).norm(dim=1).clamp_min(1e-8),
            dynamic_gate=gate.detach().mean(1), refine_gate=refine_gate.detach().mean((1, 2, 3)))
        return anchor+residual, auxiliary, stats


class PostInteractionVPR(SingleDinoVPR):
    def __init__(self, anchor, arm, **kwargs):
        # One construction, not instantiate/discard a second randomly drawn head.
        torch.nn.Module.__init__(self)
        self.anchor = anchor.requires_grad_(False).eval()
        self.fusion = PostInteractionFusion(arm, **kwargs)
