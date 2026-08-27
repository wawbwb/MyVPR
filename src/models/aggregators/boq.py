r"""
    BoQ: A Place is Worth a Bag of learnable Queries (CVPR 2024)
    
    Paper: https://arxiv.org/abs/2405.07364
    GitHub repo: https://github.com/amaralibey/Bag-of-Queries
    
    Reference:
    @InProceedings{Ali-bey_2024_CVPR,
        author    = {Ali-bey, Amar and Chaib-draa, Brahim and Gigu\`ere, Philippe},
        title     = {{BoQ}: A Place is Worth a Bag of Learnable Queries},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2024},
        pages     = {17794-17803}
    }
    
"""


import math

import torch

class BoQBlock(torch.nn.Module):
    """
    BoQ Block

    Args:
        in_dim (int): input dimension
        num_queries (int): number of queries to learn
        nheads (int): number of heads in the multihead attention. Defaults to 8.
    """
    def __init__(
        self,
        in_dim,
        num_queries,
        nheads=8,
        semantic_num_classes=None,
        semantic_bias_scale=0.2,
    ):
        super(BoQBlock, self).__init__()
        if semantic_num_classes is not None and semantic_num_classes < 2:
            raise ValueError("semantic_num_classes must be at least 2")
        if not math.isfinite(float(semantic_bias_scale)) or semantic_bias_scale < 0:
            raise ValueError("semantic_bias_scale must be finite and non-negative")
        
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        
        # the following two lines are used to add context between the learned queries during training. 
        # you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        ################
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        self.nheads = int(nheads)
        self.semantic_num_classes = semantic_num_classes
        self.semantic_bias_scale = float(semantic_bias_scale)
        self._semantic_start_unchecked = semantic_num_classes is not None
        self._last_semantic_diagnostics = None
        self.semantic_query_proj = None
        if semantic_num_classes is not None:
            self.semantic_query_proj = torch.nn.Linear(
                in_dim, semantic_num_classes
            )
            # The extended model must reproduce a historical BoQ/RU checkpoint
            # exactly before it sees a training update.
            torch.nn.init.zeros_(self.semantic_query_proj.weight)
            torch.nn.init.zeros_(self.semantic_query_proj.bias)

    def _query_semantic_bias(self, q, semantic_probabilities):
        """Build a bounded, query-specific additive attention-logit bias.

        Args:
            q: Learned query states with shape ``(B, Q, D)``.
            semantic_probabilities: Student class probabilities with shape
                ``(B, N, C)``.

        Returns:
            A signed bias with shape ``(B, Q, N)``. Every query is centred
            across its keys and its absolute value is bounded by
            ``semantic_bias_scale``.
        """
        if self.semantic_query_proj is None:
            if semantic_probabilities is not None:
                raise ValueError(
                    "semantic probabilities were supplied to a BoQ block "
                    "without semantic conditioning"
                )
            return None
        if semantic_probabilities is None:
            raise ValueError(
                "semantic-conditioned BoQ requires semantic probabilities"
            )
        expected = (q.shape[0], semantic_probabilities.shape[1], self.semantic_num_classes)
        if tuple(semantic_probabilities.shape) != expected:
            raise ValueError(
                "semantic_probabilities must have shape "
                f"{expected}, got {tuple(semantic_probabilities.shape)}"
            )
        if not bool(torch.isfinite(semantic_probabilities).all()):
            raise ValueError("semantic_probabilities must be finite")

        preferences = torch.tanh(self.semantic_query_proj(q))
        preferences = preferences - preferences.mean(dim=-1, keepdim=True)
        semantic_probabilities = semantic_probabilities.to(
            device=preferences.device, dtype=preferences.dtype
        )
        scores = torch.einsum(
            "bqc,bnc->bqn", preferences, semantic_probabilities
        )
        scores = scores - scores.mean(dim=-1, keepdim=True)
        normalizer = scores.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        return self.semantic_bias_scale * scores / normalizer
        

    def forward(
        self,
        x,
        attention_bias=None,
        semantic_probabilities=None,
    ):
        """Run one BoQ block with an optional per-key additive bias.

        ``attention_bias`` is added to the cross-attention logits through
        ``MultiheadAttention.key_padding_mask``.  It must have shape ``(B, N)``
        where ``N`` is the number of spatial input tokens.  Negative values
        suppress keys and zero leaves their logits unchanged.  The encoder and
        learned-query self-attention remain untouched.
        """
        B = x.size(0)
        x = self.encoder(x)
        
        q = self.queries.repeat(B, 1, 1)
        
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)

        if (
            semantic_probabilities is not None
            and semantic_probabilities.shape[1] != x.size(1)
        ):
            raise ValueError(
                "semantic probabilities and BoQ features must contain the "
                "same number of spatial tokens"
            )
        
        semantic_bias = self._query_semantic_bias(q, semantic_probabilities)
        if semantic_bias is None:
            self._last_semantic_diagnostics = None
        else:
            diagnostic_bias = semantic_bias.detach().float()
            self._last_semantic_diagnostics = {
                "bias_std": diagnostic_bias.std(unbiased=False),
                "bias_abs_max": diagnostic_bias.abs().amax(),
                "query_std": diagnostic_bias.std(
                    dim=1, unbiased=False
                ).mean(),
                "spatial_std": diagnostic_bias.std(
                    dim=-1, unbiased=False
                ).mean(),
            }

        if attention_bias is not None:
            expected = (B, x.size(1))
            if tuple(attention_bias.shape) != expected:
                raise ValueError(
                    f"attention_bias must have shape {expected}, got "
                    f"{tuple(attention_bias.shape)}"
                )
            attention_bias = attention_bias.to(device=x.device, dtype=x.dtype)
            if not bool(torch.isfinite(attention_bias).all()):
                raise ValueError("attention_bias must contain only finite values")
            if bool((attention_bias > 0).any()):
                raise ValueError(
                    "attention_bias is a negative prior and must be non-positive"
                )

        if semantic_bias is None:
            if attention_bias is None:
                # Keep the historical no-prior path byte-for-byte equivalent.
                out, attn = self.cross_attn(q, x, x)
            else:
                out, attn = self.cross_attn(
                    q, x, x, key_padding_mask=attention_bias
                )
        else:
            semantic_bias = semantic_bias.to(device=x.device, dtype=x.dtype)
            attention_mask = (
                semantic_bias[:, None]
                .expand(B, self.nheads, q.size(1), x.size(1))
                .reshape(B * self.nheads, q.size(1), x.size(1))
            )
            semantic_is_zero = False
            if self._semantic_start_unchecked:
                semantic_is_zero = not bool(
                    torch.count_nonzero(
                        self.semantic_query_proj.weight.detach()
                    )
                    + torch.count_nonzero(
                        self.semantic_query_proj.bias.detach()
                    )
                )
                # A freshly initialized training run needs the compatibility
                # branch only for its first forward. A loaded trained adapter
                # is detected as non-zero here and never pays this sync again.
                if self.training or not semantic_is_zero:
                    self._semantic_start_unchecked = False
            if semantic_is_zero:
                # A zero mask can select a different CUDA MHA kernel than no
                # mask. Return the exact historical value while using a
                # zero-valued straight-through term to give the adapters a
                # useful gradient on their first update.
                if attention_bias is None:
                    legacy_out, legacy_attn = self.cross_attn(q, x, x)
                else:
                    legacy_out, legacy_attn = self.cross_attn(
                        q, x, x, key_padding_mask=attention_bias
                    )
                if torch.is_grad_enabled() and semantic_bias.requires_grad:
                    semantic_out, _ = self.cross_attn(
                        q,
                        x,
                        x,
                        key_padding_mask=attention_bias,
                        attn_mask=attention_mask,
                    )
                    out = legacy_out + (
                        semantic_out - semantic_out.detach()
                    )
                else:
                    out = legacy_out
                attn = legacy_attn
            else:
                out, attn = self.cross_attn(
                    q,
                    x,
                    x,
                    key_padding_mask=attention_bias,
                    attn_mask=attention_mask,
                )
        out = self.norm_out(out)
        return x, out, attn.detach()


class BoQ(torch.nn.Module):
    """
    Bag-of-Queries module

    Args:
        in_channels (int): Number of input channels (depth of input feature maps).
        proj_channels (int): Number of channels after the projection layer. Defaults to 512.
        num_queries (int): Number of queries to learn. Defaults to 32.
        num_layers (int): Number of BoQ blocks. Defaults to 2.
        row_dim (int): Row-wise projection dimension. Defaults to 32.
    """
    def __init__(
        self,
        in_channels=1024,
        proj_channels=512,
        num_queries=32,
        num_layers=2,
        row_dim=32,
        semantic_num_classes=None,
        semantic_bias_scale=0.2,
        semantic_temperature=1.0,
        semantic_head_hidden=256,
    ):
        super().__init__()
        if semantic_num_classes is not None and semantic_num_classes < 2:
            raise ValueError("semantic_num_classes must be at least 2")
        if not math.isfinite(float(semantic_bias_scale)) or semantic_bias_scale < 0:
            raise ValueError("semantic_bias_scale must be finite and non-negative")
        if not math.isfinite(float(semantic_temperature)) or semantic_temperature <= 0:
            raise ValueError("semantic_temperature must be finite and positive")
        if semantic_num_classes is not None and semantic_head_hidden < 1:
            raise ValueError("semantic_head_hidden must be positive")
        self.semantic_num_classes = semantic_num_classes
        self.semantic_bias_scale = float(semantic_bias_scale)
        self.semantic_temperature = float(semantic_temperature)
        self.semantic_head = None
        if semantic_num_classes is not None:
            self.semantic_head = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels, semantic_head_hidden, kernel_size=1, bias=False
                ),
                torch.nn.GELU(),
                torch.nn.Conv2d(
                    semantic_head_hidden,
                    semantic_num_classes,
                    kernel_size=1,
                ),
            )
        
        # reduce input dimension using 3x3 conv
        self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        
        # normalize the input to the BoQ blocks
        self.norm_input = torch.nn.LayerNorm(proj_channels)
        
        # now the BoQ blocks input dimension is proj_channels
        boq_in_dim = proj_channels
        
        # create the BoQ blocks (each head of the self-attention has a dimension of 64)
        self.boqs = torch.nn.ModuleList([
            BoQBlock(
                boq_in_dim,
                num_queries,
                nheads=boq_in_dim // 64,
                semantic_num_classes=semantic_num_classes,
                semantic_bias_scale=semantic_bias_scale,
            )
            for _ in range(num_layers)
        ])
        
        # the outputs of all BoQ blocks are concatenated and projected to row_dim
        self.fc = torch.nn.Linear(num_layers*num_queries, row_dim)
        
    @staticmethod
    def _flatten_attention_bias(attention_bias, batch_size, height, width):
        """Validate and flatten a spatial additive attention bias."""

        if attention_bias is None:
            return None
        if attention_bias.ndim == 4:
            if attention_bias.shape[1] != 1:
                raise ValueError(
                    "4D attention_bias must have shape (B, 1, H, W)"
                )
            attention_bias = attention_bias[:, 0]
        if attention_bias.ndim == 3:
            if tuple(attention_bias.shape) != (batch_size, height, width):
                raise ValueError(
                    "3D attention_bias must match the BoQ input spatial shape "
                    f"{(batch_size, height, width)}, got "
                    f"{tuple(attention_bias.shape)}"
                )
            attention_bias = attention_bias.flatten(1)
        elif attention_bias.ndim == 2:
            expected = (batch_size, height * width)
            if tuple(attention_bias.shape) != expected:
                raise ValueError(
                    "2D attention_bias must have shape "
                    f"{expected}, got {tuple(attention_bias.shape)}"
                )
        else:
            raise ValueError(
                "attention_bias must have shape (B, N), (B, H, W), or "
                "(B, 1, H, W)"
            )
        return attention_bias

    def semantic_parameters(self):
        """Yield only query-conditioning parameters for frozen-base screens."""
        if self.semantic_head is not None:
            yield from self.semantic_head.parameters()
        for block in self.boqs:
            if block.semantic_query_proj is not None:
                yield from block.semantic_query_proj.parameters()

    def predict_semantics(self, x):
        """Predict local class logits without requiring a teacher at inference."""
        if self.semantic_head is None:
            raise RuntimeError("this BoQ instance has no semantic student head")
        if x.ndim != 4:
            raise ValueError("semantic student input must have shape (B,C,H,W)")
        return self.semantic_head(x)

    def semantic_diagnostics(self):
        """Return detached evidence that semantics changes query attention."""
        diagnostics = [
            block._last_semantic_diagnostics
            for block in self.boqs
            if block._last_semantic_diagnostics is not None
        ]
        if not diagnostics:
            return {}
        output = {
            f"query_semantic_{name}": torch.stack(
                [values[name] for values in diagnostics]
            ).mean()
            for name in (
                "bias_std",
                "bias_abs_max",
                "query_std",
                "spatial_std",
            )
        }
        parameters = [
            parameter.detach().float()
            for block in self.boqs
            for parameter in block.semantic_query_proj.parameters()
        ]
        square_sum = sum(parameter.square().sum() for parameter in parameters)
        element_count = sum(parameter.numel() for parameter in parameters)
        output["query_semantic_adapter_rms"] = (
            square_sum / element_count
        ).sqrt()
        return output

    def forward(self, x, attention_bias=None, semantic_logits=None):
        """Aggregate a feature map, optionally suppressing spatial keys.

        Args:
            x: Feature map with shape ``(B, C, H, W)``.
            attention_bias: Optional additive cross-attention-logit bias with
                shape ``(B, H, W)``, ``(B, 1, H, W)``, or ``(B, H*W)``.
                Negative values reduce attention to the corresponding patch.
            semantic_logits: Student semantic logits with shape ``(B, C, H, W)``.
                When semantic conditioning is configured these are converted to
                probabilities and combined with each learned query's independent
                class preference.

        Existing callers omit ``attention_bias`` and therefore execute the
        original BoQ path exactly.
        """
        batch_size, _, height, width = x.shape
        attention_bias = self._flatten_attention_bias(
            attention_bias,
            batch_size=batch_size,
            height=height,
            width=width,
        )

        semantic_probabilities = None
        if self.semantic_num_classes is None:
            if semantic_logits is not None:
                raise ValueError(
                    "semantic_logits were supplied to a BoQ model without "
                    "semantic conditioning"
                )
        else:
            if semantic_logits is None:
                semantic_logits = self.predict_semantics(x)
            expected = (
                batch_size,
                self.semantic_num_classes,
                height,
                width,
            )
            if tuple(semantic_logits.shape) != expected:
                raise ValueError(
                    f"semantic_logits must have shape {expected}, got "
                    f"{tuple(semantic_logits.shape)}"
                )
            if not bool(torch.isfinite(semantic_logits).all()):
                raise ValueError("semantic_logits must contain only finite values")
            semantic_probabilities = (
                semantic_logits.float() / self.semantic_temperature
            ).softmax(dim=1)
            semantic_probabilities = semantic_probabilities.flatten(2).transpose(1, 2)

        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)

        if attention_bias is not None:
            attention_bias = attention_bias.to(device=x.device, dtype=x.dtype)
            if not bool(torch.isfinite(attention_bias).all()):
                raise ValueError("attention_bias must contain only finite values")
            if bool((attention_bias > 0).any()):
                raise ValueError(
                    "attention_bias is a negative prior and must be non-positive"
                )
        
        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](
                x,
                attention_bias=attention_bias,
                semantic_probabilities=semantic_probabilities,
            )
            outs.append(out)
            attns.append(attn)

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns
