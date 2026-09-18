"""Causal 1-D deformable temporal aggregation.

InternImage/DCNv3 idea transplanted to the streaming time axis: instead of a
fixed conv kernel, each frame predicts WHERE in its past to look (continuous
offsets, linearly interpolated) and HOW MUCH each sampled point matters
(softmax modulation), per channel group.

Causality: offsets are squashed to (-max_offset, 0], so a frame can only
aggregate the past. Streaming therefore only needs a rolling cache of the
last `max_offset` frames.

Shapes:
    input  x:   [B, T, D]
    output y:   [B, T, D]
    cache:      [B, <=max_offset, D]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class CausalDeformableTemporalLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_points: int = 4,     # K sampling points per group
        num_groups: int = 4,     # G channel groups (DCNv3-style)
        max_offset: int = 16,    # farthest look-back, in 25 Hz frames (16 = 640 ms)
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_groups == 0, (d_model, num_groups)
        self.d_model = d_model
        self.num_points = num_points
        self.num_groups = num_groups
        self.max_offset = max_offset
        self.d_group = d_model // num_groups
        ffn_dim = ffn_dim or 2 * d_model

        # Query features from a short causal depthwise conv (current + 2 past frames)
        self.query_conv = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, padding=0)
        self.offset_proj = nn.Linear(d_model, num_groups * num_points)
        self.modul_proj = nn.Linear(d_model, num_groups * num_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)

        # Bias init: spread initial look-back positions like a dilated kernel
        with torch.no_grad():
            init = torch.linspace(-2.0, 2.0, num_points).repeat(num_groups)
            self.offset_proj.bias.copy_(init)
            self.offset_proj.weight.mul_(0.1)
            self.modul_proj.weight.mul_(0.1)
            self.modul_proj.bias.zero_()

    def forward(
        self, x: torch.Tensor, cache: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, D] current frames (queries).
        cache: [B, T_c, D] previous frames (values only), T_c <= max_offset.
        Returns (y [B, T, D], new_cache [B, min(S,max_offset), D]).
        """
        B, T, D = x.shape
        G, K, Dg = self.num_groups, self.num_points, self.d_group
        if cache is None:
            cache = x.new_zeros(B, 0, D)
        Tc = cache.size(1)
        ctx = torch.cat([cache, x], dim=1)  # [B, S, D]
        S = ctx.size(1)

        # ---- query features (causal conv over ctx, sliced to the T queries) ----
        q = F.pad(ctx.transpose(1, 2), (2, 0))            # [B, D, S+2]
        q = self.query_conv(q).transpose(1, 2)            # [B, S, D]
        q = F.gelu(q[:, Tc:Tc + T])                       # [B, T, D]

        # ---- offsets in (-max_offset, 0], modulation softmax over K points ----
        off = -self.max_offset * torch.sigmoid(self.offset_proj(q))          # [B, T, G*K]
        off = off.view(B, T, G, K)
        m = self.modul_proj(q).view(B, T, G, K).softmax(dim=-1)              # [B, T, G, K]

        # ---- values, grouped ----
        v = self.value_proj(ctx).view(B, S, G, Dg).permute(0, 2, 1, 3)       # [B, G, S, Dg]

        # ---- absolute (fractional) sampling positions inside ctx ----
        base = torch.arange(T, device=x.device, dtype=x.dtype) + Tc          # [T]
        pos = base.view(1, T, 1, 1) + off                                    # [B, T, G, K]
        pos = pos.clamp(min=0.0, max=S - 1)

        p0 = pos.floor()
        w = (pos - p0).permute(0, 2, 1, 3).unsqueeze(-1)                     # [B, G, T, K, 1]
        p0 = p0.long()
        p1 = (p0 + 1).clamp(max=S - 1)

        def gather(p: torch.Tensor) -> torch.Tensor:
            # p: [B, T, G, K] int -> sampled [B, G, T, K, Dg]
            idx = p.permute(0, 2, 1, 3).reshape(B, G, T * K, 1).expand(-1, -1, -1, Dg)
            return torch.gather(v, 2, idx).view(B, G, T, K, Dg)

        sampled = gather(p0) * (1.0 - w) + gather(p1) * w                    # [B, G, T, K, Dg]

        agg = (sampled * m.permute(0, 2, 1, 3).unsqueeze(-1)).sum(dim=3)     # [B, G, T, Dg]
        agg = agg.permute(0, 2, 1, 3).reshape(B, T, D)

        y = self.norm1(x + self.dropout(self.out_proj(agg)))
        y = self.norm2(y + self.dropout(self.ffn(y)))

        new_cache = ctx[:, -self.max_offset:] if self.max_offset > 0 else ctx[:, :0]
        return y, new_cache


class CausalDeformableTemporalBlock(nn.Module):
    """Stack of deformable layers with per-layer streaming caches."""

    def __init__(self, d_model: int, num_layers: int = 2, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [CausalDeformableTemporalLayer(d_model, **layer_kwargs) for _ in range(num_layers)]
        )

    def forward(
        self, x: torch.Tensor, caches: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if caches is None:
            caches = [None] * len(self.layers)
        new_caches = []
        for layer, c in zip(self.layers, caches):
            x, nc = layer(x, cache=c)
            new_caches.append(nc)
        return x, new_caches
