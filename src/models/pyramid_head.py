"""Causal multiscale ordinal segmentation head.

This module deliberately lives beside ``heads.py`` instead of changing any of
the established heads.  The old model/config paths therefore retain their
parameter names and numerical behaviour.
"""

from typing import List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import CausalDualDilationBlock


class GatedPyramidOutput(NamedTuple):
    """Stable output contract for the pyramid decoder.

    The first three fields intentionally match every existing ZipCount head.
    ``count_logits`` is either unconstrained four-way logits (``softmax`` mode)
    or normalized log class probabilities (``corn`` mode); both are valid
    inputs to ``cross_entropy``.  Boundary direction is conditional: positive
    means UP and negative means DOWN, and should only be supervised/evaluated
    on boundary frames.
    """

    count_logits: torch.Tensor
    vad_logits: torch.Tensor
    overlap_logits: torch.Tensor
    stage1_count_logits: torch.Tensor
    count_log_probs: torch.Tensor
    stage1_count_log_probs: torch.Tensor
    ordinal_logits: torch.Tensor
    stage1_ordinal_logits: torch.Tensor
    boundary_logits: torch.Tensor
    direction_logits: torch.Tensor
    stage1_boundary_logits: torch.Tensor
    stage1_direction_logits: torch.Tensor
    fusion_gates: torch.Tensor


class GatedPyramidCache(NamedTuple):
    """All state required by chunked causal decoding."""

    stage1_caches: List[torch.Tensor]
    stage2_caches: List[torch.Tensor]
    prev_edge_hidden: torch.Tensor


def _validated_stack_input_mask(
    stack_input_mask: Optional[Sequence[int]], num_stacks: int
) -> Tuple[int, ...]:
    """Validate a stack view mask with the same rules as ``heads.py``."""

    if stack_input_mask is None:
        return tuple([1] * num_stacks)
    if len(stack_input_mask) != num_stacks:
        raise ValueError(
            "stack_input_mask must have one entry per stack: "
            f"expected {num_stacks}, got {len(stack_input_mask)}"
        )
    if any(
        not isinstance(value, (bool, int, float)) or value not in (0, 1)
        for value in stack_input_mask
    ):
        raise ValueError(
            "stack_input_mask entries must be numeric 0/1 values, "
            f"got {list(stack_input_mask)}"
        )
    mask = tuple(int(value) for value in stack_input_mask)
    if not any(mask):
        raise ValueError("stack_input_mask must keep at least one stack")
    return mask


class FramewisePyramidNeck(nn.Module):
    """Project and adaptively fuse the six Zipformer stack outputs.

    Each stack has its own LayerNorm/lateral projection because the encoder
    levels differ in width and activation statistics.  The optional final
    Zipformer output is not a seventh gate candidate: it is projected through
    a separate residual path, preserving the encoder's own learned U-Net mix.
    All operations are framewise, so this neck adds no temporal lookahead.

    ``stack_input_mask`` suppresses selected stack *views* without removing
    modules, parameters, or persistent state, so masked and unmasked builds
    stay byte-identical in state_dict schema and seeded initialization.
    Unlike the concat head in ``heads.py`` (mask post-norm, pre-projection),
    the mask here zeroes the stack's *projected* feature: every stack owns a
    biased lateral projection, and zeroing before it would leak
    ``silu(bias)`` into the fusion as a learnable constant channel.
    """

    def __init__(
        self,
        stack_dims: Sequence[int],
        d_model: int,
        final_dim: Optional[int] = None,
        use_framewise_gate: bool = True,
        gate_mode: Optional[str] = None,
        dropout: float = 0.1,
        stack_input_mask: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if len(stack_dims) != 6:
            raise ValueError(
                "gated_pyramid_ordinal requires exactly six Zipformer stacks; "
                f"got {len(stack_dims)} ({list(stack_dims)})"
            )
        if any(int(d) <= 0 for d in stack_dims):
            raise ValueError(f"stack_dims must all be positive, got {list(stack_dims)}")
        if final_dim is not None and int(final_dim) <= 0:
            raise ValueError(f"final_dim must be positive or None, got {final_dim}")

        self.stack_dims = [int(d) for d in stack_dims]
        self.final_dim = int(final_dim) if final_dim is not None else None
        self.d_model = int(d_model)
        self.gate_mode = (
            str(gate_mode).lower()
            if gate_mode is not None
            else ("framewise" if use_framewise_gate else "global")
        )
        if self.gate_mode not in {"framewise", "global", "uniform"}:
            raise ValueError(
                "gate_mode must be framewise, global, or uniform; got "
                f"{gate_mode!r}"
            )
        self.expected_dim = sum(self.stack_dims) + (self.final_dim or 0)
        # Neither a Parameter nor a persistent buffer: matched view ablations
        # require identical checkpoint schema and parameter counts.
        self.stack_input_mask = _validated_stack_input_mask(
            stack_input_mask, len(self.stack_dims)
        )

        self.stack_norms = nn.ModuleList(
            [nn.LayerNorm(d) for d in self.stack_dims]
        )
        self.stack_projections = nn.ModuleList(
            [nn.Linear(d, self.d_model) for d in self.stack_dims]
        )
        self.gate_projections = nn.ModuleList(
            [nn.Linear(self.d_model, 1) for _ in self.stack_dims]
            if self.gate_mode == "framewise"
            else []
        )
        if self.gate_mode == "global":
            self.global_gate_logits = nn.Parameter(
                torch.zeros(len(self.stack_dims))
            )
        else:
            self.register_parameter("global_gate_logits", None)

        if self.final_dim is not None:
            self.final_norm = nn.LayerNorm(self.final_dim)
            self.final_projection = nn.Linear(self.final_dim, self.d_model)
        else:
            self.final_norm = None
            self.final_projection = None

        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.expected_dim:
            raise ValueError(
                f"pyramid input has dim {x.shape[-1]}, expected "
                f"{self.expected_dim} = sum({self.stack_dims})"
                + (
                    f" + final_dim({self.final_dim})"
                    if self.final_dim is not None
                    else ""
                )
            )

        projected: List[torch.Tensor] = []
        offset = 0
        for i, (dim, norm, projection) in enumerate(
            zip(self.stack_dims, self.stack_norms, self.stack_projections)
        ):
            stack = x[..., offset : offset + dim]
            piece = F.silu(projection(norm(stack)))
            if not self.stack_input_mask[i]:
                # Replace, never multiply: NaN * 0 is still NaN.  A disabled
                # stack must contribute an exact zero vector to the fusion and
                # must not poison it even when that backbone slice is
                # non-finite.  ``zeros_like`` also keeps the AMP dtype of the
                # live branches so the stack below stays homogeneous.
                piece = torch.zeros_like(piece)
            projected.append(piece)
            offset += dim

        stacked = torch.stack(projected, dim=-2)  # [B, T, 6, D]
        if self.gate_mode == "framewise":
            scores = torch.cat(
                [
                    gate(stack)
                    for gate, stack in zip(self.gate_projections, projected)
                ],
                dim=-1,
            )  # [B, T, 6]
            weights = scores.softmax(dim=-1)
        elif self.gate_mode == "global":
            assert self.global_gate_logits is not None
            weights = self.global_gate_logits.softmax(dim=-1).view(1, 1, -1)
            weights = weights.expand(
                stacked.shape[0], stacked.shape[1], len(self.stack_dims)
            )
        else:
            weights = stacked.new_full(
                stacked.shape[:-1], 1.0 / len(self.stack_dims)
            )

        fused = (stacked * weights.unsqueeze(-1)).sum(dim=-2)
        if self.final_dim is not None:
            final = x[..., offset : offset + self.final_dim]
            assert self.final_norm is not None and self.final_projection is not None
            fused = fused + self.final_projection(self.final_norm(final))

        return self.output_norm(self.dropout(fused)), weights


class _OrdinalProjection(nn.Module):
    """Four-way softmax or true CORN conditional ordinal projection."""

    def __init__(self, d_model: int, ordinal_mode: str):
        super().__init__()
        mode = str(ordinal_mode).lower()
        if mode not in {"softmax", "corn"}:
            raise ValueError(
                f"ordinal_mode must be 'softmax' or 'corn', got {ordinal_mode!r}"
            )
        self.ordinal_mode = mode
        self.projection = nn.Linear(d_model, 4 if mode == "softmax" else 3)

    @staticmethod
    def _corn_log_probs(conditional_logits: torch.Tensor) -> torch.Tensor:
        """Convert three conditional logits to a valid p(count=0..3).

        q0 = P(Y>0), q1 = P(Y>1 | Y>0), q2 = P(Y>2 | Y>1).
        Computing in log space avoids products underflowing under AMP.
        """

        log_q = F.logsigmoid(conditional_logits.float())
        log_not_q = F.logsigmoid(-conditional_logits.float())
        return torch.stack(
            [
                log_not_q[..., 0],
                log_q[..., 0] + log_not_q[..., 1],
                log_q[..., 0] + log_q[..., 1] + log_not_q[..., 2],
                log_q.sum(dim=-1),
            ],
            dim=-1,
        )

    def decode(self, ordinal_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return public count logits and normalized log class probabilities."""

        if self.ordinal_mode == "softmax":
            return ordinal_logits, F.log_softmax(ordinal_logits.float(), dim=-1)
        log_probs = self._corn_log_probs(ordinal_logits)
        # The construction already sums to one analytically; normalize once to
        # bound accumulated fp error and make this invariant testable.
        log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
        return log_probs, log_probs

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ordinal_logits = self.projection(x)
        count_logits, log_probs = self.decode(ordinal_logits)
        return count_logits, log_probs, ordinal_logits


class _CausalEdgeHead(nn.Module):
    """Boundary and signed-direction prediction from a causal feature edge."""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        edge_dim = 3 * d_model
        self.trunk = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
        )
        self.boundary_projection = nn.Linear(hidden_dim, 1)
        self.direction_projection = nn.Linear(hidden_dim, 1)

    def forward(
        self, hidden: torch.Tensor, previous: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        previous_frames = torch.cat([previous, hidden[:, :-1]], dim=1)
        edge = torch.cat(
            [hidden, previous_frames, hidden - previous_frames], dim=-1
        )
        edge_hidden = self.trunk(edge)
        return (
            self.boundary_projection(edge_hidden),
            self.direction_projection(edge_hidden),
        )


def _threshold_logits(log_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive VAD/OSD logits deterministically from the count distribution."""

    probs = log_probs.exp()
    eps = torch.finfo(probs.dtype).eps
    vad_prob = (1.0 - probs[..., 0:1]).clamp(eps, 1.0 - eps)
    overlap_prob = probs[..., 2:].sum(dim=-1, keepdim=True).clamp(
        eps, 1.0 - eps
    )
    return (
        torch.logit(vad_prob),
        torch.logit(overlap_prob),
    )


class GatedPyramidOrdinalHead(nn.Module):
    """Adaptive pyramid neck + causal multi-stage ordinal decoder.

    Stage 1 supplies both frame count and explicit causal edge predictions.
    Optional stage 2 sees ``[z, stage1 log p, boundary, direction]`` and emits
    residual corrections.  Every correction projection is zero-initialized,
    making the initial public outputs exactly equal to stage 1 without relying
    on a learned gate.  There is intentionally no Markov/state filter.
    """

    def __init__(
        self,
        d_in: int,
        stack_dims: Sequence[int],
        final_dim: Optional[int] = None,
        num_classes: int = 4,
        d_model: int = 192,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        kernel: int = 3,
        dropout: float = 0.1,
        use_framewise_gate: bool = True,
        gate_mode: Optional[str] = None,
        ordinal_mode: str = "corn",
        edge_hidden_dim: int = 128,
        use_stage2: bool = True,
        stack_input_mask: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if num_classes != 4:
            raise ValueError(
                "gated_pyramid_ordinal currently defines count states 0..3 "
                f"and therefore requires num_classes=4, got {num_classes}"
            )
        if len(dilations) != 6:
            raise ValueError(
                "each pyramid decoder stage requires six dilation blocks; "
                f"got {list(dilations)}"
            )
        expected_dim = sum(int(d) for d in stack_dims) + (
            int(final_dim) if final_dim is not None else 0
        )
        if int(d_in) != expected_dim:
            raise ValueError(
                f"d_in={d_in} but stack/final dimensions sum to {expected_dim}"
            )

        self.d_in = int(d_in)
        self.d_model = int(d_model)
        self.ordinal_mode = str(ordinal_mode).lower()
        self.use_stage2 = bool(use_stage2)

        self.neck = FramewisePyramidNeck(
            stack_dims=stack_dims,
            d_model=self.d_model,
            final_dim=final_dim,
            use_framewise_gate=use_framewise_gate,
            gate_mode=gate_mode,
            dropout=dropout,
            stack_input_mask=stack_input_mask,
        )
        self.stage1_blocks = nn.ModuleList(
            [
                CausalDualDilationBlock(self.d_model, int(d), kernel, dropout)
                for d in dilations
            ]
        )
        self.stage1_ordinal = _OrdinalProjection(
            self.d_model, self.ordinal_mode
        )
        self.stage1_edge = _CausalEdgeHead(self.d_model, edge_hidden_dim)

        if self.use_stage2:
            refinement_dim = self.d_model + 4 + 1 + 1
            self.stage2_input = nn.Sequential(
                nn.LayerNorm(refinement_dim),
                nn.Linear(refinement_dim, self.d_model),
                nn.SiLU(),
            )
            self.stage2_blocks = nn.ModuleList(
                [
                    CausalDualDilationBlock(
                        self.d_model, int(d), kernel, dropout
                    )
                    for d in dilations
                ]
            )
            ordinal_width = 4 if self.ordinal_mode == "softmax" else 3
            self.ordinal_correction = nn.Linear(
                self.d_model, ordinal_width
            )
            self.boundary_correction = nn.Linear(self.d_model, 1)
            self.direction_correction = nn.Linear(self.d_model, 1)
            for correction in (
                self.ordinal_correction,
                self.boundary_correction,
                self.direction_correction,
            ):
                nn.init.zeros_(correction.weight)
                nn.init.zeros_(correction.bias)
        else:
            self.stage2_input = None
            self.stage2_blocks = nn.ModuleList()
            self.ordinal_correction = None
            self.boundary_correction = None
            self.direction_correction = None

    @staticmethod
    def _run_blocks(
        hidden: torch.Tensor, blocks: nn.ModuleList
    ) -> torch.Tensor:
        for block in blocks:
            hidden = block(hidden)
        return hidden

    @staticmethod
    def _init_block_caches(
        hidden: torch.Tensor, blocks: nn.ModuleList
    ) -> List[torch.Tensor]:
        return [
            hidden.new_zeros(hidden.shape[0], block.pad_ctx, hidden.shape[-1])
            for block in blocks
        ]

    @staticmethod
    def _run_blocks_streaming(
        hidden: torch.Tensor,
        blocks: nn.ModuleList,
        caches: Optional[List[torch.Tensor]],
        cache_name: str,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if caches is None:
            caches = GatedPyramidOrdinalHead._init_block_caches(hidden, blocks)
        if len(caches) != len(blocks):
            raise ValueError(
                f"{cache_name} has {len(caches)} entries, expected {len(blocks)}"
            )
        new_caches: List[torch.Tensor] = []
        for block, cache in zip(blocks, caches):
            hidden, new_cache = block.forward_chunk(hidden, cache)
            new_caches.append(new_cache)
        return hidden, new_caches

    def _finish(
        self,
        z: torch.Tensor,
        stage1_hidden: torch.Tensor,
        previous_edge_hidden: torch.Tensor,
        fusion_gates: torch.Tensor,
        stage2_caches: Optional[List[torch.Tensor]] = None,
        streaming: bool = False,
    ) -> Tuple[GatedPyramidOutput, List[torch.Tensor]]:
        (
            stage1_count_logits,
            stage1_log_probs,
            stage1_ordinal_logits,
        ) = self.stage1_ordinal(stage1_hidden)
        stage1_boundary, stage1_direction = self.stage1_edge(
            stage1_hidden, previous_edge_hidden
        )

        new_stage2_caches: List[torch.Tensor] = []
        if self.use_stage2:
            assert self.stage2_input is not None
            assert self.ordinal_correction is not None
            assert self.boundary_correction is not None
            assert self.direction_correction is not None
            refinement = self.stage2_input(
                torch.cat(
                    [
                        z,
                        stage1_log_probs.to(z.dtype),
                        stage1_boundary.to(z.dtype),
                        stage1_direction.to(z.dtype),
                    ],
                    dim=-1,
                )
            )
            if streaming:
                refinement, new_stage2_caches = self._run_blocks_streaming(
                    refinement,
                    self.stage2_blocks,
                    stage2_caches,
                    "stage2_caches",
                )
            else:
                refinement = self._run_blocks(refinement, self.stage2_blocks)

            ordinal_logits = (
                stage1_ordinal_logits
                + self.ordinal_correction(refinement).to(
                    stage1_ordinal_logits.dtype
                )
            )
            boundary = stage1_boundary + self.boundary_correction(
                refinement
            ).to(stage1_boundary.dtype)
            direction = stage1_direction + self.direction_correction(
                refinement
            ).to(stage1_direction.dtype)
            count_logits, log_probs = self.stage1_ordinal.decode(
                ordinal_logits
            )
        else:
            ordinal_logits = stage1_ordinal_logits
            count_logits = stage1_count_logits
            log_probs = stage1_log_probs
            boundary = stage1_boundary
            direction = stage1_direction

        vad_logits, overlap_logits = _threshold_logits(log_probs)
        return (
            GatedPyramidOutput(
                count_logits=count_logits,
                vad_logits=vad_logits,
                overlap_logits=overlap_logits,
                stage1_count_logits=stage1_count_logits,
                count_log_probs=log_probs,
                stage1_count_log_probs=stage1_log_probs,
                ordinal_logits=ordinal_logits,
                stage1_ordinal_logits=stage1_ordinal_logits,
                boundary_logits=boundary,
                direction_logits=direction,
                stage1_boundary_logits=stage1_boundary,
                stage1_direction_logits=stage1_direction,
                fusion_gates=fusion_gates,
            ),
            new_stage2_caches,
        )

    def forward(self, x: torch.Tensor) -> GatedPyramidOutput:
        z, fusion_gates = self.neck(x)
        stage1_hidden = self._run_blocks(z, self.stage1_blocks)
        previous = stage1_hidden.new_zeros(
            stage1_hidden.shape[0], 1, stage1_hidden.shape[-1]
        )
        output, _ = self._finish(
            z, stage1_hidden, previous, fusion_gates
        )
        return output

    def forward_streaming(
        self,
        x: torch.Tensor,
        cache: Optional[GatedPyramidCache] = None,
    ) -> Tuple[GatedPyramidOutput, GatedPyramidCache]:
        z, fusion_gates = self.neck(x)
        if cache is None:
            stage1_caches = None
            stage2_caches = None
            previous = z.new_zeros(z.shape[0], 1, z.shape[-1])
        else:
            stage1_caches = cache.stage1_caches
            stage2_caches = cache.stage2_caches
            previous = cache.prev_edge_hidden
            if previous.shape != (z.shape[0], 1, z.shape[-1]):
                raise ValueError(
                    "prev_edge_hidden has shape "
                    f"{tuple(previous.shape)}, expected "
                    f"{(z.shape[0], 1, z.shape[-1])}"
                )

        stage1_hidden, new_stage1_caches = self._run_blocks_streaming(
            z,
            self.stage1_blocks,
            stage1_caches,
            "stage1_caches",
        )
        output, new_stage2_caches = self._finish(
            z,
            stage1_hidden,
            previous,
            fusion_gates,
            stage2_caches=stage2_caches,
            streaming=True,
        )
        new_cache = GatedPyramidCache(
            stage1_caches=new_stage1_caches,
            stage2_caches=new_stage2_caches,
            prev_edge_hidden=stage1_hidden[:, -1:].contiguous(),
        )
        return output, new_cache
