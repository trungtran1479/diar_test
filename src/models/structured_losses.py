"""Structured objectives for causal ordinal speaker-count segmentation.

This module is deliberately separate from :mod:`src.models.losses`.  The
legacy ``ZipCountLoss`` remains the default for every existing experiment;
these objectives are selected only with ``loss.type: pyramid_structured``.

Two conventions are important throughout:

* ``valid_frame[b, t]`` says that frame ``t`` has both encoder support and a
  real label.
* ``valid_pair[b, t]`` says that the transition ``t-1 -> t`` is valid.  Its
  first column is always false.  Boundary, direction, delta, and temporal
  smoothness all use this one shared definition, so an invalid gap can never
  manufacture a transition.

All probability-space calculations run in float32.  Invalid entries are
replaced with ``torch.where`` before nonlinear operations; multiplying a NaN
by a false mask is not a valid way to remove it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _differentiable_zero(reference: torch.Tensor) -> torch.Tensor:
    """Return an exact float32 zero connected to ``reference``'s graph.

    An empty reduction avoids ``NaN * 0`` if an entirely invalid batch also
    happens to contain non-finite values in padded output rows.
    """

    return reference.float().reshape(-1)[:0].sum()


def _require_btc(
    tensor: torch.Tensor,
    name: str,
    batch: int,
    frames: int,
    channels: Optional[int] = None,
) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a tensor, got {type(tensor).__name__}")
    if tensor.ndim != 3 or tensor.shape[:2] != (batch, frames):
        raise ValueError(
            f"{name} must have shape [B,T,C] with B={batch}, T={frames}; "
            f"got {tuple(tensor.shape)}"
        )
    if channels is not None and tensor.shape[-1] != channels:
        raise ValueError(
            f"{name} must have {channels} channels, got {tensor.shape[-1]}"
        )
    return tensor


def build_valid_masks(
    labels: torch.Tensor,
    h_lens: torch.Tensor,
    expected_T: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build strict frame and transition masks.

    Args:
        labels:
            Integer ``[B,T]`` count labels.  Negative entries are padding or an
            explicitly invalid gap.  Real labels must be in ``0..3``.
        h_lens:
            Integer ``[B]`` encoder lengths, each in ``0..T``.
        expected_T:
            If supplied, require ``labels.shape[1] == expected_T``.  There is
            intentionally no slicing or padding fallback.

    Returns:
        ``(valid_frame, valid_pair)``, both boolean ``[B,T]``.  Pair index
        ``t`` describes the transition ``t-1 -> t`` and index zero is false.
    """

    if not torch.is_tensor(labels) or labels.ndim != 2:
        shape = tuple(labels.shape) if torch.is_tensor(labels) else None
        raise ValueError(f"labels must be an integer [B,T] tensor, got {shape}")
    if labels.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"labels must have integer dtype, got {labels.dtype}")
    batch, frames = labels.shape
    if expected_T is not None and frames != int(expected_T):
        raise ValueError(
            f"labels/output time mismatch: labels T={frames}, output T={expected_T}"
        )
    if not torch.is_tensor(h_lens) or h_lens.ndim != 1 or h_lens.shape[0] != batch:
        shape = tuple(h_lens.shape) if torch.is_tensor(h_lens) else None
        raise ValueError(f"h_lens must have shape [{batch}], got {shape}")
    if h_lens.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"h_lens must have integer dtype, got {h_lens.dtype}")

    lens = h_lens.to(device=labels.device)
    if bool(((lens < 0) | (lens > frames)).any()):
        bad = lens[((lens < 0) | (lens > frames))].detach().cpu().tolist()
        raise ValueError(f"h_lens entries must be in [0,{frames}], got {bad}")

    in_length = torch.arange(frames, device=labels.device)[None, :] < lens[:, None]
    valid_frame = in_length & (labels >= 0)

    out_of_range = valid_frame & (labels > 3)
    if bool(out_of_range.any()):
        bad = torch.unique(labels[out_of_range]).detach().cpu().tolist()
        raise ValueError(f"valid speaker-count labels must be in [0,3], got {bad}")

    valid_pair = torch.zeros_like(valid_frame)
    if frames > 1:
        valid_pair[:, 1:] = valid_frame[:, :-1] & valid_frame[:, 1:]
    return valid_frame, valid_pair


def corn_log_probs(conditional_logits: torch.Tensor) -> torch.Tensor:
    """Convert three CORN conditional logits into normalized four-class log-p.

    The logits represent

    ``c1=P(N>=1)``, ``c2=P(N>=2 | N>=1)``, and
    ``c3=P(N>=3 | N>=2)``.

    Thus cumulative probabilities are ``q1=c1``, ``q2=c1*c2``,
    ``q3=c1*c2*c3``.  The returned class posterior is
    ``[1-q1, q1-q2, q2-q3, q3]``.  It is constructed in log space in float32,
    which is both AMP-safe and monotone by construction.
    """

    if not torch.is_tensor(conditional_logits):
        raise TypeError("conditional_logits must be a tensor")
    if conditional_logits.ndim < 1 or conditional_logits.shape[-1] != 3:
        raise ValueError(
            "CORN conditional logits must end in three thresholds, got "
            f"{tuple(conditional_logits.shape)}"
        )

    logits = conditional_logits.float()
    log_c = F.logsigmoid(logits)
    log_not_c = F.logsigmoid(-logits)
    log_q1 = log_c[..., 0]
    log_q2 = log_q1 + log_c[..., 1]
    log_q3 = log_q2 + log_c[..., 2]
    log_probs = torch.stack(
        (
            log_not_c[..., 0],
            log_q1 + log_not_c[..., 1],
            log_q2 + log_not_c[..., 2],
            log_q3,
        ),
        dim=-1,
    )
    # The four terms are analytically normalized.  Renormalizing in log-space
    # removes only round-off and gives downstream tests an exact probability
    # simplex even for extreme logits.
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


def corn_probs(conditional_logits: torch.Tensor) -> torch.Tensor:
    """Float32 four-class posterior corresponding to ``corn_log_probs``."""

    return corn_log_probs(conditional_logits).exp()


def _shift_with_contiguous_support(
    signal: torch.Tensor,
    support: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    """Shift ``signal`` without traversing a false entry in ``support``."""

    if signal.shape != support.shape or signal.ndim != 2:
        raise ValueError("signal and support must be aligned [B,T] tensors")
    frames = signal.shape[1]
    shifted = torch.zeros_like(signal)
    distance = abs(int(offset))
    if distance == 0:
        return torch.where(support, signal, torch.zeros_like(signal))
    if distance >= frames:
        return shifted

    width = frames - distance
    path_valid = torch.ones(
        signal.shape[0], width, dtype=torch.bool, device=signal.device
    )
    # A value may spread from transition i to i+d only if every transition
    # slot on that path is valid.  This is what prevents leakage across an
    # invalid/padded gap.
    for step in range(distance + 1):
        path_valid = path_valid & support[:, step : step + width]

    if offset > 0:
        value = signal[:, :width]
        shifted[:, distance:] = torch.where(
            path_valid, value, torch.zeros_like(value)
        )
    else:
        value = signal[:, distance:]
        shifted[:, :width] = torch.where(
            path_valid, value, torch.zeros_like(value)
        )
    return shifted


def soft_boundary_targets(
    labels: torch.Tensor,
    valid_pair: torch.Tensor,
    kernel: Sequence[float] = (0.25, 0.75, 1.0, 0.75, 0.25),
) -> torch.Tensor:
    """Create a max-envelope soft boundary target on transition indices.

    A hard boundary is located at index ``t`` when ``labels[t-1] != labels[t]``.
    Kernel mass is allowed to spread only through contiguous valid transition
    slots.  Consequently it cannot cross an internal ``-100`` gap, sequence
    start, or padded tail.
    """

    if not torch.is_tensor(labels) or labels.ndim != 2:
        raise ValueError("labels must be [B,T]")
    if (
        not torch.is_tensor(valid_pair)
        or valid_pair.dtype != torch.bool
        or valid_pair.shape != labels.shape
    ):
        raise ValueError("valid_pair must be a boolean tensor aligned with labels")
    weights = tuple(float(value) for value in kernel)
    if not weights or len(weights) % 2 != 1:
        raise ValueError("boundary kernel must have a positive odd length")
    if any((value < 0.0 or value > 1.0) for value in weights):
        raise ValueError("boundary kernel weights must lie in [0,1]")
    center = len(weights) // 2
    if weights[center] != max(weights):
        raise ValueError("boundary kernel center must be its maximum")

    hard = torch.zeros_like(labels, dtype=torch.float32)
    if labels.shape[1] > 1:
        change = valid_pair[:, 1:] & (labels[:, 1:] != labels[:, :-1])
        hard[:, 1:] = change.float()

    target = torch.zeros_like(hard)
    for index, weight in enumerate(weights):
        if weight == 0.0:
            continue
        spread = _shift_with_contiguous_support(
            hard, valid_pair, offset=index - center
        )
        target = torch.maximum(target, spread * weight)
    return torch.where(valid_pair, target, torch.zeros_like(target))


def balanced_soft_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Balanced BCE for sparse soft targets, normalized by actual mass.

    Positive and negative terms are normalized separately, then averaged over
    the components which actually exist.  An all-negative recording therefore
    still supplies a well-scaled negative loss instead of creating an infinite
    ``pos_weight``; an all-invalid batch returns differentiable zero.
    """

    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError(
            "balanced BCE expects aligned logits/targets/mask, got "
            f"{tuple(logits.shape)}, {tuple(targets.shape)}, {tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise TypeError("balanced BCE mask must be boolean")
    if bool(((targets < 0.0) | (targets > 1.0)).any()):
        raise ValueError("soft BCE targets must lie in [0,1]")

    raw = logits.float()
    safe = torch.where(mask, raw, torch.zeros_like(raw))
    target = torch.where(mask, targets.float(), torch.zeros_like(raw))
    negative_target = torch.where(mask, 1.0 - targets.float(), torch.zeros_like(raw))

    pos_mass = target.sum()
    neg_mass = negative_target.sum()
    pos_loss = -(target * F.logsigmoid(safe)).sum() / pos_mass.clamp_min(1.0)
    neg_loss = -(negative_target * F.logsigmoid(-safe)).sum() / neg_mass.clamp_min(1.0)

    has_pos = (pos_mass > 0).to(raw.dtype)
    has_neg = (neg_mass > 0).to(raw.dtype)
    components = has_pos + has_neg
    result = (has_pos * pos_loss + has_neg * neg_loss) / components.clamp_min(1.0)
    return torch.where(
        components > 0, result, _differentiable_zero(raw)
    )


def direction_loss(
    direction_logits: torch.Tensor,
    labels: torch.Tensor,
    valid_pair: torch.Tensor,
    direction_class_weights: torch.Tensor,
) -> torch.Tensor:
    """Balanced BCE on the signed up/down event at each label transition."""

    batch, frames = labels.shape
    logits = PyramidStructuredLoss._squeeze_binary_head(
        direction_logits, "direction_logits", batch, frames
    ).float()
    change = torch.zeros_like(valid_pair)
    target_up = torch.zeros_like(labels, dtype=torch.float32)
    if frames > 1:
        delta = labels[:, 1:] - labels[:, :-1]
        change[:, 1:] = valid_pair[:, 1:] & (delta != 0)
        target_up[:, 1:] = (delta > 0).float()
    safe_logits = torch.where(change, logits, torch.zeros_like(logits))
    per_change = F.binary_cross_entropy_with_logits(
        safe_logits, target_up, reduction="none"
    )
    direction_index = target_up.long()
    weights = direction_class_weights[direction_index]
    weights = torch.where(change, weights, torch.zeros_like(weights))
    per_change = torch.where(change, per_change, torch.zeros_like(per_change))
    return (per_change * weights).sum() / weights.sum().clamp_min(1.0)


def temporal_mse(
    log_probs: torch.Tensor,
    soft_boundary: torch.Tensor,
    valid_pair: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Boundary-aware smoothness: penalize log-prob drift away from edges."""

    if log_probs.shape[1] < 2:
        return _differentiable_zero(log_probs)
    pair = valid_pair[:, 1:]
    safe_left = torch.where(
        pair.unsqueeze(-1),
        log_probs[:, :-1].float(),
        torch.zeros_like(log_probs[:, :-1].float()),
    )
    safe_right = torch.where(
        pair.unsqueeze(-1),
        log_probs[:, 1:].float(),
        torch.zeros_like(log_probs[:, 1:].float()),
    )
    squared = (safe_right - safe_left).pow(2).mean(dim=-1)
    squared = squared.clamp_max(tau)
    weights = torch.where(
        pair, 1.0 - soft_boundary[:, 1:].float(), torch.zeros_like(squared)
    )
    values = torch.where(pair, squared, torch.zeros_like(squared))
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


@dataclass(frozen=True)
class _StructuredView:
    count_logits: Optional[torch.Tensor] = None
    count_log_probs: Optional[torch.Tensor] = None
    ordinal_logits: Optional[torch.Tensor] = None
    boundary_logits: Optional[torch.Tensor] = None
    direction_logits: Optional[torch.Tensor] = None
    stage1_count_logits: Optional[torch.Tensor] = None
    stage1_count_log_probs: Optional[torch.Tensor] = None
    stage1_ordinal_logits: Optional[torch.Tensor] = None


def _lookup(output: Any, names: Sequence[str]) -> Any:
    if isinstance(output, Mapping):
        for name in names:
            if name in output:
                return output[name]
    for name in names:
        if hasattr(output, name):
            return getattr(output, name)
    return None


def _adapt_output(output: Any) -> _StructuredView:
    """Adapt the explicit pyramid NamedTuple or a dict-like equivalent."""

    count_logits = _lookup(output, ("count_logits", "logits"))
    # Preserve the project's long-standing first-three tuple contract.  Plain
    # tuples do not carry structured auxiliary fields, but their count logits
    # remain usable for CE/SORD controls.
    if count_logits is None and isinstance(output, (tuple, list)) and output:
        count_logits = output[0]

    return _StructuredView(
        count_logits=count_logits,
        count_log_probs=_lookup(
            output, ("count_log_probs", "log_probs", "count_logp")
        ),
        ordinal_logits=_lookup(
            output,
            (
                "ordinal_logits",
                "corn_logits",
                "corn_conditional_logits",
                "threshold_logits",
            ),
        ),
        boundary_logits=_lookup(output, ("boundary_logits", "change_logits")),
        direction_logits=_lookup(
            output, ("direction_logits", "event_direction_logits")
        ),
        stage1_count_logits=_lookup(output, ("stage1_count_logits",)),
        stage1_count_log_probs=_lookup(output, ("stage1_count_log_probs",)),
        stage1_ordinal_logits=_lookup(
            output,
            (
                "stage1_ordinal_logits",
                "stage1_corn_logits",
                "stage1_threshold_logits",
            ),
        ),
    )


class PyramidStructuredLoss(nn.Module):
    """Ordinal, boundary, and segment-structured speaker-count objective.

    Auxiliary coefficients default to zero so each scientific comparison can
    introduce exactly one new supervision path.  The main objective is always
    present and selected with ``count_loss_type``.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        count_loss_type: str = "corn",
        focal_gamma: float = 2.0,
        sord_alpha: float = 1.5,
        lambda_stage1: float = 0.0,
        lambda_boundary: float = 0.0,
        lambda_direction: float = 0.0,
        lambda_t_mse: float = 0.0,
        lambda_segment: float = 0.0,
        lambda_delta: float = 0.0,
        boundary_kernel: Sequence[float] = (0.25, 0.75, 1.0, 0.75, 0.25),
        t_mse_tau: float = 4.0,
        direction_class_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        objective = str(count_loss_type).lower()
        if objective not in {"ce", "focal", "sord", "cumulative", "corn"}:
            raise ValueError(
                "PyramidStructuredLoss count_loss_type must be one of "
                f"ce/focal/sord/cumulative/corn, got {count_loss_type!r}"
            )
        self.count_loss_type = objective
        self.focal_gamma = float(focal_gamma)
        self.sord_alpha = float(sord_alpha)
        if not math.isfinite(self.focal_gamma) or self.focal_gamma < 0.0:
            raise ValueError("focal_gamma must be finite and non-negative")
        if not math.isfinite(self.sord_alpha) or self.sord_alpha <= 0.0:
            raise ValueError("sord_alpha must be finite and positive")
        self.lambda_stage1 = self._nonnegative(lambda_stage1, "lambda_stage1")
        self.lambda_boundary = self._nonnegative(
            lambda_boundary, "lambda_boundary"
        )
        self.lambda_direction = self._nonnegative(
            lambda_direction, "lambda_direction"
        )
        self.lambda_t_mse = self._nonnegative(lambda_t_mse, "lambda_t_mse")
        self.lambda_segment = self._nonnegative(
            lambda_segment, "lambda_segment"
        )
        self.lambda_delta = self._nonnegative(lambda_delta, "lambda_delta")
        if not math.isfinite(float(t_mse_tau)) or float(t_mse_tau) <= 0:
            raise ValueError("t_mse_tau must be finite and positive")
        self.t_mse_tau = float(t_mse_tau)

        kernel = torch.as_tensor(tuple(boundary_kernel), dtype=torch.float32)
        if kernel.ndim != 1 or kernel.numel() == 0 or kernel.numel() % 2 != 1:
            raise ValueError("boundary_kernel must be a non-empty odd-length vector")
        if (
            not bool(torch.isfinite(kernel).all())
            or bool(((kernel < 0) | (kernel > 1)).any())
        ):
            raise ValueError("boundary_kernel values must be finite and lie in [0,1]")
        if kernel[kernel.numel() // 2] != kernel.max():
            raise ValueError("boundary_kernel center must be its maximum")
        self.register_buffer("boundary_kernel", kernel)
        # Keep the tiny immutable CPU representation for target construction.
        # Calling ``.cpu().tolist()`` on the registered CUDA buffer in every
        # forward pass would otherwise impose a hidden device synchronization.
        self.boundary_kernel_values = tuple(float(value) for value in kernel)

        if class_weights is None:
            class_weights = torch.ones(4, dtype=torch.float32)
        class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if (
            class_weights.shape != (4,)
            or not bool(torch.isfinite(class_weights).all())
            or bool((class_weights <= 0).any())
        ):
            raise ValueError("class_weights must contain four positive finite values")
        self.register_buffer("class_weights", class_weights)

        if direction_class_weights is None:
            direction_class_weights = (1.0, 1.0)
        direction_weights = torch.as_tensor(
            tuple(direction_class_weights), dtype=torch.float32
        )
        if (
            direction_weights.shape != (2,)
            or not bool(torch.isfinite(direction_weights).all())
            or bool((direction_weights <= 0).any())
        ):
            raise ValueError(
                "direction_class_weights must be two positive finite values "
                "in DOWN/UP order"
            )
        self.register_buffer("direction_class_weights", direction_weights)

    @staticmethod
    def _nonnegative(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    @staticmethod
    def _squeeze_binary_head(
        logits: torch.Tensor,
        name: str,
        batch: int,
        frames: int,
    ) -> torch.Tensor:
        if not torch.is_tensor(logits):
            raise TypeError(f"{name} must be a tensor")
        if logits.ndim == 3 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        if logits.ndim != 2 or logits.shape != (batch, frames):
            raise ValueError(
                f"{name} must be [B,T] or [B,T,1], got {tuple(logits.shape)}"
            )
        return logits

    def _posterior(
        self,
        view: _StructuredView,
        batch: int,
        frames: int,
        valid_frame: torch.Tensor,
        prefix: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ordinal = (
            view.stage1_ordinal_logits if prefix == "stage1_" else view.ordinal_logits
        )
        explicit_logp = (
            view.stage1_count_log_probs
            if prefix == "stage1_"
            else view.count_log_probs
        )
        count_logits = (
            view.stage1_count_logits if prefix == "stage1_" else view.count_logits
        )

        if self.count_loss_type == "corn":
            if ordinal is None:
                raise ValueError(
                    f"{prefix or 'final '}CORN objective requires explicit "
                    f"{prefix}ordinal_logits [B,T,3]"
                )
            ordinal = _require_btc(
                ordinal, f"{prefix}ordinal_logits", batch, frames, channels=3
            )
            # Sanitize before sigmoid/log operations.  Masking the resulting
            # NaN posterior later is insufficient: a zero upstream gradient
            # times a NaN local derivative can still poison backward.
            safe_ordinal = torch.where(
                valid_frame.unsqueeze(-1),
                ordinal.float(),
                torch.zeros_like(ordinal.float()),
            )
            computed_logp = corn_log_probs(safe_ordinal)
            if explicit_logp is not None:
                _require_btc(
                    explicit_logp,
                    f"{prefix}count_log_probs",
                    batch,
                    frames,
                    channels=4,
                )
                # The loss is intentionally recomputed from conditional logits
                # rather than trusting a detached/cached posterior.  Equality
                # between the public posterior and this construction belongs
                # in the architecture invariant tests; checking it here would
                # synchronize the GPU on every training step.
            return computed_logp, safe_ordinal

        if count_logits is None:
            raise ValueError(
                f"{prefix or 'final '}count objective {self.count_loss_type!r} "
                f"requires {prefix}count_logits [B,T,4]"
            )
        count_logits = _require_btc(
            count_logits, f"{prefix}count_logits", batch, frames, channels=4
        )
        safe_count_logits = torch.where(
            valid_frame.unsqueeze(-1),
            count_logits.float(),
            torch.zeros_like(count_logits.float()),
        )
        return F.log_softmax(safe_count_logits, dim=-1), safe_count_logits

    def _weighted_frame_mean(
        self,
        values: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        safe_labels = torch.where(mask, labels, torch.zeros_like(labels))
        weights = self.class_weights[safe_labels]
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        safe_values = torch.where(mask, values.float(), torch.zeros_like(values.float()))
        return (safe_values * weights).sum() / weights.sum().clamp_min(1.0)

    def _corn_loss(
        self,
        conditional_logits: torch.Tensor,
        labels: torch.Tensor,
        valid_frame: torch.Tensor,
    ) -> torch.Tensor:
        logits = conditional_logits.float()
        # Conditional risk sets:
        # k=1: every valid frame,       target N>=1
        # k=2: only frames with N>=1,   target N>=2
        # k=3: only frames with N>=2,   target N>=3
        risk = torch.stack(
            (
                valid_frame,
                valid_frame & (labels >= 1),
                valid_frame & (labels >= 2),
            ),
            dim=-1,
        )
        targets = torch.stack(
            (labels >= 1, labels >= 2, labels >= 3), dim=-1
        ).float()
        safe_logits = torch.where(risk, logits, torch.zeros_like(logits))
        per_risk = F.binary_cross_entropy_with_logits(
            safe_logits, targets, reduction="none"
        )

        safe_labels = torch.where(
            valid_frame, labels, torch.zeros_like(labels)
        )
        frame_weights = self.class_weights[safe_labels].unsqueeze(-1)
        weights = torch.where(
            risk, frame_weights.expand_as(logits), torch.zeros_like(logits)
        )
        safe_loss = torch.where(risk, per_risk, torch.zeros_like(per_risk))
        # Each threshold is a separate conditional risk.  Normalize it by its
        # own active weighted population, then average the supported risks.
        # Pooling all cells into one denominator would let abundant k=1 frames
        # drown the rare but scientifically central k=3 risk.
        numerator = (safe_loss * weights).sum(dim=(0, 1))
        denominator = weights.sum(dim=(0, 1))
        per_threshold = numerator / denominator.clamp_min(1.0)
        supported = denominator > 0
        return torch.where(
            supported,
            per_threshold,
            torch.zeros_like(per_threshold),
        ).sum() / supported.sum().clamp_min(1)

    def _cumulative_softmax_loss(
        self,
        log_probs: torch.Tensor,
        labels: torch.Tensor,
        valid_frame: torch.Tensor,
    ) -> torch.Tensor:
        """Unconditional cumulative BCE derived from the same four logits.

        This is the pure loss-geometry control between SORD/CE and native CORN:
        no output parameterization changes.  Each threshold sees all valid
        frames and predicts ``P(N>=k)=sum_{c>=k} softmax(logits)[c]``.
        """

        threshold_losses = []
        safe_labels = torch.where(
            valid_frame, labels, torch.zeros_like(labels)
        )
        frame_weights = self.class_weights[safe_labels]
        weights = torch.where(
            valid_frame, frame_weights, torch.zeros_like(frame_weights)
        )
        denominator = weights.sum()
        for threshold in (1, 2, 3):
            log_positive = torch.logsumexp(
                log_probs[..., threshold:], dim=-1
            )
            log_negative = torch.logsumexp(
                log_probs[..., :threshold], dim=-1
            )
            target = labels >= threshold
            per_frame = torch.where(
                target, -log_positive, -log_negative
            )
            per_frame = torch.where(
                valid_frame, per_frame, torch.zeros_like(per_frame)
            )
            threshold_losses.append(
                (per_frame * weights).sum() / denominator.clamp_min(1.0)
            )
        losses = torch.stack(threshold_losses)
        # All three unconditional risks have support iff at least one frame is
        # valid.  Keeping the explicit condition preserves differentiable zero.
        return torch.where(
            denominator > 0, losses.mean(), _differentiable_zero(log_probs)
        )

    def _count_loss(
        self,
        view: _StructuredView,
        labels: torch.Tensor,
        valid_frame: torch.Tensor,
        prefix: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, frames = labels.shape
        log_probs, objective_logits = self._posterior(
            view, batch, frames, valid_frame, prefix=prefix
        )
        probs = log_probs.exp()

        if self.count_loss_type == "corn":
            loss = self._corn_loss(objective_logits, labels, valid_frame)
            return loss, log_probs, probs

        if self.count_loss_type == "cumulative":
            loss = self._cumulative_softmax_loss(
                log_probs, labels, valid_frame
            )
            return loss, log_probs, probs

        safe_labels = torch.where(
            valid_frame, labels, torch.zeros_like(labels)
        )
        hard_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        if self.count_loss_type == "ce":
            per_frame = -hard_logp
        elif self.count_loss_type == "focal":
            # ``pt`` is the unweighted model probability.  Deriving it from a
            # class-weighted CE (legacy behavior) changes focal semantics.
            per_frame = -(1.0 - hard_logp.exp()).pow(self.focal_gamma) * hard_logp
        else:
            classes = torch.arange(
                4, device=labels.device, dtype=log_probs.dtype
            ).view(1, 1, 4)
            distances = (
                classes - safe_labels.to(log_probs.dtype).unsqueeze(-1)
            ).abs()
            soft_targets = F.softmax(-self.sord_alpha * distances, dim=-1)
            per_frame = -(soft_targets * log_probs).sum(dim=-1)
        loss = self._weighted_frame_mean(
            per_frame, safe_labels, valid_frame
        )
        return loss, log_probs, probs

    def _direction_loss(
        self,
        direction_logits: torch.Tensor,
        labels: torch.Tensor,
        valid_pair: torch.Tensor,
    ) -> torch.Tensor:
        return direction_loss(
            direction_logits, labels, valid_pair, self.direction_class_weights
        )

    def _temporal_mse(
        self,
        log_probs: torch.Tensor,
        soft_boundary: torch.Tensor,
        valid_pair: torch.Tensor,
    ) -> torch.Tensor:
        return temporal_mse(log_probs, soft_boundary, valid_pair, self.t_mse_tau)

    @staticmethod
    def _segment_consistency(
        probs: torch.Tensor,
        labels: torch.Tensor,
        valid_frame: torch.Tensor,
    ) -> torch.Tensor:
        """Mean within-segment variance, with every GT segment weighted equally."""

        batch, frames, classes = probs.shape
        if not bool(valid_frame.any()):
            return _differentiable_zero(probs)

        starts = torch.zeros_like(valid_frame)
        if frames > 0:
            starts[:, 0] = valid_frame[:, 0]
        if frames > 1:
            starts[:, 1:] = valid_frame[:, 1:] & (
                (~valid_frame[:, :-1]) | (labels[:, 1:] != labels[:, :-1])
            )
        local_ids = starts.long().cumsum(dim=1) - 1
        batch_offsets = (
            torch.arange(batch, device=labels.device, dtype=torch.long) * frames
        )[:, None]
        segment_ids = local_ids + batch_offsets

        flat_valid = valid_frame.reshape(-1)
        ids = segment_ids.reshape(-1)[flat_valid]
        selected = probs.float().reshape(-1, classes)[flat_valid]
        slots = batch * frames
        sums = torch.zeros(
            slots, classes, device=probs.device, dtype=torch.float32
        ).index_add(0, ids, selected)
        counts = torch.zeros(
            slots, device=probs.device, dtype=torch.float32
        ).index_add(0, ids, torch.ones_like(ids, dtype=torch.float32))
        means = sums / counts.clamp_min(1.0).unsqueeze(-1)
        frame_error = (selected - means[ids]).pow(2).mean(dim=-1)
        segment_error = torch.zeros_like(counts).index_add(0, ids, frame_error)
        segment_error = segment_error / counts.clamp_min(1.0)
        present = counts > 0
        return segment_error[present].mean()

    def _delta_consistency(
        self,
        probs: torch.Tensor,
        boundary_logits: torch.Tensor,
        direction_logits: torch.Tensor,
        valid_pair: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, _ = probs.shape
        if frames < 2:
            return _differentiable_zero(probs)
        boundary = self._squeeze_binary_head(
            boundary_logits, "boundary_logits", batch, frames
        ).float()
        direction = self._squeeze_binary_head(
            direction_logits, "direction_logits", batch, frames
        ).float()
        pair = valid_pair[:, 1:]

        classes = torch.arange(
            4, device=probs.device, dtype=torch.float32
        ).view(1, 1, 4)
        mean_count = (probs.float() * classes).sum(dim=-1)
        count_delta = (mean_count[:, 1:] - mean_count[:, :-1]).clamp(-1.0, 1.0)

        safe_boundary = torch.where(
            valid_pair, boundary, torch.zeros_like(boundary)
        )
        safe_direction = torch.where(
            valid_pair, direction, torch.zeros_like(direction)
        )
        # Direction is conditional on a change.  Multiplying its signed mean by
        # P(change) gives the expected signed event, including STAY frames.
        event_delta = (
            torch.sigmoid(safe_boundary[:, 1:])
            * (2.0 * torch.sigmoid(safe_direction[:, 1:]) - 1.0)
        )

        safe_count = torch.where(pair, count_delta, torch.zeros_like(count_delta))
        safe_event = torch.where(pair, event_delta, torch.zeros_like(event_delta))
        # Symmetric stop-gradient: one half updates the count posterior toward
        # the event branch, the other updates the event branch toward count.
        per_pair = 0.5 * F.smooth_l1_loss(
            safe_count, safe_event.detach(), reduction="none"
        ) + 0.5 * F.smooth_l1_loss(
            safe_count.detach(), safe_event, reduction="none"
        )
        per_pair = torch.where(pair, per_pair, torch.zeros_like(per_pair))
        return per_pair.sum() / pair.sum().clamp_min(1)

    def forward(
        self,
        outputs: Any,
        labels: torch.Tensor,
        h_lens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        view = _adapt_output(outputs)
        reference = view.ordinal_logits
        if reference is None:
            reference = view.count_logits
        if reference is None:
            raise ValueError(
                "structured model output needs count_logits or ordinal_logits"
            )
        if not torch.is_tensor(reference) or reference.ndim != 3:
            raise ValueError("primary model output must be [B,T,C]")
        batch, frames = reference.shape[:2]
        valid_frame, valid_pair = build_valid_masks(
            labels, h_lens, expected_T=frames
        )
        labels = labels.to(device=reference.device)
        valid_frame = valid_frame.to(device=reference.device)
        valid_pair = valid_pair.to(device=reference.device)

        # Require all output tensors and labels to share a device.  Moving
        # labels is safe, but silently copying output branches would hide a
        # broken model contract and sever expected device placement.
        loss_count, log_probs, probs = self._count_loss(
            view, labels, valid_frame
        )
        zero = _differentiable_zero(reference)

        loss_stage1 = zero
        if self.lambda_stage1 > 0:
            loss_stage1, _, _ = self._count_loss(
                view, labels, valid_frame, prefix="stage1_"
            )

        need_soft_boundary = self.lambda_boundary > 0 or self.lambda_t_mse > 0
        soft_boundary = None
        if need_soft_boundary:
            soft_boundary = soft_boundary_targets(
                labels,
                valid_pair,
                kernel=self.boundary_kernel_values,
            )

        loss_boundary = zero
        boundary_logits = view.boundary_logits
        if self.lambda_boundary > 0:
            if boundary_logits is None:
                raise ValueError(
                    "lambda_boundary > 0 requires boundary_logits [B,T,1]"
                )
            boundary = self._squeeze_binary_head(
                boundary_logits, "boundary_logits", batch, frames
            )
            loss_boundary = balanced_soft_bce_with_logits(
                boundary, soft_boundary, valid_pair
            )

        loss_direction = zero
        if self.lambda_direction > 0:
            if view.direction_logits is None:
                raise ValueError(
                    "lambda_direction > 0 requires direction_logits [B,T,1]"
                )
            loss_direction = self._direction_loss(
                view.direction_logits, labels, valid_pair
            )

        loss_t_mse = zero
        if self.lambda_t_mse > 0:
            loss_t_mse = self._temporal_mse(
                log_probs, soft_boundary, valid_pair
            )

        loss_segment = zero
        if self.lambda_segment > 0:
            loss_segment = self._segment_consistency(
                probs, labels, valid_frame
            )

        loss_delta = zero
        if self.lambda_delta > 0:
            if boundary_logits is None or view.direction_logits is None:
                raise ValueError(
                    "lambda_delta > 0 requires boundary_logits and "
                    "direction_logits"
                )
            loss_delta = self._delta_consistency(
                probs,
                boundary_logits,
                view.direction_logits,
                valid_pair,
            )

        loss = (
            loss_count
            + self.lambda_stage1 * loss_stage1
            + self.lambda_boundary * loss_boundary
            + self.lambda_direction * loss_direction
            + self.lambda_t_mse * loss_t_mse
            + self.lambda_segment * loss_segment
            + self.lambda_delta * loss_delta
        )
        return {
            "loss": loss,
            "loss_count": loss_count,
            "loss_stage1": loss_stage1,
            "loss_boundary": loss_boundary,
            "loss_direction": loss_direction,
            "loss_t_mse": loss_t_mse,
            "loss_segment": loss_segment,
            "loss_delta": loss_delta,
        }


class PyramidAuxiliaryLoss(nn.Module):
    """Boundary / direction / smoothing / segment terms ONLY.

    Deliberately independent of the count objective: this module adds no
    count-loss term and never looks at ``count_logits`` beyond using
    ``count_log_probs`` for t_mse/segment.  It exists so the boundary- and
    event-consistency ideas from the pyramid redesign can be tested as pure
    additions on top of the unmodified legacy ``ZipCountLoss`` total, instead
    of requiring the count objective itself to switch to
    ``pyramid_structured`` (SORD/CORN) first -- that switch failed its own
    non-inferiority bar in the Phase-8 chain (phase b: -0.0087/-0.0122),
    which structurally blocked boundary/direction/t_mse/segment/stage2 since
    they were defined as increments on top of that switch.  Composing this
    loss with the legacy total re-opens those ideas as independently
    testable without re-litigating the ordinal encoding.
    """

    def __init__(
        self,
        lambda_boundary: float = 0.0,
        lambda_direction: float = 0.0,
        lambda_t_mse: float = 0.0,
        lambda_segment: float = 0.0,
        boundary_kernel: Sequence[float] = (0.25, 0.75, 1.0, 0.75, 0.25),
        t_mse_tau: float = 4.0,
        direction_class_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.lambda_boundary = PyramidStructuredLoss._nonnegative(
            lambda_boundary, "lambda_boundary"
        )
        self.lambda_direction = PyramidStructuredLoss._nonnegative(
            lambda_direction, "lambda_direction"
        )
        self.lambda_t_mse = PyramidStructuredLoss._nonnegative(
            lambda_t_mse, "lambda_t_mse"
        )
        self.lambda_segment = PyramidStructuredLoss._nonnegative(
            lambda_segment, "lambda_segment"
        )
        if not math.isfinite(float(t_mse_tau)) or float(t_mse_tau) <= 0:
            raise ValueError("t_mse_tau must be finite and positive")
        self.t_mse_tau = float(t_mse_tau)

        kernel = torch.as_tensor(tuple(boundary_kernel), dtype=torch.float32)
        if kernel.ndim != 1 or kernel.numel() == 0 or kernel.numel() % 2 != 1:
            raise ValueError(
                "boundary_kernel must be a non-empty odd-length vector"
            )
        if (
            not bool(torch.isfinite(kernel).all())
            or bool(((kernel < 0) | (kernel > 1)).any())
        ):
            raise ValueError(
                "boundary_kernel values must be finite and lie in [0,1]"
            )
        if kernel[kernel.numel() // 2] != kernel.max():
            raise ValueError("boundary_kernel center must be its maximum")
        self.register_buffer("boundary_kernel", kernel)
        self.boundary_kernel_values = tuple(float(value) for value in kernel)

        if direction_class_weights is None:
            direction_class_weights = (1.0, 1.0)
        direction_weights = torch.as_tensor(
            tuple(direction_class_weights), dtype=torch.float32
        )
        if (
            direction_weights.shape != (2,)
            or not bool(torch.isfinite(direction_weights).all())
            or bool((direction_weights <= 0).any())
        ):
            raise ValueError(
                "direction_class_weights must be two positive finite values "
                "in DOWN/UP order"
            )
        self.register_buffer("direction_class_weights", direction_weights)

    def forward(
        self,
        outputs: Any,
        labels: torch.Tensor,
        h_lens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        view = _adapt_output(outputs)
        reference = view.count_log_probs
        if reference is None:
            reference = view.count_logits
        if reference is None:
            raise ValueError(
                "PyramidAuxiliaryLoss requires count_log_probs or "
                "count_logits on the model output to build the frame mask"
            )
        if not torch.is_tensor(reference) or reference.ndim != 3:
            raise ValueError("primary model output must be [B,T,C]")
        batch, frames = reference.shape[:2]
        valid_frame, valid_pair = build_valid_masks(
            labels, h_lens, expected_T=frames
        )
        labels = labels.to(device=reference.device)
        valid_frame = valid_frame.to(device=reference.device)
        valid_pair = valid_pair.to(device=reference.device)

        zero = _differentiable_zero(reference)

        need_soft_boundary = self.lambda_boundary > 0 or self.lambda_t_mse > 0
        soft_boundary = None
        if need_soft_boundary:
            soft_boundary = soft_boundary_targets(
                labels, valid_pair, kernel=self.boundary_kernel_values
            )

        loss_boundary = zero
        if self.lambda_boundary > 0:
            if view.boundary_logits is None:
                raise ValueError(
                    "lambda_boundary > 0 requires boundary_logits [B,T,1]"
                )
            boundary = PyramidStructuredLoss._squeeze_binary_head(
                view.boundary_logits, "boundary_logits", batch, frames
            )
            loss_boundary = balanced_soft_bce_with_logits(
                boundary, soft_boundary, valid_pair
            )

        loss_direction = zero
        if self.lambda_direction > 0:
            if view.direction_logits is None:
                raise ValueError(
                    "lambda_direction > 0 requires direction_logits [B,T,1]"
                )
            loss_direction = direction_loss(
                view.direction_logits, labels, valid_pair,
                self.direction_class_weights,
            )

        loss_t_mse = zero
        if self.lambda_t_mse > 0:
            if view.count_log_probs is None:
                raise ValueError(
                    "lambda_t_mse > 0 requires count_log_probs [B,T,C]"
                )
            loss_t_mse = temporal_mse(
                view.count_log_probs, soft_boundary, valid_pair,
                self.t_mse_tau,
            )

        loss_segment = zero
        if self.lambda_segment > 0:
            if view.count_log_probs is None:
                raise ValueError(
                    "lambda_segment > 0 requires count_log_probs [B,T,C]"
                )
            probs = view.count_log_probs.exp()
            loss_segment = PyramidStructuredLoss._segment_consistency(
                probs, labels, valid_frame
            )

        loss = (
            self.lambda_boundary * loss_boundary
            + self.lambda_direction * loss_direction
            + self.lambda_t_mse * loss_t_mse
            + self.lambda_segment * loss_segment
        )
        return {
            "loss": loss,
            "loss_boundary": loss_boundary,
            "loss_direction": loss_direction,
            "loss_t_mse": loss_t_mse,
            "loss_segment": loss_segment,
        }
