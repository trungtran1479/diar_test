import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Optional, Tuple

from .deformable import CausalDeformableTemporalBlock

class TemporalAdaptiveCountAdapter(nn.Module):
    """Causal temporal adapter. Preserves shape [B, T, D] -> [B, T, D].
    
    Architecture:
    1. Causal depthwise temporal conv1d (kernel=5, groups=D)
    2. Pointwise projection: Linear(D, adapter_dim) -> GELU -> Linear(adapter_dim, D)
    3. Gated residual: output = x + gate * adapted
    """
    def __init__(self, d_model: int = 512, adapter_dim: int = 256, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        
        # Causal Depthwise Conv
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=0 # We will pad manually for causal
        )
        
        # Pointwise
        self.fc1 = nn.Linear(d_model, adapter_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(adapter_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Gating
        self.gate = nn.Sequential(
            nn.Linear(d_model, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        """
        B, T, D = x.shape
        
        # [B, T, D] -> [B, D, T]
        x_t = x.transpose(1, 2)
        
        # Causal padding (pad left only)
        # padding is (pad_left, pad_right) for last dim
        x_pad = F.pad(x_t, (self.kernel_size - 1, 0))
        
        # [B, D, T]
        conv_out = self.conv(x_pad)
        
        # [B, T, D]
        conv_out = conv_out.transpose(1, 2)
        
        # Pointwise
        h = self.fc2(self.act(self.fc1(conv_out)))
        h = self.dropout(h)
        
        # Gating
        g = self.gate(x)
        
        # Residual
        out = x + g * h
        
        return out

class LinearCountHead(nn.Module):
    """Simple linear head without adaptation."""
    def __init__(self, d_model: int, num_classes: int = 4):
        super().__init__()
        self.count_proj = nn.Linear(d_model, num_classes)
        self.vad_proj = nn.Linear(d_model, 1)
        self.overlap_proj = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor):
        count_logits = self.count_proj(x)
        vad_logits = self.vad_proj(x)
        overlap_logits = self.overlap_proj(x)
        return count_logits, vad_logits, overlap_logits

class TemporalAdaptiveCountHead(nn.Module):
    """Head with temporal adaptation."""
    def __init__(self, d_model: int, num_classes: int = 4, adapter_dim: int = 256, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.adapter = TemporalAdaptiveCountAdapter(
            d_model=d_model,
            adapter_dim=adapter_dim,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.linear_head = LinearCountHead(d_model, num_classes)

    def forward(self, x: torch.Tensor):
        h = self.adapter(x)
        return self.linear_head(h)


class StackGatedProjection(nn.Module):
    """Hypercolumn fusion (FPN-lite): per-stack learnable gates + shared projection.

    Input is the concatenation of Zipformer per-stack outputs
    [B, T, sum(stack_dims)]. Each stack slice is LayerNorm-ed (the stacks have
    very different activation scales), scaled by a softmax gate, then projected
    to d_out. The learned gates are interpretable: they show WHICH scale of the
    U-Net carries speaker-count information.
    """
    def __init__(
        self,
        stack_dims: List[int],
        d_out: int = 256,
        dropout: float = 0.1,
        stack_input_mask: Optional[List[int]] = None,
        renormalize_active_gate: bool = False,
    ):
        super().__init__()
        self.renormalize_active_gate = bool(renormalize_active_gate)
        self.stack_dims = list(stack_dims)
        if stack_input_mask is None:
            mask = [1] * len(self.stack_dims)
        else:
            if len(stack_input_mask) != len(self.stack_dims):
                raise ValueError(
                    "stack_input_mask must have one entry per stack: "
                    f"expected {len(self.stack_dims)}, got {len(stack_input_mask)}")
            if any(
                not isinstance(value, (bool, int, float))
                or value not in (0, 1)
                for value in stack_input_mask
            ):
                raise ValueError(
                    "stack_input_mask entries must be numeric 0/1 values, "
                    f"got {stack_input_mask}")
            mask = [int(value) for value in stack_input_mask]
            if not any(mask):
                raise ValueError("stack_input_mask must keep at least one stack")
        # Deliberately neither a Parameter nor a persistent buffer.  Matched
        # stack ablations must retain byte-identical state_dict keys, shapes,
        # parameter count and seeded initialization.  Python scalars also
        # follow the input across devices/dtypes without adding checkpoint
        # state.
        self.stack_input_mask = tuple(mask)
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in self.stack_dims])
        self.gates = nn.Parameter(torch.zeros(len(self.stack_dims)))
        self.proj = nn.Linear(sum(self.stack_dims), d_out)
        self.dropout = nn.Dropout(dropout)

    def gate_weights(self) -> torch.Tensor:
        if not self.renormalize_active_gate:
            return self.gates.softmax(dim=0)
        # Ablation arm (review request): exclude masked stacks from the
        # softmax denominator entirely, instead of letting them absorb gate
        # mass that is then thrown away by the zero-block multiply in
        # forward(). Masked logits get -inf so their softmax weight is
        # exactly 0 -- gate mass is redistributed over the active stacks
        # only, rather than left inert on masked ones.
        mask = torch.tensor(self.stack_input_mask, dtype=torch.bool,
                             device=self.gates.device)
        masked_logits = self.gates.masked_fill(~mask, float("-inf"))
        return masked_logits.softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, sum(stack_dims)] -> [B, T, d_out]
        w = self.gate_weights() * len(self.stack_dims)  # keep initial scale ~1 per slice
        pieces, ofs = [], 0
        for i, d in enumerate(self.stack_dims):
            piece = self.norms[i](x[..., ofs:ofs + d])
            # Mask AFTER the per-stack LayerNorm and BEFORE both the learned
            # gate and shared projection.  A disabled stack is thus a literal
            # zero block: it cannot affect the output, and its LayerNorm plus
            # the corresponding projection columns receive zero data
            # gradients.  Keeping every module instantiated is what makes the
            # comparison parameter- and initialization-matched.
            if not self.stack_input_mask[i]:
                # Do not multiply by zero: NaN * 0 is still NaN.  A disabled
                # stack must remain unable to poison the fused representation
                # even if that backbone slice is non-finite.
                piece = torch.zeros_like(piece)
            pieces.append(piece * w[i])
            ofs += d
        return self.dropout(self.proj(torch.cat(pieces, dim=-1)))


class OrdinalConsistentHead(nn.Module):
    """Count head with an ordinal-consistent auxiliary branch.

    - count_proj: 4-way softmax logits (powerset-style multi-class, the
      primary output — cf. pyannote powerset > multilabel on overlap).
    - cum_proj:   2 cumulative logits [P(count>=1), P(count>=2)] = (VAD, OSD).
      CORAL/CORN-style ordinal view of the same variable; the loss ties the
      two branches together (consistency + monotonicity).
    """
    def __init__(self, d_model: int, num_classes: int = 4, hidden_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.trunk = nn.Sequential(
                nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout))
            d_head = hidden_dim
        else:
            self.trunk = nn.Identity()
            d_head = d_model
        self.count_proj = nn.Linear(d_head, num_classes)
        self.cum_proj = nn.Linear(d_head, 2)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        count_logits = self.count_proj(h)          # [B, T, 4]
        cum = self.cum_proj(h)                     # [B, T, 2]
        vad_logits = cum[..., 0:1]                 # [B, T, 1]  P(count >= 1)
        overlap_logits = cum[..., 1:2]             # [B, T, 1]  P(count >= 2)
        return count_logits, vad_logits, overlap_logits


class CausalDualDilationBlock(nn.Module):
    """One MS-TCN++-style dual-dilated residual block, made strictly causal.

    Two depthwise branches read the SAME normalised input: dilation 1 for
    onset/offset detail, dilation d for segment-scale context (an overlap is
    confirmed by seconds of evidence, not one frame). Their concat goes
    through a GLU so the block can gate context against detail per frame.
    Causality is left-padding only; a future-frame perturbation must not
    change past outputs (unit-tested).
    """
    def __init__(self, d_model: int, dilation: int, kernel: int = 3, dropout: float = 0.1,
                 causal: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.causal = bool(causal)
        self.pad_local = kernel - 1
        self.pad_ctx = (kernel - 1) * dilation
        self.dw_local = nn.Conv1d(d_model, d_model, kernel, groups=d_model)
        self.dw_ctx = nn.Conv1d(d_model, d_model, kernel, dilation=dilation, groups=d_model)
        self.glu = nn.Linear(2 * d_model, 2 * d_model)
        self.drop = nn.Dropout(dropout)

    def _pad(self, h: torch.Tensor, total: int) -> torch.Tensor:
        # causal: everything on the left. non-causal (context-oracle only):
        # split symmetrically, so each frame sees an equal past/future window.
        if self.causal:
            return F.pad(h, (total, 0))
        return F.pad(h, (total // 2, total - total // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        h = self.norm(x).transpose(1, 2)                       # [B, D, T]
        a = self.dw_local(self._pad(h, self.pad_local))
        b = self.dw_ctx(self._pad(h, self.pad_ctx))
        g = torch.cat([a, b], dim=1).transpose(1, 2)           # [B, T, 2D]
        return x + self.drop(F.glu(self.glu(g), dim=-1))

    def forward_chunk(self, x: torch.Tensor, cache: torch.Tensor):
        """Chunk-wise forward with a cache of the last pad_ctx POST-NORM frames.

        Only defined for the causal block: a non-causal block needs future
        frames and cannot stream.

        The cache MUST hold post-LayerNorm values, and the initial cache must
        be literal zeros, because `forward` pads AFTER the norm. Caching raw
        inputs and prepending raw zeros looks equivalent but is not: a zero
        frame passes through LayerNorm as its bias β, and in any trained
        checkpoint β≠0 (measured 0.057–0.092 here), which skewed the first
        RF-length of every stream by up to 0.014 logit while unit tests on a
        fresh head (β=0) passed. Test file: tests/test_tcn_streaming.py sets
        β≠0 explicitly so this cannot silently regress.
        """
        # x: [B, T, D], cache: [B, pad_ctx, D] post-norm
        if not self.causal:
            raise RuntimeError(
                "forward_chunk is undefined for a non-causal block — the "
                "non-causal variant exists only for the offline context oracle")
        z = self.norm(x)
        zin = torch.cat([cache, z], dim=1)                     # [B, pad_ctx+T, D]
        h = zin.transpose(1, 2)
        a = self.dw_local(h[:, :, self.pad_ctx - self.pad_local:])   # -> T
        b = self.dw_ctx(h)                                            # -> T
        g = torch.cat([a, b], dim=1).transpose(1, 2)
        # NOT `zin[:, -self.pad_ctx:]`: with kernel=1 pad_ctx is 0 and `-0:`
        # silently returns the WHOLE sequence as the next cache. Explicit
        # arithmetic yields the correct empty cache instead.
        new_cache = zin[:, zin.shape[1] - self.pad_ctx:].contiguous()
        return x + self.drop(F.glu(self.glu(g), dim=-1)), new_cache


class TCNCountHead(nn.Module):
    """Architecture-ablation head: identical fusion (StackGatedProjection) and
    identical output head (OrdinalConsistentHead) to DeformableCountHead —
    ONLY the temporal module differs: causal deformable block -> stack of
    dual-dilation TCN blocks (dilations 1..32, kernel 3 => 127-frame receptive
    field = 5.08 s of past at the encoder's 25 Hz).

    Why: segment_diagnosis measured the deficit as structural (fragmentation
    2.59x teacher's, boundary F1 0.329 vs 0.540) while overlap recall already
    beats the teacher. Deformable sampling answers "where to look"; it has no
    bias toward piecewise-constant outputs. Dilated TCNs are the standard
    remedy for exactly this failure in temporal action segmentation (MS-TCN).
    """
    def __init__(
        self,
        d_in: int,
        num_classes: int = 4,
        d_model: int = 256,
        dilations=(1, 2, 4, 8, 16, 32),
        kernel: int = 3,
        dropout: float = 0.1,
        stack_dims: Optional[List[int]] = None,
        head_hidden_dim: int = 0,
        causal: bool = True,
        stack_indices: Optional[List[int]] = None,
        stack_input_mask: Optional[List[int]] = None,
        renormalize_active_gate: bool = False,
    ):
        """``causal=False`` (context oracle only) pads convolutions
        symmetrically; streaming then raises.  ``stack_indices`` (stack probe,
        e.g. ``[3]``) slices the hypercolumn to the chosen encoder stacks
        before fusion — the input stays the full hypercolumn so the encoder
        path is byte-identical across probe arms and only the head's view
        changes.
        """
        super().__init__()
        self.causal = bool(causal)
        self._slices: Optional[List[tuple]] = None
        if stack_input_mask is not None and stack_indices is not None:
            raise ValueError(
                "stack_input_mask is the parameter-matched ablation; it cannot "
                "be combined with shape-changing stack_indices")
        if stack_input_mask is not None and not stack_dims:
            raise ValueError("stack_input_mask requires stack_dims")
        if stack_indices is not None:
            if not stack_dims:
                raise ValueError("stack_indices requires stack_dims")
            idx = [int(i) for i in stack_indices]
            if len(idx) == 0 or len(set(idx)) != len(idx) \
                    or any(i < 0 or i >= len(stack_dims) for i in idx):
                raise ValueError(
                    f"stack_indices must be distinct indices into "
                    f"{len(stack_dims)} stacks, got {stack_indices}")
            offsets = [0]
            for d in stack_dims:
                offsets.append(offsets[-1] + int(d))
            self._slices = [(offsets[i], offsets[i + 1]) for i in idx]
            stack_dims = [int(stack_dims[i]) for i in idx]
            d_sel = sum(stack_dims)
        else:
            d_sel = d_in
        if stack_dims:
            assert sum(stack_dims) == d_sel, (stack_dims, d_sel)
            self.fusion = StackGatedProjection(
                stack_dims,
                d_out=d_model,
                dropout=dropout,
                stack_input_mask=stack_input_mask,
                renormalize_active_gate=renormalize_active_gate,
            )
        else:
            self.fusion = nn.Sequential(nn.LayerNorm(d_sel), nn.Linear(d_sel, d_model))
        self.blocks = nn.ModuleList(
            [CausalDualDilationBlock(d_model, d, kernel, dropout, causal=self.causal)
             for d in dilations])
        self.head = OrdinalConsistentHead(d_model, num_classes,
                                          hidden_dim=head_hidden_dim, dropout=dropout)

    def _select(self, x: torch.Tensor) -> torch.Tensor:
        if self._slices is None:
            return x
        return torch.cat([x[..., s:e] for s, e in self._slices], dim=-1)

    def forward(self, x: torch.Tensor):
        h = self.fusion(self._select(x))
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)

    def forward_streaming(self, x: torch.Tensor, caches: Optional[List[torch.Tensor]] = None):
        """Chunk-wise forward carrying one input-cache per TCN block, matching
        the DeformableCountHead streaming interface used by infer_streaming.

        Without this, each 16-frame chunk would re-zero-pad every block and the
        effective receptive field would collapse from 127 frames (5.08 s) to
        the chunk length (0.64 s). Fusion and the output head are per-frame,
        so the block caches are the ONLY state.
        """
        if not self.causal:
            raise RuntimeError(
                "forward_streaming is undefined for causal=False — the "
                "non-causal variant exists only for the offline context oracle")
        h = self.fusion(self._select(x))
        if caches is None:
            caches = [h.new_zeros(h.shape[0], blk.pad_ctx, h.shape[-1])
                      for blk in self.blocks]
        # zip() stops at the shorter sequence: caches=[] would silently skip
        # EVERY TCN block and pass fusion output straight to the head.
        if len(caches) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} caches, got {len(caches)}")
        for i, (blk, c) in enumerate(zip(self.blocks, caches)):
            want = (h.shape[0], blk.pad_ctx, h.shape[-1])
            if tuple(c.shape) != want or c.dtype != h.dtype or c.device != h.device:
                raise ValueError(
                    f"cache[{i}]: shape {tuple(c.shape)} dtype {c.dtype} device {c.device}, "
                    f"expected {want} {h.dtype} {h.device}")
        new_caches = []
        for blk, c in zip(self.blocks, caches):
            h, c2 = blk.forward_chunk(h, c)
            new_caches.append(c2)
        return self.head(h), new_caches


class EventStateOutput(NamedTuple):
    """Stable output contract for :class:`TCNEventStateCountHead`.

    The first three fields deliberately match every existing ZipCount head so
    metric/evaluation code that indexes ``logits[0:3]`` remains valid.  The
    final two fields are training auxiliaries:

      0. ``count_logits``: filtered, normalized log-probabilities [B,T,C]
      1. ``vad_logits``: existing cumulative P(count>=1) logits [B,T,1]
      2. ``overlap_logits``: existing P(count>=2) logits [B,T,1]
      3. ``raw_count_logits``: unfiltered absolute-count emissions [B,T,C]
      4. ``event_logits``: DOWN/STAY/UP logits [B,T,3]

    ``count_logits`` are valid logits even though they are already normalized:
    applying softmax recovers the filtered state posterior exactly.
    """

    count_logits: torch.Tensor
    vad_logits: torch.Tensor
    overlap_logits: torch.Tensor
    raw_count_logits: torch.Tensor
    event_logits: torch.Tensor


class EventStateCache(NamedTuple):
    """Opaque streaming state for ``TCNEventStateCountHead``.

    ``log_q`` is always float32, even under autocast/fp16 inference.  It is the
    posterior at the final emitted frame and must be reset only at a real
    recording boundary, not at an encoder chunk boundary.
    """

    tcn_caches: List[torch.Tensor]
    log_q: torch.Tensor


class CausalSignedStickyFilter(nn.Module):
    """Differentiable causal Markov filter for piecewise-constant counts.

    At frame ``t`` the event posterior supplies transition mass in three
    directions (DOWN, STAY, UP).  For previous count ``i``, DOWN moves to
    ``i-1`` and UP moves to ``i+1``; impossible mass at the two edges is
    folded into STAY.  This tridiagonal prior matches the measured data
    invariant that 99.812% of count changes have magnitude one.  The tiny
    numerical floor plus the absolute emission still permit recovery from a
    missed event or a rare larger jump.  The resulting row-stochastic
    transition is combined with the current absolute-count emission using a
    standard Bayesian filtering update.

    The recurrence deliberately runs in float32/log-space.  This prevents
    long streams and strongly sticky event logits from underflowing in fp16.
    It is strictly left-to-right and therefore adds no lookahead.
    """

    DOWN = 0
    STAY = 1
    UP = 2

    def __init__(self, num_classes: int = 4, emission_scale: float = 1.0,
                 eps: float = 1.0e-6,
                 event_class_weights: Optional[List[float]] = None):
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >=2, got {num_classes}")
        if emission_scale <= 0:
            raise ValueError(f"emission_scale must be >0, got {emission_scale}")
        if eps <= 0:
            raise ValueError(f"eps must be >0, got {eps}")
        self.num_classes = int(num_classes)
        self.emission_scale = float(emission_scale)
        self.eps = float(eps)
        if event_class_weights is None:
            event_class_weights = [1.0, 1.0, 1.0]
        correction = torch.as_tensor(event_class_weights, dtype=torch.float32)
        if correction.shape != (3,) or (correction <= 0).any():
            raise ValueError(
                "event_class_weights must contain three positive values in "
                "DOWN/STAY/UP order")
        # Weighted CE learns q(k|x) proportional to w_k p(k|x).  The Markov
        # transition needs the calibrated p, not the cost-sensitive q, so
        # subtract log(w_k) before softmax.  Non-persistent: this is experiment
        # configuration, not a learned weight, and must not break old TCN
        # checkpoint initialization.
        self.register_buffer(
            "event_log_weight", correction.log(), persistent=False)

    def transition_matrix(self, event_logits: torch.Tensor) -> torch.Tensor:
        """Return row-stochastic ``A[..., previous_count, next_count]``.

        Args:
            event_logits: [..., 3] in stable DOWN/STAY/UP order.
        Returns:
            Float32 tensor [..., C, C].
        """
        if event_logits.ndim < 1 or event_logits.shape[-1] != 3:
            raise ValueError(
                f"event_logits must end in 3 DOWN/STAY/UP logits, got "
                f"{tuple(event_logits.shape)}")
        g = torch.softmax(
            event_logits.float() - self.event_log_weight, dim=-1)
        out_shape = (*g.shape[:-1], self.num_classes, self.num_classes)
        a = g.new_zeros(out_shape)
        down, stay, up = g[..., self.DOWN], g[..., self.STAY], g[..., self.UP]

        for i in range(self.num_classes):
            diagonal = stay
            if i == 0:
                diagonal = diagonal + down
            else:
                a[..., i, i - 1] = down
            if i == self.num_classes - 1:
                diagonal = diagonal + up
            else:
                a[..., i, i + 1] = up
            a[..., i, i] = diagonal

        # Softmax can produce exact zeros for extreme fp32 logits.  A tiny
        # floor keeps the following log finite; re-normalization preserves the
        # row-stochastic invariant after applying that floor.
        a = a.clamp_min(self.eps)
        return a / a.sum(dim=-1, keepdim=True)

    def _step(
        self,
        log_q: torch.Tensor,
        raw_count_logits: torch.Tensor,
        event_logits: torch.Tensor,
    ) -> torch.Tensor:
        log_a = self.transition_matrix(event_logits).log()       # [B,C,C]
        # Previous state is axis i, next state is axis j.
        log_prior = torch.logsumexp(log_q.unsqueeze(-1) + log_a, dim=-2)
        log_emission = torch.log_softmax(raw_count_logits.float(), dim=-1)
        return torch.log_softmax(
            log_prior + self.emission_scale * log_emission, dim=-1)

    def forward(
        self,
        raw_count_logits: torch.Tensor,
        event_logits: torch.Tensor,
        initial_log_q: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter a full sequence or one streaming chunk.

        When ``initial_log_q`` is ``None``, frame zero initializes the state
        directly from its absolute emission and no transition is fabricated
        across the crop/recording boundary.  When it is provided, every frame
        transitions from the carried state, including the first chunk frame.
        """
        if raw_count_logits.ndim != 3:
            raise ValueError(
                f"raw_count_logits must be [B,T,C], got {tuple(raw_count_logits.shape)}")
        if event_logits.shape[:2] != raw_count_logits.shape[:2] or event_logits.shape[-1] != 3:
            raise ValueError(
                f"event_logits must be [B,T,3] aligned with emissions, got "
                f"{tuple(event_logits.shape)} vs {tuple(raw_count_logits.shape)}")
        if raw_count_logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"expected {self.num_classes} count classes, got "
                f"{raw_count_logits.shape[-1]}")
        batch, time, _ = raw_count_logits.shape
        if time == 0:
            raise ValueError("event-state filter requires at least one frame")

        if initial_log_q is None:
            log_q = torch.log_softmax(raw_count_logits[:, 0].float(), dim=-1)
            outputs = [log_q]
            start = 1
        else:
            want = (batch, self.num_classes)
            if tuple(initial_log_q.shape) != want:
                raise ValueError(
                    f"initial_log_q has shape {tuple(initial_log_q.shape)}, expected {want}")
            if initial_log_q.device != raw_count_logits.device:
                raise ValueError(
                    f"initial_log_q device {initial_log_q.device}, expected "
                    f"{raw_count_logits.device}")
            if initial_log_q.dtype != torch.float32:
                raise ValueError(
                    f"initial_log_q dtype {initial_log_q.dtype}, expected torch.float32")
            # Accept any finite log-score vector from a caller, but normalize
            # it so the cache contract is unambiguous.
            if not torch.isfinite(initial_log_q).all():
                raise ValueError("initial_log_q must be finite")
            log_q = torch.log_softmax(initial_log_q, dim=-1)
            outputs = []
            start = 0

        for t in range(start, time):
            log_q = self._step(log_q, raw_count_logits[:, t], event_logits[:, t])
            outputs.append(log_q)

        return torch.stack(outputs, dim=1), log_q


class TCNEventStateCountHead(TCNCountHead):
    """Dual-dilation TCN plus a causal DOWN/STAY/UP count-state decoder.

    This is a new opt-in head; ``TCNCountHead`` and all legacy checkpoint
    behavior remain unchanged.  The inherited fusion, TCN blocks and ordinal
    projections retain the exact same parameter names, so a trained
    ``tcn_ordinal`` checkpoint can initialize this head with only
    ``event_proj.*`` missing.

    See :class:`EventStateOutput` for the stable five-field output order and
    :class:`EventStateCache` for the streaming cache contract.
    """

    def __init__(
        self,
        d_in: int,
        num_classes: int = 4,
        d_model: int = 256,
        dilations=(1, 2, 4, 8, 16, 32),
        kernel: int = 3,
        dropout: float = 0.1,
        stack_dims: Optional[List[int]] = None,
        head_hidden_dim: int = 0,
        event_stay_bias: float = 2.9,
        state_emission_scale: float = 1.0,
        use_state_filter: bool = True,
        event_filter_class_weights: Optional[List[float]] = None,
    ):
        super().__init__(
            d_in=d_in,
            num_classes=num_classes,
            d_model=d_model,
            dilations=dilations,
            kernel=kernel,
            dropout=dropout,
            stack_dims=stack_dims,
            head_hidden_dim=head_hidden_dim,
        )
        self.num_classes = int(num_classes)
        self.use_state_filter = bool(use_state_filter)
        self.event_proj = nn.Linear(d_model, 3)
        with torch.no_grad():
            # A newly-added random event head must not inject random state
            # jumps into a strong pretrained count model before seeing one
            # update.  Start from the measured sticky prior; supervision can
            # then learn feature-dependent deviations immediately.
            self.event_proj.weight.zero_()
            self.event_proj.bias.zero_()
            self.event_proj.bias[CausalSignedStickyFilter.STAY] = float(event_stay_bias)
        self.state_filter = CausalSignedStickyFilter(
            num_classes=num_classes,
            emission_scale=state_emission_scale,
            event_class_weights=event_filter_class_weights,
        )

    def _project(self, h: torch.Tensor, initial_log_q: Optional[torch.Tensor] = None):
        raw_count, vad, overlap = self.head(h)
        event = self.event_proj(h)
        filtered, last_log_q = self.state_filter(raw_count, event, initial_log_q)
        # C1 (event auxiliary) and C2 (event-conditioned filter) instantiate
        # exactly the same module and parameter set.  This switch changes only
        # which count view is public, so C1 is a clean supervision/parameter
        # control for C2.  We still compute/carry the filter in C1 to keep the
        # streaming-state contract and runtime path identical.
        public_count = filtered if self.use_state_filter else raw_count
        return EventStateOutput(public_count, vad, overlap, raw_count, event), last_log_q

    def forward(self, x: torch.Tensor) -> EventStateOutput:
        h = self.fusion(x)
        for blk in self.blocks:
            h = blk(h)
        output, _ = self._project(h)
        return output

    def forward_streaming(
        self,
        x: torch.Tensor,
        caches: Optional[EventStateCache] = None,
    ) -> Tuple[EventStateOutput, EventStateCache]:
        """Streaming forward with TCN history and the final count posterior."""
        h = self.fusion(x)
        if h.shape[1] == 0:
            raise ValueError("streaming chunks must contain at least one frame")

        if caches is None:
            block_caches = [
                h.new_zeros(h.shape[0], blk.pad_ctx, h.shape[-1])
                for blk in self.blocks
            ]
            initial_log_q = None
        else:
            if not isinstance(caches, EventStateCache):
                raise ValueError(
                    f"caches must be EventStateCache or None, got {type(caches).__name__}")
            block_caches = caches.tcn_caches
            initial_log_q = caches.log_q

        if len(block_caches) != len(self.blocks):
            raise ValueError(
                f"expected {len(self.blocks)} TCN caches, got {len(block_caches)}")
        for i, (blk, cache) in enumerate(zip(self.blocks, block_caches)):
            want = (h.shape[0], blk.pad_ctx, h.shape[-1])
            if (tuple(cache.shape) != want or cache.dtype != h.dtype
                    or cache.device != h.device):
                raise ValueError(
                    f"tcn_caches[{i}]: shape {tuple(cache.shape)} dtype "
                    f"{cache.dtype} device {cache.device}, expected {want} "
                    f"{h.dtype} {h.device}")

        new_block_caches = []
        for blk, cache in zip(self.blocks, block_caches):
            h, new_cache = blk.forward_chunk(h, cache)
            new_block_caches.append(new_cache)

        output, last_log_q = self._project(h, initial_log_q)
        return output, EventStateCache(new_block_caches, last_log_q)


class DeformableCountHead(nn.Module):
    """v2 head: hypercolumn fusion -> causal deformable temporal block ->
    ordinal-consistent count head. All trainable parts live here; the
    Zipformer backbone stays frozen.

    d_in: backbone output dim (sum(stack_dims) hypercolumn, or plain 512).
    """
    def __init__(
        self,
        d_in: int,
        num_classes: int = 4,
        d_model: int = 256,
        num_layers: int = 2,
        num_points: int = 4,
        num_groups: int = 4,
        max_offset: int = 16,
        dropout: float = 0.1,
        stack_dims: Optional[List[int]] = None,
        head_hidden_dim: int = 0,
    ):
        super().__init__()
        if stack_dims:
            assert sum(stack_dims) == d_in, (stack_dims, d_in)
            self.fusion = StackGatedProjection(stack_dims, d_out=d_model, dropout=dropout)
        else:
            self.fusion = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_model))
        self.temporal = CausalDeformableTemporalBlock(
            d_model, num_layers=num_layers, num_points=num_points,
            num_groups=num_groups, max_offset=max_offset, dropout=dropout,
        )
        self.head = OrdinalConsistentHead(d_model, num_classes, hidden_dim=head_hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor):
        # x: [B, T, d_in]
        h = self.fusion(x)                # [B, T, d_model]
        h, _ = self.temporal(h)           # [B, T, d_model]
        return self.head(h)

    def forward_streaming(self, x: torch.Tensor, caches: Optional[List[torch.Tensor]] = None):
        """Chunk-wise forward carrying deformable caches across chunks."""
        h = self.fusion(x)
        h, new_caches = self.temporal(h, caches)
        return self.head(h), new_caches
