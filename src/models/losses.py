import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Sequence, Tuple

class ZipCountLoss(nn.Module):
    """Multi-task loss for frame-level speaker counting.

    Core (v1): weighted CE/focal on 4-class count + BCE on VAD/overlap
    aux heads + temporal smoothness.

    v2 additions (all default-off, enable via config):
      count_loss_type "sord": soft ordinal targets q_c ∝ exp(-α|c−y|)
          (SORD; count is ordinal — being off by 2 must cost more than by 1).
      lambda_emae:       |E_{c~p}[c] − y| expected-count regularizer (MAE-aware).
      lambda_consistency: ties the count softmax to the cumulative aux heads:
          P(count≥1) from softmax ↔ sigmoid(vad), P(count≥2) ↔ sigmoid(overlap).
      lambda_monotonic:  relu(σ(overlap) − σ(vad)) — ordinal rank consistency
          between the cumulative heads (CORAL-style constraint as a penalty).
      lambda_dice:       soft dice on the overlap channel (rare-class shape
          loss, cf. mask losses in Mask2Former/EEND-M2F).
    """
    def __init__(
        self,
        class_weights: torch.Tensor = None,
        count_loss_type: str = "focal",
        focal_gamma: float = 2.0,
        lambda_vad: float = 0.3,
        lambda_overlap: float = 0.5,
        lambda_smooth: float = 0.05,
        smooth_type: str = "kl",
        sord_alpha: float = 1.5,
        lambda_emae: float = 0.0,
        lambda_consistency: float = 0.0,
        lambda_monotonic: float = 0.0,
        lambda_dice: float = 0.0,
        lambda_event: float = 0.0,
        event_class_weights: Optional[Sequence[float]] = None,
        lambda_raw_anchor: float = 0.0,
    ):
        super().__init__()
        self.count_loss_type = count_loss_type
        self.focal_gamma = focal_gamma
        self.lambda_vad = lambda_vad
        self.lambda_overlap = lambda_overlap
        self.lambda_smooth = lambda_smooth
        self.smooth_type = smooth_type
        self.sord_alpha = sord_alpha
        self.lambda_emae = lambda_emae
        self.lambda_consistency = lambda_consistency
        self.lambda_monotonic = lambda_monotonic
        self.lambda_dice = lambda_dice
        self.lambda_event = float(lambda_event)
        self.lambda_raw_anchor = float(lambda_raw_anchor)

        # Register class weights as buffer so it moves with the model
        if class_weights is None:
            class_weights = torch.ones(4)
        self.register_buffer("class_weights", class_weights)
        if event_class_weights is None:
            event_class_weights = (1.0, 1.0, 1.0)
        event_class_weights = torch.as_tensor(event_class_weights, dtype=torch.float32)
        if event_class_weights.shape != (3,) or (event_class_weights <= 0).any():
            raise ValueError(
                "event_class_weights must contain three positive values in "
                "DOWN/STAY/UP order")
        self.register_buffer("event_class_weights", event_class_weights)
        
    def _focal_loss(self, logits, targets, mask):
        # logits: [B, T, C], targets: [B, T], mask: [B, T]
        ce_loss = F.cross_entropy(
            logits.transpose(1, 2), 
            targets, 
            weight=self.class_weights, 
            reduction='none'
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
        
        return (focal_loss * mask).sum() / mask.sum().clamp(min=1)
        
    def _ce_loss(self, logits, targets, mask):
        ce_loss = F.cross_entropy(
            logits.transpose(1, 2),
            targets,
            weight=self.class_weights,
            reduction='none'
        )
        return (ce_loss * mask).sum() / mask.sum().clamp(min=1)

    def _sord_loss(self, logits, targets, mask):
        """Soft ORDinal cross-entropy: targets are q_c ∝ exp(-α|c − y|),
        so mass leaks to neighboring counts and far misses cost more.
        logits: [B, T, C], targets: [B, T] int, mask: [B, T]"""
        C = logits.size(-1)
        classes = torch.arange(C, device=logits.device, dtype=logits.dtype)  # [C]
        dist = (classes.view(1, 1, C) - targets.unsqueeze(-1).to(logits.dtype)).abs()
        q = F.softmax(-self.sord_alpha * dist, dim=-1)                        # [B, T, C]
        logp = F.log_softmax(logits, dim=-1)
        per_frame = -(q * logp).sum(dim=-1)                                   # [B, T]
        # Per-frame class weighting by the hard label (same role as in CE)
        w = self.class_weights[targets.clamp(min=0, max=C - 1)]
        per_frame = per_frame * w
        return (per_frame * mask).sum() / mask.sum().clamp(min=1)

    def _expected_count_mae(self, logits, targets, mask):
        """|E_{c~p}[c] − y|: differentiable, ordinal-aware count regression."""
        C = logits.size(-1)
        probs = F.softmax(logits, dim=-1)
        classes = torch.arange(C, device=logits.device, dtype=logits.dtype)
        expected = (probs * classes.view(1, 1, C)).sum(dim=-1)                # [B, T]
        err = (expected - targets.to(logits.dtype)).abs()
        return (err * mask).sum() / mask.sum().clamp(min=1)

    @staticmethod
    def _dice_loss(probs, targets, mask, eps: float = 1.0):
        """Soft dice over the masked frames. probs/targets/mask: [B, T]."""
        p = probs * mask
        g = targets * mask
        inter = (p * g).sum()
        return 1.0 - (2.0 * inter + eps) / (p.sum() + g.sum() + eps)
        
    def _smoothness_loss(self, logits, mask):
        # logits: [B, T, C]
        if logits.size(1) < 2:
            return torch.tensor(0.0, device=logits.device)

        mask_pair = mask[:, 1:] & mask[:, :-1]  # both frames of the pair must be valid

        if self.smooth_type == "kl":
            # Symmetric KL == Jeffreys divergence: KL(p||q) + KL(q||p) =
            # sum (p-q)(log p - log q). Computed this way (never calling
            # F.kl_div, never taking log(p) of a raw softmax output) so no
            # term ever needs log(0): log_softmax is numerically stable for
            # arbitrarily extreme logits, and the (p-q) factor damps the
            # product to 0 exactly where p and q both vanish together.
            # F.kl_div(log(p+eps), q) does NOT have this property -- its
            # backward pass differentiates through log(target) directly, and
            # a softmax output that underflows to an exact 0.0 (routine at
            # fp16, and reachable at fp32 with large-magnitude logits) makes
            # that gradient term -inf/nan even though the eps-clamped
            # forward value looks finite. Reproduced directly: logits in
            # [-100, 100] gave a finite fp32 loss but a non-finite fp32
            # gradient under the old F.kl_div formulation.
            logp = F.log_softmax(logits.float(), dim=-1)
            p = logp.exp()
            loss = ((p[:, 1:] - p[:, :-1]) * (logp[:, 1:] - logp[:, :-1])).sum(-1)
        else:  # l1
            probs = F.softmax(logits, dim=-1)
            loss = F.l1_loss(probs[:, 1:], probs[:, :-1], reduction='none').mean(dim=-1)

        loss = torch.where(mask_pair, loss, torch.zeros_like(loss))
        return loss.sum() / mask_pair.sum().clamp(min=1)

    def _count_objective(self, logits, targets, mask):
        """Apply the configured frame-count objective to one set of logits."""
        if self.count_loss_type == "focal":
            return self._focal_loss(logits, targets, mask)
        if self.count_loss_type == "sord":
            return self._sord_loss(logits, targets, mask)
        return self._ce_loss(logits, targets, mask)

    def _event_loss(self, event_logits, labels, final_mask):
        """Signed count-change CE in stable DOWN/STAY/UP order.

        Event ``t`` describes the observed transition ``labels[t-1] ->
        labels[t]``.  The first crop frame has no predecessor and is therefore
        always excluded.  Requiring both frames to be valid also prevents a
        padded tail from creating a fake event.  Sign supervision deliberately
        keeps the rare >1 jumps without inventing an unsupported magnitude
        class; the absolute-count emission remains responsible for magnitude.
        """
        if event_logits.ndim != 3 or event_logits.shape[-1] != 3:
            raise ValueError(
                f"event_logits must be [B,T,3] DOWN/STAY/UP, got "
                f"{tuple(event_logits.shape)}")
        if event_logits.shape[:2] != labels.shape:
            raise ValueError(
                f"event logits/labels are not aligned: {tuple(event_logits.shape)} "
                f"vs {tuple(labels.shape)}")
        if labels.shape[1] < 2:
            return event_logits.float().sum() * 0.0

        pair_mask = final_mask[:, 1:] & final_mask[:, :-1]
        delta = labels[:, 1:] - labels[:, :-1]
        # 0=DOWN, 1=STAY, 2=UP.  Padding values are harmless because pair_mask
        # removes them before reduction.
        targets = torch.ones_like(delta, dtype=torch.long)
        targets = torch.where(delta < 0, torch.zeros_like(targets), targets)
        targets = torch.where(delta > 0, torch.full_like(targets, 2), targets)
        per_pair = F.cross_entropy(
            event_logits[:, 1:].float().transpose(1, 2),
            targets,
            weight=self.event_class_weights,
            reduction="none",
        )
        # torch.where avoids the NaN*False failure mode that previously hid in
        # KD masking if a future loss implementation ever produces a bad row.
        per_pair = torch.where(pair_mask, per_pair, torch.zeros_like(per_pair))
        return per_pair.sum() / pair_mask.sum().clamp_min(1)
        
    def forward(self, logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], labels: torch.Tensor, h_lens: torch.Tensor) -> Dict[str, torch.Tensor]:
        if len(logits) < 3:
            raise ValueError(f"expected at least 3 model outputs, got {len(logits)}")
        count_logits, vad_logits, overlap_logits = logits[:3]
        raw_count_logits = logits[3] if len(logits) >= 4 else None
        event_logits = logits[4] if len(logits) >= 5 else None
        
        # count_logits: [B, T, 4]
        # vad_logits: [B, T, 1]
        # overlap_logits: [B, T, 1]
        # labels: [B, T_label]
        # h_lens: [B]
        
        B, T, _ = count_logits.shape
        
        # Make sure labels match T
        if labels.shape[1] != T:
            # Note: label alignment should happen in the model or dataset before this
            # This is a fallback
            labels = labels[:, :T]
            
        # Create mask based on h_lens
        mask = torch.arange(T, device=count_logits.device).unsqueeze(0) < h_lens.unsqueeze(1)
        
        # Valid labels only (ignore padding -100)
        valid_label_mask = (labels >= 0)
        final_mask = mask & valid_label_mask
        
        # 1. Count Loss
        # Need to clamp labels to 0-3 just in case, -100 will be ignored by mask
        clamped_labels = labels.clamp(min=0, max=3)
        
        loss_count = self._count_objective(
            count_logits, clamped_labels, final_mask)
            
        # 2. VAD Loss (Binary: count >= 1)
        vad_labels = (clamped_labels >= 1).float()
        vad_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        loss_vad = vad_loss_fn(vad_logits.squeeze(-1), vad_labels)
        loss_vad = (loss_vad * final_mask).sum() / final_mask.sum().clamp(min=1)
        
        # 3. Overlap Loss (Binary: count >= 2)
        overlap_labels = (clamped_labels >= 2).float()
        loss_overlap = vad_loss_fn(overlap_logits.squeeze(-1), overlap_labels)
        loss_overlap = (loss_overlap * final_mask).sum() / final_mask.sum().clamp(min=1)
        
        # 4. Smoothness Loss
        loss_smooth = torch.tensor(0.0, device=count_logits.device)
        if self.lambda_smooth > 0:
            loss_smooth = self._smoothness_loss(count_logits, final_mask)

        fmask = final_mask.float()
        zero = torch.tensor(0.0, device=count_logits.device)

        # 5. Expected-count MAE (ordinal-aware regression on the softmax)
        loss_emae = self._expected_count_mae(count_logits, clamped_labels, fmask) \
            if self.lambda_emae > 0 else zero

        # 6. Consistency: softmax-derived cumulative probs vs the aux heads
        loss_cons = zero
        if self.lambda_consistency > 0:
            probs = F.softmax(count_logits, dim=-1)                     # [B, T, 4]
            p_ge1 = 1.0 - probs[..., 0]                                 # P(count >= 1)
            p_ge2 = probs[..., 2] + probs[..., 3]                       # P(count >= 2)
            s_vad = torch.sigmoid(vad_logits.squeeze(-1))
            s_ov = torch.sigmoid(overlap_logits.squeeze(-1))
            loss_cons = (((p_ge1 - s_vad) ** 2 + (p_ge2 - s_ov) ** 2) * fmask).sum() \
                / fmask.sum().clamp(min=1)

        # 7. Ordinal monotonicity between cumulative heads: P(>=2) <= P(>=1)
        loss_mono = zero
        if self.lambda_monotonic > 0:
            s_vad = torch.sigmoid(vad_logits.squeeze(-1))
            s_ov = torch.sigmoid(overlap_logits.squeeze(-1))
            loss_mono = (F.relu(s_ov - s_vad) * fmask).sum() / fmask.sum().clamp(min=1)

        # 8. Dice on the overlap channel (rare positive class)
        loss_dice = zero
        if self.lambda_dice > 0:
            s_ov = torch.sigmoid(overlap_logits.squeeze(-1))
            loss_dice = self._dice_loss(s_ov, overlap_labels, fmask)

        # 9. Signed sparse count-change supervision and an absolute-emission
        # anchor for the recurrent state filter.  Fail loudly if a config turns
        # either loss on for a legacy three-output head.
        loss_event = zero
        if self.lambda_event > 0:
            if event_logits is None:
                raise ValueError(
                    "lambda_event > 0 requires a tcn_state_ordinal head with "
                    "event logits")
            loss_event = self._event_loss(event_logits, labels, final_mask)

        loss_raw_anchor = zero
        if self.lambda_raw_anchor > 0:
            if raw_count_logits is None:
                raise ValueError(
                    "lambda_raw_anchor > 0 requires a tcn_state_ordinal head "
                    "with raw count emissions")
            loss_raw_anchor = self._count_objective(
                raw_count_logits, clamped_labels, final_mask)

        # Total
        loss = (
            loss_count
            + self.lambda_vad * loss_vad
            + self.lambda_overlap * loss_overlap
            + self.lambda_smooth * loss_smooth
            + self.lambda_emae * loss_emae
            + self.lambda_consistency * loss_cons
            + self.lambda_monotonic * loss_mono
            + self.lambda_dice * loss_dice
            + self.lambda_event * loss_event
            + self.lambda_raw_anchor * loss_raw_anchor
        )

        return {
            "loss": loss,
            "loss_count": loss_count,
            "loss_vad": loss_vad,
            "loss_overlap": loss_overlap,
            "loss_smooth": loss_smooth,
            "loss_emae": loss_emae,
            "loss_consistency": loss_cons,
            "loss_monotonic": loss_mono,
            "loss_dice": loss_dice,
            "loss_event": loss_event,
            "loss_raw_anchor": loss_raw_anchor,
        }
