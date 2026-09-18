import os
import yaml
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("Warning: tensorboard not found. Using a dummy writer.")
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

from tqdm import tqdm
from src.data.dataset import SpeakerCountDataset, collate_fn
from src.data.feature_extractor import feature_extractor_for_config
from src.data.label_utils import align_batch_labels, align_batch_posteriors
from src.models.zipcount_v1 import build_model
from src.models.losses import ZipCountLoss
from src.models.structured_losses import PyramidStructuredLoss, PyramidAuxiliaryLoss
from src.utils.metrics import compute_metrics, aggregate_metrics, format_confusion
from src.utils.train_mode import set_backbone_partial_train_mode

def boundary_aware_smooth_loss(count_logits, aligned_labels, h_lens,
                               tau: float = 4.0, guard: int = 2):
    """Truncated MSE on adjacent log-probs (MS-TCN's T-MSE), but weighted to
    ZERO within +-guard frames of a ground-truth label change.

    Motivation (measured, vox_sel): the model's overlap RECALL already beats
    the DiariZen teacher (onset 0.843 vs 0.707, missed segments 13.6% vs
    25.8%) — the deficit is structural: fragmentation 2.59x vs 1.27x and
    boundary F1 0.329 vs 0.540. Plain smoothing would fix flicker by erasing
    short overlaps too; gating it off around true boundaries penalises
    flicker inside segments while leaving transitions free to be sharp.
    """
    logp = torch.log_softmax(count_logits.float(), dim=-1)
    d = (logp[:, 1:] - logp[:, :-1]).pow(2).mean(-1).clamp_max(tau)   # [B,T-1]
    lab = aligned_labels
    T = lab.shape[1]
    inlen = torch.arange(T, device=lab.device)[None, :] < h_lens.to(lab.device)[:, None]
    valid = (lab[:, 1:] >= 0) & (lab[:, :-1] >= 0) & inlen[:, 1:] & inlen[:, :-1]
    bnd = ((lab[:, 1:] != lab[:, :-1]) & valid).float().unsqueeze(1)
    if guard > 0:
        bnd = torch.nn.functional.max_pool1d(
            bnd, kernel_size=2 * guard + 1, stride=1, padding=guard)
    w = (1.0 - bnd.squeeze(1)) * valid.float()
    return (d * w).sum() / w.sum().clamp_min(1)


def kd_loss(count_logits, teacher_post, teacher_mask, aligned_labels,
            label_lens, h_lens, temperature: float = 1.0, return_stats: bool = False,
            mask_mode: str = "agree", uncertain_max_prob: float = 0.9):
    """Knowledge distillation from DiariZen (offline teacher) to the causal
    streaming student, on frames where the teacher is VERIFIED CORRECT.

    Two guards make this safe:
      1. `teacher_mask` — only windows that have a teacher posterior at all.
         Synthetic LibriMix windows deliberately have none: the teacher scores
         only ~25% there (it calls fully-overlapped mixes "1 speaker") and
         would teach the student the opposite of the truth.
      2. agreement mask — only frames where the teacher's argmax matches the
         ground-truth label. Where the teacher is wrong we simply do not learn
         from it; where it is right, its SOFT distribution still adds what a
         hard label cannot (how confident, and what the runner-up is).

    Forward KL( teacher || student ), teacher held fixed.
    """
    B, T, C = count_logits.shape
    dev = count_logits.device
    # MUST be float32: under AMP `count_logits` is fp16, where 1e-8 underflows to
    # exactly 0, so clamping cannot protect the log/divide and all-zero teacher
    # rows (padding, or a window with no teacher) produce 0/0 -> NaN. NaN then
    # survives any `* mask` because NaN * False is still NaN, poisoning the whole
    # batch loss and every gradient, which GradScaler silently turns into a
    # skipped step while the scheduler keeps advancing.
    t_post = align_batch_posteriors(teacher_post, label_lens, h_lens).float()

    # A row only carries teacher information if it actually received probability
    # mass. This single test covers temporal padding, windows without a teacher,
    # frames padded to fix a +-1 length mismatch, and empty alignment buckets.
    #
    # Deliberately MAJORITY coverage, not "no fake frame at all": a bucket that
    # averaged 3 real frames and 1 zero row has mass 0.75 and is kept. That is
    # sound because align_batch_posteriors divides by the full bucket size and
    # we renormalise by the same mass below, so the surviving posterior is the
    # mean over the COVERED frames only — the zero row never dilutes it. Length
    # mismatches are +-1 frame, so at most the final bucket is ever partial.
    has_mass = t_post.sum(-1) > 0.5

    valid = torch.arange(T, device=dev)[None, :] < h_lens.to(dev)[:, None]
    valid = valid & teacher_mask.to(dev)[:, None]
    valid = valid & (aligned_labels >= 0)
    valid = valid & has_mass
    # keep only frames where the teacher agrees with ground truth
    valid = valid & (t_post.argmax(-1) == aligned_labels)
    # Phase 2 (3 paired seeds) showed plain "agree" adds nothing measurable: it
    # keeps 95.6% of class-1 frames but only 57.5% of class-3, and what survives
    # is near one-hot (mean max-prob 0.937, 55.3% above 0.99). A KL against an
    # almost-one-hot target on frames the student already gets right is just the
    # hard label again at lower weight. "agree_uncertain" additionally drops the
    # saturated frames, keeping only where the teacher is RIGHT but HESITANT —
    # the frames whose runner-up mass is the actual dark knowledge. This shrinks
    # the mask a lot, so lambda_kd usually needs to go up to compensate.
    if mask_mode == "agree_uncertain":
        # renormalise before thresholding: a partially covered bucket sums to
        # <1, so raw max would look under-confident purely from missing mass
        conf = t_post.max(-1).values / t_post.sum(-1).clamp_min(1e-6)
        valid = valid & (conf < uncertain_max_prob)
    elif mask_mode != "agree":
        raise ValueError(f"unknown distill.mask_mode: {mask_mode!r}")
    if not valid.any():
        return count_logits.sum() * 0.0

    log_p_student = torch.log_softmax(count_logits.float() / temperature, dim=-1)
    # normalise only where safe; invalid rows are replaced by a dummy uniform so
    # no NaN/Inf can be created before the mask is applied
    denom = t_post.sum(-1, keepdim=True)
    t_post = torch.where(valid.unsqueeze(-1),
                         t_post / denom.clamp_min(1e-6),
                         torch.full_like(t_post, 1.0 / C))
    t_post = t_post.clamp_min(1e-6)

    # The teacher is stored as probabilities at T=1, so it must be softened by
    # the SAME temperature as the student. Softening only the student inverts
    # the intent: at the optimum softmax(z_s/T)=q implies softmax(z_s) ∝ q^T,
    # i.e. a teacher at 0.94/0.06 trains a student that predicts 0.996/0.004 at
    # inference. KD then transfers the teacher's *confidence* rather than its
    # structure. For probabilities (no logits available) the equivalent of
    # dividing logits by T is q^(1/T), renormalised.
    if temperature != 1.0:
        t_post = t_post.pow(1.0 / temperature)
        t_post = t_post / t_post.sum(-1, keepdim=True).clamp_min(1e-6)
    kl = (t_post * (t_post.log() - log_p_student)).sum(-1)          # [B, T]
    kl = torch.where(valid, kl, torch.zeros_like(kl))               # not `kl * valid`
    # T^2 keeps gradient magnitude comparable across temperatures
    loss = kl.sum() / valid.sum().clamp_min(1) * (temperature ** 2)
    if return_stats:
        with torch.no_grad():
            conf = torch.where(valid, t_post.max(-1).values, torch.zeros_like(kl))
            stats = {
                "kd_valid_frac": valid.float().mean(),
                "kd_teacher_conf": conf.sum() / valid.sum().clamp_min(1),
                "kd_kept_per_class": torch.stack([
                    (valid & (aligned_labels == c)).sum() for c in range(C)]).float(),
            }
        return loss, stats
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--max-steps", type=int, default=0, help="Override max_steps for dry-run")
    parser.add_argument("--encoder-type", type=str, default="", help="Override encoder type")
    parser.add_argument("--class-weights", type=str, default="",
                        help="Override loss.class_weights, e.g. '1,1,1,1' or 'auto'")
    parser.add_argument("--count-loss-type", type=str, default="",
                        help="Override loss.count_loss_type "
                             "(ce/focal/sord/cumulative/corn)")
    parser.add_argument("--log-dir", type=str, default="", help="Override training.log_dir")
    parser.add_argument("--init-from", type=str, default="",
                        help="Init model WEIGHTS from a checkpoint (fresh optimizer/step) — stage-2 finetune")
    parser.add_argument("--resume-from", type=str, default="",
                        help="Full resume: weights + optimizer + step from a checkpoint")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.max_steps > 0:
        config["training"]["max_steps"] = args.max_steps
    if args.encoder_type:
        config["model"]["encoder"]["type"] = args.encoder_type
    if args.class_weights:
        config["loss"]["class_weights"] = (
            "auto" if args.class_weights == "auto"
            else [float(x) for x in args.class_weights.split(",")]
        )
    if args.count_loss_type:
        config["loss"]["count_loss_type"] = args.count_loss_type
    if args.log_dir:
        config["training"]["log_dir"] = args.log_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # TF32 matmuls: ~free speedup on Ampere+, precision is fine for training
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Class weights: explicit list in config wins; "auto" reads stats.json;
    # anything else -> uniform (None).
    cw_cfg = config["loss"].get("class_weights", "auto")
    class_weights = None
    if isinstance(cw_cfg, (list, tuple)):
        class_weights = torch.tensor(cw_cfg, dtype=torch.float32).to(device)
        print(f"Using class weights from config: {class_weights}")
    elif cw_cfg == "auto":
        stats_file = config["data"].get("stats_file", "")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                if "class_weights" in stats:
                    class_weights = torch.tensor(stats["class_weights"], dtype=torch.float32).to(device)
                    print(f"Loaded auto class weights: {class_weights}")
    else:
        print("Class weights disabled (uniform).")
    
    # 1. Dataloaders
    train_manifest = config["data"]["train_manifest"]
    augment = config["data"].get("augment")  # train only, never val
    if augment:
        print(f"Waveform augmentation ON: {augment}")
    # Knowledge distillation (optional): teacher count-posteriors on disk.
    # Windows with no teacher file simply train on ground truth alone.
    teacher_dir = config.get("distill", {}).get("teacher_dir", "")
    lambda_kd = float(config.get("distill", {}).get("lambda_kd", 0.0))
    kd_temperature = float(config.get("distill", {}).get("temperature", 1.0))
    # NOTE: key is deliberately NOT `lambda_smooth` — that name belongs to the
    # legacy symmetric-KL smoothness inside ZipCountLoss (losses.py), which the
    # base configs set to 0.02. Sharing the key silently activated BOTH losses
    # in Phase 4/5/6 and turned Phase 4 into a confounded comparison
    # (legacy 0.02->0.15 AND T-MSE 0.15 in one arm). Distinct key, default 0.
    lambda_bsmooth = float(config.get("loss", {}).get("lambda_boundary_smooth", 0.0))
    bsmooth_tau = float(config.get("loss", {}).get("boundary_smooth_tau", 4.0))
    bsmooth_guard = int(config.get("loss", {}).get("boundary_smooth_guard", 2))
    if lambda_bsmooth > 0:
        print(f"Boundary-aware smoothing ON: lambda={lambda_bsmooth} "
              f"tau={bsmooth_tau} guard=±{bsmooth_guard} frames")
    kd_mask_mode = str(config.get("distill", {}).get("mask_mode", "agree"))
    kd_uncertain_max_prob = float(
        config.get("distill", {}).get("uncertain_max_prob", 0.9))
    if lambda_kd > 0:
        where = ("teacher agrees with ground truth" if kd_mask_mode == "agree"
                 else f"teacher agrees AND max-prob < {kd_uncertain_max_prob}")
        print(f"Distillation ON: teacher_dir={teacher_dir} "
              f"lambda_kd={lambda_kd} T={kd_temperature} mask={kd_mask_mode} "
              f"(KD applied only where {where})")

    feature_extractor = feature_extractor_for_config(config)

    train_dataset = SpeakerCountDataset(train_manifest, augment=augment,
                                        teacher_dir=teacher_dir,
                                        feature_extractor=feature_extractor)
    num_workers = int(config["training"].get("num_workers", 8))
    # A fixed seed makes ablations comparable: same init, same shuffle order,
    # same augmentation draws, so a difference between runs is attributable to
    # the thing being ablated rather than to batch order.
    seed = config["training"].get("seed", None)
    loader_gen = None
    if seed is not None:
        seed = int(seed)
        import random as _random
        import numpy as _np
        _random.seed(seed); _np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        loader_gen = torch.Generator(); loader_gen.manual_seed(seed)
        print(f"Seeded run: seed={seed}")

    def _worker_init(wid):
        if seed is not None:
            import random as _random
            import numpy as _np
            _random.seed(seed + wid); _np.random.seed(seed + wid)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
        generator=loader_gen,
        worker_init_fn=_worker_init if seed is not None else None,
    )
    
    val_manifest = config["data"].get("val_manifest", "")
    val_loader = None
    if val_manifest and os.path.exists(val_manifest):
        val_dataset = SpeakerCountDataset(val_manifest, feature_extractor=feature_extractor)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    # Model & Loss
    model = build_model(config)
    if bool(config["model"]["encoder"].get("gradient_checkpointing", False)):
        if not hasattr(model.backbone, "enable_gradient_checkpointing"):
            raise ValueError(
                f"model.encoder.gradient_checkpointing=true but "
                f"{type(model.backbone).__name__} has no enable_gradient_checkpointing()")
        model.backbone.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled on the backbone (active only for "
              "layers actually in train mode once set_train_mode() runs).")
    lambda_event_cfg = float(config["loss"].get("lambda_event", 0.0))
    lambda_raw_anchor_cfg = float(config["loss"].get("lambda_raw_anchor", 0.0))
    if lambda_event_cfg > 0 or lambda_raw_anchor_cfg > 0:
        if config["model"].get("head", {}).get("type") != "tcn_state_ordinal":
            raise ValueError(
                "event/raw-anchor losses require model.head.type=tcn_state_ordinal")
        head_cfg = config["model"].get("head", {})
        if bool(head_cfg.get("use_state_filter", True)):
            loss_event_weights = [float(x) for x in config["loss"].get(
                "event_class_weights", [1, 1, 1])]
            filter_event_weights = [float(x) for x in head_cfg.get(
                "event_filter_class_weights", [1, 1, 1])]
            if filter_event_weights != loss_event_weights:
                raise ValueError(
                    "filtered event-state head must undo the weighted-CE "
                    "calibration shift: model.head.event_filter_class_weights "
                    f"{filter_event_weights} != loss.event_class_weights "
                    f"{loss_event_weights}")
        print(
            "Event-state training: "
            f"lambda_event={lambda_event_cfg} "
            f"weights={config['loss'].get('event_class_weights', [1, 1, 1])} "
            f"lambda_raw_anchor={lambda_raw_anchor_cfg} "
            f"filter={bool(getattr(model.head, 'use_state_filter', False))} "
            f"filter_weight_correction={getattr(model.head.state_filter, 'event_log_weight', None)}"
        )
    loss_cfg = config["loss"]
    loss_type = str(loss_cfg.get("type", "legacy")).lower()
    if loss_type in {"legacy", "zipcount"}:
        # This is the historical path.  Keep its defaults, config keys, and
        # numerical behavior unchanged so old checkpoints/runs remain exactly
        # reproducible.
        loss_fn = ZipCountLoss(
            class_weights=class_weights,
            count_loss_type=loss_cfg["count_loss_type"],
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            lambda_vad=loss_cfg["lambda_vad"],
            lambda_overlap=loss_cfg["lambda_overlap"],
            lambda_smooth=loss_cfg["lambda_smooth"],
            smooth_type=loss_cfg.get("smooth_type", "kl"),
            sord_alpha=loss_cfg.get("sord_alpha", 1.5),
            lambda_emae=loss_cfg.get("lambda_emae", 0.0),
            lambda_consistency=loss_cfg.get("lambda_consistency", 0.0),
            lambda_monotonic=loss_cfg.get("lambda_monotonic", 0.0),
            lambda_dice=loss_cfg.get("lambda_dice", 0.0),
            lambda_event=loss_cfg.get("lambda_event", 0.0),
            event_class_weights=loss_cfg.get("event_class_weights"),
            lambda_raw_anchor=loss_cfg.get("lambda_raw_anchor", 0.0),
        ).to(device)
        # Additive boundary/direction/t_mse/segment terms on TOP of the
        # legacy total, independent of the count objective (see
        # PyramidAuxiliaryLoss docstring: the pyramid_structured SORD/CORN
        # bridge failed its own non-inferiority bar, which structurally
        # blocked these ideas; composing with the unchanged legacy loss
        # re-opens them without touching the ordinal encoding). Absent from
        # every existing config, so old runs are byte-identical.
        aux_lambda_boundary = float(loss_cfg.get("aux_lambda_boundary", 0.0))
        aux_lambda_direction = float(loss_cfg.get("aux_lambda_direction", 0.0))
        aux_lambda_t_mse = float(loss_cfg.get("aux_lambda_t_mse", 0.0))
        aux_lambda_segment = float(loss_cfg.get("aux_lambda_segment", 0.0))
        aux_loss_fn = None
        if any((
            aux_lambda_boundary > 0, aux_lambda_direction > 0,
            aux_lambda_t_mse > 0, aux_lambda_segment > 0,
        )):
            head_type = str(
                config["model"].get("head", {}).get("type", "")
            ).lower()
            if head_type != "gated_pyramid_ordinal":
                raise ValueError(
                    "loss.aux_lambda_* requires "
                    "model.head.type=gated_pyramid_ordinal, got "
                    f"{head_type!r}"
                )
            aux_loss_fn = PyramidAuxiliaryLoss(
                lambda_boundary=aux_lambda_boundary,
                lambda_direction=aux_lambda_direction,
                lambda_t_mse=aux_lambda_t_mse,
                lambda_segment=aux_lambda_segment,
                boundary_kernel=loss_cfg.get(
                    "aux_boundary_kernel", [0.25, 0.75, 1.0, 0.75, 0.25]
                ),
                t_mse_tau=loss_cfg.get("aux_t_mse_tau", 4.0),
                direction_class_weights=loss_cfg.get(
                    "aux_direction_class_weights", [1.0, 1.0]
                ),
            ).to(device)
            print(
                "Pyramid auxiliary loss ON (legacy count objective "
                "unchanged): "
                f"boundary={aux_lambda_boundary:g} "
                f"direction={aux_lambda_direction:g} "
                f"t_mse={aux_lambda_t_mse:g} "
                f"segment={aux_lambda_segment:g}"
            )
    elif loss_type == "pyramid_structured":
        aux_loss_fn = None
        for key in (
            "aux_lambda_boundary", "aux_lambda_direction",
            "aux_lambda_t_mse", "aux_lambda_segment",
        ):
            if float(loss_cfg.get(key, 0.0)) != 0.0:
                raise ValueError(
                    f"loss.{key} is the legacy-bridge auxiliary path and "
                    "cannot be combined with pyramid_structured, which has "
                    "its own native boundary/direction/t_mse/segment lambdas"
                )
        if lambda_bsmooth > 0:
            raise ValueError(
                "loss.lambda_boundary_smooth is the legacy external T-MSE "
                "path and cannot be combined with pyramid_structured. Use "
                "loss.lambda_t_mse instead so smoothing is applied exactly once."
            )
        ignored_legacy = {
            key: float(loss_cfg.get(key, 0.0))
            for key in (
                "lambda_vad",
                "lambda_overlap",
                "lambda_smooth",
                "lambda_emae",
                "lambda_consistency",
                "lambda_monotonic",
                "lambda_dice",
                "lambda_event",
                "lambda_raw_anchor",
            )
            if float(loss_cfg.get(key, 0.0)) != 0.0
        }
        if ignored_legacy:
            raise ValueError(
                "pyramid_structured does not consume legacy auxiliary losses; "
                "set these explicitly to zero/remove them instead of silently "
                f"ignoring active-looking knobs: {ignored_legacy}"
            )
        head_cfg = config["model"].get("head", {})
        head_type = str(head_cfg.get("type", "")).lower()
        ordinal_mode = str(head_cfg.get("ordinal_mode", "corn")).lower()
        objective = str(loss_cfg.get("count_loss_type", "corn")).lower()
        supported_objectives = {"ce", "focal", "sord", "cumulative", "corn"}
        if objective not in supported_objectives:
            raise ValueError(
                "pyramid_structured loss.count_loss_type must be one of "
                f"{sorted(supported_objectives)}, got {objective!r}"
            )
        if head_type != "gated_pyramid_ordinal":
            raise ValueError(
                "pyramid_structured requires "
                "model.head.type=gated_pyramid_ordinal"
            )
        expected_mode = "corn" if objective == "corn" else "softmax"
        if ordinal_mode != expected_mode:
            raise ValueError(
                f"loss.count_loss_type={objective!r} requires "
                f"model.head.ordinal_mode={expected_mode!r}, got "
                f"{ordinal_mode!r}. Use count_loss_type=cumulative for the "
                "pure cumulative-loss arm on unchanged four-way softmax logits."
            )
        loss_fn = PyramidStructuredLoss(
            class_weights=class_weights,
            count_loss_type=loss_cfg.get("count_loss_type", "corn"),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            sord_alpha=loss_cfg.get("sord_alpha", 1.5),
            lambda_stage1=loss_cfg.get("lambda_stage1", 0.0),
            lambda_boundary=loss_cfg.get("lambda_boundary", 0.0),
            lambda_direction=loss_cfg.get("lambda_direction", 0.0),
            lambda_t_mse=loss_cfg.get("lambda_t_mse", 0.0),
            lambda_segment=loss_cfg.get("lambda_segment", 0.0),
            lambda_delta=loss_cfg.get("lambda_delta", 0.0),
            boundary_kernel=loss_cfg.get(
                "boundary_kernel", [0.25, 0.75, 1.0, 0.75, 0.25]
            ),
            t_mse_tau=loss_cfg.get("t_mse_tau", 4.0),
            direction_class_weights=loss_cfg.get(
                "direction_class_weights", [1.0, 1.0]
            ),
        ).to(device)
        print(
            "Structured pyramid loss: "
            f"count={loss_fn.count_loss_type} "
            f"stage1={loss_fn.lambda_stage1:g} "
            f"boundary={loss_fn.lambda_boundary:g} "
            f"direction={loss_fn.lambda_direction:g} "
            f"t_mse={loss_fn.lambda_t_mse:g} "
            f"segment={loss_fn.lambda_segment:g} "
            f"delta={loss_fn.lambda_delta:g}"
        )
    else:
        raise ValueError(
            f"Unknown loss.type {loss_type!r}; expected legacy or "
            "pyramid_structured"
        )
    model = model.to(device)

    # Model-WEIGHT loading happens HERE, before the freeze/optimizer setup
    # below, specifically so LoRA (if enabled) can wrap the backbone AFTER
    # pretrained weights are in place but BEFORE the optimizer is built --
    # the optimizer must see LoRA's parameters, and peft's wrapping renames
    # every parameter's qualified name, so wrapping before loading would
    # break `--init-from`'s state_dict key matching (see
    # WavLMBackbone.apply_lora's docstring). `--resume-from`'s OPTIMIZER
    # state is restored later, after the optimizer object exists (it
    # inherently cannot be restored before that).
    start_step = 0
    resume_ckpt = None
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        # Architecture ablations: load ONLY backbone weights so every head —
        # including the incumbent — starts from the same random init. Loading
        # a pretrained incumbent head against a random challenger measures
        # head PRETRAINING, not head architecture.
        if bool(config["training"].get("init_backbone_only", False)):
            n_all = len(sd)
            sd = {k: v for k, v in sd.items() if k.startswith("backbone.")}
            print(f"init_backbone_only: keeping {len(sd)}/{n_all} tensors (head random)")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Init weights from {args.init_from} (step {ckpt.get('step', '?')}): "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if bool(config["training"].get("require_backbone_complete", False)):
            # `init_backbone_only` legitimately leaves HEAD keys missing, and
            # that same tolerance can hide a genuinely absent backbone tensor
            # (e.g. a topology drift renaming one conv). This knob asserts the
            # backbone specifically loaded completely.
            backbone_missing = [k for k in missing if k.startswith("backbone.")]
            if backbone_missing:
                raise RuntimeError(
                    f"require_backbone_complete: {len(backbone_missing)} "
                    f"backbone keys were NOT loaded from the init checkpoint; "
                    f"first: {backbone_missing[:3]}")
        # Presence of the key enables an exact invariant, including the
        # important empty-set case used by a control arm.  Testing truthiness
        # here would silently accept missing control weights when the expected
        # list is deliberately empty.
        check_expected_missing = "expected_init_missing" in config["training"]
        expected_missing = set(config["training"].get("expected_init_missing", []))
        if check_expected_missing and set(missing) != expected_missing:
            raise RuntimeError(
                "--init-from missing-key invariant failed: "
                f"expected={sorted(expected_missing)} actual={sorted(missing)}")
        if config["training"].get("require_no_unexpected_init_keys", False) and unexpected:
            raise RuntimeError(
                f"--init-from unexpected keys: {sorted(unexpected)}")
        if not config["training"].get("init_backbone_only", False) \
                and len(missing) > len(sd) * 0.1:
            raise RuntimeError(f"--init-from: too many missing keys ({len(missing)}) — wrong config/arch?")
    elif args.resume_from:
        resume_ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        start_step = int(resume_ckpt.get("step", 0))
        print(f"Resumed model weights from {args.resume_from} at step {start_step} "
              "(optimizer state restored after the optimizer is built)")

    lora_cfg = config["model"]["encoder"].get("lora", {})
    if bool(lora_cfg.get("enabled", False)):
        if not hasattr(model.backbone, "apply_lora"):
            raise ValueError(
                f"model.encoder.lora.enabled=true but "
                f"{type(model.backbone).__name__} has no apply_lora()")
        model.backbone.apply_lora(
            target_modules=lora_cfg["target_modules"],
            layer_indices=lora_cfg.get("layer_indices"),
            rank=int(lora_cfg.get("rank", 8)),
            alpha=int(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.0)),
        )
        # Activation-level (not just parameter-level) check, two-directional:
        # 1) B=0 composes to an exact no-op through the REAL forward path.
        # 2) the wrapped Linears are actually reachable from that forward
        #    path at all -- perturbing lora_B and seeing NO output change
        #    would mean apply_lora() silently did nothing, exactly the
        #    failure class found with peft (WavLMAttention reads
        #    `self.q_proj.weight` directly, bypassing a wrapped Linear's
        #    forward() entirely; a peft-wrapped q_proj's LoRA path was
        #    completely inert -- layer-0 output requires_grad was False
        #    despite "successful" wrapping). Checking B=0-is-a-no-op alone
        #    cannot catch that: it would have passed vacuously too.
        model.backbone.eval()
        probe_feat = torch.randn(1, 16000, device=device)
        probe_len = torch.tensor([16000], device=device)
        wrapped = [
            getattr(model.backbone.encoder.encoder.layers[i].attention, name)
            for i in model.backbone.lora_layer_indices
            for name in model.backbone.lora_target_modules
        ]
        with torch.no_grad():
            h_zero, _ = model.backbone.forward_features(probe_feat, probe_len)
            for w in wrapped:
                w.lora_B.add_(1.0)  # arbitrary nonzero probe
            h_perturbed, _ = model.backbone.forward_features(probe_feat, probe_len)
            for w in wrapped:
                w.lora_B.zero_()
            h_reset, _ = model.backbone.forward_features(probe_feat, probe_len)
        no_op_diff = (h_zero - h_reset).abs().max().item()
        perturb_diff = (h_zero - h_perturbed).abs().max().item()
        print(f"[diag] LoRA activation check: B=0-is-noop max_abs_diff={no_op_diff:.2e}, "
              f"B!=0-changes-output max_abs_diff={perturb_diff:.2e}")
        if no_op_diff > 1e-6:
            raise RuntimeError(
                f"LoRA B=0 init invariant failed: max_abs_diff={no_op_diff:.3e} > 1e-6 "
                "-- LoRA is perturbing the output before any training step"
            )
        if perturb_diff < 1e-4:
            raise RuntimeError(
                f"LoRA wiring invariant failed: perturbing lora_B changed the output by "
                f"only {perturb_diff:.3e} -- the wrapped Linears are not reachable from "
                "the model's real forward path (this is the exact failure mode peft's "
                "wrapping had for WavLMAttention's direct .weight access)"
            )

    lora_active = bool(getattr(model.backbone, "lora_applied", False))

    stage = config["training"].get("stage", "stage1_freeze_backbone")
    unfreeze_last_n = config["model"]["encoder"].get("unfreeze_last_n", 0)

    # apply_lora() above already set the exact requires_grad pattern LoRA
    # needs (base frozen, lora_A/B trainable) and verified it. Every OTHER
    # stage branch below mutates model.backbone's requires_grad wholesale
    # and would silently clobber that pattern (stage1 re-freezes the LoRA
    # params too; stage3/the unknown-stage fallback unfreezes the entire
    # base backbone, defeating the point of LoRA). Require an explicit,
    # LoRA-only stage instead of letting any of those run against it.
    if lora_active and stage != "stage4_lora":
        raise ValueError(
            f"model.encoder.lora.enabled=true requires training.stage=stage4_lora, "
            f"got stage={stage!r} -- every other stage branch would overwrite "
            "apply_lora()'s requires_grad pattern"
        )

    if stage == "stage1_freeze_backbone":
        print("Stage 1: Freezing entire backbone, training only head.")
        for p in model.backbone.parameters():
            p.requires_grad = False
    elif stage == "stage2_unfreeze_last_stacks":
        if unfreeze_last_n <= 0:
            raise ValueError("stage2_unfreeze_last_stacks requires unfreeze_last_n > 0")
        print(f"Stage 2: Unfreezing last {unfreeze_last_n} stacks of backbone and training head.")
        # unfreeze_last_n is applied inside the backbone's own __init__ (its
        # _freeze_params(unfreeze_last_n) method), via model.encoder.unfreeze_last_n
        # in the YAML -- StreamingZipformerEncoder and WavLMBackbone both
        # implement this; verify the same holds before adding a new backbone type.
    elif stage == "stage3_finetune":
        print("Stage 3: Full finetuning of all parameters.")
        for p in model.parameters():
            p.requires_grad = True
    elif stage == "stage4_lora":
        if not lora_active:
            raise ValueError("training.stage=stage4_lora requires model.encoder.lora.enabled=true")
        print("Stage 4: LoRA adapters trainable (base backbone frozen), training head + LoRA.")
    else:
        print(f"Unknown stage '{stage}', training all parameters.")
        for p in model.parameters():
            p.requires_grad = True

    # Head-only arm: freeze the whole backbone regardless of `stage`. This is the
    # cleanest anchor for the drift question — if head-only keeps the Vox gain
    # while holding AMI, no differential-LR tuning is needed at all.
    if bool(config["training"].get("freeze_backbone_all", False)):
        if lora_active:
            raise ValueError(
                "freeze_backbone_all=true would re-freeze the LoRA params "
                "apply_lora() just unfroze -- do not set this for a LoRA run"
            )
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("freeze_backbone_all: backbone frozen, training head only.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")

    # The backbone must stay in eval mode wherever it's frozen: dropout/
    # whitening/balancer noise in train mode would randomize "frozen"
    # features at every step even though their weights never update. This
    # matters for PARTIAL freezing too, not just the fully-frozen case: a
    # naive `model.train()` puts the ENTIRE model (including frozen early
    # WavLM/Zipformer layers) into train mode, so control (head_only) and
    # candidate (unfreeze_last_n>0) arms would differ in both adaptation AND
    # stochastic-backbone noise, confounding the comparison. Fix: eval() the
    # whole backbone first, then train() only the modules that actually own
    # a trainable parameter -- nn.Module.train() recurses into children, so
    # this correctly re-enables dropout inside an unfrozen layer's own
    # sub-modules (attention/FFN dropout etc.) without touching frozen
    # layers or the CNN feature-extractor/projection.
    def set_train_mode():
        model.train()
        set_backbone_partial_train_mode(model.backbone)

    # Mandatory pre-run diagnostics for any partial-unfreeze/adaptation arm:
    # printed once, before training starts, so a triage decision never has
    # to trust a config's INTENT (e.g. "stage2_unfreeze_last_stacks") over
    # what the model actually ended up with -- see zipcount-wavlm-loss-
    # unfreeze-fixes memory for the bug class this guards against (a config
    # claiming a partial unfreeze that silently did nothing).
    backbone_trainable_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    head_trainable_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    print(f"[diag] trainable_backbone_params={backbone_trainable_params} "
          f"trainable_head_params={head_trainable_params}")
    trainable_layer_indices = getattr(model.backbone, "trainable_layer_indices", None)
    if trainable_layer_indices is not None:
        print(f"[diag] trainable WavLM layer indices: {trainable_layer_indices}")
    set_train_mode()
    layer_training_modes = None
    wavlm_layers = getattr(getattr(getattr(model.backbone, "encoder", None), "encoder", None), "layers", None)
    if wavlm_layers is not None:
        layer_training_modes = [layer.training for layer in wavlm_layers]
        print(f"[diag] WavLM per-layer .training mode after set_train_mode(): {layer_training_modes}")

    # Differential learning rates. Continuing to train a strong checkpoint with a
    # single LR across all 66M params costs AMI accuracy (the lambda_kd=0 control
    # loses it too, so it is forgetting, not distillation). A smaller backbone LR
    # — or a frozen backbone — tests whether that drift can be avoided while
    # still adapting the head.
    base_lr = float(config["training"]["lr"])
    # `lora_lr` is the LoRA-run spelling of the same knob `backbone_lr`
    # already is for direct-unfreeze runs -- both set the LR for whatever
    # trainable params live in model.backbone.parameters(), just named to
    # match how each experiment's config reads naturally.
    backbone_lr = float(config["training"].get(
        "lora_lr", config["training"].get("backbone_lr", base_lr)))
    head_lr = float(config["training"].get("head_lr", base_lr))
    # A LoRA run must ALWAYS go through the grouped-optimizer path, even
    # when lora_lr/head_lr/lr happen to be numerically equal (this
    # project's LoRA config sets all three to 1e-4) -- the split isn't
    # only about LR, it's the only way to give the LoRA group its own
    # weight_decay=0.0 below.
    if backbone_lr != base_lr or head_lr != base_lr or lora_active:
        bb = [p for p in model.backbone.parameters() if p.requires_grad]
        hd = [p for p in model.head.parameters() if p.requires_grad]
        seen = {id(p) for p in bb} | {id(p) for p in hd}
        rest = [p for p in trainable_params if id(p) not in seen]
        bb_group = {"params": bb, "lr": backbone_lr}
        if lora_active:
            # LoRA's A/B matrices are a small, freshly-initialized low-rank
            # update, not a full weight tensor with the usual overfitting
            # profile weight decay is meant to guard against -- decaying
            # them pulls the adapter back toward "no adaptation" every step,
            # fighting the thing being measured. AdamW's implicit default
            # (0.01, unset anywhere in this project) would otherwise apply
            # here same as every other group.
            bb_group["weight_decay"] = 0.0
        groups = [bb_group, {"params": hd, "lr": head_lr}]
        if rest:
            groups.append({"params": rest, "lr": head_lr})
        optimizer = torch.optim.AdamW(groups, lr=base_lr)
        print(f"Differential LR: backbone={backbone_lr} ({sum(p.numel() for p in bb)} params"
              f"{', weight_decay=0.0' if lora_active else ''}), "
              f"head={head_lr} ({sum(p.numel() for p in hd)} params)"
              + (f", other={head_lr} ({sum(p.numel() for p in rest)})" if rest else ""))
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=base_lr)

    warmup_steps = int(config["training"].get("warmup_steps", 0))
    lr_schedule = config["training"].get("lr_schedule", "constant")
    sched_max_steps = int(config["training"]["max_steps"])

    def lr_lambda(s):
        if s < warmup_steps:
            return (s + 1) / max(warmup_steps, 1)
        if lr_schedule == "cosine" and sched_max_steps > warmup_steps:
            import math
            prog = min((s - warmup_steps) / (sched_max_steps - warmup_steps), 1.0)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))  # -> 10% floor
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    use_amp = config["training"].get("use_amp", True) and device.type == "cuda"
    amp_init_scale = float(config["training"].get("amp_init_scale", 65536.0))
    amp_growth_interval = int(config["training"].get("amp_growth_interval", 2000))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp, init_scale=amp_init_scale,
        growth_interval=amp_growth_interval)
    if use_amp:
        print(f"AMP GradScaler: init_scale={amp_init_scale:g} "
              f"growth_interval={amp_growth_interval}")

    if args.resume_from:
        # Model weights + start_step were already restored in the early
        # weight-loading block above (before LoRA/optimizer setup); only the
        # optimizer's own state can be restored here, since the optimizer
        # object didn't exist yet at that point.
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        print(f"Restored optimizer state from {args.resume_from}")

    # Adding an event projection consumes RNG during module initialization.  A
    # paired architecture ablation must not consequently shift every dropout
    # mask in the shared TCN/backbone.  The DataLoader has its own generator, so
    # resetting the training RNG here preserves both the paired batch order and
    # the paired stochastic-model path.
    if seed is not None and bool(config["training"].get("reset_rng_after_init", False)):
        import random as _random
        import numpy as _np
        _random.seed(seed); _np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        print(f"Reset model/augmentation RNG after checkpoint init: seed={seed}")
    
    max_steps = config["training"]["max_steps"]
    grad_clip = config["training"].get("grad_clip", 5.0)
    grad_accum_steps = int(config["training"].get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
    micro_batch_size = int(config["training"]["batch_size"])
    effective_batch_size = micro_batch_size * grad_accum_steps
    if grad_accum_steps > 1:
        print(f"[diag] Gradient accumulation: micro_batch_size={micro_batch_size} x "
              f"grad_accum_steps={grad_accum_steps} -> effective_batch_size={effective_batch_size}")
    else:
        print(f"[diag] effective_batch_size={effective_batch_size} (no accumulation)")
    # AMP overflow budget: a few scaler-handled skips are normal; a stream of
    # them is the silent-no-training failure and must still fail fast.
    max_amp_overflows = int(config["training"].get("max_amp_overflows", 5))
    amp_overflow_count = 0
    eval_interval = config["training"].get("eval_interval", 1000)
    # When finetuning from an already-strong checkpoint the peak arrives almost
    # immediately (v3 peaked at its very FIRST eval, step 1000, then declined
    # for 19k steps), so a flat 1000-step interval can step straight over the
    # best model. Evaluate densely early, then fall back to the normal interval.
    eval_interval_early = config["training"].get("eval_interval_early", eval_interval)
    early_phase_steps = config["training"].get("early_phase_steps", 0)

    def should_eval(s):
        iv = eval_interval_early if s <= early_phase_steps else eval_interval
        return s % iv == 0
    label_downsample_method = config["data"].get("label_downsample", "majority_vote")
    
    log_dir = config["training"].get("log_dir", "logs/zipcount")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    set_train_mode()
    step = start_step
    if start_step > 0:
        for _ in range(start_step):  # fast-forward warmup schedule
            scheduler.step()
    pbar = tqdm(total=max_steps, initial=start_step)
    
    best_macro_f1 = -1.0
    best_overlap_f1 = -1.0
    
    if len(train_loader) == 0:
        raise ValueError("Train loader is empty! Check dataset size and batch_size (drop_last=True).")
        
    optimizer.zero_grad()
    micro_step = 0
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            features = batch["features"].to(device)
            feature_lens = batch["feature_lens"].to(device)
            labels = batch["labels"].to(device)
            label_lens = batch["label_lens"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, h_lens = model(features, feature_lens)
                aligned_labels = align_batch_labels(labels, label_lens, h_lens, method=label_downsample_method)
                loss_dict = loss_fn(logits, aligned_labels, h_lens)
                loss = loss_dict["loss"]

                if lambda_kd > 0 and batch["teacher_mask"].any():
                    kd = kd_loss(
                        logits[0], batch["teacher"].to(device),
                        batch["teacher_mask"].to(device),
                        aligned_labels, label_lens, h_lens, kd_temperature,
                        mask_mode=kd_mask_mode,
                        uncertain_max_prob=kd_uncertain_max_prob,
                    )
                    loss = loss + lambda_kd * kd
                    loss_dict["loss_kd"] = kd.detach()

                if lambda_bsmooth > 0:
                    sm = boundary_aware_smooth_loss(
                        logits[0], aligned_labels, h_lens,
                        tau=bsmooth_tau, guard=bsmooth_guard)
                    loss = loss + lambda_bsmooth * sm
                    loss_dict["loss_bsmooth"] = sm.detach()

                if aux_loss_fn is not None:
                    aux_dict = aux_loss_fn(logits, aligned_labels, h_lens)
                    loss = loss + aux_dict["loss"]
                    for key, value in aux_dict.items():
                        if key != "loss":
                            # key is "loss_boundary" etc.; rename so the
                            # "Loss/" TB tag stripping (k[5:]) below still
                            # produces a clean "aux_boundary" leaf.
                            loss_dict[f"loss_aux_{key[5:]}"] = value.detach()

            # Fail fast on a non-finite loss. GradScaler quietly skips the
            # optimizer step on NaN/Inf grads while the scheduler and step
            # counter keep advancing, so a broken loss looks exactly like a
            # healthy run for thousands of steps and updates nothing.
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at step {step}: "
                    + ", ".join(f"{k}={float(v):.4g}" for k, v in loss_dict.items())
                )

            # Scale DOWN for backward (so grad_accum_steps micro-batches sum to
            # the same magnitude as one effective_batch_size batch) but log the
            # unscaled per-micro-batch loss below, so TensorBoard values stay
            # comparable across configs with different grad_accum_steps.
            scaler.scale(loss / grad_accum_steps).backward()
            micro_step += 1
            if micro_step % grad_accum_steps != 0:
                continue  # keep accumulating; no optimizer step, no eval, no step increment yet

            # Unscale and inspect gradients on every path, including configs
            # which deliberately disable clipping.  Otherwise GradScaler can
            # skip an Inf/NaN step while the scheduler and visible counter keep
            # advancing, recreating the silent no-training failure seen in KD.
            scaler.unscale_(optimizer)
            max_grad_norm = grad_clip if grad_clip > 0 else float("inf")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                # Under AMP an occasional overflow at high loss-scale is the
                # scaler's NORMAL self-healing event (skip step, halve scale) —
                # oracle O1 died at step 193 to a single such overflow. The
                # failure this guard exists for (the KD bug) was THOUSANDS of
                # silent skips, so allow a small logged budget and still fail
                # fast beyond it or when AMP is off (then inf is a real bug).
                amp_overflow_count += 1
                if (not use_amp) or amp_overflow_count > max_amp_overflows:
                    raise RuntimeError(
                        f"non-finite gradient norm at step {step}: "
                        f"{float(grad_norm)} (overflow #{amp_overflow_count}, "
                        f"budget {max_amp_overflows})")
                print(f"[amp] overflow #{amp_overflow_count}/{max_amp_overflows} "
                      f"at step {step}; skipping step, scaler will reduce scale")
                scaler.step(optimizer)   # skips: grads are non-finite
                scaler.update()          # reduces the scale
                optimizer.zero_grad()
                continue                 # scheduler must NOT advance on a skip

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                writer.add_scalar("Loss/total", loss.item(), step)
                for k, v in loss_dict.items():
                    if k != "loss":
                        writer.add_scalar(f"Loss/{k[5:]}", v.item(), step)
                # Interpretable per-stack gates of the hypercolumn fusion
                fusion = getattr(model.head, "fusion", None)
                if fusion is not None and hasattr(fusion, "gate_weights"):
                    for i, g in enumerate(fusion.gate_weights().tolist()):
                        writer.add_scalar(f"FusionGate/stack{i}", g, step)
                
            step += 1
            pbar.update(1)
            
            # Validation
            if should_eval(step) or step == max_steps:
                if device.type == "cuda":
                    print(f"[diag@{step}] peak GPU memory: "
                          f"allocated={torch.cuda.max_memory_allocated(device) / 1e9:.2f}GB "
                          f"reserved={torch.cuda.max_memory_reserved(device) / 1e9:.2f}GB")
                if val_loader is not None:
                    model.eval()
                    val_metrics_list = []
                    with torch.no_grad():
                        for v_batch in tqdm(val_loader, desc="Validation", leave=False):
                            v_feat = v_batch["features"].to(device)
                            v_feat_lens = v_batch["feature_lens"].to(device)
                            v_labels = v_batch["labels"].to(device)
                            v_label_lens = v_batch["label_lens"].to(device)
                            
                            with torch.amp.autocast("cuda", enabled=use_amp):
                                v_logits, v_h_lens = model(v_feat, v_feat_lens)
                                v_aligned_labels = align_batch_labels(v_labels, v_label_lens, v_h_lens, method=label_downsample_method)
                            
                            m = compute_metrics(v_logits[0], v_logits[1], v_logits[2], v_aligned_labels, v_h_lens)
                            if m:
                                val_metrics_list.append(m)
                                
                    agg_metrics = aggregate_metrics(val_metrics_list)
                    for k, v in agg_metrics.items():
                        writer.add_scalar(f"Val/{k}", v, step)

                    m = agg_metrics
                    print(f"\n[val@{step}] acc={m.get('acc',0):.3f} mae={m.get('mae',0):.3f} "
                          f"macroF1={m.get('macro_f1',0):.3f} "
                          f"f1/cls=[{m.get('f1_0',0):.2f} {m.get('f1_1',0):.2f} "
                          f"{m.get('f1_2',0):.2f} {m.get('f1_3',0):.2f}] "
                          f"ovF1={m.get('f1_ov_count',0):.3f} vadF1={m.get('f1_vad',0):.3f}")
                    print(format_confusion(m))

                    macro_f1 = agg_metrics.get("macro_f1", 0.0)
                    overlap_f1 = agg_metrics.get("f1_ov_count", 0.0)
                    
                    state = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "config": config,
                    }
                    weights_state = {
                        "model_state_dict": state["model_state_dict"],
                        "step": step,
                        "config": config,
                    }

                    if bool(config["training"].get("save_last", True)):
                        torch.save(state, os.path.join(log_dir, "last.pt"))
                    # Snapshot every eval when asked: the best model may simply be
                    # an early one, in which case early stopping — not optimizer
                    # surgery — is the fix.
                    if bool(config["training"].get("save_every_eval", False)):
                        snapshot = (weights_state if bool(config["training"].get(
                            "snapshot_weights_only", False)) else state)
                        torch.save(snapshot, os.path.join(log_dir, f"step{step}.pt"))

                    save_best = bool(config["training"].get("save_best", True))
                    if save_best and macro_f1 > best_macro_f1:
                        best_macro_f1 = macro_f1
                        torch.save(state, os.path.join(log_dir, "best_macro_f1.pt"))

                    if save_best and overlap_f1 > best_overlap_f1:
                        best_overlap_f1 = overlap_f1
                        torch.save(state, os.path.join(log_dir, "best_overlap_f1.pt"))

                    set_train_mode()
                    
    print("Training finished.")

if __name__ == "__main__":
    main()
