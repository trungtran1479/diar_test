"""Paper-grade segment diagnostics for speaker-count predictions.

This script deliberately scores ZipCount and the teacher on the SAME recordings
and the SAME frame support.  Window predictions are first placed at their
``source_offset`` on a 100-Hz recording timeline, then each recording is scored
once.  Therefore a real transition at a 30-s join is retained, a constant state
across a join is not split into two segments, and a missing window is a masked
gap rather than an invented adjacency.  Earlier versions scored each window
independently, let ZipCount see items for which the teacher was absent, averaged
boundary F1 per window (making empty windows ambiguous), and credited an overlap
onset whenever the prediction was merely positive somewhere near the reference
event.  Those choices can all make a structurally bad prediction look better
than it is.

Metrics reported here:

* overlap fragmentation = number of positive ``count >= 2`` prediction runs /
  number of positive reference runs (not the number of generic label segments);
* pooled boundary precision/recall/F1 from aggregate TP/FP/FN;
* pooled signed-boundary F1, where an upward count transition cannot match a
  downward transition;
* strict overlap-onset and overlap-offset recall: a reference event is matched
  one-to-one only to a predicted event with the same direction;
* signed timing error for matched events (positive = prediction is late);
* overlap-segment MISSED/PARTIAL/DETECTED coverage and within-segment positive
  run counts.

The low-rate ZipCount output is expanded to 100 Hz with the exact inverse of
the ``np.array_split`` buckets used to align labels during training.  Any
remaining teacher/reference length mismatch is handled only by right-tail
truncation, never by shifting a window or repeating an unseen teacher frame;
the discarded frames and any resulting timeline gaps are printed.  The legacy
CLI remains valid.  ``--tol-ms`` is still the primary tolerance;
``--tolerances-ms`` adds a paper-friendly tolerance sweep and always includes
the primary value.
"""
import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import yaml

N = 4
FRAME_HZ = 100
WINDOW_ID_RE = re.compile(r"^(?P<recording>.+)_w(?P<window>\d+)$")


@dataclass
class TimelineWindow:
    """One common-support window positioned on its source recording."""

    recording_id: str
    window_id: str
    start_frame: int
    reference: np.ndarray
    zipcount: np.ndarray
    teacher: np.ndarray


@dataclass
class RecordingTimeline:
    """Three aligned systems plus the scored support on one 100-Hz timeline."""

    recording_id: str
    origin_frame: int
    reference: np.ndarray
    zipcount: np.ndarray
    teacher: np.ndarray
    valid: np.ndarray
    windows: int

    @property
    def gap_frames(self) -> int:
        return int(len(self.valid) - np.count_nonzero(self.valid))


def recording_position(item: Mapping) -> Tuple[str, int]:
    """Return stable recording id and absolute 100-Hz window start.

    Evaluation manifests created by ``prepare_meeting.py`` and
    ``prep_diarizen_dataset.py`` store the source offset on their single track
    and end ids in ``_w####``.  Explicit ``recording_id``/``source_offset``
    fields are also accepted.  Refusing to guess a missing offset is important:
    concatenating in manifest order would silently move a window in time.
    """
    wid = str(item.get("id", ""))
    match = WINDOW_ID_RE.fullmatch(wid)
    recording_id = item.get("recording_id")
    if recording_id is None and match:
        recording_id = match.group("recording")

    tracks = item.get("tracks") or []
    if recording_id is None and len(tracks) == 1 and tracks[0].get("source"):
        recording_id = os.path.abspath(str(tracks[0]["source"]))
    if recording_id is None:
        raise ValueError(
            f"{wid or '<missing id>'}: cannot infer recording id; provide "
            "recording_id or an id ending in _w####")

    offset = item.get("source_offset")
    if offset is None and len(tracks) == 1:
        offset = tracks[0].get("source_offset")
    if offset is None:
        raise ValueError(
            f"{wid}: cannot place window on recording timeline; provide "
            "source_offset (top-level or on its single track)")
    offset = float(offset)
    if not np.isfinite(offset) or offset < 0:
        raise ValueError(f"{wid}: invalid source_offset={offset}")
    return str(recording_id), int(round(offset * FRAME_HZ))


def expand_output_to_reference(pred: np.ndarray, reference_frames: int) -> np.ndarray:
    """Expand low-rate predictions onto the exact training-label buckets.

    Training maps ``reference_frames`` to ``len(pred)`` with
    ``np.array_split``.  Repeating every output four times is only equivalent
    when the lengths divide exactly; at ragged tails it can shorten the window
    or move bucket boundaries.  Repeating by the same quotient/remainder sizes
    is the exact inverse and always ends on the reference right edge.
    """
    pred = np.asarray(pred, dtype=np.int64)
    if pred.ndim != 1 or len(pred) == 0:
        raise ValueError(f"expected a non-empty 1-D prediction, got {pred.shape}")
    if reference_frames < len(pred):
        raise ValueError(
            f"cannot map {len(pred)} outputs onto only {reference_frames} frames")
    base, rem = divmod(int(reference_frames), len(pred))
    sizes = np.full(len(pred), base, dtype=np.int64)
    sizes[:rem] += 1
    return np.repeat(pred, sizes).astype(np.int64, copy=False)


def stitch_recording(windows: Sequence[TimelineWindow]) -> RecordingTimeline:
    """Place sorted windows on a recording clock without collapsing gaps.

    Adjacent windows share a normal frame-to-frame edge, so transitions at the
    join remain measurable.  Missing common support remains ``valid=False``;
    metrics split there and therefore never compare labels across unknown time.
    Overlap is rejected because choosing one of two context-dependent window
    predictions would be an unregistered scoring policy.
    """
    if not windows:
        raise ValueError("cannot stitch an empty recording")
    ordered = sorted(windows, key=lambda w: (w.start_frame, w.window_id))
    recording_id = ordered[0].recording_id
    if any(w.recording_id != recording_id for w in ordered):
        raise ValueError("stitch_recording received multiple recording ids")

    for w in ordered:
        shapes = (np.asarray(w.reference).shape, np.asarray(w.zipcount).shape,
                  np.asarray(w.teacher).shape)
        if any(len(shape) != 1 for shape in shapes) or len(set(shapes)) != 1:
            raise ValueError(f"{w.window_id}: unequal/non-1-D timeline arrays {shapes}")
        if shapes[0][0] == 0:
            raise ValueError(f"{w.window_id}: empty common support")
        if w.start_frame < 0:
            raise ValueError(f"{w.window_id}: negative start frame {w.start_frame}")

    origin = min(w.start_frame for w in ordered)
    end = max(w.start_frame + len(w.reference) for w in ordered)
    length = end - origin
    reference = np.zeros(length, dtype=np.int64)
    zipcount = np.zeros(length, dtype=np.int64)
    teacher = np.zeros(length, dtype=np.int64)
    valid = np.zeros(length, dtype=bool)
    for w in ordered:
        s = w.start_frame - origin
        e = s + len(w.reference)
        if valid[s:e].any():
            raise ValueError(
                f"{recording_id}: overlapping window at {w.window_id}; "
                "overlap resolution is intentionally undefined")
        reference[s:e] = w.reference
        zipcount[s:e] = w.zipcount
        teacher[s:e] = w.teacher
        valid[s:e] = True
    return RecordingTimeline(
        recording_id=recording_id,
        origin_frame=origin,
        reference=reference,
        zipcount=zipcount,
        teacher=teacher,
        valid=valid,
        windows=len(ordered),
    )


def segments(y: np.ndarray) -> List[Tuple[int, int, int]]:
    """Return ``(start, end_exclusive, label)`` runs for a 1-D array."""
    y = np.asarray(y)
    if len(y) == 0:
        return []
    cut = np.flatnonzero(np.diff(y)) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(y)]
    return [(int(s), int(e), int(y[s])) for s, e in zip(starts, ends)]


def positive_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return runs where a boolean mask is True; all-False has zero runs."""
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    padded = np.pad(mask.astype(np.int8), (1, 1))
    delta = np.diff(padded)
    starts = np.flatnonzero(delta == 1)
    ends = np.flatnonzero(delta == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def change_events(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """All label-change locations and their sign (-1 or +1).

    Count jumps larger than one still have a single direction: the task here
    is boundary timing, not decomposing a 0->2 jump into two invented events.
    """
    y = np.asarray(y)
    if len(y) < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)
    d = np.diff(y.astype(np.int64))
    idx = np.flatnonzero(d != 0) + 1
    return idx.astype(np.int64), np.sign(d[idx - 1]).astype(np.int8)


def overlap_transition_events(y: np.ndarray, direction: int) -> np.ndarray:
    """Strict transitions into (+1) or out of (-1) ``count >= 2``."""
    if direction not in (-1, 1):
        raise ValueError(f"direction must be -1 or +1, got {direction}")
    ov = (np.asarray(y) >= 2).astype(np.int8)
    if len(ov) < 2:
        return np.empty(0, dtype=np.int64)
    return (np.flatnonzero(np.diff(ov) == direction) + 1).astype(np.int64)


@dataclass
class EventMatch:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    delays: List[int] = field(default_factory=list)

    def add(self, other: "EventMatch") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.delays.extend(other.delays)


def match_events(gt_events: Sequence[int], pred_events: Sequence[int], tol: int) -> EventMatch:
    """Maximum-cardinality chronological one-to-one matching on a time line.

    Both inputs are sorted event indices.  A match is permitted when the
    absolute distance is at most ``tol``.  The two-pointer construction is the
    standard maximum-cardinality matching for ordered interval events and,
    unlike a per-reference nearest-neighbour check, never reuses a prediction.
    Delays are ``pred - gt`` frames, so positive means late.
    """
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}")
    gt = np.asarray(gt_events, dtype=np.int64)
    pred = np.asarray(pred_events, dtype=np.int64)
    if len(gt) > 1 and np.any(np.diff(gt) < 0):
        raise ValueError("gt_events must be sorted")
    if len(pred) > 1 and np.any(np.diff(pred) < 0):
        raise ValueError("pred_events must be sorted")

    i = j = 0
    delays: List[int] = []
    while i < len(gt) and j < len(pred):
        if pred[j] < gt[i] - tol:
            j += 1                         # too early for this or any later GT
        elif gt[i] < pred[j] - tol:
            i += 1                         # this GT is too early for this pred
        else:
            delays.append(int(pred[j] - gt[i]))
            i += 1
            j += 1
    tp = len(delays)
    return EventMatch(tp=tp, fp=int(len(pred) - tp), fn=int(len(gt) - tp),
                      delays=delays)


def match_signed_events(
    gt_idx: Sequence[int],
    gt_sign: Sequence[int],
    pred_idx: Sequence[int],
    pred_sign: Sequence[int],
    tol: int,
) -> EventMatch:
    """Match boundaries one-to-one, separately for upward/downward signs."""
    gt_idx = np.asarray(gt_idx, dtype=np.int64)
    gt_sign = np.asarray(gt_sign, dtype=np.int8)
    pred_idx = np.asarray(pred_idx, dtype=np.int64)
    pred_sign = np.asarray(pred_sign, dtype=np.int8)
    if len(gt_idx) != len(gt_sign) or len(pred_idx) != len(pred_sign):
        raise ValueError("event index/sign lengths differ")
    out = EventMatch()
    for sign in (-1, 1):
        out.add(match_events(gt_idx[gt_sign == sign], pred_idx[pred_sign == sign], tol))
    return out


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """P/R/F1 with an explicit perfect empty-empty convention.

    Empty reference plus empty prediction is a correct negative, not F1=0.
    In a non-empty corpus aggregate, empty windows simply add no events and do
    not receive disproportionate weight.
    """
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return float(precision), float(recall), float(f1)


def boundary_f1(gt: np.ndarray, pred: np.ndarray, tol: int) -> Tuple[float, int, int]:
    """Backward-compatible helper: unsigned F1, #GT and #pred boundaries."""
    gb, _ = change_events(gt)
    pb, _ = change_events(pred)
    m = match_events(gb, pb, tol)
    return precision_recall_f1(m.tp, m.fp, m.fn)[2], len(gb), len(pb)


def transition_recall(
    gt: np.ndarray, pred: np.ndarray, tol: int, into_overlap: bool = True
) -> Tuple[int, int]:
    """Backward-compatible strict onset/offset recall helper.

    Merely being in the requested state near a reference transition is not a
    hit: a predicted transition of the same direction must exist and can be
    consumed only once.
    """
    direction = 1 if into_overlap else -1
    ge = overlap_transition_events(gt, direction)
    pe = overlap_transition_events(pred, direction)
    m = match_events(ge, pe, tol)
    return m.tp, len(ge)


def parse_tolerances(primary_ms: int, values: str) -> List[int]:
    """Parse millisecond tolerances, include the primary, return sorted unique."""
    vals = [int(x.strip()) for x in str(values).split(",") if x.strip()]
    vals.append(int(primary_ms))
    if any(x < 0 for x in vals):
        raise ValueError("tolerances must be non-negative")
    return sorted(set(vals))


def ms_to_frames(ms: int) -> int:
    return int(round(ms * FRAME_HZ / 1000.0))


def delay_summary(delays: Sequence[int]) -> Dict[str, float]:
    """Delay statistics in milliseconds; positive delay means late."""
    if not delays:
        return {"n": 0, "median_ms": float("nan"), "mean_ms": float("nan"),
                "p90_abs_ms": float("nan"), "late_frac": float("nan")}
    d = np.asarray(delays, dtype=np.float64) * (1000.0 / FRAME_HZ)
    return {
        "n": int(len(d)),
        "median_ms": float(np.median(d)),
        "mean_ms": float(np.mean(d)),
        "p90_abs_ms": float(np.percentile(np.abs(d), 90)),
        "late_frac": float(np.mean(d > 0)),
    }


def _fmt_delay(d: Dict[str, float]) -> str:
    if not d["n"]:
        return "n=0"
    return (f"n={d['n']} med={d['median_ms']:+.0f}ms "
            f"mean={d['mean_ms']:+.0f}ms p90|e|={d['p90_abs_ms']:.0f}ms "
            f"late={100*d['late_frac']:.1f}%")


def _empty_system(tolerances_ms: Iterable[int]) -> dict:
    return {
        "items": 0,
        "frames": 0,
        "gt_overlap_runs": 0,
        "pred_overlap_runs": 0,
        "boundary": {t: EventMatch() for t in tolerances_ms},
        "signed_boundary": {t: EventMatch() for t in tolerances_ms},
        "onset": {t: EventMatch() for t in tolerances_ms},
        "offset": {t: EventMatch() for t in tolerances_ms},
        "missed": 0,
        "partial": 0,
        "detected": 0,
        "fragments_in_nonmissed": [],
    }


def _update_contiguous_system(a: dict, gt: np.ndarray, pred: np.ndarray,
                              tolerances_ms: Iterable[int]) -> None:
    """Accumulate metrics for one contiguous valid span (no item counting)."""
    g2, p2 = gt >= 2, pred >= 2
    gt_runs = positive_runs(g2)
    a["gt_overlap_runs"] += len(gt_runs)
    a["pred_overlap_runs"] += len(positive_runs(p2))

    gb, gs = change_events(gt)
    pb, ps = change_events(pred)
    gon = overlap_transition_events(gt, 1)
    pon = overlap_transition_events(pred, 1)
    goff = overlap_transition_events(gt, -1)
    poff = overlap_transition_events(pred, -1)
    for tol_ms in tolerances_ms:
        tol = ms_to_frames(tol_ms)
        a["boundary"][tol_ms].add(match_events(gb, pb, tol))
        a["signed_boundary"][tol_ms].add(
            match_signed_events(gb, gs, pb, ps, tol))
        a["onset"][tol_ms].add(match_events(gon, pon, tol))
        a["offset"][tol_ms].add(match_events(goff, poff, tol))

    for s0, e0 in gt_runs:
        if e0 - s0 < 5:                  # retain legacy >=50 ms threshold
            continue
        cov = float(p2[s0:e0].mean())
        if cov < 0.10:
            a["missed"] += 1
        elif cov > 0.90:
            a["detected"] += 1
            a["fragments_in_nonmissed"].append(len(positive_runs(p2[s0:e0])))
        else:
            a["partial"] += 1
            a["fragments_in_nonmissed"].append(len(positive_runs(p2[s0:e0])))


def update_system(a: dict, gt: np.ndarray, pred: np.ndarray,
                  tolerances_ms: Iterable[int],
                  valid: np.ndarray = None) -> None:
    """Accumulate one recording, splitting metrics only at unknown gaps.

    ``valid`` is the common-support mask on the recording timeline.  A normal
    window join has adjacent True frames and is therefore scored continuously;
    a missing window or truncated non-final tail produces a False gap and no
    transition is invented across it.
    """
    gt = np.asarray(gt, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if gt.ndim != 1 or pred.ndim != 1 or len(gt) != len(pred):
        raise ValueError(f"expected equal 1-D arrays, got {gt.shape}, {pred.shape}")
    if valid is None:
        valid = np.ones(len(gt), dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.ndim != 1 or len(valid) != len(gt):
            raise ValueError(
                f"valid mask must match timeline, got {valid.shape} vs {gt.shape}")
    a["items"] += 1
    a["frames"] += int(np.count_nonzero(valid))
    for s0, e0 in positive_runs(valid):
        _update_contiguous_system(a, gt[s0:e0], pred[s0:e0], tolerances_ms)


def _event_prf(m: EventMatch) -> Tuple[float, float, float]:
    return precision_recall_f1(m.tp, m.fp, m.fn)


def print_system(name: str, a: dict, primary_ms: int,
                 tolerances_ms: Sequence[int]) -> None:
    print(f"\n===== {name} =====")
    print(f"common support           {a['items']:,} recordings / "
          f"{a['frames']:,} frames")
    print(f"overlap fragmentation   {a['pred_overlap_runs']/max(a['gt_overlap_runs'],1):.3f} "
          f"({a['pred_overlap_runs']:,} pred positive runs / "
          f"{a['gt_overlap_runs']:,} gt positive runs)")

    b = a["boundary"][primary_ms]
    bp, br, bf = _event_prf(b)
    sb = a["signed_boundary"][primary_ms]
    sp, sr, sf = _event_prf(sb)
    onset = a["onset"][primary_ms]
    offset = a["offset"][primary_ms]
    _, on_r, _ = _event_prf(onset)
    _, off_r, _ = _event_prf(offset)
    print(f"boundary F1 pooled       {bf:.3f}  P={bp:.3f} R={br:.3f} "
          f"TP/FP/FN={b.tp}/{b.fp}/{b.fn} @ +/-{primary_ms}ms")
    print(f"signed-boundary F1       {sf:.3f}  P={sp:.3f} R={sr:.3f} "
          f"TP/FP/FN={sb.tp}/{sb.fp}/{sb.fn}")
    print(f"strict overlap-onset R   {on_r:.3f}  ({onset.tp}/{onset.tp + onset.fn})")
    print(f"strict overlap-offset R  {off_r:.3f}  ({offset.tp}/{offset.tp + offset.fn})")
    print(f"onset timing error       {_fmt_delay(delay_summary(onset.delays))}")
    print(f"offset timing error      {_fmt_delay(delay_summary(offset.delays))}")

    tot = a["missed"] + a["partial"] + a["detected"]
    print(f"gt overlap runs (>=50ms) {tot:,}")
    print(f"  MISSED  (<10% cov)     {a['missed']:>6,}  "
          f"({100*a['missed']/max(tot,1):.1f}%)")
    print(f"  PARTIAL (10-90%)       {a['partial']:>6,}  "
          f"({100*a['partial']/max(tot,1):.1f}%)")
    print(f"  DETECTED(>90%)         {a['detected']:>6,}  "
          f"({100*a['detected']/max(tot,1):.1f}%)")
    frags = a["fragments_in_nonmissed"]
    if frags:
        print(f"  positive runs/non-missed overlap: mean {np.mean(frags):.2f} "
              f"median {np.median(frags):.1f}")

    print("tolerance sweep (pooled event counts; delay is pred-gt)")
    print("  tol    boundaryF1 signedBF1 onsetR offsetR  onset-med offset-med")
    for tol_ms in tolerances_ms:
        bm = a["boundary"][tol_ms]
        sm = a["signed_boundary"][tol_ms]
        om = a["onset"][tol_ms]
        fm = a["offset"][tol_ms]
        bf = _event_prf(bm)[2]
        sf = _event_prf(sm)[2]
        ore = _event_prf(om)[1]
        fre = _event_prf(fm)[1]
        od = delay_summary(om.delays)["median_ms"]
        fd = delay_summary(fm.delays)["median_ms"]
        od_s = "nan" if np.isnan(od) else f"{od:+.0f}ms"
        fd_s = "nan" if np.isnan(fd) else f"{fd:+.0f}ms"
        print(f"  {tol_ms:>4}ms   {bf:>8.3f}   {sf:>8.3f} "
              f"{ore:>6.3f} {fre:>7.3f} {od_s:>10} {fd_s:>10}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default="configs/zipcount_v4_distill.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--teacher-dir", required=True)
    ap.add_argument("--tol-ms", type=int, default=250,
                    help="Primary boundary tolerance (legacy-compatible)")
    ap.add_argument("--tolerances-ms", default="100,250,500",
                    help="Comma-separated tolerance sweep; --tol-ms is always included")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    tolerances_ms = parse_tolerances(args.tol_ms, args.tolerances_ms)

    from src.data.dataset import SpeakerCountDataset, collate_fn
    from src.data.feature_extractor import feature_extractor_for_config
    from src.models.zipcount_v1 import build_model

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model = model.to(device).eval()

    with open(args.manifest) as f:
        items = [json.loads(line) for line in f if line.strip()]
    by_id = {it["id"]: it for it in items}
    if len(by_id) != len(items):
        raise ValueError("manifest contains duplicate item ids")

    # Common-item restriction is load-bearing: without it ZipCount is scored
    # on every item while the teacher silently gets an easier/different subset.
    teacher_path = {
        wid: os.path.join(args.teacher_dir, wid + ".npy") for wid in by_id
    }
    common_ids = {wid for wid, path in teacher_path.items() if os.path.isfile(path)}
    if not common_ids:
        raise RuntimeError("No manifest items have teacher posteriors")

    ds = SpeakerCountDataset(args.manifest, feature_extractor=feature_extractor_for_config(config))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=4)

    skipped_no_teacher = 0
    truncated_frames = {"reference": 0, "zipcount": 0, "teacher": 0}
    windows_by_recording: Dict[str, List[TimelineWindow]] = defaultdict(list)
    collected_ids = set()

    with torch.no_grad():
        for batch in loader:
            logits, h_lens = model(batch["features"].to(device),
                                   batch["feature_lens"].to(device))
            pred25 = logits[0].float().argmax(-1).cpu().numpy()
            for i, wid in enumerate(batch["ids"]):
                if wid not in common_ids:
                    skipped_no_teacher += 1
                    continue
                it = by_id[wid]
                gt = np.clip(np.load(it["label_filepath"]).astype(np.int64), 0, 3)
                # Invert the exact np.array_split grouping used for hard-label
                # training.  This reaches the reference right edge without a
                # repeat(4)-then-truncate timing artifact on ragged windows.
                zp = expand_output_to_reference(
                    pred25[i, :int(h_lens[i])], len(gt))
                teacher_post = np.load(teacher_path[wid])
                if teacher_post.ndim != 2 or teacher_post.shape[1] < N:
                    raise ValueError(
                        f"{teacher_path[wid]}: expected [T, >={N}], got {teacher_post.shape}")
                tp = teacher_post[:, :N].argmax(-1).astype(np.int64)

                # Same window is not enough: score both systems on exactly the
                # same time support. A rare length disagreement is resolved at
                # the RIGHT TAIL only. We neither shift the following window
                # nor repeat a teacher frame it never produced; stitch_recording
                # leaves any non-final shortfall as an explicit masked gap.
                n = min(len(gt), len(zp), len(tp))
                if n <= 0:
                    raise ValueError(f"{wid}: empty common time support")
                truncated_frames["reference"] += len(gt) - n
                truncated_frames["zipcount"] += len(zp) - n
                truncated_frames["teacher"] += len(tp) - n
                recording_id, start_frame = recording_position(it)
                windows_by_recording[recording_id].append(TimelineWindow(
                    recording_id=recording_id,
                    window_id=wid,
                    start_frame=start_frame,
                    reference=gt[:n],
                    zipcount=zp[:n],
                    teacher=tp[:n],
                ))
                collected_ids.add(wid)

    if collected_ids != common_ids:
        missing = sorted(common_ids - collected_ids)
        raise RuntimeError(
            f"collected {len(collected_ids)} common windows, expected "
            f"{len(common_ids)}; missing examples={missing[:5]}")

    # This is the load-bearing recording-level pass.  update_system is called
    # once per recording, not once per 30-s window, so joins participate in
    # event matching and segment/run counting exactly like ordinary frames.
    agg = {s: _empty_system(tolerances_ms) for s in ("zipcount", "teacher")}
    timeline_gap_frames = 0
    timelines = []
    for recording_id in sorted(windows_by_recording):
        timeline = stitch_recording(windows_by_recording[recording_id])
        timelines.append(timeline)
        timeline_gap_frames += timeline.gap_frames
        update_system(agg["zipcount"], timeline.reference, timeline.zipcount,
                      tolerances_ms, valid=timeline.valid)
        update_system(agg["teacher"], timeline.reference, timeline.teacher,
                      tolerances_ms, valid=timeline.valid)

    if agg["zipcount"]["items"] != len(windows_by_recording):
        raise RuntimeError("internal error: recording aggregate count mismatch")

    print(f"primary tolerance +/-{args.tol_ms} ms | checkpoint "
          f"{os.path.basename(args.checkpoint)}")
    print(f"common teacher set: {len(common_ids):,}/{len(items):,} manifest windows; "
          f"skipped without teacher={skipped_no_teacher:,}")
    print(f"recording timelines: {len(timelines):,} recordings at {FRAME_HZ} Hz; "
          f"masked internal gaps={timeline_gap_frames:,} frames")
    print(f"right-tail-only support trims (no shift/pad): "
          f"reference={truncated_frames['reference']:,}, "
          f"zipcount={truncated_frames['zipcount']:,}, "
          f"teacher={truncated_frames['teacher']:,} frames")
    for name in ("zipcount", "teacher"):
        print_system(name, agg[name], args.tol_ms, tolerances_ms)


if __name__ == "__main__":
    main()
