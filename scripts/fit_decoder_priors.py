"""Fit the decoder priors used by viterbi_decode / semimarkov_decode
(src/utils/decoders.py) from TRAINING labels only -- never from DEV or the
held-out corpora, to avoid leaking eval-set structure into what is
effectively a model parameter (same methodological standard as fitting the
model itself only on train).

Produces TWO different transition matrices, not one, because Viterbi and
semi-Markov need genuinely different priors:

  log_trans_frame   fit per FRAME (self-loop-heavy, diagonal ~0.93-0.98) --
                     for viterbi_decode, where the self-transition
                     probability is what implicitly encodes a geometric
                     segment-duration distribution.
  log_trans_segment fit per SEGMENT, i.e. only over adjacent-and-DIFFERENT-
                     class run pairs, diagonal forced to -inf -- for
                     semimarkov_decode, whose whole point is an EXPLICIT
                     duration prior; reusing the self-loop-heavy frame
                     matrix there let the DP carve one long true run into
                     several same-class hidden segments purely to re-farm
                     the duration prior's peak (a real bug caught in
                     review before this ever ran on real data -- see
                     src/utils/decoders.py's semimarkov_decode docstring).

Restricted to REAL conversational sources (excludes libri2mix/libri3mix
synthetic mixes, solo_1spk, musan_noise, silence -- same spirit as
SpeakerCountDataset.TEACHER_DENY_SOURCES): those exist purely as training
augmentation/regularisation and would skew segment-duration statistics
toward degenerate all-silence or all-single-speaker runs not representative
of the AMI/DipCo/MSDWild/VoxConverse domains this prior is meant to help on.

Labels are read directly from each item's `label_filepath` (100Hz, no audio
needed) and downsampled to the model's 25Hz output rate via the SAME
majority_vote convention training itself uses (label_utils.
align_labels_to_output_len), so the fitted statistics are in the same units
the decoders will consume them in.

Usage:
  PYTHONPATH=. python scripts/fit_decoder_priors.py \
      --manifest "/media/.../train_manifest_v3.json" \
      --out artifacts/wavlm_offline/decoder_priors.npz \
      --frame-rate-hz 25 --max-dur-frames 750 --sample-every 10
"""
import argparse
import hashlib
import json
import os

import numpy as np

from src.data.label_utils import align_labels_to_output_len

REAL_SOURCE_PREFIXES = (
    "ami_train", "alimeeting_train", "aishell4_train", "msdwild_train",
    "ramc_train", "voxconverse_train", "nsf_train",
)
NUM_CLASSES = 4


def _sha_of_file(path: str, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame-rate-hz", type=float, default=25.0)
    ap.add_argument("--max-dur-frames", type=int, default=750)
    ap.add_argument("--sample-every", type=int, default=10,
                     help="use every Nth matching window (speed; the "
                          "statistics fit here are low-dimensional and "
                          "converge long before using all ~127k real-source "
                          "windows)")
    ap.add_argument("--laplace", type=float, default=1.0,
                     help="additive (Laplace) smoothing TOTAL count per row "
                          "for the transition matrices (only 4 categories "
                          "per row, so a flat +1 per cell is reasonable)")
    ap.add_argument("--laplace-duration-total", type=float, default=1.0,
                     help="additive (Laplace) smoothing TOTAL pseudo-count "
                          "per class for the duration histogram, SPREAD "
                          "UNIFORMLY over all max-dur-frames bins (i.e. "
                          "each bin gets this / max_dur_frames added) -- "
                          "deliberately NOT a flat +1 PER BIN like the "
                          "transition matrix: with 750 bins, a flat +1/bin "
                          "adds 750 pseudo-segments per class, which can "
                          "swamp a real (possibly much smaller) count in "
                          "the tail, and the total added mass would keep "
                          "growing every time --max-dur-frames is raised "
                          "even though nothing about the real data changed "
                          "(caught in review: the max_dur bump from 250->"
                          "750 also silently tripled total smoothing mass "
                          "under the old flat-per-bin scheme)")
    args = ap.parse_args()

    manifest_sha = _sha_of_file(args.manifest)
    frame_trans_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    seg_trans_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    dur_counts_raw = np.zeros((NUM_CLASSES, args.max_dur_frames), dtype=np.float64)
    n_segments_per_class = np.zeros(NUM_CLASSES, dtype=np.int64)  # excl. edge-censored
    n_capped_per_class = np.zeros(NUM_CLASSES, dtype=np.int64)    # raw length > max_dur_frames
    n_edge_censored_runs = 0  # first/last run per window, excluded from duration stats

    n_used = 0
    n_seen = 0
    n_missing = 0
    with open(args.manifest) as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            src = str(item.get("source", ""))
            if not src.startswith(REAL_SOURCE_PREFIXES):
                continue
            n_seen += 1
            if n_seen % args.sample_every != 0:
                continue
            lab_path = item.get("label_filepath")
            if not lab_path or not os.path.exists(lab_path):
                n_missing += 1
                continue
            labels_100hz = np.load(lab_path).astype(np.int64)
            if labels_100hz.size == 0:
                continue
            target_len = max(1, int(round(item["duration"] * args.frame_rate_hz)))
            seq = align_labels_to_output_len(labels_100hz, target_len, method="majority_vote")
            seq = np.clip(seq, 0, NUM_CLASSES - 1)

            # frame transitions (within-window only; a window boundary is
            # not a real transition, so consecutive windows are NOT
            # stitched here -- this is a statistical fit, not the
            # decode-time stitching eval_sequence_metrics.py does for a
            # full recording)
            if len(seq) > 1:
                frm = seq[:-1]
                to = seq[1:]
                np.add.at(frame_trans_counts, (frm, to), 1.0)

            # run-length durations + segment transitions (adjacent runs are
            # by construction always a DIFFERENT class, so this naturally
            # only ever populates seg_trans_counts' off-diagonal)
            change = np.flatnonzero(np.diff(seq) != 0)
            starts = np.r_[0, change + 1]
            ends = np.r_[change + 1, len(seq)]
            run_classes = [int(seq[s]) for s in starts]
            for c_from, c_to in zip(run_classes[:-1], run_classes[1:]):
                seg_trans_counts[c_from, c_to] += 1.0

            # duration stats: EXCLUDE the first and last run of this window
            # from the histogram -- both are right/left-CENSORED by the
            # window edge (we only know they lasted AT LEAST this long, not
            # their true completed length), so counting them as complete
            # observations biases the fitted distribution toward short
            # durations. A window with only 1-2 runs contributes nothing
            # here (both/its one run is an edge run) -- expected, not a bug.
            interior = range(1, len(starts) - 1) if len(starts) > 2 else range(0)
            for i in interior:
                s, e, c = starts[i], ends[i], run_classes[i]
                raw_len = int(e - s)
                n_segments_per_class[c] += 1
                if raw_len > args.max_dur_frames:
                    n_capped_per_class[c] += 1
                d = min(raw_len, args.max_dur_frames)
                dur_counts_raw[c, d - 1] += 1.0
            n_edge_censored_runs += min(2, len(starts))

            n_used += 1

    if n_used == 0:
        raise SystemExit("no usable windows found -- check --manifest / source filters")

    frame_trans_counts += args.laplace
    frame_trans = frame_trans_counts / frame_trans_counts.sum(axis=1, keepdims=True)
    log_trans_frame = np.log(frame_trans)

    # Laplace-smooth ONLY the off-diagonal (the diagonal is structurally
    # forbidden, not merely rare -- smoothing it with the same count as a
    # real observable pair would let a large enough --laplace value leak a
    # non-negligible self-transition probability back in).
    off_diag_mask = ~np.eye(NUM_CLASSES, dtype=bool)
    seg_trans_counts_smoothed = seg_trans_counts.copy()
    seg_trans_counts_smoothed[off_diag_mask] += args.laplace
    seg_trans = np.zeros_like(seg_trans_counts_smoothed)
    row_sums = seg_trans_counts_smoothed.sum(axis=1, keepdims=True)
    seg_trans[off_diag_mask] = (seg_trans_counts_smoothed / row_sums)[off_diag_mask]
    log_trans_segment = np.log(np.clip(seg_trans, 1e-300, 1.0))
    np.fill_diagonal(log_trans_segment, -1e18)

    # p99 / cap-hit stats computed from RAW counts, BEFORE any smoothing --
    # smoothing (especially spread over 750 bins) would otherwise distort
    # what "p99 of the real data" means (caught in review: printing p99
    # after adding Laplace mass to every bin isn't p99 of the data anymore).
    p99_dur_frames_raw = []
    for c in range(NUM_CLASSES):
        total = dur_counts_raw[c].sum()
        if total == 0:
            p99_dur_frames_raw.append(None)
            continue
        p99_dur_frames_raw.append(int(np.searchsorted(np.cumsum(dur_counts_raw[c]) / total, 0.99)) + 1)

    # duration smoothing: total pseudo-mass per class is FIXED at
    # --laplace-duration-total regardless of --max-dur-frames, spread
    # uniformly over all bins -- see the argparse help above for why a flat
    # +1/bin (the transition-matrix scheme) is wrong here.
    per_bin = args.laplace_duration_total / args.max_dur_frames
    dur_counts = dur_counts_raw + per_bin
    dur_pmf = dur_counts / dur_counts.sum(axis=1, keepdims=True)
    dur_log_pmf = np.log(dur_pmf)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        log_trans_frame=log_trans_frame,
        log_trans_segment=log_trans_segment,
        dur_log_pmf=dur_log_pmf,
        max_dur_frames=args.max_dur_frames,
        frame_rate_hz=args.frame_rate_hz,
    )
    priors_sha256 = _sha_of_file(args.out if args.out.endswith(".npz") else args.out + ".npz")
    meta = {
        "manifest": args.manifest,
        "manifest_sha256": manifest_sha,
        "source_prefixes": list(REAL_SOURCE_PREFIXES),
        "sample_every": args.sample_every,
        "laplace_transition": args.laplace,
        "laplace_duration_total_per_class": args.laplace_duration_total,
        "n_windows_seen_matching_source": n_seen,
        "n_windows_used": n_used,
        "n_windows_missing_label_file": n_missing,
        "n_edge_censored_runs_excluded": int(n_edge_censored_runs),
        "n_segments_per_class_used_for_duration": n_segments_per_class.tolist(),
        "n_segments_capped_per_class": n_capped_per_class.tolist(),
        "p99_duration_frames_per_class_raw": p99_dur_frames_raw,
        "frame_rate_hz": args.frame_rate_hz,
        "max_dur_frames": args.max_dur_frames,
        "priors_npz_sha256": priors_sha256,
    }
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"fit priors from {n_used} windows ({n_seen} matched source filter, "
          f"sampled 1/{args.sample_every}, {n_missing} missing label files)")
    print(f"excluded {n_edge_censored_runs} edge-censored (first/last-per-window) runs "
          f"from duration stats; {n_segments_per_class.tolist()} interior segments/class used")
    print("frame-transition matrix (self-loop-heavy; for viterbi_decode), rows=from cols=to:")
    print(np.array2string(frame_trans, precision=4, suppress_small=True))
    print("segment-transition matrix (diagonal forbidden; for semimarkov_decode), rows=from cols=to:")
    print(np.array2string(seg_trans, precision=4, suppress_small=True))
    mean_dur = (dur_pmf * (np.arange(args.max_dur_frames) + 1)).sum(axis=1) / args.frame_rate_hz
    print(f"mean segment duration per class (s, post-smoothing): {mean_dur}")
    print(f"p99 segment duration per class (frames, RAW pre-smoothing data, "
          f"capped at max_dur={args.max_dur_frames}): {p99_dur_frames_raw}")
    print(f"segments whose RAW length exceeded max_dur_frames (cap actually binding): "
          f"{n_capped_per_class.tolist()} of {n_segments_per_class.tolist()} per class")
    print(f"priors_npz_sha256={priors_sha256}")
    print(f"-> {args.out} (+ .meta.json)")


if __name__ == "__main__":
    main()
