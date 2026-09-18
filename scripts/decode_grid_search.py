"""Select and LOCK exactly ONE decoder (hysteresis, Viterbi, OR semi-Markov
-- not all three) on the DEV cache only. The held-out 4-corpus caches are
never touched by this script, so nothing here can leak DEV-set structure
into what gets reported as the final held-out result.

Locking a single winner (not "whichever of the three happen to pass the
adoption bar on held-out") matters methodologically: letting held-out data
help choose AMONG decoder families, even implicitly by reporting all three
and only discussing whichever looks best, is a form of multiple-comparison
leakage -- the held-out set must only ever be used to CONFIRM or REJECT one
pre-committed choice (caught in review; an earlier version's held-out
scorer tested and separately "adopted" each of the three families).

Selection: highest DEV macro-F1 among the grid, subject to not reducing
mean_edit_score OR mean_boundary_f1 below the raw-argmax DEV baseline (a
light guard against a hyperparameter that wins on frame accuracy purely by
over-smoothing away real segment structure). Within a family, ties broken
by higher mean_edit_score. ACROSS families, the single overall winner is
whichever family's best surviving grid point has the highest DEV macro-F1;
ties (to 4 decimal places) broken by preferring the SIMPLER family in the
fixed order hysteresis < viterbi < semimarkov (fewer/looser modelling
assumptions first), never by an ad hoc post-hoc judgement call.

Usage:
  PYTHONPATH=. python scripts/decode_grid_search.py \
      --dev-cache logs/logit_cache/ami_dev_s1234 \
      --priors artifacts/wavlm_offline/decoder_priors.npz \
      --out artifacts/wavlm_offline/locked_decoder_configs.json
"""
import argparse
import itertools
import json

import numpy as np

from src.utils.decode_scoring import decode_protocol_sha256, load_cache, score_cache, sha_of_file
from src.utils.decoders import hysteresis_min_duration, semimarkov_decode, viterbi_decode
from src.utils.duration_decoder_protocol import (
    BOUNDARY_TOL_FRAMES,
    FRAME_RATE_HZ,
    PREREGISTERED_DEV,
    PREREGISTERED_PRIORS,
    validate_meta_against_protocol,
    validate_priors_provenance,
)

FAMILY_ORDER = ["hysteresis", "viterbi", "semimarkov"]  # simplicity tie-break order


def select_winner(candidates: dict, family_order=FAMILY_ORDER):
    """Pick the single overall winning decoder family from `candidates`
    (family -> (macro_f1, edit_score, params, result)).

    Highest DEV macro-F1 across families; ties -- rounded to 4 decimal
    places, NOT near-exact float equality (see the module docstring's
    "Selection" paragraph for why 1e-9 was the wrong threshold) -- broken by
    FAMILY_ORDER (simpler first), never by edit-score or any other
    secondary metric.

    Extracted as a standalone, pure, importable function (rather than
    inlined in main()) specifically so it can be unit-tested directly --
    tests/test_decoders.py asserts this function's actual behaviour on a
    constructed tie, rather than re-implementing the tie-break logic
    separately in a test and only checking that reimplementation.

    Returns (winner_family: str, tied: List[str]) or (None, []) if
    `candidates` is empty.
    """
    if not candidates:
        return None, []
    best_f1_rounded = round(max(v[0] for v in candidates.values()), 4)
    tied = [fam for fam, v in candidates.items() if round(v[0], 4) == best_f1_rounded]
    winner = min(tied, key=lambda fam: family_order.index(fam))
    return winner, tied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-cache", required=True)
    ap.add_argument("--priors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--boundary-tol-frames", type=int, default=BOUNDARY_TOL_FRAMES,
        choices=[BOUNDARY_TOL_FRAMES],
        help="fixed by preregistration; retained as a visible CLI field, not a tunable knob",
    )
    args = ap.parse_args()

    cache, cache_meta = load_cache(args.dev_cache)
    validate_meta_against_protocol(cache_meta, PREREGISTERED_DEV, "DEV cache")
    priors_meta = validate_priors_provenance(args.priors, PREREGISTERED_PRIORS)
    priors_sha256 = sha_of_file(args.priors)
    protocol_sha256 = decode_protocol_sha256()
    with np.load(args.priors) as priors:
        priors_frame_rate = float(priors["frame_rate_hz"])
        max_dur = int(priors["max_dur_frames"])
        if priors_frame_rate != FRAME_RATE_HZ:
            raise SystemExit(
                f"--priors was fit at {priors_frame_rate}Hz, expected {FRAME_RATE_HZ}Hz"
            )
        if max_dur != PREREGISTERED_PRIORS["max_dur_frames"]:
            raise SystemExit(
                f"--priors max_dur_frames={max_dur}, expected preregistered "
                f"{PREREGISTERED_PRIORS['max_dur_frames']}"
            )
        log_trans_frame = priors["log_trans_frame"].copy()
        log_trans_segment = priors["log_trans_segment"].copy()
        dur_log_pmf = priors["dur_log_pmf"].copy()

    def log_probs_of(probs):
        return np.log(np.clip(probs, 1e-8, 1.0))

    baseline = score_cache(cache, decode_fn=None, boundary_tol_frames=args.boundary_tol_frames)
    print(f"DEV baseline (argmax): macroF1={baseline['macro_f1']:.4f} "
          f"edit={baseline['mean_edit_score']:.2f} frag={baseline['mean_fragmentation_rate']:.3f} "
          f"boundaryF1={baseline['mean_boundary_f1']:.4f}")

    candidates = {}  # family -> (macro_f1, edit_score, params, result)

    # --- 1. hysteresis + min-duration ---
    best = None
    for margin, min_dur in itertools.product([0.05, 0.1, 0.15, 0.2, 0.3], [3, 5, 8, 12]):
        def fn(probs, valid, m=margin, d=min_dur):
            return hysteresis_min_duration(probs, valid, switch_margin=m, min_duration_frames=d)
        r = score_cache(cache, fn, args.boundary_tol_frames)
        ok = r["mean_edit_score"] >= baseline["mean_edit_score"] and r["mean_boundary_f1"] >= baseline["mean_boundary_f1"]
        print(f"  hysteresis margin={margin} min_dur={min_dur}: macroF1={r['macro_f1']:.4f} "
              f"edit={r['mean_edit_score']:.2f} boundaryF1={r['mean_boundary_f1']:.4f} "
              f"{'OK' if ok else 'REJECTED(regresses edit/boundary)'}")
        if not ok:
            continue
        key = (r["macro_f1"], r["mean_edit_score"])
        if best is None or key > best[0]:
            best = (key, {"switch_margin": margin, "min_duration_frames": min_dur}, r)
    if best:
        candidates["hysteresis"] = (best[0][0], best[0][1], best[1], best[2])
        print(f"hysteresis best-on-grid: {best[1]} -> DEV macroF1={best[2]['macro_f1']:.4f}")
    else:
        print("hysteresis: no grid point beat the edit/boundary guard")

    # --- 2. Viterbi (frame-transition prior, self-loop-heavy) ---
    best = None
    for tw in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]:
        def fn(probs, valid, tw=tw):
            return viterbi_decode(log_probs_of(probs), valid, log_trans_frame, trans_weight=tw)
        r = score_cache(cache, fn, args.boundary_tol_frames)
        ok = r["mean_edit_score"] >= baseline["mean_edit_score"] and r["mean_boundary_f1"] >= baseline["mean_boundary_f1"]
        print(f"  viterbi trans_weight={tw}: macroF1={r['macro_f1']:.4f} "
              f"edit={r['mean_edit_score']:.2f} boundaryF1={r['mean_boundary_f1']:.4f} "
              f"{'OK' if ok else 'REJECTED(regresses edit/boundary)'}")
        if not ok:
            continue
        key = (r["macro_f1"], r["mean_edit_score"])
        if best is None or key > best[0]:
            best = (key, {"trans_weight": tw}, r)
    if best:
        candidates["viterbi"] = (best[0][0], best[0][1], best[1], best[2])
        print(f"viterbi best-on-grid: {best[1]} -> DEV macroF1={best[2]['macro_f1']:.4f}")
    else:
        print("viterbi: no grid point beat the edit/boundary guard")

    # --- 3. semi-Markov (segment-transition prior, self-transition forbidden) ---
    best = None
    for tw, dw in itertools.product([0.0, 0.5, 1.0, 2.0], [0.5, 1.0, 2.0]):
        def fn(probs, valid, tw=tw, dw=dw):
            return semimarkov_decode(log_probs_of(probs), valid, log_trans_segment, dur_log_pmf,
                                      max_dur=max_dur, trans_weight=tw, duration_weight=dw)
        r = score_cache(cache, fn, args.boundary_tol_frames)
        ok = r["mean_edit_score"] >= baseline["mean_edit_score"] and r["mean_boundary_f1"] >= baseline["mean_boundary_f1"]
        print(f"  semimarkov trans_weight={tw} duration_weight={dw}: macroF1={r['macro_f1']:.4f} "
              f"edit={r['mean_edit_score']:.2f} boundaryF1={r['mean_boundary_f1']:.4f} "
              f"{'OK' if ok else 'REJECTED(regresses edit/boundary)'}")
        if not ok:
            continue
        key = (r["macro_f1"], r["mean_edit_score"])
        if best is None or key > best[0]:
            best = (key, {"trans_weight": tw, "duration_weight": dw, "max_dur": max_dur}, r)
    if best:
        candidates["semimarkov"] = (best[0][0], best[0][1], best[1], best[2])
        print(f"semimarkov best-on-grid: {best[1]} -> DEV macroF1={best[2]['macro_f1']:.4f}")
    else:
        print("semimarkov: no grid point beat the edit/boundary guard")

    if not candidates:
        print("\nNO decoder family produced a grid point beating the edit/boundary "
              "guard -- locking NOTHING. This is itself a valid (null) result.")
        out = {
            "dev_cache": args.dev_cache, "priors": args.priors,
            "priors_sha256": priors_sha256, "decode_protocol_sha256": protocol_sha256,
            "priors_meta": priors_meta,
            "selection_protocol": {
                "dev": PREREGISTERED_DEV,
                "boundary_tol_frames": BOUNDARY_TOL_FRAMES,
            },
            "dev_cache_meta": cache_meta,
            "dev_baseline_macro_f1": baseline["macro_f1"],
            "winning_family": None, "locked_hyperparams": {},
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"-> {args.out}")
        return

    winner, tied = select_winner(candidates)
    win_f1, win_edit, win_params, win_result = candidates[winner]

    print(f"\n=== WINNER: {winner} ===  DEV macroF1={win_f1:.4f} (baseline {baseline['macro_f1']:.4f}, "
          f"delta {win_f1 - baseline['macro_f1']:+.4f}) params={win_params}")
    if len(tied) > 1:
        print(f"(tie-break applied among {tied}, by fixed simplicity order {FAMILY_ORDER})")

    out = {
        "dev_cache": args.dev_cache,
        "priors": args.priors,
        "priors_sha256": priors_sha256,
        "decode_protocol_sha256": protocol_sha256,
        "priors_meta": priors_meta,
        "selection_protocol": {
            "dev": PREREGISTERED_DEV,
            "boundary_tol_frames": BOUNDARY_TOL_FRAMES,
        },
        "dev_cache_meta": cache_meta,
        "dev_baseline_macro_f1": baseline["macro_f1"],
        "winning_family": winner,
        "locked_hyperparams": {winner: win_params},
        "dev_winner_result": {k: v for k, v in win_result.items() if k != "confusion"},
        "dev_all_candidates": {
            fam: {k: v for k, v in res.items() if k != "confusion"}
            for fam, (_, _, _, res) in candidates.items()
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
