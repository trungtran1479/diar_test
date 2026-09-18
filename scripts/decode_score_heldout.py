"""Apply the SINGLE LOCKED decoder (from scripts/decode_grid_search.py --
never re-tuned here, and never more than one family: an earlier version
tested and separately "adopted" all three locked families against held-out
data, which is multiple-comparison leakage -- held-out may only confirm or
reject the one pre-committed winner) to the 4 held-out corpora across 3
seeds, and check the pre-registered adoption bar:

  mean delta (decoder - argmax) over the 4-corpus mean >= +0.005
  AND same sign in >= 2 of 3 seeds
  AND no single corpus's mean delta < -0.002

The seed/corpus SET this bar is computed over is itself pre-registered
(PREREGISTERED_SEEDS / PREREGISTERED_CORPORA below) and enforced exactly --
see check 0. An earlier version let `--seeds`/`--corpora` be freely
overridden on the command line while the adoption bar's `same_sign >= 2`
stayed hardcoded, so e.g. running only 2 seeds (both positive) would still
report "2/3" and pass, or scoring only one favourable corpus could ADOPT
while silently ignoring the other three.

If the locked decoder does not pass, it is reported as REJECTED, not
silently dropped -- a null result here is exactly the kind of finding this
ablation exists to produce (see zipcount-wavlm-offline-progress memory).

This scores at the cache's native 25Hz -- a DIAGNOSTIC result, not yet a
unified-protocol (100Hz) adoption claim; see src/utils/decode_scoring.py's
module docstring. A decoder passing here should be re-confirmed through the
100Hz protocol before being reported as improving the project's headline
numbers.

STRICT TWO-PASS STRUCTURE, on purpose: Pass 1 loads every required cache and
verifies EVERY lineage invariant below with ZERO scores computed or printed.
Only after "lineage OK" is printed does Pass 2 decode/score anything. An
earlier version interleaved loading-and-scoring in one loop, printing each
corpus/seed's macro-F1 as it went and only running the cross-checks
afterward -- so a lineage failure was reported only after the (invalid)
held-out numbers had already been shown, defeating the entire point of
"held-out only confirms or rejects a pre-committed decision."

LINEAGE, checked automatically (no manual --checkpoint-shas flag -- an
earlier version had one, but it was optional and easy to simply not pass,
so verification silently didn't happen by default; everything below is
derived from the caches' OWN recorded metadata instead):

  0. `--seeds`/`--corpora` must be EXACTLY PREREGISTERED_SEEDS/
     PREREGISTERED_CORPORA (as sets, no duplicates) -- this is what makes
     "same_sign >= 2 of 3" and "4-corpus mean" mean what the module
     docstring says they mean; a caller cannot silently narrow the
     evaluated set and still trigger the same adoption bar.
  1. `_cache_meta.json` must exist, pass schema validation (every required
     field present/typed/non-empty, `frame_rate_hz` exactly 25.0 -- see
     decode_scoring.validate_cache_meta_schema), its recorded file list
     must match the directory's actual .npz files, AND its `support_sha256`
     must match one RECOMPUTED from the actual .npz contents just read (not
     merely the string in the JSON) -- all in load_cache.
  2. `meta["corpus_id"]` must equal the corpus name implied by the cache's
     OWN directory (`<corpus_id>_s<seed>`) -- otherwise a cache dumped from
     an entirely different corpus's manifest could sit in a directory named
     e.g. "ami_test_s1234" and be scored as if it were AMI (a review
     counterexample constructed exactly this: three internally-consistent
     seeds all claiming one corpus that none of them actually were).
  3. This script's priors (`--priors`) must SHA256-match the priors the
     locked decoder was tuned against (`locked_doc["priors_sha256"]`), and
     the priors' own recorded `frame_rate_hz` must be 25.0.
  4. The decoding LOGIC (src/utils/decoders.py + decode_scoring.py) must
     SHA256-match what it was when DEV grid search ran
     (`locked_doc["decode_protocol_sha256"]`).
  5. `locked_doc["dev_cache_meta"]` itself is schema-validated (an earlier
     version read it directly with no validation at all, so a DEV lock
     produced from a broken/incomplete cache could still be "trusted").
     For the ONE seed that DEV tuning actually used, that seed's held-out
     caches must use the EXACT SAME checkpoint DEV was tuned on.
  6. Within each seed, all 4 corpora must share the same checkpoint_sha256
     and config_sha256.
  7. ACROSS ALL THREE SEEDS AND ALL FOUR CORPORA: `canonical_config_sha256`
     (proves the same experimental arm -- see scripts/dump_logits.py),
     `protocol_sha256` (proves the same prediction-generating code), and
     `source` must all be identical.
  8. Within each corpus, all 3 seeds must share the same `manifest_sha256`
     (same corpus manifest file), `restrict_set_sha256`, and IDENTICAL
     (recording set, ground truth, valid-frame mask) support.

A missing cache directory, or any lineage mismatch above, is a HARD
FAILURE, not a silent skip. The final report embeds every loaded cache's
key fingerprints (checkpoint/config/support SHA per corpus+seed) so it can
later be verified against the exact artifacts that produced it, without
having to keep the raw cache directories around indefinitely.

Usage:
  PYTHONPATH=. python scripts/decode_score_heldout.py \
      --cache-root logs/logit_cache \
      --locked artifacts/wavlm_offline/locked_decoder_configs.json \
      --priors artifacts/wavlm_offline/decoder_priors.npz \
      --out results/decode_heldout_report.json
"""
import argparse
import json
import os

import numpy as np

from src.utils.decode_scoring import (
    decode_protocol_sha256, load_cache, score_cache, sha_of_file,
    verify_consistent_across, verify_corpus_id, verify_gt_consistent_across_seeds,
    validate_cache_meta_schema,
)
from src.utils.decoders import hysteresis_min_duration, semimarkov_decode, viterbi_decode
from src.utils.duration_decoder_protocol import (
    BOUNDARY_TOL_FRAMES,
    FRAME_RATE_HZ,
    PREREGISTERED_CORPORA as _PROTOCOL_CORPORA,
    PREREGISTERED_DEV,
    PREREGISTERED_HELDOUT,
    PREREGISTERED_PRIORS,
    PREREGISTERED_SEEDS as _PROTOCOL_SEEDS,
    validate_meta_against_protocol,
    validate_priors_provenance,
)

# The one pre-registered protocol this script will ever score. Not
# overridable via CLI on purpose -- see the module docstring's check 0.
PREREGISTERED_SEEDS = _PROTOCOL_SEEDS
PREREGISTERED_CORPORA = _PROTOCOL_CORPORA
PREREGISTERED_INPUTS = PREREGISTERED_HELDOUT


def make_decoder(winning_family, params, log_trans_frame, log_trans_segment, dur_log_pmf, max_dur):
    def log_probs_of(probs):
        return np.log(np.clip(probs, 1e-8, 1.0))

    if winning_family == "hysteresis":
        p = params
        return lambda probs, valid, p=p: hysteresis_min_duration(
            probs, valid, switch_margin=p["switch_margin"], min_duration_frames=p["min_duration_frames"])
    if winning_family == "viterbi":
        p = params
        return lambda probs, valid, p=p: viterbi_decode(
            log_probs_of(probs), valid, log_trans_frame, trans_weight=p["trans_weight"])
    if winning_family == "semimarkov":
        p = params
        return lambda probs, valid, p=p: semimarkov_decode(
            log_probs_of(probs), valid, log_trans_segment, dur_log_pmf,
            max_dur=p.get("max_dur", max_dur), trans_weight=p["trans_weight"], duration_weight=p["duration_weight"])
    raise ValueError(f"unknown decoder family: {winning_family!r}")


def load_and_verify_lineage(args, locked_doc):
    """PASS 1: load every required cache and verify every invariant listed
    in the module docstring. Returns (caches, metas) keyed by (corpus,
    seed) ONLY if everything checks out -- raises SystemExit otherwise.
    Computes/prints nothing about scores; that is Pass 2's job entirely."""
    # check 3 (priors)
    priors_sha256 = sha_of_file(args.priors)
    if priors_sha256 != locked_doc.get("priors_sha256"):
        raise SystemExit(
            f"--priors {args.priors!r} has sha256={priors_sha256}, but the locked "
            f"decision in {args.locked!r} was tuned against sha256="
            f"{locked_doc.get('priors_sha256')} -- the priors file changed since "
            "DEV grid search ran."
        )
    priors_meta = validate_priors_provenance(args.priors, PREREGISTERED_PRIORS)
    if locked_doc.get("priors_meta") != priors_meta:
        raise SystemExit(
            "the priors provenance sidecar no longer matches the copy embedded when "
            "DEV selection was locked"
        )
    with np.load(args.priors) as priors_npz:
        priors_frame_rate = float(priors_npz["frame_rate_hz"])
        priors_max_dur = int(priors_npz["max_dur_frames"])
    if priors_frame_rate != FRAME_RATE_HZ:
        raise SystemExit(
            f"--priors {args.priors!r} was fit at {priors_frame_rate}Hz, "
            f"not {FRAME_RATE_HZ}Hz."
        )
    if priors_max_dur != PREREGISTERED_PRIORS["max_dur_frames"]:
        raise SystemExit(
            f"--priors max_dur_frames={priors_max_dur}, expected preregistered "
            f"{PREREGISTERED_PRIORS['max_dur_frames']}"
        )

    # check 4 (decode protocol)
    protocol_sha256 = decode_protocol_sha256()
    if protocol_sha256 != locked_doc.get("decode_protocol_sha256"):
        raise SystemExit(
            f"src/utils/decoders.py + decode_scoring.py currently hash to "
            f"{protocol_sha256}, but {args.locked!r} was locked against "
            f"{locked_doc.get('decode_protocol_sha256')} -- the decoding logic "
            "changed since DEV grid search ran."
        )

    # check 5 (DEV meta itself must be schema-valid, not read on faith)
    dev_meta = locked_doc["dev_cache_meta"]
    validate_cache_meta_schema(dev_meta, f"{args.locked!r}['dev_cache_meta']")
    validate_meta_against_protocol(dev_meta, PREREGISTERED_DEV, "locked DEV cache")
    expected_selection_protocol = {
        "dev": PREREGISTERED_DEV,
        "boundary_tol_frames": BOUNDARY_TOL_FRAMES,
    }
    if locked_doc.get("selection_protocol") != expected_selection_protocol:
        raise SystemExit(
            f"locked selection_protocol={locked_doc.get('selection_protocol')} does not "
            f"match the current preregistered protocol={expected_selection_protocol}"
        )
    dev_seed = str(dev_meta["seed"])
    dev_checkpoint_sha256 = dev_meta["checkpoint_sha256"]
    dev_canonical_config_sha256 = dev_meta["canonical_config_sha256"]
    dev_protocol_sha256 = dev_meta["protocol_sha256"]

    caches, metas = {}, {}
    for corpus in args.corpora:
        for seed in args.seeds:
            cache_dir = os.path.join(args.cache_root, f"{corpus}_s{seed}")
            if not os.path.isdir(cache_dir):
                raise SystemExit(
                    f"REQUIRED cache dir missing: {cache_dir} -- refusing to produce a "
                    "held-out report silently missing a seed/corpus."
                )
            cache, meta = load_cache(cache_dir)  # schema- and support-hash-validates internally
            verify_corpus_id(meta, corpus, cache_dir)  # check 2
            validate_meta_against_protocol(
                meta,
                {"corpus_id": corpus, **PREREGISTERED_INPUTS[corpus]},
                f"held-out cache {cache_dir}",
            )
            caches[(corpus, seed)] = cache
            metas[(corpus, seed)] = meta

            if seed == dev_seed and meta.get("checkpoint_sha256") != dev_checkpoint_sha256:
                raise SystemExit(
                    f"{cache_dir}: checkpoint_sha256={meta.get('checkpoint_sha256')}, but "
                    f"DEV grid search (seed {dev_seed}) was tuned against checkpoint_sha256="
                    f"{dev_checkpoint_sha256} -- these must be the SAME checkpoint."
                )

    # check 6: within a seed, all corpora share checkpoint/config
    for seed in args.seeds:
        seed_metas = {corpus: metas[(corpus, seed)] for corpus in args.corpora}
        verify_consistent_across(seed_metas, "checkpoint_sha256", f"seed {seed} across corpora")
        verify_consistent_across(seed_metas, "config_sha256", f"seed {seed} across corpora")

    # check 7: ACROSS ALL SEEDS AND CORPORA -- canonical config, decode
    # protocol, and source must all be identical. Canonical-config catches
    # one seed belonging to a different experimental arm even though it's
    # internally self-consistent across its own 4 corpora (check 6 alone
    # cannot catch this).
    all_metas_flat = {f"{corpus}_s{seed}": metas[(corpus, seed)]
                       for corpus in args.corpora for seed in args.seeds}
    verify_consistent_across(all_metas_flat, "canonical_config_sha256", "all seeds/corpora")
    verify_consistent_across(all_metas_flat, "protocol_sha256", "all seeds/corpora")
    verify_consistent_across(all_metas_flat, "source", "all seeds/corpora")
    one_meta = next(iter(all_metas_flat.values()))
    if one_meta.get("canonical_config_sha256") != dev_canonical_config_sha256:
        raise SystemExit(
            f"held-out caches' canonical_config_sha256="
            f"{one_meta.get('canonical_config_sha256')} does not match DEV's "
            f"({dev_canonical_config_sha256}) -- DEV tuning and held-out scoring "
            "are not the same experimental arm (modulo seed/log_dir)."
        )
    if one_meta.get("protocol_sha256") != dev_protocol_sha256:
        raise SystemExit(
            f"held-out caches' protocol_sha256={one_meta.get('protocol_sha256')} does not "
            f"match DEV's ({dev_protocol_sha256}) -- DEV tuning and held-out scoring used "
            "different prediction-generating code."
        )

    # check 8: within a corpus, all seeds share the corpus manifest,
    # restrict-set, and full support (recording set + gt + valid).
    for corpus in args.corpora:
        corpus_metas = {seed: metas[(corpus, seed)] for seed in args.seeds}
        verify_consistent_across(corpus_metas, "manifest_sha256", f"corpus {corpus} across seeds")
        verify_consistent_across(corpus_metas, "restrict_set_sha256", f"corpus {corpus} across seeds")
        verify_consistent_across(corpus_metas, "support_sha256", f"corpus {corpus} across seeds")
        corpus_caches = {seed: caches[(corpus, seed)] for seed in args.seeds}
        verify_gt_consistent_across_seeds(corpus_caches, corpus)

    print("lineage OK: preregistered seed/corpus set; schema-valid + support-hash-verified "
          "caches; correct corpus_id per cache; priors/decode-protocol match the lock; "
          "DEV-seed checkpoint matches; checkpoint/config consistent within each seed; "
          "canonical experimental arm + protocol + source consistent across ALL seeds/corpora; "
          "manifest/restrict-set/support consistent within each corpus")
    return caches, metas, priors_sha256, protocol_sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True,
                     help="dir containing <corpus>_s<seed> subdirs from dump_logits.py")
    ap.add_argument("--locked", required=True)
    ap.add_argument("--priors", required=True)
    ap.add_argument("--seeds", nargs="+", default=list(PREREGISTERED_SEEDS))
    ap.add_argument("--corpora", nargs="+", default=list(PREREGISTERED_CORPORA))
    ap.add_argument(
        "--boundary-tol-frames", type=int, default=BOUNDARY_TOL_FRAMES,
        choices=[BOUNDARY_TOL_FRAMES],
        help="fixed by preregistration; retained as a visible CLI field, not a tunable knob",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # check 0: the seed/corpus SET is not a free parameter -- it IS the
    # preregistered protocol the adoption bar's thresholds were chosen for.
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit(f"--seeds contains duplicates: {args.seeds}")
    if set(args.seeds) != set(PREREGISTERED_SEEDS):
        raise SystemExit(
            f"--seeds={args.seeds} does not exactly match the preregistered protocol "
            f"{list(PREREGISTERED_SEEDS)}. The adoption bar ('same sign in >=2 of 3 seeds') "
            "is only meaningful for this exact set -- running a subset, a superset, or a "
            "different set changes what 'same_sign>=2' and the 4-corpus mean actually mean, "
            "which this script refuses to do silently."
        )
    if len(set(args.corpora)) != len(args.corpora):
        raise SystemExit(f"--corpora contains duplicates: {args.corpora}")
    if set(args.corpora) != set(PREREGISTERED_CORPORA):
        raise SystemExit(
            f"--corpora={args.corpora} does not exactly match the preregistered protocol "
            f"{list(PREREGISTERED_CORPORA)}."
        )

    with open(args.locked) as f:
        locked_doc = json.load(f)
    winning_family = locked_doc.get("winning_family")
    if not winning_family:
        raise SystemExit(
            f"{args.locked} has no winning_family (DEV grid search locked nothing) "
            "-- there is no decoder to score on held-out."
        )
    params = locked_doc["locked_hyperparams"][winning_family]

    dev_seed_check = str(locked_doc.get("dev_cache_meta", {}).get("seed"))
    if dev_seed_check not in args.seeds:
        raise SystemExit(
            f"DEV grid search was tuned on seed {dev_seed_check}, which is not in the "
            f"held-out seed set {args.seeds} -- the DEV-checkpoint-matches-held-out check "
            "(module docstring's check 5) would be vacuous otherwise."
        )

    # ===== PASS 1: verify everything, reveal nothing =====
    # Several of the checks inside (verify_corpus_id, verify_consistent_
    # across, verify_gt_consistent_across_seeds, validate_cache_meta_schema)
    # raise plain ValueError rather than SystemExit -- catch and re-raise
    # uniformly so every lineage failure exits the same clean way (a
    # message, not a raw traceback), regardless of which specific check
    # caught it.
    try:
        caches, metas, priors_sha256, protocol_sha256 = load_and_verify_lineage(args, locked_doc)
    except ValueError as e:
        raise SystemExit(f"LINEAGE CHECK FAILED: {e}")

    with np.load(args.priors) as priors:
        log_trans_frame = priors["log_trans_frame"].copy()
        log_trans_segment = priors["log_trans_segment"].copy()
        dur_log_pmf = priors["dur_log_pmf"].copy()
        max_dur = int(priors["max_dur_frames"])
    decode_fn = make_decoder(winning_family, params, log_trans_frame, log_trans_segment, dur_log_pmf, max_dur)
    print(f"Locked decoder under test: {winning_family} params={params}")

    # ===== PASS 2: decode, score, print =====
    methods = ["argmax", winning_family]
    results = {m: {c: {} for c in args.corpora} for m in methods}
    for corpus in args.corpora:
        for seed in args.seeds:
            cache = caches[(corpus, seed)]
            results["argmax"][corpus][seed] = score_cache(cache, None, args.boundary_tol_frames)
            results[winning_family][corpus][seed] = score_cache(cache, decode_fn, args.boundary_tol_frames)
            print(f"scored {corpus}/seed{seed}: "
                  + " | ".join(f"{m}={results[m][corpus][seed]['macro_f1']:.4f}" for m in methods))

    # --- report table ---
    print("\n=== macro-F1 by corpus (3-seed mean) ===")
    header = "method".ljust(14) + "".join(c[:14].ljust(16) for c in args.corpora) + "mean".ljust(10)
    print(header)
    corpus_means = {m: {} for m in methods}
    seed_means = {m: {} for m in methods}
    for m in methods:
        row = m.ljust(14)
        per_corpus_vals = []
        for c in args.corpora:
            vals = [results[m][c][s]["macro_f1"] for s in args.seeds]
            mean = float(np.mean(vals))
            corpus_means[m][c] = mean
            per_corpus_vals.append(mean)
            row += f"{mean:.4f}".ljust(16)
        overall = float(np.mean(per_corpus_vals))
        row += f"{overall:.4f}"
        print(row)
        for s in args.seeds:
            svals = [results[m][c][s]["macro_f1"] for c in args.corpora]
            seed_means[m][s] = float(np.mean(svals))

    # --- adoption bar (single decoder, single verdict) ---
    print("\n=== adoption bar check (mean>=+0.005, same-sign>=2/3 seeds, no corpus<-0.002) ===")
    seed_deltas = {s: seed_means[winning_family][s] - seed_means["argmax"][s] for s in args.seeds}
    mean_delta = float(np.mean(list(seed_deltas.values())))
    same_sign = sum(1 for d in seed_deltas.values() if np.sign(d) == np.sign(mean_delta) and d != 0)
    corpus_deltas = {c: corpus_means[winning_family][c] - corpus_means["argmax"][c] for c in args.corpora}
    worst_corpus = min(corpus_deltas.values())
    passed = (mean_delta >= 0.005) and (same_sign >= 2) and (worst_corpus >= -0.002)
    verdict = {
        "decoder": winning_family, "params": params,
        "mean_delta": mean_delta, "seed_deltas": seed_deltas,
        "same_sign_count": same_sign, "corpus_deltas": corpus_deltas,
        "worst_corpus_delta": worst_corpus, "adopted": passed,
    }
    print(f"{winning_family}: mean_delta={mean_delta:+.4f} same_sign={same_sign}/{len(args.seeds)} "
          f"worst_corpus_delta={worst_corpus:+.4f} -> {'ADOPTED' if passed else 'REJECTED'}")
    print(f"    per-seed delta: {seed_deltas}")
    print(f"    per-corpus delta: {corpus_deltas}")
    print("\nNOTE: this is the 25Hz diagnostic protocol. Adoption here is NOT yet "
          "a claim about the unified (100Hz) headline protocol -- confirm there "
          "before reporting any macro-F1 change against the 0.6026-family numbers.")

    out = {
        "winning_family": winning_family,
        "locked_hyperparams": params,
        "lineage": {
            "priors_sha256": priors_sha256,
            "priors_meta": locked_doc["priors_meta"],
            "decode_protocol_sha256": protocol_sha256,
            "selection_protocol": locked_doc["selection_protocol"],
            "heldout_protocol": PREREGISTERED_INPUTS,
            "dev_seed": str(locked_doc["dev_cache_meta"].get("seed")),
            "dev_checkpoint_sha256": locked_doc["dev_cache_meta"].get("checkpoint_sha256"),
            "canonical_config_sha256": locked_doc["dev_cache_meta"].get("canonical_config_sha256"),
            # per-cache fingerprints, so this report can be verified against
            # the exact artifacts that produced it even after the raw cache
            # directories are cleaned up.
            "cache_fingerprints": {
                f"{corpus}_s{seed}": {
                    "checkpoint_sha256": metas[(corpus, seed)]["checkpoint_sha256"],
                    "config_sha256": metas[(corpus, seed)]["config_sha256"],
                    "manifest_sha256": metas[(corpus, seed)]["manifest_sha256"],
                    "support_sha256": metas[(corpus, seed)]["support_sha256"],
                    "content_sha256": metas[(corpus, seed)]["content_sha256"],
                    "protocol_sha256": metas[(corpus, seed)]["protocol_sha256"],
                    "restrict_set_sha256": metas[(corpus, seed)]["restrict_set_sha256"],
                    "source": metas[(corpus, seed)]["source"],
                    "frame_rate_hz": metas[(corpus, seed)]["frame_rate_hz"],
                    "corpus_id": metas[(corpus, seed)]["corpus_id"],
                }
                for corpus in args.corpora for seed in args.seeds
            },
        },
        "raw_results": {m: {c: {s: {kk: vv for kk, vv in r.items() if kk != "confusion"}
                                 for s, r in sd.items()} for c, sd in cd.items()}
                        for m, cd in results.items()},
        "corpus_means_macro_f1": corpus_means,
        "seed_means_macro_f1": seed_means,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
