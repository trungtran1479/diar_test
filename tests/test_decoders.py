"""Regression tests for the duration-aware decoding ablation
(src/utils/decoders.py, src/utils/decode_scoring.py, scripts/dump_logits.py's
position-keyed assembly). Each test here freezes a real bug caught in code
review before any of this ran on real data -- see zipcount-wavlm-offline-
progress memory for the full history. Run:

  PYTHONPATH=. python -m pytest tests/test_decoders.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from src.utils.decoders import (  # noqa: E402
    NEG_INF,
    _safe_log,
    contiguous_valid_spans,
    decode_respecting_gaps,
    hysteresis_min_duration,
    semimarkov_decode,
    viterbi_decode,
)
from src.utils.decode_scoring import (  # noqa: E402
    cache_content_fingerprint,
    decode_protocol_sha256,
    load_cache,
    recording_content_fingerprint,
    score_cache,
    support_fingerprint,
    validate_cache_meta_schema,
    verify_consistent_across,
    verify_corpus_id,
    verify_gt_consistent_across_seeds,
)
from src.utils.duration_decoder_protocol import (  # noqa: E402
    validate_meta_against_protocol,
    validate_priors_provenance,
)
from decode_grid_search import FAMILY_ORDER, main as grid_main, select_winner  # noqa: E402
from decode_score_heldout import PREREGISTERED_CORPORA, PREREGISTERED_SEEDS, main as heldout_main  # noqa: E402
from dump_logits import (  # noqa: E402
    _PREDICTION_PROTOCOL_FILES,
    assemble_ordered,
    canonical_config_sha256,
    main as dump_main,
)


def _write_fake_cache_dir(out_dir, corpus_id, seed, n_recs=2,
                           checkpoint_sha="ck", config_sha="cfg", canon_sha="canon",
                           manifest_sha="man", protocol_sha="proto", restrict_sha="restrict",
                           source="nemo", frame_rate_hz=25.0,
                           tamper_npz_after_meta=False, tamper_probs_after_meta=False):
    """Build a real, on-disk, schema-compliant cache directory (matching
    exactly what scripts/dump_logits.py writes) for CLI/load_cache-level
    tests, so lineage bugs are caught the same way they would be against a
    real cache, not just against an in-memory dict a test hand-assembles."""
    os.makedirs(out_dir, exist_ok=True)
    # Ground-truth/support is corpus-derived and therefore identical across
    # seeds; probabilities remain seed-specific, matching the real cache
    # contract closely enough for end-to-end lineage tests.
    corpus_rng = np.random.RandomState(sum(corpus_id.encode("utf-8")) + 17)
    pred_rng = np.random.RandomState(
        (sum(corpus_id.encode("utf-8")) * 1009 + int(seed)) % (2**31)
    )
    recordings = []
    support_records = {}
    content_record_hashes = {}
    for i in range(n_recs):
        T = corpus_rng.randint(50, 100)
        gt = corpus_rng.randint(0, 4, size=T).astype(np.int64)
        valid = np.ones(T, dtype=bool)
        probs = pred_rng.dirichlet(np.ones(4), size=T).astype(np.float16)
        rec_id = f"rec{i}"
        np.savez_compressed(os.path.join(out_dir, f"{rec_id}.npz"), probs=probs, valid=valid, gt_25hz=gt)
        recordings.append(rec_id)
        support_records[rec_id] = (gt.tobytes(), valid.tobytes())
        content_record_hashes[rec_id] = recording_content_fingerprint(probs, gt, valid)
    meta = {
        "source": source, "corpus_id": corpus_id, "seed": int(seed),
        "manifest": "fake", "manifest_sha256": manifest_sha,
        "config": "fake", "config_sha256": config_sha,
        "canonical_config_sha256": canon_sha,
        "checkpoint": "fake", "checkpoint_sha256": checkpoint_sha,
        "restrict_to_dir": "fake_dir" if restrict_sha else None, "restrict_set_sha256": restrict_sha,
        "support_sha256": support_fingerprint(support_records),
        "content_sha256": cache_content_fingerprint(content_record_hashes),
        "protocol_sha256": protocol_sha,
        "frame_rate_hz": frame_rate_hz,
        "n_recordings": len(recordings), "recordings": recordings,
    }
    import json as _json
    with open(os.path.join(out_dir, "_cache_meta.json"), "w") as f:
        _json.dump(meta, f)
    if tamper_npz_after_meta:
        # modify a .npz AFTER the meta (and its support_sha256) was written
        p = os.path.join(out_dir, "rec0.npz")
        data = dict(np.load(p))
        data["gt_25hz"] = (data["gt_25hz"] + 1) % 4
        np.savez_compressed(p, **data)
    if tamper_probs_after_meta:
        p = os.path.join(out_dir, "rec0.npz")
        data = dict(np.load(p))
        data["probs"] = np.full_like(data["probs"], 0.25)
        np.savez_compressed(p, **data)
    return meta


def _self_loop_trans(diag=0.9, n=4):
    off = (1.0 - diag) / (n - 1)
    return np.log(np.where(np.eye(n, dtype=bool), diag, off))


def _forbidden_diag_trans(n=4):
    """A segment-transition matrix: uniform off-diagonal, -inf diagonal."""
    lt = np.full((n, n), np.log(1.0 / (n - 1)))
    np.fill_diagonal(lt, NEG_INF)
    return lt


def test_contiguous_valid_spans_basic():
    valid = np.array([True, True, False, False, True, True, True, False, True])
    assert contiguous_valid_spans(valid) == [(0, 2), (4, 7), (8, 9)]


def test_contiguous_valid_spans_all_valid():
    valid = np.ones(5, dtype=bool)
    assert contiguous_valid_spans(valid) == [(0, 5)]


def test_contiguous_valid_spans_all_invalid():
    valid = np.zeros(5, dtype=bool)
    assert contiguous_valid_spans(valid) == []


def test_viterbi_gap_independence():
    """A change to span1's emission must NOT change span2's decoded output
    when routed through decode_respecting_gaps -- regression for the bug
    where every decoder's DP state trivially crossed a masked-out gap even
    though the gap's own emission was zeroed."""
    rng = np.random.RandomState(0)
    log_trans = _self_loop_trans(0.9)

    def mk_probs(seed, n=40):
        return np.random.RandomState(seed).dirichlet(np.ones(4) * 0.3, size=n)

    T = 100
    valid = np.ones(T, dtype=bool)
    valid[40:60] = False

    probs_a = np.zeros((T, 4))
    probs_a[:40] = mk_probs(1)
    probs_a[60:] = mk_probs(2)
    probs_b = probs_a.copy()
    probs_b[:40] = mk_probs(3)  # change ONLY the first span

    def fn(p, v):
        return viterbi_decode(_safe_log(p), v, log_trans, trans_weight=2.0)

    dec_a = decode_respecting_gaps(fn, probs_a, valid)
    dec_b = decode_respecting_gaps(fn, probs_b, valid)
    assert np.array_equal(dec_a[60:], dec_b[60:])


def test_semimarkov_forbids_same_class_adjacency_and_respects_max_dur():
    """With max_dur=2 and near-certain class-0 emission throughout, no
    DECODED (visible) run may exceed 2 frames -- regression for the bug
    where an unconstrained self-transition let the DP carve one long true
    run into several same-class hidden segments to re-farm the duration
    prior's peak, defeating the whole point of an explicit duration prior."""
    T = 12
    probs = np.zeros((T, 4))
    probs[:, 0] = 0.9
    probs[:, 1:] = 0.1 / 3
    valid = np.ones(T, dtype=bool)
    log_trans_segment = _forbidden_diag_trans()
    dur_pmf = np.ones((4, 2))
    dur_pmf /= dur_pmf.sum(-1, keepdims=True)
    dur_log_pmf = _safe_log(dur_pmf)

    decoded = semimarkov_decode(_safe_log(probs), valid, log_trans_segment, dur_log_pmf,
                                 max_dur=2, trans_weight=1.0, duration_weight=1.0)
    run_lengths = np.diff(np.flatnonzero(np.r_[1, np.diff(decoded) != 0, 1]))
    assert run_lengths.max() <= 2


def test_semimarkov_diagonal_defensively_enforced():
    """Even if a caller passes a transition matrix with a non-forbidden
    diagonal, semimarkov_decode must still forbid same-class adjacency
    itself (defensive `np.fill_diagonal(lt, NEG_INF)` inside the function),
    not merely rely on the caller having fit it correctly."""
    T = 10
    probs = np.zeros((T, 4))
    probs[:, 0] = 0.9
    probs[:, 1:] = 0.1 / 3
    valid = np.ones(T, dtype=bool)
    # deliberately NOT forbidding the diagonal here
    log_trans_frame_like = _self_loop_trans(0.9)
    dur_pmf = np.ones((4, 2))
    dur_pmf /= dur_pmf.sum(-1, keepdims=True)
    dur_log_pmf = _safe_log(dur_pmf)

    decoded = semimarkov_decode(_safe_log(probs), valid, log_trans_frame_like, dur_log_pmf,
                                 max_dur=2, trans_weight=1.0, duration_weight=1.0)
    run_lengths = np.diff(np.flatnonzero(np.r_[1, np.diff(decoded) != 0, 1]))
    assert run_lengths.max() <= 2


def test_hysteresis_merge_uses_short_runs_own_support():
    """The short run's OWN frames must decide which neighbour it merges
    into (compare probs[short_run, left_class] vs probs[short_run,
    right_class]), NOT each neighbour's confidence over its own frames --
    regression for a bug where the merge target was picked backwards.
    Constructed so the two criteria disagree: the left neighbour is very
    confident in itself, but the short run's own frames support the RIGHT
    class far more (0.40 vs 0.05) -- correct behaviour merges right."""
    probs = np.zeros((20, 4))
    probs[0:8, 0] = 0.9
    probs[0:8, 1:] = 0.0333
    probs[11:20, 1] = 0.9
    probs[11:20, [0, 2, 3]] = 0.0333
    # short run's own frames: low support for left_class(0), high for right_class(1)
    probs[8:11, 0] = 0.05
    probs[8:11, 1] = 0.40
    probs[8:11, 2] = 0.50
    probs[8:11, 3] = 0.05
    probs = probs / probs.sum(-1, keepdims=True)
    valid = np.ones(20, dtype=bool)

    decoded = hysteresis_min_duration(probs, valid, switch_margin=0.15, min_duration_frames=5)
    assert decoded[9] == 1  # merged toward the class its OWN frames support


def test_scorer_does_not_manufacture_boundary_across_gap():
    """score_cache must treat two gap-separated valid spans as independent
    sequences for edit/boundary metrics, not concatenate them into one
    sequence with a fake transition exactly at the join."""
    T = 200
    gt = np.zeros(T, dtype=np.int64)
    gt[150:] = 3
    probs = np.zeros((T, 4), dtype=np.float32)
    probs[:100, 0] = 1.0
    probs[100:150, 0] = 1.0  # invalid region, content irrelevant
    probs[150:, 3] = 1.0
    valid = np.ones(T, dtype=bool)
    valid[100:150] = False

    cache = {"rec0": {"probs": probs, "valid": valid, "gt": gt}}
    r = score_cache(cache, decode_fn=None, boundary_tol_frames=3)
    assert r["n_spans"] == 2
    assert r["mean_boundary_f1"] == pytest.approx(1.0)
    assert r["mean_edit_score"] == pytest.approx(100.0)


def test_select_winner_ties_prefer_simpler_family():
    """Calls the ACTUAL production select_winner() (scripts/
    decode_grid_search.py), not a reimplementation of its logic -- an
    earlier test only asserted that two Python floats round to the same
    value, which would keep passing even if select_winner()'s tie-break
    were changed or deleted entirely. Constructs a genuine tie (values that
    round identically to 4dp but differ by far more than the old, too-
    strict `abs(diff) < 1e-9` criterion would have tolerated) and asserts
    the simpler family (hysteresis) wins over viterbi/semimarkov."""
    candidates = {
        "semimarkov": (0.600049, 90.0, {"trans_weight": 1.0}, {}),
        "viterbi": (0.600030, 88.0, {"trans_weight": 2.0}, {}),
        "hysteresis": (0.600001, 85.0, {"switch_margin": 0.1}, {}),
    }
    assert round(candidates["semimarkov"][0], 4) == round(candidates["hysteresis"][0], 4)
    assert abs(candidates["semimarkov"][0] - candidates["hysteresis"][0]) > 1e-9

    winner, tied = select_winner(candidates)
    assert winner == "hysteresis"
    assert set(tied) == {"hysteresis", "viterbi", "semimarkov"}


def test_select_winner_no_candidates():
    winner, tied = select_winner({})
    assert winner is None
    assert tied == []


def test_select_winner_no_tie_picks_highest_f1():
    candidates = {
        "hysteresis": (0.50, 80.0, {}, {}),
        "viterbi": (0.90, 80.0, {}, {}),
        "semimarkov": (0.70, 80.0, {}, {}),
    }
    winner, tied = select_winner(candidates)
    assert winner == "viterbi"
    assert tied == ["viterbi"]


def test_assemble_ordered_is_insertion_order_independent():
    """The core of the position-keyed assembly fix: reading back by a
    caller-supplied, explicitly sorted index list must reconstruct the
    correct order regardless of what order items were INSERTED into the
    slot dict -- regression for the bug where gap placeholders were
    appended to the output list immediately while valid windows sat
    buffered until a batch flush, silently reordering the timeline."""
    def mk(val, n=3):
        return (np.full((n, 4), val, dtype=np.float32),
                np.full(n, val, dtype=np.int64),
                np.ones(n, dtype=bool))

    # inserted in a SCRAMBLED order: 2, 0, 1
    slot = {}
    slot[2] = mk(20)
    slot[0] = mk(0)
    slot[1] = mk(10)

    probs, gt, valid = assemble_ordered(slot, [0, 1, 2])
    assert list(gt[::3]) == [0, 10, 20]  # ascending index order, not insertion order
    assert probs.shape == (9, 4)
    assert valid.all()


def test_validate_cache_meta_schema_rejects_all_missing_key_as_consistent():
    """Regression for the bug where `verify_consistent_across` used
    `m.get(key)` and three caches all missing the SAME key looked
    "consistent" (their values were all None). Schema validation must
    reject an incomplete meta BEFORE it ever reaches a consistency check,
    so three such caches can never even load."""
    incomplete_meta = {
        "source": "nemo", "seed": 1234, "manifest_sha256": "m", "config_sha256": "c",
        "canonical_config_sha256": "cc", "checkpoint_sha256": "ck",
        "support_sha256": "s", "protocol_sha256": "p", "frame_rate_hz": 25.0,
        "recordings": ["rec0"],
        # restrict_set_sha256 deliberately OMITTED -- required for source="nemo"
    }
    with pytest.raises(ValueError, match="restrict_set_sha256"):
        validate_cache_meta_schema(incomplete_meta, "/fake/dir")


def test_verify_consistent_across_rejects_shared_none():
    """Even if validate_cache_meta_schema is somehow bypassed,
    verify_consistent_across itself must not treat "everyone has the same
    missing/None value" as consistent."""
    metas = {
        "a": {"restrict_set_sha256": None},
        "b": {"restrict_set_sha256": None},
        "c": {"restrict_set_sha256": None},
    }
    with pytest.raises(ValueError, match="missing/None"):
        verify_consistent_across(metas, "restrict_set_sha256", "test group")


def test_verify_consistent_across_rejects_real_mismatch():
    metas = {"a": {"k": "x"}, "b": {"k": "x"}, "c": {"k": "y"}}
    with pytest.raises(ValueError, match="not consistent"):
        verify_consistent_across(metas, "k", "test group")


def test_verify_consistent_across_passes_when_truly_consistent():
    metas = {"a": {"k": "x"}, "b": {"k": "x"}}
    verify_consistent_across(metas, "k", "test group")  # must not raise


def test_verify_gt_consistent_across_seeds_catches_valid_mask_mismatch():
    """Regression: an earlier version compared `gt` only. Two caches with
    IDENTICAL gt values but a DIFFERENT valid mask are scoring different
    SUPPORT and must be rejected too."""
    T = 50
    gt = np.zeros(T, dtype=np.int64)
    valid_a = np.ones(T, dtype=bool)
    valid_b = np.ones(T, dtype=bool)
    valid_b[10:20] = False  # same gt, different support

    caches = {
        "1234": {"rec0": {"gt": gt, "valid": valid_a}},
        "2345": {"rec0": {"gt": gt, "valid": valid_b}},
    }
    with pytest.raises(ValueError, match="valid-frame mask differs"):
        verify_gt_consistent_across_seeds(caches, "fake_corpus")


def test_verify_gt_consistent_across_seeds_passes_when_identical():
    T = 50
    gt = np.arange(T) % 4
    valid = np.ones(T, dtype=bool)
    caches = {
        "1234": {"rec0": {"gt": gt.copy(), "valid": valid.copy()}},
        "2345": {"rec0": {"gt": gt.copy(), "valid": valid.copy()}},
    }
    verify_gt_consistent_across_seeds(caches, "fake_corpus")  # must not raise


def test_canonical_config_sha256_ignores_seed_and_log_dir_only():
    base = {"model": {"encoder": {"type": "wavlm"}}, "training": {"seed": 1234, "log_dir": "logs/a", "lr": 0.001}}
    same_arm_diff_seed = {"model": {"encoder": {"type": "wavlm"}},
                           "training": {"seed": 9999, "log_dir": "logs/b", "lr": 0.001}}
    different_arm = {"model": {"encoder": {"type": "wavlm"}},
                      "training": {"seed": 1234, "log_dir": "logs/a", "lr": 0.999}}

    assert canonical_config_sha256(base) == canonical_config_sha256(same_arm_diff_seed)
    assert canonical_config_sha256(base) != canonical_config_sha256(different_arm)


def test_verify_corpus_id_rejects_mismatch():
    """The exact counterexample from review: a cache whose OWN metadata
    claims a different corpus than the directory it was loaded as."""
    meta = {"corpus_id": "some_other_corpus"}
    with pytest.raises(ValueError, match="corpus_id"):
        verify_corpus_id(meta, "ami_test", "/fake/ami_test_s1234")


def test_verify_corpus_id_passes_when_matching():
    meta = {"corpus_id": "ami_test"}
    verify_corpus_id(meta, "ami_test", "/fake/ami_test_s1234")  # must not raise


def test_load_cache_recomputes_support_sha256_catches_tampering(tmp_path):
    """Regression: an earlier version trusted the support_sha256 STRING in
    _cache_meta.json without ever recomputing it from the actual .npz
    contents, so a stale meta or a modified .npz after the fact would pass
    silently. load_cache must recompute and compare."""
    out_dir = str(tmp_path / "ami_test_s1234")
    _write_fake_cache_dir(out_dir, corpus_id="ami_test", seed="1234", tamper_npz_after_meta=True)
    with pytest.raises(ValueError, match="support_sha256"):
        load_cache(out_dir)


def test_load_cache_recomputes_content_sha256_catches_probability_tampering(tmp_path):
    """Changing predictions alone must be detected; support_sha256 only
    covers gt/valid and cannot provide this guarantee."""
    out_dir = str(tmp_path / "ami_test_s1234")
    _write_fake_cache_dir(
        out_dir, corpus_id="ami_test", seed="1234", tamper_probs_after_meta=True
    )
    with pytest.raises(ValueError, match="content_sha256"):
        load_cache(out_dir)


def test_load_cache_passes_for_untampered_real_cache(tmp_path):
    out_dir = str(tmp_path / "ami_test_s1234")
    _write_fake_cache_dir(out_dir, corpus_id="ami_test", seed="1234")
    cache, meta = load_cache(out_dir)  # must not raise
    assert meta["corpus_id"] == "ami_test"
    assert len(cache) == 2


def test_preregistered_meta_rejects_consistently_wrong_manifest():
    """A self-reported corpus_id is not evidence that the manifest is the
    sealed corpus input; all three seeds can repeat the same copy/paste
    error, so every cache must also match the external protocol anchor."""
    expected = {
        "corpus_id": "ami_test",
        "manifest_sha256": "sealed-ami-manifest",
        "restrict_set_sha256": "sealed-common-window-set",
    }
    wrong_but_self_consistent = {
        "corpus_id": "ami_test",
        "manifest_sha256": "same-wrong-manifest-in-all-three-seeds",
        "restrict_set_sha256": "sealed-common-window-set",
    }
    with pytest.raises(ValueError, match="manifest_sha256"):
        validate_meta_against_protocol(
            wrong_but_self_consistent, expected, "ami_test cache"
        )


def test_prediction_protocol_covers_wavlm_backbone_and_tcn_head():
    assert "src/models/wavlm_wrapper.py" in _PREDICTION_PROTOCOL_FILES
    assert "src/models/heads.py" in _PREDICTION_PROTOCOL_FILES


def test_dump_rejects_wrong_dev_manifest_before_touching_output(tmp_path, monkeypatch):
    wrong_manifest = tmp_path / "wrong_dev.json"
    wrong_manifest.write_text("{}\n")
    out_dir = tmp_path / "ami_dev_s1234"
    monkeypatch.setattr(sys, "argv", [
        "dump_logits.py",
        "--source", "dataset",
        "--manifest", str(wrong_manifest),
        "--config", "/does/not/need/to/exist.yaml",
        "--checkpoint", "/does/not/need/to/exist.pt",
        "--seed", "1234",
        "--corpus-id", "ami_dev",
        "--out-dir", str(out_dir),
    ])
    with pytest.raises(SystemExit, match="INPUT PROTOCOL MISMATCH"):
        dump_main()
    assert not out_dir.exists()


def test_grid_search_rejects_non_dev_cache_before_opening_priors(tmp_path, monkeypatch):
    wrong_cache = tmp_path / "ami_test_s1234"
    _write_fake_cache_dir(
        str(wrong_cache), corpus_id="ami_test", seed="1234", source="nemo"
    )
    out_path = tmp_path / "must_not_exist.json"
    monkeypatch.setattr(sys, "argv", [
        "decode_grid_search.py",
        "--dev-cache", str(wrong_cache),
        "--priors", "/does/not/need/to/exist.npz",
        "--out", str(out_path),
    ])
    with pytest.raises(ValueError, match="DEV cache"):
        grid_main()
    assert not out_path.exists()


def test_priors_provenance_rejects_wrong_training_manifest(tmp_path):
    import json as _json
    from src.utils.duration_decoder_protocol import sha256_file

    priors = tmp_path / "priors.npz"
    np.savez(priors, frame_rate_hz=25.0, max_dur_frames=10)
    sha = sha256_file(str(priors))
    expected = {
        "priors_npz_sha256": sha,
        "manifest_sha256": "sealed-training-manifest",
        "frame_rate_hz": 25.0,
        "max_dur_frames": 10,
    }
    sidecar = dict(expected)
    sidecar["manifest_sha256"] = "dev-or-heldout-manifest-by-mistake"
    (tmp_path / "priors.npz.meta.json").write_text(_json.dumps(sidecar))
    with pytest.raises(ValueError, match="manifest_sha256"):
        validate_priors_provenance(str(priors), expected)


def test_heldout_main_rejects_corpus_id_mismatch_end_to_end(tmp_path, monkeypatch):
    """End-to-end reproduction of the review's counterexample: three seeds
    of a cache directory named 'claimed_ami_s<seed>' whose OWN metadata
    says corpus_id='not_actually_ami'. Even though the three seeds are
    perfectly self-consistent with EACH OTHER, decode_score_heldout.py must
    still refuse to score them as if they were really the claimed corpus.
    Uses a single fake corpus/seed pair standing in for the full
    preregistered grid (the corpus_id check fires before the full-grid
    requirement would even matter) -- reached via directly patching
    PREREGISTERED_CORPORA/PREREGISTERED_SEEDS for this test only, since the
    real preregistered set is a separate, already-tested invariant (see
    test_heldout_main_rejects_nonpreregistered_seeds below)."""
    import decode_score_heldout as heldout_mod
    import json as _json

    cache_root = tmp_path / "cache_root"
    _write_fake_cache_dir(str(cache_root / "claimed_ami_s1234"), corpus_id="not_actually_ami", seed="1234")

    priors_path = tmp_path / "priors.npz"
    np.savez(str(priors_path),
             log_trans_frame=np.zeros((4, 4)), log_trans_segment=np.zeros((4, 4)),
             dur_log_pmf=np.zeros((4, 10)), max_dur_frames=10, frame_rate_hz=25.0)
    from src.utils.decode_scoring import sha_of_file as _sha_of_file
    priors_sha = _sha_of_file(str(priors_path) if str(priors_path).endswith(".npz") else str(priors_path) + ".npz")
    priors_meta = {
        "priors_npz_sha256": priors_sha,
        "manifest_sha256": "train-man",
        "source_prefixes": ["real_train"],
        "sample_every": 1,
        "laplace_transition": 1.0,
        "laplace_duration_total_per_class": 1.0,
        "n_windows_missing_label_file": 0,
        "frame_rate_hz": 25.0,
        "max_dur_frames": 10,
    }
    (tmp_path / "priors.npz.meta.json").write_text(_json.dumps(priors_meta))

    dev_protocol = {
        "source": "nemo", "corpus_id": "claimed_ami", "seed": 1234,
        "manifest_sha256": "man", "frame_rate_hz": 25.0,
    }

    locked = {
        "winning_family": "viterbi",
        "locked_hyperparams": {"viterbi": {"trans_weight": 1.0}},
        "priors_sha256": priors_sha, "decode_protocol_sha256": decode_protocol_sha256(),
        "priors_meta": priors_meta,
        "selection_protocol": {"dev": dev_protocol, "boundary_tol_frames": 3},
        # a FULL, schema-valid DEV meta (matching _write_fake_cache_dir's
        # defaults for checkpoint/canonical-config/protocol) so every check
        # EXCEPT corpus_id passes, isolating exactly the check this test
        # targets.
        "dev_cache_meta": {
            "source": "nemo", "corpus_id": "claimed_ami", "seed": 1234,
            "manifest_sha256": "man", "config_sha256": "cfg",
            "canonical_config_sha256": "canon", "checkpoint_sha256": "ck",
            "support_sha256": "dev_support", "content_sha256": "dev_content",
            "protocol_sha256": "proto",
            "frame_rate_hz": 25.0, "recordings": ["dev_rec0"],
            "restrict_set_sha256": "dev_restrict",
        },
    }
    locked_path = tmp_path / "locked.json"
    locked_path.write_text(_json.dumps(locked))

    monkeypatch.setattr(heldout_mod, "PREREGISTERED_SEEDS", ("1234",))
    monkeypatch.setattr(heldout_mod, "PREREGISTERED_CORPORA", ("claimed_ami",))
    monkeypatch.setattr(heldout_mod, "PREREGISTERED_PRIORS", priors_meta)
    monkeypatch.setattr(heldout_mod, "PREREGISTERED_DEV", dev_protocol)
    monkeypatch.setattr(heldout_mod, "PREREGISTERED_INPUTS", {
        "claimed_ami": {
            "source": "nemo", "manifest_sha256": "man",
            "restrict_set_sha256": "restrict", "frame_rate_hz": 25.0,
        }
    })
    monkeypatch.setattr(sys, "argv", [
        "decode_score_heldout.py",
        "--cache-root", str(cache_root),
        "--locked", str(locked_path),
        "--priors", str(priors_path),
        "--seeds", "1234",
        "--corpora", "claimed_ami",
        "--out", str(tmp_path / "report.json"),
    ])
    with pytest.raises(SystemExit, match="corpus_id"):
        heldout_mod.main()


def test_full_preregistered_heldout_fixture_passes_and_binds_content(tmp_path, monkeypatch):
    import decode_score_heldout as heldout_mod
    import json as _json
    from src.utils.decode_scoring import sha_of_file as _sha_of_file

    cache_root = tmp_path / "cache_root"
    expected_inputs = {}
    for corpus in PREREGISTERED_CORPORA:
        expected_inputs[corpus] = {
            "source": "nemo",
            "manifest_sha256": f"manifest-{corpus}",
            "restrict_set_sha256": f"restrict-{corpus}",
            "frame_rate_hz": 25.0,
        }
        for seed in PREREGISTERED_SEEDS:
            _write_fake_cache_dir(
                str(cache_root / f"{corpus}_s{seed}"),
                corpus_id=corpus,
                seed=seed,
                checkpoint_sha=f"ck-{seed}",
                config_sha=f"cfg-{seed}",
                canon_sha="same-arm",
                manifest_sha=f"manifest-{corpus}",
                protocol_sha="same-prediction-protocol",
                restrict_sha=f"restrict-{corpus}",
            )

    priors_path = tmp_path / "priors.npz"
    frame_trans = np.full((4, 4), 0.1 / 3.0)
    np.fill_diagonal(frame_trans, 0.9)
    segment_trans = np.full((4, 4), 1.0 / 3.0)
    np.fill_diagonal(segment_trans, 0.0)
    np.savez(
        priors_path,
        log_trans_frame=np.log(frame_trans),
        log_trans_segment=np.log(np.clip(segment_trans, 1e-300, 1.0)),
        dur_log_pmf=np.log(np.full((4, 5), 0.2)),
        max_dur_frames=5,
        frame_rate_hz=25.0,
    )
    priors_sha = _sha_of_file(str(priors_path))
    priors_meta = {
        "priors_npz_sha256": priors_sha,
        "manifest_sha256": "sealed-training-manifest",
        "source_prefixes": ["real_train"],
        "sample_every": 1,
        "laplace_transition": 1.0,
        "laplace_duration_total_per_class": 1.0,
        "n_windows_missing_label_file": 0,
        "frame_rate_hz": 25.0,
        "max_dur_frames": 5,
    }
    (tmp_path / "priors.npz.meta.json").write_text(_json.dumps(priors_meta))

    dev_protocol = {
        "source": "dataset", "corpus_id": "ami_dev", "seed": 1234,
        "manifest_sha256": "manifest-ami-dev", "frame_rate_hz": 25.0,
    }
    dev_meta = {
        **dev_protocol,
        "config_sha256": "cfg-1234",
        "canonical_config_sha256": "same-arm",
        "checkpoint_sha256": "ck-1234",
        "support_sha256": "dev-support",
        "content_sha256": "dev-content",
        "protocol_sha256": "same-prediction-protocol",
        "recordings": ["dev-rec"],
    }
    locked = {
        "winning_family": "viterbi",
        "locked_hyperparams": {"viterbi": {"trans_weight": 0.1}},
        "priors_sha256": priors_sha,
        "priors_meta": priors_meta,
        "decode_protocol_sha256": decode_protocol_sha256(),
        "selection_protocol": {"dev": dev_protocol, "boundary_tol_frames": 3},
        "dev_cache_meta": dev_meta,
    }
    locked_path = tmp_path / "locked.json"
    locked_path.write_text(_json.dumps(locked))
    report_path = tmp_path / "report.json"

    monkeypatch.setattr(heldout_mod, "PREREGISTERED_INPUTS", expected_inputs)
    monkeypatch.setattr(heldout_mod, "PREREGISTERED_PRIORS", priors_meta)
    monkeypatch.setattr(heldout_mod, "PREREGISTERED_DEV", dev_protocol)
    monkeypatch.setattr(sys, "argv", [
        "decode_score_heldout.py",
        "--cache-root", str(cache_root),
        "--locked", str(locked_path),
        "--priors", str(priors_path),
        "--seeds", *PREREGISTERED_SEEDS,
        "--corpora", *PREREGISTERED_CORPORA,
        "--out", str(report_path),
    ])
    heldout_mod.main()
    report = _json.loads(report_path.read_text())
    fingerprints = report["lineage"]["cache_fingerprints"]
    assert len(fingerprints) == 12
    assert all(item["content_sha256"] for item in fingerprints.values())


def test_heldout_main_rejects_nonpreregistered_seeds(tmp_path, monkeypatch):
    """--seeds subsetting must be rejected BEFORE any file is opened --
    regression for a version where the adoption bar's 'same_sign>=2' stayed
    hardcoded regardless of how many seeds were actually passed, so running
    e.g. only 2 (both positive) seeds could still report '2/3' and pass."""
    monkeypatch.setattr(sys, "argv", [
        "decode_score_heldout.py",
        "--cache-root", "/nonexistent",
        "--locked", "/nonexistent/locked.json",
        "--priors", "/nonexistent/priors.npz",
        "--seeds", "1234", "2345",  # only 2 of the preregistered 3
        "--corpora", *PREREGISTERED_CORPORA,
        "--out", "/nonexistent/report.json",
    ])
    with pytest.raises(SystemExit, match="preregistered"):
        heldout_main()


def test_heldout_main_rejects_duplicate_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "decode_score_heldout.py",
        "--cache-root", "/nonexistent",
        "--locked", "/nonexistent/locked.json",
        "--priors", "/nonexistent/priors.npz",
        "--seeds", "1234", "1234", "2345",
        "--corpora", *PREREGISTERED_CORPORA,
        "--out", "/nonexistent/report.json",
    ])
    with pytest.raises(SystemExit, match="duplicates"):
        heldout_main()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
