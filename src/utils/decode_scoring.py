"""Score a decoded (or raw-argmax) per-recording prediction set against the
cached ground truth from scripts/dump_logits.py, reporting the full metric
suite the duration-aware decoding ablation needs together: macro-F1, OSD,
segmental edit score, fragmentation rate, boundary F1.

Scoring happens at the model's native 25Hz rate throughout (the cache's own
rate) -- NOT upsampled to 100Hz like scripts/eval_zipcount_nemo.py's
headline-comparison protocol. This is a deliberate, self-consistent choice
for a same-checkpoint decoder-vs-decoder comparison (argmax vs hysteresis vs
Viterbi vs semi-Markov all scored the same way), not an attempt to reproduce
the headline macro-F1 numbers exactly -- do not compare these macro-F1
values directly against the 0.6026-family numbers reported elsewhere, and
do not treat a pass here as a unified-protocol adoption claim on its own;
that needs a separate 100Hz confirmation pass (see zipcount-wavlm-offline-
progress memory).

Gap handling: `gt[valid]`/`pred[valid]` alone is NOT enough for the
SEQUENCE-level metrics (edit score, fragmentation, boundary F1) -- naively
concatenating two separated valid spans manufactures a fake boundary/
adjacency exactly at the join, the same class of bug the decoders
themselves had to be fixed for (see src/utils/decoders.py). This module
scores each contiguous valid span independently for those three metrics and
averages per-span (matching the project's existing per-RECORDING averaging
convention in scripts/eval_sequence_metrics.py, just one level finer).
macro-F1/OSD/accuracy are frame-pooled, order-independent metrics, so they
are unaffected by this and are still computed by simply concatenating all
valid frames globally.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.eval_sortformer_ami import binary_f1, f1_from_conf
from src.utils.decoders import contiguous_valid_spans, decode_respecting_gaps
from src.utils.sequence_metrics import fragmentation_rate, segmental_edit_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from segment_diagnosis import boundary_f1 as _boundary_f1  # noqa: E402

N_CLS = 4
# (span_probs, span_valid_all_true) -> span_decoded -- see
# src.utils.decoders.decode_respecting_gaps for why this must only ever be
# called on a single contiguous, gap-free span.
DecodeFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def sha_of_file(path: str, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def support_fingerprint(support_records: Dict[str, Tuple[bytes, bytes]]) -> str:
    """SHA256 over (recording_id, gt bytes, valid bytes) for every recording,
    sorted by recording_id. The single canonical implementation -- both
    scripts/dump_logits.py (when WRITING a cache's metadata) and
    load_cache below (when READING one back, to catch tampering/staleness)
    call this exact function, so there is no way for the two to compute the
    fingerprint differently and paper over a real mismatch."""
    digest = hashlib.sha256()
    for rec_id in sorted(support_records.keys()):
        gt_bytes, valid_bytes = support_records[rec_id]
        for blob in (rec_id.encode("utf-8"), gt_bytes, valid_bytes):
            digest.update(len(blob).to_bytes(8, "big"))
            digest.update(blob)
    return digest.hexdigest()


def recording_content_fingerprint(
    probs: np.ndarray, gt: np.ndarray, valid: np.ndarray
) -> str:
    """Fingerprint one recording's exact cached tensors, including schema.

    Dtype and shape are included so byte-identical buffers with different
    interpretations cannot collide at the serialization boundary. Arrays
    are converted to contiguous C order only for a stable byte view; values
    are not cast, so changing e.g. float16 probabilities to float32 is a
    content change too.
    """
    digest = hashlib.sha256()
    for name, value in (("probs", probs), ("gt", gt), ("valid", valid)):
        array = np.ascontiguousarray(value)
        header = json.dumps(
            {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload = array.tobytes(order="C")
        for blob in (header, payload):
            digest.update(len(blob).to_bytes(8, "big"))
            digest.update(blob)
    return digest.hexdigest()


def cache_content_fingerprint(recording_hashes: Dict[str, str]) -> str:
    """Fingerprint the complete recording->tensor-content mapping."""
    digest = hashlib.sha256()
    for rec_id in sorted(recording_hashes):
        for blob in (rec_id.encode("utf-8"), recording_hashes[rec_id].encode("ascii")):
            digest.update(len(blob).to_bytes(8, "big"))
            digest.update(blob)
    return digest.hexdigest()


def decode_protocol_sha256() -> str:
    """SHA256 over this module's own source AND src/utils/decoders.py's --
    the shared decoding LOGIC both scripts/decode_grid_search.py (DEV
    tuning) and scripts/decode_score_heldout.py (held-out scoring) import.
    If either file changes between when a decoder is locked and when it is
    scored on held-out, this hash changes too, and decode_score_heldout.py
    refuses to proceed silently on a decision that may no longer mean what
    it meant when it was locked.

    Deliberately does NOT include decode_grid_search.py or
    decode_score_heldout.py themselves: those two drivers are SUPPOSED to
    differ from each other (one tunes, one scores), so hashing them into
    the same value would make this always mismatch even with zero relevant
    changes.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    paths = sorted([
        os.path.join(here, "decode_scoring.py"),
        os.path.join(here, "decoders.py"),
        os.path.join(here, "duration_decoder_protocol.py"),
    ])
    digest = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()


_MISSING = object()  # sentinel distinct from any real (including None) meta value

# required in EVERY cache's _cache_meta.json, regardless of source mode.
# (name, validator) -- validator(value) -> True if acceptable.
_REQUIRED_META_FIELDS = [
    ("source", lambda v: v in ("dataset", "nemo")),
    ("corpus_id", lambda v: isinstance(v, str) and len(v) > 0),
    ("seed", lambda v: isinstance(v, int)),
    ("manifest_sha256", lambda v: isinstance(v, str) and len(v) > 0),
    ("config_sha256", lambda v: isinstance(v, str) and len(v) > 0),
    ("canonical_config_sha256", lambda v: isinstance(v, str) and len(v) > 0),
    ("checkpoint_sha256", lambda v: isinstance(v, str) and len(v) > 0),
    ("support_sha256", lambda v: isinstance(v, str) and len(v) > 0),
    ("content_sha256", lambda v: isinstance(v, str) and len(v) > 0),
    ("protocol_sha256", lambda v: isinstance(v, str) and len(v) > 0),
    # exactly 25.0, not merely positive -- every downstream decoder/prior
    # assumes this rate; a cache at a different rate would silently
    # misalign against duration priors fit at 25Hz (see
    # fit_decoder_priors.py) without ever comparing against a live model.
    ("frame_rate_hz", lambda v: isinstance(v, (int, float)) and float(v) == 25.0),
    ("recordings", lambda v: isinstance(v, list) and len(v) > 0),
]
# nemo-source caches additionally need a real restrict-set fingerprint --
# the whole point of the held-out corpora is scoring on DiariZen's
# common-window set, so a None here means --restrict-to-dir was forgotten.
_REQUIRED_NEMO_META_FIELDS = [
    ("restrict_set_sha256", lambda v: isinstance(v, str) and len(v) > 0),
]


def validate_cache_meta_schema(meta: dict, cache_dir: str) -> None:
    """Raise unless every required field is present, correctly typed, and
    non-empty. Load-bearing for every consistency check downstream: without
    this, three caches that all happen to be MISSING the same key would
    look "consistent" to a naive `set(values) == {None}` comparison (caught
    in review with a fixture reproducing exactly that)."""
    missing_or_invalid = [
        name for name, ok in _REQUIRED_META_FIELDS if not ok(meta.get(name, _MISSING))
    ]
    if meta.get("source") == "nemo":
        missing_or_invalid += [
            name for name, ok in _REQUIRED_NEMO_META_FIELDS if not ok(meta.get(name, _MISSING))
        ]
    if missing_or_invalid:
        raise ValueError(
            f"{cache_dir}: _cache_meta.json is missing or has an invalid value for "
            f"required field(s) {missing_or_invalid} -- refusing to load. A cache "
            "cannot be trusted for lineage checks if its own metadata is incomplete."
        )


def load_cache(cache_dir: str) -> Tuple[Dict[str, dict], dict]:
    """Returns (rec_id -> {"probs", "valid", "gt"}, cache_meta).

    FAILS CLOSED if `_cache_meta.json` (written by scripts/dump_logits.py)
    is missing, has any required field missing/invalid (see
    validate_cache_meta_schema), or its recorded recording list doesn't
    exactly match the directory's actual .npz files -- an earlier version
    silently accepted a metadata-less directory or one with holes in its
    metadata, either of which could let a hand-assembled or stale cache be
    scored with no reliable record of its lineage.
    """
    meta_path = os.path.join(cache_dir, "_cache_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{cache_dir} has no _cache_meta.json -- refusing to load. "
            "Caches must be produced by scripts/dump_logits.py, which always "
            "writes this file; a directory without one cannot be trusted to "
            "come from a known checkpoint/config/manifest."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    validate_cache_meta_schema(meta, cache_dir)

    out = {}
    support_records = {}
    content_record_hashes = {}
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        rec = os.path.splitext(os.path.basename(path))[0]
        with np.load(path) as data:
            required_arrays = {"probs", "gt_25hz", "valid"}
            missing = required_arrays - set(data.files)
            if missing:
                raise ValueError(f"{path}: missing required array(s) {sorted(missing)}")
            raw_probs = np.asarray(data["probs"])
            raw_gt = np.asarray(data["gt_25hz"])
            raw_valid = np.asarray(data["valid"])

        if raw_probs.dtype != np.float16:
            raise ValueError(f"{path}: probs dtype must be float16, got {raw_probs.dtype}")
        if raw_gt.dtype != np.int64:
            raise ValueError(f"{path}: gt_25hz dtype must be int64, got {raw_gt.dtype}")
        if raw_valid.dtype != np.bool_:
            raise ValueError(f"{path}: valid dtype must be bool, got {raw_valid.dtype}")
        if raw_probs.ndim != 2 or raw_probs.shape[1] != N_CLS:
            raise ValueError(
                f"{path}: probs must have shape [T,{N_CLS}], got {raw_probs.shape}"
            )
        n_frames = raw_probs.shape[0]
        if raw_gt.shape != (n_frames,) or raw_valid.shape != (n_frames,):
            raise ValueError(
                f"{path}: length mismatch: probs={raw_probs.shape}, "
                f"gt_25hz={raw_gt.shape}, valid={raw_valid.shape}"
            )
        if not np.isfinite(raw_probs).all():
            raise ValueError(f"{path}: probs contains NaN/Inf")
        if np.any(raw_probs < 0) or np.any(raw_probs > 1):
            raise ValueError(f"{path}: probs contains values outside [0,1]")
        if np.any((raw_gt < 0) | (raw_gt >= N_CLS)):
            raise ValueError(f"{path}: gt_25hz contains a label outside [0,{N_CLS - 1}]")
        if raw_valid.any():
            sums = raw_probs[raw_valid].astype(np.float32).sum(axis=1)
            if not np.allclose(sums, 1.0, rtol=0.0, atol=5e-3):
                raise ValueError(
                    f"{path}: valid-frame probability rows must sum to 1 "
                    f"(observed range {sums.min()}..{sums.max()})"
                )

        gt = raw_gt
        valid = raw_valid
        out[rec] = {
            "probs": raw_probs.astype(np.float32),
            "valid": valid,
            "gt": gt,
        }
        support_records[rec] = (gt.tobytes(), valid.tobytes())
        content_record_hashes[rec] = recording_content_fingerprint(
            raw_probs, gt, valid
        )
    if not out:
        raise ValueError(f"{cache_dir}: zero .npz recordings found -- refusing to load an empty cache.")

    # RECOMPUTE support_sha256 from the actual .npz contents just read,
    # rather than trusting the string dump_logits.py wrote into the JSON --
    # an earlier version never re-derived it, so a stale/hand-edited
    # _cache_meta.json, or .npz files modified after the meta was written,
    # would carry a support_sha256 that no longer matches the real data
    # with nothing to catch it.
    recomputed_support_sha256 = support_fingerprint(support_records)
    if recomputed_support_sha256 != meta.get("support_sha256"):
        raise ValueError(
            f"{cache_dir}: support_sha256 in _cache_meta.json "
            f"({meta.get('support_sha256')}) does not match the fingerprint recomputed "
            f"from the actual .npz files just loaded ({recomputed_support_sha256}) -- "
            "the metadata is stale or the .npz files were modified after it was written."
        )

    recomputed_content_sha256 = cache_content_fingerprint(content_record_hashes)
    if recomputed_content_sha256 != meta.get("content_sha256"):
        raise ValueError(
            f"{cache_dir}: content_sha256 in _cache_meta.json "
            f"({meta.get('content_sha256')}) does not match the fingerprint recomputed "
            f"from the actual probs/gt/valid tensors ({recomputed_content_sha256}) -- "
            "the metadata is stale or a cached prediction was modified."
        )

    expected = set(meta.get("recordings", []))
    got = set(out.keys())
    if expected != got:
        raise ValueError(
            f"{cache_dir}: _cache_meta.json lists {len(expected)} recordings "
            f"but {len(got)} .npz files are present (missing={expected - got}, "
            f"extra={got - expected}) -- the directory was modified after "
            "dump_logits.py wrote it, or dump_logits.py was interrupted."
        )
    return out, meta


def verify_meta(meta: dict, *, expected_checkpoint_sha256: Optional[str] = None,
                 expected_config_sha256: Optional[str] = None) -> None:
    """Raise if `meta` (from load_cache) doesn't match the expected
    checkpoint/config lineage."""
    if expected_checkpoint_sha256 is not None and meta.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError(
            f"checkpoint SHA mismatch: cache was built from "
            f"{meta.get('checkpoint_sha256')}, expected {expected_checkpoint_sha256}")
    if expected_config_sha256 is not None and meta.get("config_sha256") != expected_config_sha256:
        raise ValueError(
            f"config SHA mismatch: cache was built from "
            f"{meta.get('config_sha256')}, expected {expected_config_sha256}")


def verify_corpus_id(meta: dict, expected_corpus_id: str, cache_dir: str) -> None:
    """Raise unless `meta["corpus_id"]` matches `expected_corpus_id` (the
    corpus name implied by the cache DIRECTORY it was loaded from, e.g.
    "ami_test" for a dir named "ami_test_s1234"). Without this, a cache
    dumped from a completely different corpus's manifest -- or a fake one
    hand-assembled for testing -- could sit in a directory named
    "ami_test_s1234" and be scored as if it were really AMI, with no other
    check catching it (a counterexample review constructed exactly this:
    three internally-consistent seeds, all claiming to be one corpus, that
    were not)."""
    if meta.get("corpus_id") != expected_corpus_id:
        raise ValueError(
            f"{cache_dir}: corpus_id in _cache_meta.json is {meta.get('corpus_id')!r}, "
            f"but this cache was loaded as if it were {expected_corpus_id!r} (implied by "
            "its directory name) -- refusing to score a cache under a corpus label it "
            "does not itself claim."
        )


def verify_consistent_across(metas: Dict[str, dict], key: str, group_label: str) -> None:
    """Raise unless every meta dict in `metas` (name -> meta) agrees on
    `meta[key]` AND that shared value is not missing/None. Used to
    auto-cross-check things that SHOULD be identical across a set of caches
    without requiring the caller to supply the expected value by hand (e.g.
    all 4 corpora for one seed must share the same checkpoint_sha256; all 3
    seeds for one corpus must share the same restrict_set_sha256). The
    explicit None/missing check is defense in depth on top of
    validate_cache_meta_schema (called by load_cache for every cache this
    function's caller passes in) -- schema validation already guarantees
    these particular keys can't be None for a cache that loaded at all, but
    this function is generic over any key, so it does not assume that."""
    values = {name: m.get(key, _MISSING) for name, m in metas.items()}
    if any(v is _MISSING or v is None for v in values.values()):
        raise ValueError(
            f"{group_label}: {key!r} is missing/None in at least one of "
            f"{list(metas.keys())}: {values} -- cannot verify consistency of an absent value."
        )
    uniq = set(values.values())
    if len(uniq) > 1:
        raise ValueError(
            f"{group_label}: {key!r} is not consistent across {list(metas.keys())}: {values}"
        )


def verify_gt_consistent_across_seeds(
    caches: Dict[str, Dict[str, dict]], corpus: str
) -> None:
    """Raise unless every seed's cache for `corpus` (seed -> {rec -> {...,
    'gt', 'valid', ...}}) has an IDENTICAL recording set, IDENTICAL ground
    truth, AND IDENTICAL valid-frame mask per recording. Ground truth and
    the valid mask both come from the same audio/RTTMs/restrict-set
    regardless of which seed's checkpoint produced the predictions -- if
    either differs, something upstream used a different manifest/restrict
    set for different seeds of the "same" corpus, which would silently make
    a 3-seed aggregate meaningless. An earlier version compared `gt` only;
    two caches can have identical ground truth values while scoring a
    completely different SUPPORT (e.g. one restricted to a subset of
    frames), which this catches too."""
    seeds = list(caches.keys())
    if len(seeds) < 2:
        return
    ref_seed = seeds[0]
    ref = caches[ref_seed]
    for seed in seeds[1:]:
        cache = caches[seed]
        if set(cache.keys()) != set(ref.keys()):
            raise ValueError(
                f"{corpus}: seed {seed} has a different recording set than "
                f"seed {ref_seed} ({set(cache.keys()) ^ set(ref.keys())} differ)"
            )
        for rec in ref:
            gt_a, gt_b = ref[rec]["gt"], cache[rec]["gt"]
            valid_a, valid_b = ref[rec]["valid"], cache[rec]["valid"]
            if gt_a.shape != gt_b.shape:
                raise ValueError(
                    f"{corpus}/{rec}: gt shape differs between seed {ref_seed} "
                    f"({gt_a.shape}) and seed {seed} ({gt_b.shape})"
                )
            if not np.array_equal(gt_a, gt_b):
                raise ValueError(
                    f"{corpus}/{rec}: ground truth differs between seed {ref_seed} "
                    f"and seed {seed} -- these should be identical (same audio/RTTMs "
                    "regardless of checkpoint), so something upstream is inconsistent."
                )
            if valid_a.shape != valid_b.shape:
                raise ValueError(
                    f"{corpus}/{rec}: valid-mask shape differs between seed {ref_seed} "
                    f"({valid_a.shape}) and seed {seed} ({valid_b.shape})"
                )
            if not np.array_equal(valid_a, valid_b):
                raise ValueError(
                    f"{corpus}/{rec}: valid-frame mask differs between seed {ref_seed} "
                    f"and seed {seed} -- identical gt with a DIFFERENT valid mask means "
                    "the two seeds are scoring different SUPPORT, which is just as "
                    "invalid for a 3-seed aggregate as differing ground truth."
                )


def score_cache(
    cache: Dict[str, dict],
    decode_fn: Optional[DecodeFn],
    boundary_tol_frames: int = 3,
) -> Dict:
    """decode_fn=None scores raw argmax (the baseline). When decode_fn is
    given, it is applied via decode_respecting_gaps per recording -- NEVER
    called directly on a `valid` array that may contain internal gaps."""
    conf = np.zeros((N_CLS, N_CLS), dtype=np.int64)
    all_gt, all_pred = [], []
    per_span_edit, per_span_frag, per_span_bf1 = [], [], []

    for rec, d in cache.items():
        probs, valid, gt = d["probs"], d["valid"], d["gt"]
        if decode_fn is None:
            # argmax baseline needs no gap-awareness of its own (it is
            # already fully frame-independent), but still must not score
            # invalid frames.
            pred = probs.argmax(-1)
        else:
            pred = decode_respecting_gaps(decode_fn, probs, valid)
        pred = np.clip(pred, 0, N_CLS - 1)

        gt_v = gt[valid]
        pred_v = pred[valid]
        if gt_v.size == 0:
            continue
        np.add.at(conf, (gt_v, pred_v), 1)
        all_gt.append(gt_v)
        all_pred.append(pred_v)

        # sequence-level metrics: one contiguous span at a time, never
        # stitched across a gap.
        for s, e in contiguous_valid_spans(valid):
            gt_span = gt[s:e].tolist()
            pred_span = pred[s:e].tolist()
            if len(gt_span) == 0:
                continue
            per_span_edit.append(segmental_edit_score(gt_span, pred_span))
            per_span_frag.append(fragmentation_rate(gt_span, pred_span))
            bf1, _, _ = _boundary_f1(np.asarray(gt_span), np.asarray(pred_span), boundary_tol_frames)
            per_span_bf1.append(bf1)

    gt_flat = np.concatenate(all_gt) if all_gt else np.zeros(0, dtype=np.int64)
    pred_flat = np.concatenate(all_pred) if all_pred else np.zeros(0, dtype=np.int64)

    f1s = [f1_from_conf(conf, c)[0] for c in range(N_CLS)]
    macro_f1 = float(np.mean(f1s))
    osd = binary_f1((gt_flat >= 2).astype(int), (pred_flat >= 2).astype(int))
    acc = float(np.trace(conf) / max(conf.sum(), 1))

    return {
        "n_recordings": len(cache),
        "n_spans": len(per_span_edit),
        "n_frames": int(gt_flat.size),
        "macro_f1": macro_f1,
        "f1_per_class": f1s,
        "acc": acc,
        "osd_f1": osd[0],
        "mean_edit_score": float(np.mean(per_span_edit)) if per_span_edit else 0.0,
        "mean_fragmentation_rate": float(np.mean(per_span_frag)) if per_span_frag else 0.0,
        "mean_boundary_f1": float(np.mean(per_span_bf1)) if per_span_bf1 else 0.0,
        "confusion": conf.tolist(),
    }
