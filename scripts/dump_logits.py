"""Cache a checkpoint's per-recording, per-frame class-PROBABILITY sequence
(at the model's own 25Hz output rate) to disk, stitching consecutive windows
of the same recording into one continuous timeline so the decoders in
src/utils/decoders.py can see real segment structure across window joins
instead of resetting at every 30s/90s boundary.

This does NOT re-derive macro-F1 or any other metric -- it only produces the
raw material (probs + ground truth + a valid mask) that scripts/
decode_grid_search.py and scripts/decode_score_heldout.py consume. Every
cache directory is stamped with enough lineage metadata that a later
cross-seed/cross-corpus consistency check never has to reopen the raw .npz
files just to answer "was this the same experimental arm / same ground
truth" -- see the meta fields below and src/utils/decode_scoring.py's
schema validation.

Two source modes, matching the two manifest conventions already used
elsewhere in this project:

  --source dataset   AMI-dev val_manifest.json-style: `_w####`-suffixed
                      ids, label_filepath at 100Hz, features built via
                      SpeakerCountDataset (mirrors scripts/
                      eval_sequence_metrics.py's stitching). Used for DEV.
  --source nemo       data_ext/nemo_eval/<corpus>/manifest.json-style:
                      `<rec>_<4-digit-index>.wav` filenames, ground truth
                      via RTTM at 100Hz (mirrors scripts/
                      eval_zipcount_nemo.py). Used for the 4 held-out
                      corpora. Windows missing from --restrict-to-dir (the
                      common-window set) are left as gaps -- marked
                      invalid, not silently skipped -- so decoders still see
                      the recording's real timeline extent.

Example:
  PYTHONPATH=. python scripts/dump_logits.py --source dataset \
      --manifest data/meetings/val_manifest.json \
      --config artifacts/wavlm_offline/wavlm_hc6_s1234.yaml \
      --checkpoint logs/wavlm_hc6_s1234/step3000.pt --seed 1234 \
      --corpus-id ami_dev --out-dir logs/logit_cache/ami_dev_s1234

  PYTHONPATH=. python scripts/dump_logits.py --source nemo \
      --manifest data_ext/nemo_eval/ami_test/manifest.json \
      --restrict-to-dir logs/diarizen_ami_test/pred_rttms \
      --config artifacts/wavlm_offline/wavlm_hc6_s1234.yaml \
      --checkpoint logs/wavlm_hc6_s1234/step3000.pt --seed 1234 \
      --corpus-id ami_test --out-dir logs/logit_cache/ami_test_s1234
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Dict, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import yaml

from src.data.dataset import SpeakerCountDataset, collate_fn
from src.data.feature_extractor import feature_extractor_for_config
from src.data.label_utils import align_batch_labels, rttm_to_frame_counts
from src.models.zipcount_v1 import build_model
from src.utils.decode_scoring import (
    cache_content_fingerprint,
    recording_content_fingerprint,
    support_fingerprint,
)
from src.utils.duration_decoder_protocol import (
    PREREGISTERED_DEV,
    PREREGISTERED_HELDOUT,
    PREREGISTERED_SEEDS,
    validate_meta_against_protocol,
)

N_CLS = 4
_W_RE = re.compile(r"_w(\d+)$")
_NEMO_RE = re.compile(r"^(.*)_(\d{4})$")

# training-run bookkeeping fields that are EXPECTED to differ between seeds
# of the "same" experimental arm and must be excluded before hashing a
# config for cross-seed consistency -- see _canonical_config_sha256.
_CONFIG_FIELDS_EXCLUDED_FROM_CANONICAL_HASH = [
    ("training", "seed"),
    ("training", "log_dir"),
]


def _sha_of_file(path: str, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


# The full "what code produced these predictions" set -- deliberately
# broader than just this script. An earlier version's protocol_sha256
# hashed ONLY dump_logits.py, so a change to build_model(), the feature
# extractor, the dataset/collate path, or RTTM/label parsing would silently
# NOT be reflected in it, letting two caches be produced by materially
# different prediction-generating code while still claiming the same
# protocol. This is still not exhaustive (e.g. individual backbone wrapper
# files like src/models/wavlm_wrapper.py aren't included, since which one
# even loads is config-dependent) -- it is the fixed, always-relevant
# orchestration path, not a claim of covering every possible dependency.
_PREDICTION_PROTOCOL_FILES = [
    "scripts/dump_logits.py",
    "src/models/zipcount_v1.py",
    "src/models/wavlm_wrapper.py",
    "src/models/heads.py",
    "src/data/feature_extractor.py",
    "src/data/dataset.py",
    "src/data/label_utils.py",
]


def prediction_protocol_sha256() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    digest = hashlib.sha256()
    for rel in sorted(_PREDICTION_PROTOCOL_FILES):
        with open(os.path.join(repo_root, rel), "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()


def _sha_of_dir_listing(dir_path: str) -> str:
    """SHA256 of the sorted filename set inside ``dir_path``.

    Only membership matters here: ``--restrict-to-dir`` uses RTTM stems as
    a common-window allow-list and never consumes the RTTM contents.
    """
    names = sorted(os.listdir(dir_path))
    digest = hashlib.sha256()
    digest.update("\n".join(names).encode("utf-8"))
    return digest.hexdigest()


def canonical_config_sha256(config: dict) -> str:
    """SHA256 of the config with per-seed bookkeeping fields stripped, so
    the SAME experimental arm run at three different seeds hashes
    IDENTICALLY -- this is what proves (not just assumes) that seed 3456's
    cache wasn't accidentally dumped from a different config (e.g. the
    causal-head ablation) than seeds 1234/2345. Only `training.seed` and
    `training.log_dir` are excluded; every other field (architecture, loss,
    data, augmentation, ...) must match byte-for-byte across seeds."""
    canon = copy.deepcopy(config)
    for path in _CONFIG_FIELDS_EXCLUDED_FROM_CANONICAL_HASH:
        node = canon
        for key in path[:-1]:
            node = node.get(key, {})
        node.pop(path[-1], None)
    blob = json.dumps(canon, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def assemble_ordered(slot: Dict[int, tuple], indices: Sequence[int]):
    """Concatenate slot[i] for i in `indices` (caller-sorted), regardless of
    the ORDER items were inserted into `slot`. This is the load-bearing
    piece of the position-keyed assembly fix: both dump_dataset_mode and
    dump_nemo_mode fill `slot` in whatever order batches happen to complete
    in (DataLoader/collate_fn reordering, or simply loop order), and
    correctness depends ENTIRELY on reading back by `indices` here, never on
    insertion order. Each slot[i] is a (probs[T,C], gt[T], valid[T]) tuple.
    Unit-tested directly in tests/test_decoders.py without needing a model.
    """
    probs = np.concatenate([slot[i][0] for i in indices], axis=0).astype(np.float16)
    gt = np.concatenate([slot[i][1] for i in indices], axis=0).astype(np.int64)
    valid = np.concatenate([slot[i][2] for i in indices], axis=0).astype(bool)
    return probs, gt, valid


def dump_dataset_mode(args, model, device, feature_fn):
    """DEV-style: SpeakerCountDataset, `_w####` ids, 100Hz label files."""
    ds = SpeakerCountDataset(args.manifest, feature_extractor=feature_fn)
    ids = [ds.data[i]["id"] for i in range(len(ds))]

    def rec_of(wid):
        return wid.rsplit("_w", 1)[0]

    def widx(wid):
        m = _W_RE.search(wid)
        if not m:
            raise ValueError(f"window id has no _w#### suffix: {wid!r}")
        return int(m.group(1))

    order = sorted(range(len(ids)), key=lambda i: (rec_of(ids[i]), widx(ids[i])))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, order), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=4,
    )

    # rec -> {window_index -> (probs[T,C], gt[T], valid[T])}
    slots = defaultdict(dict)
    with torch.no_grad():
        for batch in loader:
            logits, h_lens = model(batch["features"].to(device), batch["feature_lens"].to(device))
            aligned = align_batch_labels(batch["labels"].to(device), batch["label_lens"].to(device), h_lens)
            probs = torch.softmax(logits[0].float(), dim=-1)
            for i, wid in enumerate(batch["ids"]):
                n = int(h_lens[i])
                rid = rec_of(wid)
                true = aligned[i, :n].cpu().numpy()
                slots[rid][widx(wid)] = (probs[i, :n].cpu().numpy(), np.clip(true, 0, N_CLS - 1), true >= 0)

    written = []
    support_records = {}
    content_record_hashes = {}
    for rid, idx_map in slots.items():
        # window indices need not start at 0 or be contiguous for the
        # `_w####` convention (unlike nemo mode's `<rec>_<idx4>`, no gap
        # semantics defined here) -- assemble in ascending window-index
        # order, whatever indices are actually present.
        indices = sorted(idx_map.keys())
        probs, gt, valid = assemble_ordered(idx_map, indices)
        _write_recording(args.out_dir, rid, probs, valid, gt_25hz=gt)
        support_records[rid] = (gt.tobytes(), valid.tobytes())
        content_record_hashes[rid] = recording_content_fingerprint(probs, gt, valid)
        written.append(rid)
    return written, support_records, content_record_hashes


def dump_nemo_mode(args, model, device, feature_fn):
    """Held-out corpus style: raw wav + RTTM, `<rec>_<idx4>` filenames."""
    restrict = None
    if args.restrict_to_dir:
        restrict = {os.path.splitext(f)[0] for f in os.listdir(args.restrict_to_dir) if f.endswith(".rttm")}

    items = [json.loads(l) for l in open(args.manifest)]
    by_rec = defaultdict(dict)  # rec -> {idx: item}
    for it in items:
        uid = os.path.splitext(os.path.basename(it["audio_filepath"]))[0]
        m = _NEMO_RE.match(uid)
        if not m:
            raise ValueError(f"uid does not match <rec>_<idx> convention: {uid!r}")
        rec, idx = m.group(1), int(m.group(2))
        it["_uid"] = uid
        by_rec[rec][idx] = it

    from src.data.label_utils import align_labels_to_output_len

    written = []
    support_records = {}
    content_record_hashes = {}
    for rec, idx_map in by_rec.items():
        max_idx = max(idx_map)
        ordered = [idx_map.get(i) for i in range(max_idx + 1)]  # None = gap

        slot = {}  # idx -> (probs[T,C] float32, gt[T] int64, valid[T] bool)
        batch_items, batch_idx = [], []

        def flush():
            nonlocal batch_items, batch_idx
            if not batch_items:
                return
            feats, flens = [], []
            for it in batch_items:
                wav, sr = sf.read(it["audio_filepath"], dtype="float32")
                if wav.ndim > 1:
                    wav = wav[:, 0]
                f = feature_fn(torch.from_numpy(wav).unsqueeze(0), sr)
                feats.append(f)
                flens.append(f.shape[0])
            batch = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True, padding_value=0.0)
            with torch.no_grad():
                logits, h_lens = model(batch.to(device), torch.tensor(flens, device=device))
            probs = torch.softmax(logits[0].float(), dim=-1)
            for j, it in enumerate(batch_items):
                n = int(h_lens[j])
                p = probs[j, :n].cpu().numpy()
                gt100 = rttm_to_frame_counts(it["rttm_filepath"], it["duration"])
                # downsample gt100 -> model rate via the SAME majority_vote
                # convention as training/eval (inverse of eval_zipcount_
                # nemo.py's upsample-the-prediction direction).
                gt_model_rate = align_labels_to_output_len(gt100, n, method="majority_vote")
                slot[batch_idx[j]] = (p, gt_model_rate, np.ones(n, dtype=bool))
            batch_items, batch_idx = [], []

        for idx, it in enumerate(ordered):
            if it is None or (restrict is not None and it["_uid"] not in restrict):
                # gap: unknown duration -> assume 90s (this manifest's
                # standard window length) worth of INVALID model-rate frames
                # so the recording's timeline doesn't silently shrink.
                nominal_t = int(round(90.0 * 25.0))
                slot[idx] = (
                    np.zeros((nominal_t, N_CLS), dtype=np.float32),
                    np.zeros(nominal_t, dtype=np.int64),
                    np.zeros(nominal_t, dtype=bool),
                )
                continue
            batch_items.append(it)
            batch_idx.append(idx)
            if len(batch_items) >= args.batch_size:
                flush()
        flush()

        if not slot:
            continue
        indices = list(range(max_idx + 1))
        missing = [i for i in indices if i not in slot]
        if missing:
            raise AssertionError(f"recording {rec}: missing slot indices {missing}")
        probs, gt, valid = assemble_ordered(slot, indices)
        _write_recording(args.out_dir, rec, probs, valid, gt_25hz=gt)
        support_records[rec] = (gt.tobytes(), valid.tobytes())
        content_record_hashes[rec] = recording_content_fingerprint(probs, gt, valid)
        written.append(rec)
    return written, support_records, content_record_hashes


def _write_recording(out_dir, rec_id, probs, valid, gt_25hz):
    os.makedirs(out_dir, exist_ok=True)
    safe_name = rec_id.replace("/", "_")
    np.savez_compressed(
        os.path.join(out_dir, f"{safe_name}.npz"),
        probs=probs, valid=valid, gt_25hz=gt_25hz,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["dataset", "nemo"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", required=True, type=int,
                     help="the training seed this checkpoint corresponds to -- MUST match "
                          "config['training']['seed'] (asserted below); stamped into the "
                          "cache metadata so downstream lineage checks ('does this held-out "
                          "cache use the SAME checkpoint DEV was tuned on') don't have to "
                          "parse it back out of a file path")
    ap.add_argument("--corpus-id", required=True,
                     help="canonical short name for what this cache scores, e.g. 'ami_test', "
                          "'dipco_eval', or 'ami_dev' for the DEV set -- stamped into metadata "
                          "and cross-checked by decode_score_heldout.py against the cache "
                          "directory name it was loaded from (<corpus_id>_s<seed>), so a cache "
                          "that was actually dumped from a DIFFERENT corpus's manifest cannot "
                          "silently pass as e.g. 'ami_test' just by sitting in a directory "
                          "named ami_test_s1234")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--restrict-to-dir", default=None, help="nemo mode only")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true",
                     help="required if --out-dir already exists and is non-empty -- "
                          "without it, a stale cache from a previous (possibly "
                          "different checkpoint/config) run could silently mix with "
                          "this run's fresh .npz files, since recordings not "
                          "re-written this run would be left over unnoticed")
    args = ap.parse_args()

    # Validate the sealed data protocol BEFORE deleting an old output cache,
    # building the WavLM model, or occupying the GPU. A self-reported
    # --corpus-id is not evidence that --manifest/--restrict-to-dir are the
    # intended inputs; their content hashes must match the external anchor.
    manifest_sha = _sha_of_file(args.manifest)
    actual_input = {
        "source": args.source,
        "corpus_id": args.corpus_id,
        "seed": args.seed,
        "manifest_sha256": manifest_sha,
        "frame_rate_hz": 25.0,
    }
    if args.corpus_id == PREREGISTERED_DEV["corpus_id"]:
        expected_input = PREREGISTERED_DEV
        if args.restrict_to_dir is not None:
            raise SystemExit("AMI DEV dataset cache must not use --restrict-to-dir")
    elif args.corpus_id in PREREGISTERED_HELDOUT:
        if str(args.seed) not in PREREGISTERED_SEEDS:
            raise SystemExit(
                f"held-out cache seed {args.seed} is not preregistered: "
                f"{list(PREREGISTERED_SEEDS)}"
            )
        if not args.restrict_to_dir or not os.path.isdir(args.restrict_to_dir):
            raise SystemExit(
                f"held-out corpus {args.corpus_id!r} requires an existing "
                "--restrict-to-dir"
            )
        actual_input["restrict_set_sha256"] = _sha_of_dir_listing(args.restrict_to_dir)
        expected_input = {
            "corpus_id": args.corpus_id,
            **PREREGISTERED_HELDOUT[args.corpus_id],
        }
    else:
        raise SystemExit(
            f"--corpus-id {args.corpus_id!r} is not part of the sealed duration-decoder "
            f"protocol (DEV={PREREGISTERED_DEV['corpus_id']!r}, held-out="
            f"{list(PREREGISTERED_HELDOUT)})"
        )
    try:
        validate_meta_against_protocol(
            actual_input, expected_input, f"dump request for {args.corpus_id}"
        )
    except ValueError as e:
        raise SystemExit(f"INPUT PROTOCOL MISMATCH: {e}")

    expected_out_name = f"{args.corpus_id}_s{args.seed}"
    if os.path.basename(os.path.normpath(args.out_dir)) != expected_out_name:
        raise SystemExit(
            f"--out-dir must end in {expected_out_name!r}; got {args.out_dir!r}"
        )

    config = yaml.safe_load(open(args.config))
    config_seed = config.get("training", {}).get("seed")
    if config_seed != args.seed:
        raise SystemExit(
            f"--seed {args.seed} does not match config['training']['seed']={config_seed} "
            f"in {args.config!r} -- refusing to proceed. This guards against exactly the "
            "kind of copy-paste error (dumping seed 2345's checkpoint under a --seed 1234 "
            "label) that would otherwise silently corrupt cross-seed lineage checks."
        )
    if not os.path.isfile(args.checkpoint) or os.path.getsize(args.checkpoint) == 0:
        raise SystemExit(f"--checkpoint {args.checkpoint!r} is missing or empty")

    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir):
        if not args.overwrite:
            raise SystemExit(
                f"--out-dir {args.out_dir!r} already exists and is non-empty. "
                "Refusing to write into it without --overwrite: a stale cache "
                "from an earlier (possibly different checkpoint/config) run "
                "could otherwise silently mix with this run's output."
            )
        import shutil
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model = model.to(device).eval()
    feature_fn = feature_extractor_for_config(config)

    if args.source == "dataset":
        written, support_records, content_record_hashes = dump_dataset_mode(
            args, model, device, feature_fn
        )
    else:
        written, support_records, content_record_hashes = dump_nemo_mode(
            args, model, device, feature_fn
        )

    if not written:
        raise SystemExit(f"dumped ZERO recordings from {args.manifest!r} -- refusing to "
                          "write an empty cache (would pass schema validation as a valid "
                          "but useless directory).")

    meta = {
        "source": args.source,
        "corpus_id": args.corpus_id,
        "seed": args.seed,
        "manifest": args.manifest,
        "manifest_sha256": manifest_sha,
        "config": args.config,
        "config_sha256": _sha_of_file(args.config),
        "canonical_config_sha256": canonical_config_sha256(config),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": _sha_of_file(args.checkpoint),
        "restrict_to_dir": args.restrict_to_dir,
        "restrict_set_sha256": (
            _sha_of_dir_listing(args.restrict_to_dir) if args.restrict_to_dir else None
        ),
        "support_sha256": support_fingerprint(support_records),
        "content_sha256": cache_content_fingerprint(content_record_hashes),
        "protocol_sha256": prediction_protocol_sha256(),
        "frame_rate_hz": 25.0,
        "n_recordings": len(written),
        "recordings": written,
    }
    with open(os.path.join(args.out_dir, "_cache_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"dumped {len(written)} recordings -> {args.out_dir} "
          f"(checkpoint_sha256={meta['checkpoint_sha256'][:12]}... "
          f"canonical_config_sha256={meta['canonical_config_sha256'][:12]}... "
          f"support_sha256={meta['support_sha256'][:12]}...)")


if __name__ == "__main__":
    main()
