"""Immutable protocol for the WavLM duration-decoder experiment.

The cache metadata is self-reported by ``dump_logits.py``.  Cross-seed
agreement is useful, but three runs can agree on the same *wrong* input
(for example, a copy/pasted manifest).  The constants below are therefore
the external anchor: DEV selection, held-out scoring, and the fitted priors
must match these already-sealed content hashes, not merely one another.

Changing any value here is a protocol revision and intentionally changes
``decode_protocol_sha256()`` because this file is included in that digest.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Mapping


FRAME_RATE_HZ = 25.0
BOUNDARY_TOL_FRAMES = 3
PREREGISTERED_SEEDS = ("1234", "2345", "3456")
PREREGISTERED_CORPORA = (
    "ami_test",
    "dipco_eval",
    "msdwild_manyval",
    "voxconverse_voxlock",
)

# Content hashes were sealed before DEV grid search / held-out scoring.
# Paths are deliberately not authoritative: a relocated byte-identical
# manifest is the same input, while a file at the old path with new contents
# is not.
PREREGISTERED_DEV: Dict[str, Any] = {
    "source": "dataset",
    "corpus_id": "ami_dev",
    "seed": 1234,
    "manifest_sha256": "c5e44d1e839a474049c1518fd5508f4c6ac7bca9963d4e5a2ca44fe576482961",
    "frame_rate_hz": FRAME_RATE_HZ,
}

PREREGISTERED_HELDOUT: Dict[str, Dict[str, Any]] = {
    "ami_test": {
        "source": "nemo",
        "manifest_sha256": "341ecb3a0d20f1a0e0a8b869fffce5c01346b9b468c4a71e31aa96964408250f",
        "restrict_set_sha256": "f8adad63e259d5c1184d8552e26f07489f905632dfab4d59cb3278ab3e9452af",
        "frame_rate_hz": FRAME_RATE_HZ,
    },
    "dipco_eval": {
        "source": "nemo",
        "manifest_sha256": "9bae64af1d61a488876202768a63aef830c5f512634f70082d86837288b43809",
        "restrict_set_sha256": "7b34eee8b58c54b881b4c02f1017807eaef4187cda14decac92625d6aa53bd43",
        "frame_rate_hz": FRAME_RATE_HZ,
    },
    "msdwild_manyval": {
        "source": "nemo",
        "manifest_sha256": "eb9903566b96661a69a18a546cbb2f9b9d792ef21cc6766d532d0568fe7a9d73",
        "restrict_set_sha256": "d531226bfd13fc82689903ce91dc91402658b7e283979b496e9701d263215ac5",
        "frame_rate_hz": FRAME_RATE_HZ,
    },
    "voxconverse_voxlock": {
        "source": "nemo",
        "manifest_sha256": "b5a5a77eb38b8d8fe31110cfa7ea762e763b54865b09696899990b4d0ee9b942",
        "restrict_set_sha256": "06149e9c489cf2fec0092ee63d250e51e5f439e412c329435427131641028e31",
        "frame_rate_hz": FRAME_RATE_HZ,
    },
}

PREREGISTERED_PRIORS: Dict[str, Any] = {
    "priors_npz_sha256": "cda16a416a29a6882ca01911f1b4ae081c5610895e8a6e7f0c1595eedb2cb513",
    "manifest_sha256": "fee646152c558cf14f64994927645220e571071d766c3ea3dcf3bcb08cd5474a",
    "source_prefixes": [
        "ami_train",
        "alimeeting_train",
        "aishell4_train",
        "msdwild_train",
        "ramc_train",
        "voxconverse_train",
        "nsf_train",
    ],
    "sample_every": 10,
    "laplace_transition": 1.0,
    "laplace_duration_total_per_class": 1.0,
    "n_windows_missing_label_file": 0,
    "frame_rate_hz": FRAME_RATE_HZ,
    "max_dur_frames": 750,
}


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def validate_meta_against_protocol(
    meta: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    """Fail unless every protocol-sealed field has its exact expected value."""
    mismatches = {
        key: {"actual": meta.get(key), "expected": value}
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{label}: metadata does not match the preregistered protocol: "
            f"{mismatches}"
        )


def validate_priors_provenance(
    priors_path: str, expected: Mapping[str, Any] | None = None
) -> Dict[str, Any]:
    """Validate both the priors bytes and their training-only provenance."""
    expected = PREREGISTERED_PRIORS if expected is None else expected
    meta_path = priors_path + ".meta.json"
    if not os.path.isfile(meta_path):
        raise ValueError(
            f"{priors_path}: missing required provenance sidecar {meta_path!r}"
        )
    with open(meta_path) as f:
        meta = json.load(f)
    validate_meta_against_protocol(meta, expected, f"priors sidecar {meta_path}")

    actual_sha = sha256_file(priors_path)
    if actual_sha != expected["priors_npz_sha256"]:
        raise ValueError(
            f"{priors_path}: SHA256={actual_sha}, expected sealed priors SHA256="
            f"{expected['priors_npz_sha256']}"
        )
    if meta.get("priors_npz_sha256") != actual_sha:
        raise ValueError(
            f"{meta_path}: priors_npz_sha256={meta.get('priors_npz_sha256')} "
            f"does not match the actual NPZ SHA256={actual_sha}"
        )
    return meta
