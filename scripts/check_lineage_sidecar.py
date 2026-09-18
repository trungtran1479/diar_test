"""Verify or write a minimal lineage sidecar for oracle/probe arms.

    check_lineage_sidecar.py SIDECAR CONFIG INIT           -> exit 0 iff match
    check_lineage_sidecar.py --write SIDECAR CONFIG INIT   -> write atomically

The sidecar binds the arm's config file sha and its init-checkpoint sha, so a
result can only be [skip]-reused when both still hold (5th review: existence
alone must never gate reuse).
"""
import hashlib
import json
import os
import sys


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main():
    argv = sys.argv[1:]
    write = argv and argv[0] == "--write"
    if write:
        argv = argv[1:]
    sidecar, config, init = argv
    live = {"config_sha": sha(config), "init_sha": sha(init)}
    if write:
        tmp = sidecar + f".tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(live, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, sidecar)
        return 0
    try:
        stored = json.load(open(sidecar))
    except Exception:
        return 1
    return 0 if stored == live else 1


if __name__ == "__main__":
    sys.exit(main())
