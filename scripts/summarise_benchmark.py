"""Collect run_full_benchmark.sh output into one table.

Reports every seed plus mean and range, never the best seed. Phase 2 measured
~0.02 macro-F1 of within-arm seed spread on VoxConverse, which is larger than
any effect we have measured on this model, so a single-seed row would be
indistinguishable from cherry-picking.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

SEEDS = ["ft_s1234", "ft_s2345", "ft_s3456"]
ROWS = [
    ("AMI-SDM dev",      "ami-sdm_dev",    "in-domain (AMI train was in the mix)"),
    ("AMI-SDM test",     "ami-sdm_test",   "in-domain"),
    ("DipCo-MDM dev",    "dipco-mdm_dev",  "OUT-of-domain, never trained on"),
    ("DipCo-MDM eval",   "dipco-mdm_eval", "OUT-of-domain, never trained on"),
    ("LibriCount",       "libricount",     "synthetic, clip-level"),
    ("VoxConverse test", "voxfull",        "in-domain"),
    ("vox_lock",         "voxlock",        "LOCKED until now"),
    ("MSDWild many.val", "msdwild",        "LOCKED, 0 overlap with train"),
]


def parse_log(path):
    """macro-F1 / acc / OSD out of an eval_bench log."""
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read()
    out = {}
    m = re.search(r"acc=([\d.]+)\s+mae[^\s]*=([\d.]+)\s+macroF1=([\d.]+)", txt)
    if m:
        out["acc"], out["macro_f1"] = float(m.group(1)), float(m.group(3))
    else:  # libricount has no macroF1 line
        m = re.search(r"acc=([\d.]+)\s+mae\(capped\)=([\d.]+)", txt)
        if m:
            out["acc"] = float(m.group(1))
    m = re.search(r"OSD \(count>=2\): F1=([\d.]+)", txt)
    if m:
        out["osd_f1"] = float(m.group(1))
    m = re.search(r"f1/cls = \[([\d.\s]+)\]", txt)
    if m:
        f = [float(x) for x in m.group(1).split()]
        out.update({f"f1_{i}": v for i, v in enumerate(f)})
    return out or None


def parse_json(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    p = dict(d["pooled"])
    p["macro_unw"] = d.get("macro_f1_unweighted_over_recordings")
    p["n_rec"] = d.get("n_recordings")
    return p


def get(d, model, key):
    log = os.path.join(d, f"{model}__{key}.log")
    js = os.path.join(d, f"{model}__{key}.json")
    return parse_json(js) if os.path.exists(js) else parse_log(log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/bench")
    ap.add_argument("--metric", default="macro_f1")
    args = ap.parse_args()
    m = args.metric

    w = 19
    print(f"\n{'='*92}\nZipCount — full benchmark, metric = {m}")
    print("finetuned = uniform LR 3e-5, lambda_kd=0, step 400 (the selected config)")
    print("=" * 92)
    print(f"{'benchmark':<{w}} {'v3 base':>8} {'s1234':>8} {'s2345':>8} {'s3456':>8} "
          f"{'mean':>8} {'range':>7} {'vs v3':>8}   note")
    print("-" * 92)

    for label, key, note in ROWS:
        base = get(args.dir, "v3_base", key)
        vals = []
        for s in SEEDS:
            r = get(args.dir, s, key)
            vals.append(r.get(m) if r else None)
        if all(v is None for v in vals) and base is None:
            print(f"{label:<{w}} {'— not run —':>44}   {note}")
            continue
        got = [v for v in vals if v is not None]
        bs = f"{base[m]:.4f}" if base and base.get(m) is not None else "—"
        cells = " ".join(f"{v:8.4f}" if v is not None else f"{'—':>8}" for v in vals)
        if got:
            mean = float(np.mean(got))
            rng = max(got) - min(got)
            delta = (f"{mean - base[m]:+8.4f}"
                     if base and base.get(m) is not None else f"{'—':>8}")
            print(f"{label:<{w}} {bs:>8} {cells} {mean:8.4f} {rng:7.4f} {delta}   {note}")
        else:
            print(f"{label:<{w}} {bs:>8} {cells} {'—':>8} {'—':>7} {'—':>8}   {note}")

    print("-" * 92)
    print("range = max-min across the three seeds. Any 'vs v3' smaller than the")
    print("range on the same row is inside seed noise and must not be reported")
    print("as an improvement.")


if __name__ == "__main__":
    main()
