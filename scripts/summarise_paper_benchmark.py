"""Collect run_paper_benchmark.sh output into paper-ready tables.

Reports every seed plus mean and range, never the best seed — Phase 2 of
this project measured ~0.02 macro-F1 of within-arm seed spread on
VoxConverse, larger than most effects seen since, so a single-seed number
is indistinguishable from cherry-picking. Any mask-vs-root delta smaller
than either system's own seed range is flagged, not silently reported as
an improvement.
"""
import argparse
import json
import os
import re

import numpy as np

SEEDS = [1234, 2345, 3456]
SYSTEMS = [
    ("mask", "TCN + stack mask pre012 (main system)"),
    ("root", "TCN root, all six stacks (ablation)"),
]
BENCH_ROWS = [
    ("AMI-SDM dev", "ami-sdm_dev", "in-domain (AMI train was in the mix)"),
    ("AMI-SDM test", "ami-sdm_test", "in-domain"),
    ("DipCo-MDM dev", "dipco-mdm_dev", "OUT-of-domain, never trained on"),
    ("DipCo-MDM eval", "dipco-mdm_eval", "OUT-of-domain, never trained on"),
    ("LibriCount", "libricount", "synthetic, clip-level"),
]
WINDOW_ROWS = [
    ("vox_sel", "voxsel", "dev-ish selection (every chain/graduation gate)"),
    ("VoxConverse test", "voxfull", "in-domain"),
    ("vox_lock", "voxlock", "LOCKED until the rev-8 chain finished"),
    ("MSDWild many.val", "msdwild", "LOCKED, 0 overlap with train"),
]


def parse_bench_log(path):
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read()
    out = {}
    m = re.search(r"acc=([\d.]+)\s+mae[^\s]*=([\d.]+)\s+macroF1=([\d.]+)", txt)
    if m:
        out["acc"], out["mae"], out["macro_f1"] = (
            float(m.group(1)), float(m.group(2)), float(m.group(3))
        )
    else:  # libricount has no macroF1 line
        m = re.search(r"acc=([\d.]+)\s+mae\(capped\)=([\d.]+)", txt)
        if m:
            out["acc"], out["mae"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"OSD \(count>=2\): F1=([\d.]+)", txt)
    if m:
        out["osd_f1"] = float(m.group(1))
    return out or None


def parse_perrec(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    p = dict(d["pooled"])
    p["macro_unw"] = d.get("macro_f1_unweighted_over_recordings")
    p["n_rec"] = d.get("n_recordings")
    return p


def parse_seq(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    agg = d["aggregate"]
    return {
        "edit_score": agg["mean_edit_score"],
        "fragmentation_rate": agg["mean_fragmentation_rate"],
        "ece": d["calibration"]["ece"],
        "brier": d["calibration"]["brier"],
        "mae": d["calibration"].get("mae"),
        "n_frames": d.get("n_frames"),
    }


def collect(root_dir, system, key, kind):
    vals = []
    for seed in SEEDS:
        name = f"{system}_s{seed}"
        if kind == "bench":
            row = parse_bench_log(os.path.join(root_dir, f"{name}__{key}.log"))
        elif kind == "perrec":
            row = parse_perrec(os.path.join(root_dir, f"{name}__{key}_perrec.json"))
        else:
            row = parse_seq(os.path.join(root_dir, f"{name}__{key}_seq.json"))
        vals.append(row)
    return vals


def summarize(vals, metric):
    got = [v[metric] for v in vals if v and v.get(metric) is not None]
    if not got:
        return None
    return {
        "vals": [v.get(metric) if v else None for v in vals],
        "mean": float(np.mean(got)),
        "range": max(got) - min(got) if len(got) > 1 else 0.0,
    }


def print_table(title, rows, root_dir, kind, metric, width=20):
    print(f"\n{'=' * 100}\n{title}  (metric = {metric})\n{'=' * 100}")
    header = f"{'benchmark':<{width}}"
    for sys_key, _ in SYSTEMS:
        header += f"{sys_key + ' mean':>12}{'range':>8}"
    header += f"{'delta(mask-root)':>18}   note"
    print(header)
    print("-" * 100)
    for label, key, note in rows:
        summaries = {}
        for sys_key, _ in SYSTEMS:
            vals = collect(root_dir, sys_key, key, kind)
            summaries[sys_key] = summarize(vals, metric)
        if all(s is None for s in summaries.values()):
            print(f"{label:<{width}} {'— not run —':>40}   {note}")
            continue
        line = f"{label:<{width}}"
        for sys_key, _ in SYSTEMS:
            s = summaries[sys_key]
            if s is None:
                line += f"{'—':>12}{'—':>8}"
            else:
                line += f"{s['mean']:12.4f}{s['range']:8.4f}"
        mask_s, root_s = summaries.get("mask"), summaries.get("root")
        if mask_s and root_s:
            delta = mask_s["mean"] - root_s["mean"]
            noise = max(mask_s["range"], root_s["range"])
            flag = " (within seed range)" if abs(delta) < noise else ""
            line += f"{delta:+18.4f}{flag}   {note}"
        else:
            line += f"{'—':>18}   {note}"
        print(line)
    print("-" * 100)
    print("range = max-min across the three seeds. A |delta| smaller than the")
    print("larger system's own range is flagged '(within seed range)', not")
    print("reported as a real difference.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/bench_paper")
    args = ap.parse_args()

    print_table("PRIMARY: macro-F1 across corpora", BENCH_ROWS, args.dir, "bench", "macro_f1")
    print_table("LibriCount accuracy (no macro-F1 for this bench)", [
        ("LibriCount", "libricount", "synthetic, clip-level"),
    ], args.dir, "bench", "acc")
    print_table("PRIMARY: pooled macro-F1, window-JSON corpora", WINDOW_ROWS, args.dir, "perrec", "macro_f1")
    print_table("SECONDARY: OSD F1, window-JSON corpora", WINDOW_ROWS, args.dir, "perrec", "osd_f1")
    print_table("SECONDARY: count MAE, window-JSON corpora", WINDOW_ROWS, args.dir, "seq", "mae")
    print_table("NEW (sec. 11): segmental edit score [0,100]", WINDOW_ROWS, args.dir, "seq", "edit_score")
    print_table("NEW (sec. 11): fragmentation rate (1.0 = ideal)", WINDOW_ROWS, args.dir, "seq", "fragmentation_rate")
    print_table("NEW (sec. 11): expected calibration error", WINDOW_ROWS, args.dir, "seq", "ece")
    print_table("NEW (sec. 11): Brier score", WINDOW_ROWS, args.dir, "seq", "brier")

    print(
        "\nNote: boundary AP / recall@FP-budget was NOT computed here -- "
        "eval_boundary_ap.py requires boundary_logits, which only the "
        "gated_pyramid_ordinal head exposes. Both benchmarked systems use "
        "the plain tcn_ordinal head (no boundary branch); this metric is "
        "not applicable to them."
    )


if __name__ == "__main__":
    main()
