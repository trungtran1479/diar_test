"""Summarise the v3 -> step400 interpolation sweep as a cross-domain Pareto curve.

Two questions, kept separate:

  1. SELECTION. Which alpha maximises the pre-registered primary composite
     (unweighted mean macro-F1 over AMI-dev, DipCo-dev, vox_sel), averaged over
     seeds? Reported alongside the secondary composite (worst domain), which is
     reported and NOT optimised.

  2. GEOMETRY. Does the path bow outside the chord between the endpoints? For
     each domain the chord value at alpha is (1-alpha)*score(0) + alpha*score(1).
     A point above the chord in a domain means interpolation bought accuracy
     that neither endpoint had there. If a single alpha is above the chord in
     EVERY domain, weight-space interpolation moved the frontier rather than
     sliding along it — that is the claim worth making, and it is exactly what
     a linear trade-off would forbid.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

DOMAINS = ["ami_dev", "dipco_dev", "voxsel"]
SEEDS = ["1234", "2345", "3456"]


def macro_from_log(path):
    if not os.path.exists(path):
        return None
    m = re.search(r"macroF1=([\d.]+)", open(path, errors="ignore").read())
    return float(m.group(1)) if m else None


def macro_from_json(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))["pooled"]["macro_f1"]


def endpoints(bench_dir):
    """alpha=0 is v3; alpha=1 is each seed's step400."""
    a0 = {
        "ami_dev":   macro_from_log(f"{bench_dir}/v3_base__ami-sdm_dev.log"),
        "dipco_dev": macro_from_log(f"{bench_dir}/v3_base__dipco-mdm_dev.log"),
        "voxsel":    macro_from_json(f"{bench_dir}/v3_base__voxsel.json"),
    }
    a1 = {}
    for s in SEEDS:
        a1[s] = {
            "ami_dev":   macro_from_log(f"{bench_dir}/ft_s{s}__ami-sdm_dev.log"),
            "dipco_dev": macro_from_log(f"{bench_dir}/ft_s{s}__dipco-mdm_dev.log"),
            "voxsel":    macro_from_json(f"results/p2_ctrl_s{s}_voxsel.json"),
        }
    return a0, a1


def sweep_point(d, alpha, seed):
    tag = f"a{alpha:.2f}_s{seed}"
    return {
        "ami_dev":   macro_from_log(f"{d}/{tag}__ami_dev.log"),
        "dipco_dev": macro_from_log(f"{d}/{tag}__dipco_dev.log"),
        "voxsel":    macro_from_json(f"{d}/{tag}__voxsel.json"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/interp")
    ap.add_argument("--bench-dir", default="results/bench")
    args = ap.parse_args()

    a0, a1 = endpoints(args.bench_dir)
    if any(v is None for v in a0.values()):
        raise SystemExit(f"missing alpha=0 endpoint scores: {a0}")

    alphas = sorted({float(re.match(r"a([\d.]+)_s", os.path.basename(p)).group(1))
                     for p in glob.glob(f"{args.dir}/a*_s*__voxsel.json")})
    grid = [0.0] + alphas + [1.0]

    # per-alpha, per-seed domain scores
    table = {}
    for a in grid:
        for s in SEEDS:
            if a == 0.0:
                table[(a, s)] = dict(a0)
            elif a == 1.0:
                table[(a, s)] = a1[s]
            else:
                table[(a, s)] = sweep_point(args.dir, a, s)

    print("\n" + "=" * 96)
    print("Weight-space interpolation  v3 (alpha=0)  ->  step400 fine-tune (alpha=1)")
    print("composite dev = unweighted mean macro-F1 over AMI-dev, DipCo-dev, vox_sel")
    print("=" * 96)
    print(f"{'alpha':>6} {'AMI-dev':>9} {'DipCo-dev':>10} {'vox_sel':>9} | "
          f"{'MEAN':>7} {'WORST':>7} | above-chord in")
    print("-" * 96)

    best = None
    rows = []
    for a in grid:
        got = [table[(a, s)] for s in SEEDS]
        if any(v is None for g in got for v in g.values()):
            print(f"{a:6.2f}   -- incomplete --")
            continue
        mean_dom = {d: float(np.mean([g[d] for g in got])) for d in DOMAINS}
        comp = float(np.mean([mean_dom[d] for d in DOMAINS]))
        worst = float(min(mean_dom.values()))

        # chord between the two endpoints, per domain (seed-averaged)
        a1m = {d: float(np.mean([a1[s][d] for s in SEEDS])) for d in DOMAINS}
        above = []
        for d in DOMAINS:
            chord = (1 - a) * a0[d] + a * a1m[d]
            if mean_dom[d] > chord + 1e-6:
                above.append(d)

        flag = "" if 0 < a < 1 else "  (endpoint)"
        print(f"{a:6.2f} {mean_dom['ami_dev']:9.4f} {mean_dom['dipco_dev']:10.4f} "
              f"{mean_dom['voxsel']:9.4f} | {comp:7.4f} {worst:7.4f} | "
              f"{','.join(above) if above else '-'}{flag}")
        rows.append((a, mean_dom, comp, worst, above))
        if 0 < a < 1 and (best is None or comp > best[2]):
            best = (a, mean_dom, comp, worst, above)

    print("-" * 96)
    e0 = float(np.mean([a0[d] for d in DOMAINS]))
    a1m = {d: float(np.mean([a1[s][d] for s in SEEDS])) for d in DOMAINS}
    e1 = float(np.mean([a1m[d] for d in DOMAINS]))
    print(f"endpoint composites: alpha=0 {e0:.4f}   alpha=1 {e1:.4f}")

    if best is None:
        print("\nno interior alpha scored — nothing to select.")
        return
    a, dom, comp, worst, above = best
    print(f"\nSELECTED alpha* = {a:.2f} on the primary composite "
          f"({comp:.4f} vs {max(e0, e1):.4f} for the better endpoint)")
    print(f"  worst-domain (secondary, reported not optimised): {worst:.4f}")

    dominates = all(dom[d] >= max(a0[d], a1m[d]) - 1e-6 for d in DOMAINS)
    print(f"\nGEOMETRY")
    print(f"  domains above the endpoint chord at alpha*: "
          f"{','.join(above) if above else 'none'}")
    if dominates:
        print("  alpha* DOMINATES both endpoints in every domain — the path bows")
        print("  outside the chord, so this is a genuinely better operating point,")
        print("  obtained without any retraining.")
    elif len(above) == len(DOMAINS):
        print("  alpha* beats the CHORD in every domain but does not dominate both")
        print("  endpoints outright: the trade-off is convex-favourable, i.e. mixing")
        print("  is cheaper than the linear trade-off implies, but each endpoint")
        print("  still wins its own home domain.")
    else:
        print("  the path does NOT bow outside the chord in every domain. Interpolation")
        print("  is sliding along the trade-off, not moving the frontier. Report it as")
        print("  a tunable operating point, NOT as a free improvement.")


if __name__ == "__main__":
    main()
