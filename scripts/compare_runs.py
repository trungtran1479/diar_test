"""Paired comparison of two model families (e.g. KD vs no-KD control) across
seeds AND recordings.

Two separate sources of uncertainty are reported, because they answer different
questions:
  * seed spread     — training stochasticity, from paired per-seed deltas
  * recording CI    — corpus sampling, from a paired bootstrap over recordings

A result is only called real when the paired delta is positive across every
seed AND the recording-level bootstrap CI excludes zero. A positive mean with a
CI straddling zero is "suggestive", not a win.

Example:
  python scripts/compare_runs.py \
      --a results/ctrl_s{1,2,3}.json --b results/kd_s{1,2,3}.json \
      --name-a "lambda=0" --name-b "lambda=.25 T=2" --metric macro_f1
"""
import argparse
import json

import numpy as np


def load(paths):
    return [json.load(open(p)) for p in paths]


def paired_bootstrap(a_runs, b_runs, metric, n_boot=10000, seed=0):
    """Bootstrap over RECORDINGS, averaging seeds within each recording so the
    resample unit is the recording (what we are generalising over)."""
    recs = sorted(set.intersection(*[set(r["per_recording"]) for r in a_runs + b_runs]))
    da = np.array([[r["per_recording"][x][metric] for x in recs] for r in a_runs]).mean(0)
    db = np.array([[r["per_recording"][x][metric] for x in recs] for r in b_runs]).mean(0)
    d = db - da
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(recs), size=(n_boot, len(recs)))
    boots = d[idx].mean(1)
    return d, float(d.mean()), (float(np.percentile(boots, 2.5)),
                                float(np.percentile(boots, 97.5))), recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="control run jsons")
    ap.add_argument("--b", nargs="+", required=True, help="treatment run jsons")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--metric", default="macro_f1")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    m = args.metric

    print(f"=== {args.name_b}  vs  {args.name_a}   [{m}] ===\n")
    print("per-seed (paired: same seed, same batch order):")
    n = min(len(A), len(B))
    per_seed = []
    for i in range(n):
        a, b = A[i]["pooled"][m], B[i]["pooled"][m]
        per_seed.append(b - a)
        print(f"  seed{i+1}: {args.name_a}={a:.4f}  {args.name_b}={b:.4f}  delta={b-a:+.4f}")
    per_seed = np.array(per_seed)
    print(f"  -> mean delta {per_seed.mean():+.4f}, "
          f"all same sign: {bool(np.all(per_seed > 0) or np.all(per_seed < 0))}")

    d, mean_d, ci, recs = paired_bootstrap(A, B, m)
    print(f"\nrecording-level paired bootstrap ({len(recs)} recordings, seeds averaged):")
    print(f"  mean delta {mean_d:+.4f}   95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"  recordings improved: {int((d > 0).sum())}/{len(d)}")
    excl = ci[0] > 0 or ci[1] < 0
    print(f"  CI excludes zero: {excl}")

    order = np.argsort(d)
    print(f"\n  most negative recordings: " +
          ", ".join(f"{recs[i]}({d[i]:+.3f})" for i in order[:args.top]))
    print(f"  most positive recordings: " +
          ", ".join(f"{recs[i]}({d[i]:+.3f})" for i in order[::-1][:args.top]))
    # is the effect carried by a handful of recordings?
    share = np.abs(d[order[::-1]][:args.top]).sum() / max(np.abs(d).sum(), 1e-9)
    print(f"  top-{args.top} recordings carry {share*100:.0f}% of the total |delta|")

    print("\nVERDICT: ", end="")
    if np.all(per_seed > 0) and excl and ci[0] > 0:
        print("REAL — positive across all seeds and recording CI excludes zero")
    elif per_seed.mean() > 0:
        print("SUGGESTIVE — mean positive but not established (check CI / seed signs)")
    else:
        print("NO EFFECT / NEGATIVE")


if __name__ == "__main__":
    main()
