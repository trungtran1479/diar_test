"""Paired bootstrap confidence interval for the macro-F1 DIFFERENCE between
two systems, resampling RECORDINGS (not frames).

Why recordings: windows from one recording share speakers, room and noise
conditions, so they are not independent draws. Bootstrapping frames or
windows would give an interval that is far too narrow. The unit of
resampling here is the recording, which is the coarsest unit the evaluation
protocol provides.

Why this exists: comparing two systems' pooled macro-F1 and calling a small
gap "within seed noise" conflates two different quantities -- the
seed-to-seed variance of ONE system's training, and the sampling variance of
the DIFFERENCE between two fixed systems on this test set. This script
measures the second.

Inputs are the per-window confusion dumps written by
`count_from_rttms.py --dump-conf-json` and
`eval_zipcount_nemo.py --dump-conf-json`. Both key on the same window uid
(`<recording>_<window>`), so the pairing is exact and only windows present
in BOTH files are used.

Single-corpus mode answers "do these two systems differ ON THIS CORPUS".
That is NOT the quantity reported in the main table, which is a mean over
corpora weighted EQUALLY. Testing the headline number requires
`--stratified`: pass one `corpus=path` pair per corpus for each system;
recordings are then resampled WITHIN each corpus, macro-F1 is computed per
corpus, and the four values are averaged exactly as the table does. Pooling
recordings across corpora instead would silently re-weight the metric by
corpus size (VoxConverse alone has more recordings than the rest combined).

Examples:
  # one corpus
  PYTHONPATH=. python scripts/paired_bootstrap.py \
      --a conf_zipcount.json --a-name ZipCount \
      --b conf_sortformer.json --b-name Sortformer-streaming

  # the reported corpus-equal-weighted mean
  PYTHONPATH=. python scripts/paired_bootstrap.py --stratified \
      --a ami=a_ami.json --a dipco=a_dip.json --a vox=a_vox.json --a msd=a_msd.json \
      --b ami=b_ami.json --b dipco=b_dip.json --b vox=b_vox.json --b msd=b_msd.json
"""
import argparse
import json

import numpy as np

N_CLS = 4


def macro_f1(conf):
    """conf: [4,4] int array, rows = truth, cols = prediction."""
    f1s = []
    for c in range(N_CLS):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def recording_of(uid):
    """`aepyx_0003` -> `aepyx`. Window index is the final `_`-separated field."""
    return uid.rsplit("_", 1)[0]


def load(path):
    raw = json.load(open(path))
    return {uid: np.asarray(v, dtype=np.int64).reshape(N_CLS, N_CLS)
            for uid, v in raw.items()}


def stack_by_recording(ca, cb, a_name, b_name, tag=""):
    """-> (A, B) arrays of per-recording confusions, aligned and paired."""
    common = sorted(set(ca) & set(cb))
    if not common:
        raise SystemExit(f"no shared window uids between the two dumps {tag}")
    only_a, only_b = len(set(ca) - set(cb)), len(set(cb) - set(ca))
    if only_a or only_b:
        print(f"note{tag}: dropping {only_a} windows only in {a_name}, "
              f"{only_b} only in {b_name}")
    by_rec = {}
    for uid in common:
        by_rec.setdefault(recording_of(uid), []).append(uid)
    recs = sorted(by_rec)
    A = np.stack([np.sum([ca[u] for u in by_rec[r]], axis=0) for r in recs])
    B = np.stack([np.sum([cb[u] for u in by_rec[r]], axis=0) for r in recs])
    return A, B, len(common), len(recs)


def parse_spec(values):
    """['ami=x.json', ...] -> {'ami': 'x.json'} (stratified mode)."""
    out = {}
    for v in values:
        if "=" not in v:
            raise SystemExit(f"--stratified needs corpus=path, got {v!r}")
        k, p = v.split("=", 1)
        out[k] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, action="append",
                    help="confusion JSON for system A; 'corpus=path' if --stratified")
    ap.add_argument("--b", required=True, action="append",
                    help="confusion JSON for system B; 'corpus=path' if --stratified")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--stratified", action="store_true",
                    help="resample within each corpus and average macro-F1 over "
                         "corpora with EQUAL weight, matching the reported mean")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.stratified:
        sa, sb = parse_spec(args.a), parse_spec(args.b)
        if set(sa) != set(sb):
            raise SystemExit(f"corpus sets differ: {sorted(sa)} vs {sorted(sb)}")
        corpora = sorted(sa)
        strata = []
        for c in corpora:
            A, B, nw, nr = stack_by_recording(
                load(sa[c]), load(sb[c]), args.a_name, args.b_name, f" [{c}]")
            strata.append((c, A, B))
            print(f"  {c:14s} {nw:5d} windows / {nr:4d} recordings   "
                  f"{args.a_name} {macro_f1(A.sum(0)):.4f}  "
                  f"{args.b_name} {macro_f1(B.sum(0)):.4f}")
        obs_a = float(np.mean([macro_f1(A.sum(0)) for _, A, _ in strata]))
        obs_b = float(np.mean([macro_f1(B.sum(0)) for _, _, B in strata]))
        print(f"\ncorpus-equal-weighted mean over {len(corpora)} corpora")
        print(f"{args.a_name:28s} macro-F1 = {obs_a:.4f}")
        print(f"{args.b_name:28s} macro-F1 = {obs_b:.4f}")
        print(f"observed difference (A - B) = {obs_a - obs_b:+.4f}")

        deltas = np.empty(args.n_boot)
        for i in range(args.n_boot):
            da = db = 0.0
            for _, A, B in strata:
                idx = rng.integers(0, len(A), len(A))
                da += macro_f1(A[idx].sum(0))
                db += macro_f1(B[idx].sum(0))
            deltas[i] = (da - db) / len(strata)
    else:
        if len(args.a) != 1 or len(args.b) != 1:
            raise SystemExit("pass --stratified to use more than one corpus")
        A, B, nw, nr = stack_by_recording(
            load(args.a[0]), load(args.b[0]), args.a_name, args.b_name)
        obs_a, obs_b = macro_f1(A.sum(0)), macro_f1(B.sum(0))
        print(f"{nw} windows over {nr} recordings")
        print(f"{args.a_name:28s} macro-F1 = {obs_a:.4f}")
        print(f"{args.b_name:28s} macro-F1 = {obs_b:.4f}")
        print(f"observed difference (A - B) = {obs_a - obs_b:+.4f}")

        deltas = np.empty(args.n_boot)
        n = len(A)
        for i in range(args.n_boot):
            idx = rng.integers(0, n, n)
            deltas[i] = macro_f1(A[idx].sum(0)) - macro_f1(B[idx].sum(0))

    lo, hi = np.quantile(deltas, [args.alpha / 2, 1 - args.alpha / 2])
    # two-sided bootstrap p-value: how often the resampled difference falls
    # on the other side of zero from the observed one
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    p = min(1.0, float(p))

    print(f"\npaired bootstrap over recordings ({args.n_boot} resamples)")
    print(f"  mean delta = {deltas.mean():+.4f}")
    print(f"  {100*(1-args.alpha):.0f}% CI = [{lo:+.4f}, {hi:+.4f}]")
    print(f"  two-sided p = {p:.4f}")
    verdict = ("CI excludes 0 -> difference is resolved by this test"
               if lo > 0 or hi < 0 else
               "CI includes 0 -> difference NOT resolved by this test")
    print(f"  {verdict}")


if __name__ == "__main__":
    main()
