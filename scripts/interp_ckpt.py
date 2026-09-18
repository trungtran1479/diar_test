"""Linear weight-space interpolation between two checkpoints.

    theta(alpha) = (1 - alpha) * theta_A + alpha * theta_B

Only floating-point tensors are interpolated. Integer buffers (step counters,
num_batches_tracked) have no meaningful midpoint, so they are taken from B and
verified to be shape-compatible — silently averaging them would produce a
checkpoint that loads but is subtly wrong.

Both endpoints must come from the same run lineage (B fine-tuned FROM A) for
the interpolation to stay in the same loss basin; interpolating independently
initialised models does not work.
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="alpha=0 endpoint")
    ap.add_argument("--b", required=True, help="alpha=1 endpoint")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert 0.0 <= args.alpha <= 1.0, "alpha must be in [0,1]"

    ca = torch.load(args.a, map_location="cpu", weights_only=False)
    cb = torch.load(args.b, map_location="cpu", weights_only=False)
    sa = ca.get("model_state_dict", ca)
    sb = cb.get("model_state_dict", cb)

    if set(sa) != set(sb):
        only_a, only_b = set(sa) - set(sb), set(sb) - set(sa)
        raise SystemExit(f"key mismatch: {len(only_a)} only in A, {len(only_b)} only in B")

    out, n_lerp, n_copy = {}, 0, 0
    for k in sb:
        ta, tb = sa[k], sb[k]
        if ta.shape != tb.shape:
            raise SystemExit(f"shape mismatch at {k}: {tuple(ta.shape)} vs {tuple(tb.shape)}")
        if torch.is_floating_point(tb):
            out[k] = (1.0 - args.alpha) * ta.float() + args.alpha * tb.float()
            out[k] = out[k].to(tb.dtype)
            n_lerp += 1
        else:
            out[k] = tb.clone()
            n_copy += 1

    torch.save({"model_state_dict": out,
                "interp": {"a": args.a, "b": args.b, "alpha": args.alpha}},
               args.out)
    print(f"alpha={args.alpha}: lerped {n_lerp} float tensors, copied {n_copy} "
          f"non-float from B -> {args.out}")


if __name__ == "__main__":
    main()
