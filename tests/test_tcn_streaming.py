"""Streaming equivalence tests for TCNCountHead.

    full_causal_forward == concatenate(cached_chunk_forward)

REGRESSION CONTEXT — why the bias perturbation below is load-bearing:
the first implementation cached RAW block inputs and prepended raw zeros
before LayerNorm. That equals the full forward (which pads zeros AFTER the
norm) only when LayerNorm.bias == 0 — true of a fresh head, so tests at
~1e-6 passed, while a trained checkpoint (|bias| 0.057–0.092) drifted up to
0.014 logit over the first receptive-field of every stream. Any equivalence
test run only on a fresh head is a false pass for this class of bug, so this
file explicitly sets nonzero norm biases, and additionally runs the real
Phase-5 checkpoint when it exists on this machine.

Run:  PYTHONPATH=. python tests/test_tcn_streaming.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.heads import TCNCountHead  # noqa: E402

STACK_DIMS = [192, 256, 384, 512, 384, 256]
TRAINED_CKPT = "logs/p5_tcn192_s1234/step3000.pt"


def build_head(seed=0, d_model=192, perturb_bias=True):
    torch.manual_seed(seed)
    head = TCNCountHead(sum(STACK_DIMS), stack_dims=STACK_DIMS, d_model=d_model).eval()
    if perturb_bias:
        # the load-bearing line — see module docstring
        with torch.no_grad():
            for blk in head.blocks:
                torch.nn.init.normal_(blk.norm.bias, std=0.1)
    return head


def assert_equivalent(head, x, chunk_sizes, tol=2e-5, label=""):
    with torch.no_grad():
        full = head(x)
        caches, outs, pos = None, [], 0
        for s in chunk_sizes:
            out, caches = head.forward_streaming(x[:, pos:pos + s], caches)
            outs.append(out)
            pos += s
        assert pos == x.shape[1], (pos, x.shape)
        for k, name in enumerate(["count", "vad", "overlap"]):
            chunked = torch.cat([o[k] for o in outs], dim=1)
            d = (full[k] - chunked).abs().max().item()
            assert d < tol, f"{label}/{name}: max|full-chunked|={d:.2e} >= {tol}"
    print(f"  {label:<28} OK (max err count={_err(full, outs, 0):.1e})")


def _err(full, outs, k):
    return (full[k] - torch.cat([o[k] for o in outs], dim=1)).abs().max().item()


def test_fresh_head_nonzero_bias():
    head = build_head()
    x = torch.randn(2, 203, sum(STACK_DIMS))
    assert_equivalent(head, x, [16] * 12 + [11], label="fresh+bias, chunk=16")
    assert_equivalent(head, x, [7, 1, 32, 16, 3, 64, 80], label="fresh+bias, ragged")


def test_trained_checkpoint():
    if not os.path.exists(TRAINED_CKPT):
        print(f"  (skip: {TRAINED_CKPT} not present)")
        return
    head = build_head(perturb_bias=False)
    sd = torch.load(TRAINED_CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    head_sd = {k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")}
    head.load_state_dict(head_sd)
    head.eval()
    bmax = max(float(blk.norm.bias.abs().max()) for blk in head.blocks)
    assert bmax > 0.01, "trained checkpoint unexpectedly has ~zero norm bias"
    x = torch.randn(1, 250, sum(STACK_DIMS))
    assert_equivalent(head, x, [16] * 15 + [10],
                      label=f"trained ckpt (|b|max={bmax:.3f})")


def test_kernel1_streaming():
    """kernel=1 => pad_ctx=0; the old `zin[:, -0:]` slice returned the WHOLE
    sequence as the next cache instead of an empty one."""
    torch.manual_seed(3)
    head = TCNCountHead(sum(STACK_DIMS), stack_dims=STACK_DIMS,
                        d_model=64, kernel=1).eval()
    with torch.no_grad():
        for blk in head.blocks:
            torch.nn.init.normal_(blk.norm.bias, std=0.1)
    x = torch.randn(1, 50, sum(STACK_DIMS))
    assert_equivalent(head, x, [16, 16, 16, 2], label="kernel=1, chunk=16")
    _, caches = head.forward_streaming(x[:, :16], None)
    assert all(c.shape[1] == 0 for c in caches), \
        "kernel=1 caches must be empty, not the whole sequence"


def test_cache_carries_context():
    head = build_head()
    x = torch.randn(1, 120, sum(STACK_DIMS))
    with torch.no_grad():
        (cold, _, _), _ = head.forward_streaming(x[:, 100:116], None)
        caches, pos = None, 0
        for s in [16] * 6 + [4]:
            _, caches = head.forward_streaming(x[:, pos:pos + s], caches)
            pos += s
        (warm, _, _), _ = head.forward_streaming(x[:, 100:116], caches)
    d = (cold - warm).abs().max().item()
    assert d > 1e-3, "cache has no effect — equivalence test would be vacuous"
    print(f"  cold-vs-warm divergence      OK ({d:.3f})")


def test_cache_validation():
    head = build_head()
    x = torch.randn(1, 16, sum(STACK_DIMS))
    for bad, why in [
        ([], "empty list must not silently skip all blocks"),
        (None, None),  # placeholder, replaced below
    ]:
        if bad == []:
            try:
                head.forward_streaming(x, [])
            except ValueError:
                print(f"  reject empty caches          OK")
            else:
                raise AssertionError(why)
    _, caches = head.forward_streaming(x, None)
    caches[3] = caches[3][:, :-1]  # wrong time length
    try:
        head.forward_streaming(x, caches)
    except ValueError:
        print(f"  reject wrong-shape cache     OK")
    else:
        raise AssertionError("wrong-shape cache must raise")


if __name__ == "__main__":
    print("test_tcn_streaming:")
    test_fresh_head_nonzero_bias()
    test_trained_checkpoint()
    test_cache_carries_context()
    test_cache_validation()
    print("ALL PASSED")
