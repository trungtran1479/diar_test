"""Distillation step 1: dump DiariZen's SOFT speaker-count posteriors for
ZipCount training windows, so the causal streaming student can be trained
against what the offline bidirectional teacher sees.

Why this works without information loss: DiariZen predicts a 16-class POWERSET
(which subset of 4 local speakers is active). Summing powerset probabilities
within each "number of active speakers" group marginalises exactly onto
ZipCount's 4 count classes {0,1,2,3+}:
    class0 <- {0}   class1 <- {1..4}   class2 <- {5..10}   class3 <- {11..15}
Soft probabilities are kept (not argmax): the teacher's uncertainty in the
2-vs-3 region is precisely the knowledge the student lacks.

Only the segmentation model is run (WavLM+Conformer forward). Embedding
extraction and VBx clustering are skipped — they resolve speaker IDENTITY,
which counting does not need, and they dominate the runtime.

Runs in the `diarizen` conda env (torch 2.1.1), separate from training.
Output: one float16 .npy per window, [T_teacher, 4], plus a frame-rate note.

Example:
  /home/edabk/miniconda3/envs/diarizen/bin/python scripts/extract_teacher_posteriors.py \
      --manifest "/media/.../manifests/train_manifest_v3.json" \
      --out-dir "/media/.../teacher_post" --limit 20 --benchmark
"""
import argparse
import json
import os
import time

import numpy as np
import torch

OUT_HZ = 100          # write on the same 100 Hz grid as the label .npy files


def build_count_mapping(device):
    """[16 powerset classes] -> [4 count classes] one-hot mapping matrix."""
    from pyannote.audio.utils.powerset import Powerset
    ps = Powerset(4, 4)
    counts = ps.mapping.sum(-1).long().clamp(max=3)      # e.g. [0,1,1,1,1,2,...,3]
    M = torch.zeros(ps.num_powerset_classes, 4)
    M[torch.arange(ps.num_powerset_classes), counts] = 1.0
    return M.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--sources", default=None,
                    help="comma list; only windows whose source starts with one of these")
    ap.add_argument("--seg-step", type=float, default=None,
                    help="sliding-window step in SECONDS for the teacher's 16 s "
                         "chunks. DiariZen infers at 1.6 s (10%% step) which is "
                         "~10x redundant compute; counts are averaged anyway, so "
                         "a larger step trades little accuracy for a lot of speed")
    args = ap.parse_args()

    import soundfile as sf
    from diarizen.pipelines.inference import DiariZenPipeline

    os.makedirs(args.out_dir, exist_ok=True)
    pipe = DiariZenPipeline.from_pretrained(args.repo)
    seg = pipe._segmentation                      # pyannote Inference wrapper
    # Keep the RAW 16-class powerset posteriors. pyannote builds the
    # powerset->multilabel converter in Inference.__init__, so flipping
    # skip_conversion afterwards is not enough — replace the module itself.
    # Multilabel would collapse the joint distribution we need to marginalise.
    import torch.nn as nn

    class _KeepPowerset(nn.Module):
        # pyannote calls conversion(output, soft=...), so a plain Identity
        # (which takes no kwargs) will not do
        def forward(self, x, soft: bool = False):
            return x

    seg.skip_conversion = True
    seg.conversion = _KeepPowerset().to(seg.device)
    if args.seg_step:
        # Assigning after the constructor bypasses pyannote's own validation. A
        # step longer than the chunk leaves uncovered seconds that get silently
        # filled with "silence" posteriors, which look perfectly well-formed.
        if args.seg_step > seg.duration:
            raise SystemExit(
                f"--seg-step {args.seg_step}s exceeds the teacher's chunk "
                f"duration {seg.duration}s: that would leave gaps in the "
                f"timeline filled with fake silence posteriors.")
        seg.step = args.seg_step
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = build_count_mapping(device)
    rf = pipe._segmentation.model._receptive_field       # 50 Hz frames
    rf_start, rf_step = rf.start, rf.step

    items = [json.loads(l) for l in open(args.manifest)]
    if args.sources:
        pref = tuple(args.sources.split(","))
        items = [it for it in items if str(it.get("source", "")).startswith(pref)]
    items = items[args.start:]
    if args.limit:
        items = items[:args.limit]
    print(f"teacher posteriors: {len(items)} windows -> {args.out_dir}", flush=True)

    t0 = time.time()
    n_done = n_skip = 0
    for i, it in enumerate(items):
        out_path = os.path.join(args.out_dir, it["id"] + ".npy")
        if os.path.exists(out_path):
            n_skip += 1
            continue
        tr = it["tracks"][0]
        x, sr = sf.read(tr["source"], dtype="float32",
                        start=int(round(tr.get("source_offset", 0.0) * 16000)),
                        frames=int(round(tr["duration"] * 16000)))
        if x.ndim > 1:
            x = x.mean(axis=1)
        wav = torch.from_numpy(x).unsqueeze(0)            # [1, T]
        with torch.no_grad():
            # soft=True -> powerset posteriors, not binarised decisions
            segm = seg({"waveform": wav, "sample_rate": sr})
        # segm.data is [n_chunks, n_frames, 16] from a HEAVILY OVERLAPPING
        # sliding window (step 1.6 s over 16 s chunks). Powerset classes are
        # speaker-permutation dependent across chunks, so they must NOT be
        # averaged directly. Speaker COUNT is permutation invariant, so we
        # marginalise 16->4 per chunk FIRST, then overlap-add on the time axis.
        p = torch.from_numpy(np.asarray(segm.data)).float().to(device)
        if p.min() < 0:                                    # logits, not probs
            p = torch.softmax(p, dim=-1)
        cnt_chunks = p @ M                                 # [chunks, frames, 4]
        cnt_chunks = cnt_chunks / cnt_chunks.sum(-1, keepdim=True).clamp_min(1e-8)

        n_out = int(round(it["duration"] * OUT_HZ))        # match label npy length
        acc = torch.zeros(n_out, 4, device=device)
        hits = torch.zeros(n_out, 1, device=device)
        chunk_step = segm.sliding_window.step
        n_ch, n_fr, _ = cnt_chunks.shape
        f_idx = torch.arange(n_fr, device=device)
        for c in range(n_ch):
            f_t = c * chunk_step + rf_start + f_idx * rf_step       # frame start times
            b0 = torch.round(f_t * OUT_HZ).long()
            b1 = torch.round((f_t + rf_step) * OUT_HZ).long()
            for b_lo, b_hi, v in zip(b0.tolist(), b1.tolist(), cnt_chunks[c]):
                lo, hi = max(b_lo, 0), min(max(b_hi, b_lo + 1), n_out)
                if lo < hi:
                    acc[lo:hi] += v
                    hits[lo:hi] += 1
        cnt = acc / hits.clamp_min(1e-8)
        empty = (hits.squeeze(-1) == 0)
        if empty.any():                                    # edges never covered
            cnt[empty] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        cnt = cnt / cnt.sum(-1, keepdim=True).clamp_min(1e-8)
        np.save(out_path, cnt.cpu().numpy().astype(np.float16))
        n_done += 1
        if args.benchmark and n_done == 1:
            print(f"  teacher frames={cnt.shape[0]} for {it['duration']}s "
                  f"-> {cnt.shape[0]/it['duration']:.1f} Hz", flush=True)
        if n_done % 50 == 0:
            el = time.time() - t0
            print(f"  {n_done} done ({el/n_done:.2f}s/window, "
                  f"eta_full={el/n_done*len(items)/3600:.1f}h)", flush=True)
    el = float(time.time() - t0)
    if n_done:
        per = el / n_done
        print(f"Done. {n_done} written, {n_skip} skipped. "
              f"{per:.2f}s/window -> {per * len(items) / 3600:.2f}h per {len(items)}")
    else:
        print(f"Done. nothing to do ({n_skip} already existed)")


if __name__ == "__main__":
    main()
