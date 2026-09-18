"""Run DiariZen (BUT-FIT WavLM-Large + VBx) on the SAME windowed wavs used for
the Sortformer official benchmark, writing predicted RTTMs so they can be
scored with scripts/count_from_rttms.py (counting) and pyannote.metrics (DER),
against the identical ground-truth RTTMs.

Runs in the `diarizen` conda env (torch 2.1.1), separate from .zipformer.

Example:
  python scripts/run_diarizen.py \
      --manifest data_ext/nemo_eval/ami_test/manifest.json \
      --out-rttm-dir logs/diarizen_ami_test/pred_rttms \
      --repo BUT-FIT/diarizen-wavlm-large-s80-md-v2
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-rttm-dir", required=True)
    ap.add_argument("--repo", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from diarizen.pipelines.inference import DiariZenPipeline
    os.makedirs(args.out_rttm_dir, exist_ok=True)
    pipe = DiariZenPipeline.from_pretrained(args.repo, rttm_out_dir=args.out_rttm_dir)

    items = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        items = items[:args.limit]
    print(f"DiariZen {args.repo}: {len(items)} windows -> {args.out_rttm_dir}")

    n_fail = 0
    for i, item in enumerate(items):
        wav = item["audio_filepath"]
        uid = os.path.splitext(os.path.basename(wav))[0]
        try:
            pipe(wav, sess_name=uid)
        except Exception as e:
            # Windows shorter than DiariZen's 16 s seg_duration (recording
            # tails) hit pyannote's "negative dimensions" in reconstruct.
            # This is a tooling limit, not a quality result, so we write NO
            # file -> the scorer skips the window for ALL systems' fairness
            # rather than charging DiariZen an all-miss. Count + report them.
            n_fail += 1
            print(f"  !! SKIP {uid} (dur={item['duration']}s): "
                  f"{type(e).__name__}: {e}", flush=True)
        if (i + 1) % 20 == 0 or i + 1 == len(items):
            print(f"  [{i + 1}/{len(items)}] {uid}", flush=True)
    print(f"Done. {n_fail} windows skipped (no RTTM written; scorer excludes them).")


if __name__ == "__main__":
    main()
