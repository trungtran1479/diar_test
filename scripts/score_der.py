"""Score DER of predicted RTTMs against ground-truth RTTMs listed in a NeMo
eval manifest, using pyannote.metrics (runs in the `diarizen` env). collar and
overlap handling are explicit so the number is comparable across systems.

Example:
  python scripts/score_der.py \
      --manifest data_ext/nemo_eval/ami_test/manifest.json \
      --pred-rttm-dir logs/diarizen_ami_test/pred_rttms \
      --collar 0.25 --name "DiariZen AMI test"

Use --restrict-to-dir to score only the common-window set every system in a
table produced output for (same convention as count_from_rttms.py /
eval_zipcount_nemo.py), so no system is credited for windows another
system skipped.
"""
import argparse
import json
import os

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


def load_rttm(path, uid):
    ann = Annotation(uri=uid)
    if not os.path.exists(path):
        return ann
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 8 and p[0] == "SPEAKER":
                start, dur, spk = float(p[3]), float(p[4]), p[7]
                if dur > 0:
                    ann[Segment(start, start + dur)] = spk
    return ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred-rttm-dir", required=True)
    ap.add_argument("--collar", type=float, default=0.25)
    ap.add_argument("--skip-overlap", action="store_true")
    ap.add_argument("--restrict-to-dir", default=None,
                    help="only score windows whose uid has an RTTM in this dir "
                         "(for common-window comparison across systems)")
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    restrict = None
    if args.restrict_to_dir:
        restrict = {os.path.splitext(f)[0] for f in os.listdir(args.restrict_to_dir)
                    if f.endswith(".rttm")}

    metric = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
    n = 0
    n_skip = 0
    for line in open(args.manifest):
        item = json.loads(line)
        uid = os.path.splitext(os.path.basename(item["audio_filepath"]))[0]
        if restrict is not None and uid not in restrict:
            continue
        pred_path = os.path.join(args.pred_rttm_dir, uid + ".rttm")
        if not os.path.exists(pred_path):
            # window the system could not process (e.g. DiariZen tail < 16 s);
            # skip for fairness rather than charge an all-miss
            n_skip += 1
            continue
        ref = load_rttm(item["rttm_filepath"], uid)
        hyp = load_rttm(pred_path, uid)
        metric(ref, hyp)
        n += 1
    if n_skip:
        print(f"(skipped {n_skip} windows with no predicted RTTM)")

    der = abs(metric)
    comp = metric[:]
    total = comp["total"]
    print(f"\n===== DER {args.name} ({n} windows, collar={args.collar}, "
          f"skip_overlap={args.skip_overlap}) =====")
    print(f"DER = {der * 100:.2f}%")
    print(f"  miss      = {comp['missed detection'] / total * 100:.2f}%")
    print(f"  false pos = {comp['false alarm'] / total * 100:.2f}%")
    print(f"  confusion = {comp['confusion'] / total * 100:.2f}%")


if __name__ == "__main__":
    main()
