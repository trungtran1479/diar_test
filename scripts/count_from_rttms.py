"""Post-process predicted RTTMs (from NVIDIA's official e2e_diarize_speech.py)
into frame-level speaker-COUNT metrics, scored against the ground-truth RTTMs
in the NeMo eval manifest built by make_nemo_eval_assets.py.

Counts are rasterized at 100 Hz from RTTM segments (both gt and pred), capped
at 3+ — identical to the ZipCount / eval_sortformer_ami protocol.

Example:
  PYTHONPATH=. python scripts/count_from_rttms.py \
      --manifest data_ext/nemo_eval/ami_test/manifest.json \
      --pred-rttm-dir logs/sortformer_official_ami_test/pred_rttms
"""
import argparse
import json
import os

import numpy as np

from src.data.label_utils import rttm_to_frame_counts
from src.eval_sortformer_ami import binary_f1, f1_from_conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred-rttm-dir", required=True)
    ap.add_argument("--restrict-to-dir", default=None,
                    help="only score windows whose uid has an RTTM in this dir "
                         "(for common-window comparison across systems)")
    ap.add_argument("--name", default="")
    ap.add_argument("--dump-conf-json", default=None,
                    help="also write {uid: 4x4 confusion} here, for "
                         "scripts/paired_bootstrap.py")
    args = ap.parse_args()

    restrict = None
    if args.restrict_to_dir:
        restrict = {os.path.splitext(f)[0] for f in os.listdir(args.restrict_to_dir)
                    if f.endswith(".rttm")}

    gt_all, pred_all = [], []
    per_uid = {}
    n_missing = 0
    with open(args.manifest) as f:
        for line in f:
            item = json.loads(line)
            uid = os.path.splitext(os.path.basename(item["audio_filepath"]))[0]
            if restrict is not None and uid not in restrict:
                continue
            pred_path = os.path.join(args.pred_rttm_dir, uid + ".rttm")
            if not os.path.exists(pred_path):
                n_missing += 1
                continue
            dur = item["duration"]
            g = rttm_to_frame_counts(item["rttm_filepath"], dur)
            p = rttm_to_frame_counts(pred_path, dur)
            gt_all.append(g)
            pred_all.append(p)
            if args.dump_conf_json:
                cw = np.zeros((4, 4), dtype=np.int64)
                np.add.at(cw, (g, p), 1)
                per_uid[uid] = cw.ravel().tolist()
    if n_missing:
        print(f"WARNING: {n_missing} windows had no predicted RTTM")
    if args.dump_conf_json:
        with open(args.dump_conf_json, "w") as fh:
            json.dump(per_uid, fh)
        print(f"wrote per-window confusions for {len(per_uid)} windows "
              f"-> {args.dump_conf_json}")

    gt = np.concatenate(gt_all)
    pred = np.concatenate(pred_all)
    conf = np.zeros((4, 4), dtype=np.int64)
    np.add.at(conf, (gt, pred), 1)

    acc = np.trace(conf) / conf.sum()
    mae = np.abs(gt - pred).mean()
    f1s = [f1_from_conf(conf, c)[0] for c in range(4)]
    vad_f1, vad_p, vad_r = binary_f1((gt >= 1).astype(int), (pred >= 1).astype(int))
    osd_f1, osd_p, osd_r = binary_f1((gt >= 2).astype(int), (pred >= 2).astype(int))

    print(f"\n===== Counting from official RTTMs {args.name} "
          f"({len(gt_all)} windows, 100 Hz) =====")
    print(f"acc={acc:.3f}  mae={mae:.3f}  macroF1={np.mean(f1s):.3f}")
    print(f"f1/cls = [" + " ".join(f"{f:.3f}" for f in f1s) + "]")
    print(f"VAD:  F1={vad_f1:.3f} (P={vad_p:.3f} R={vad_r:.3f})")
    print(f"OSD (count>=2): F1={osd_f1:.3f} (P={osd_p:.3f} R={osd_r:.3f})")
    print("confusion P(pred|true):  pred0   pred1   pred2   pred3")
    for t in range(4):
        row = conf[t] / max(conf[t].sum(), 1)
        print(f"  true{t}:               " + "   ".join(f"{v:.3f}" for v in row))


if __name__ == "__main__":
    main()
