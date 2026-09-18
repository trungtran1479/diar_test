"""How much teacher signal survives each KD mask, and what is in it?

Phase 2 showed `agree` adds nothing measurable. The hypothesis is that the mask
keeps mostly saturated frames, so the KL target is effectively the hard label.
Before spending six more training runs, measure the masks directly on the
TRAINING manifest (no dev/test involved, so this cannot leak selection):

  * what fraction of frames each mask keeps, overall and per class
  * how confident the surviving teacher rows are
  * how much probability mass sits on the runner-up — this is the only thing a
    soft target can teach that a hard label cannot

If `agree_uncertain` keeps a tiny fraction, lambda_kd must be raised to keep the
gradient contribution comparable, otherwise the arm tests "no KD" twice.
"""
import argparse
import json

import numpy as np
import torch
import yaml

from src.data.dataset import SpeakerCountDataset, collate_fn
from src.data.feature_extractor import feature_extractor_for_config
from src.data.label_utils import align_batch_labels, align_batch_posteriors

N_CLS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/zipcount_v4_distill.yaml")
    ap.add_argument("--max-batches", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--thresholds", default="0.8,0.9,0.95,0.99")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    manifest = config["data"]["train_manifest"]
    teacher_dir = config["distill"]["teacher_dir"]
    ths = [float(x) for x in args.thresholds.split(",")]

    ds = SpeakerCountDataset(manifest, teacher_dir=teacher_dir,
                              feature_extractor=feature_extractor_for_config(config))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=4, generator=torch.Generator().manual_seed(0))

    tot = np.zeros(N_CLS)          # labelled frames per class
    agree = np.zeros(N_CLS)        # teacher argmax == label
    unc = {t: np.zeros(N_CLS) for t in ths}   # agree AND max-prob < t
    conf_sum = 0.0
    runner_sum = 0.0
    n_agree = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.max_batches:
                break
            if not batch["teacher_mask"].any():
                continue
            labels, label_lens = batch["labels"], batch["label_lens"]
            # the model downsamples; here labels are already at teacher rate, so
            # use the label length itself as the target length
            h_lens = label_lens
            t_post = align_batch_posteriors(batch["teacher"], label_lens, h_lens).float()
            al = align_batch_labels(labels, label_lens, h_lens)

            valid = torch.arange(al.shape[1])[None, :] < h_lens[:, None]
            valid &= batch["teacher_mask"][:, None]
            valid &= al >= 0
            valid &= t_post.sum(-1) > 0.5

            s = t_post.sum(-1).clamp_min(1e-6)
            p = t_post / s.unsqueeze(-1)
            top2 = p.topk(2, dim=-1)
            conf = top2.values[..., 0]
            runner = top2.values[..., 1]
            ag = valid & (p.argmax(-1) == al)

            for c in range(N_CLS):
                cm = valid & (al == c)
                tot[c] += int(cm.sum())
                agree[c] += int((ag & (al == c)).sum())
                for t in ths:
                    unc[t][c] += int((ag & (al == c) & (conf < t)).sum())
            conf_sum += float(conf[ag].sum())
            runner_sum += float(runner[ag].sum())
            n_agree += int(ag.sum())

    print(f"manifest: {manifest}")
    print(f"frames with a label and a teacher: {int(tot.sum()):,}\n")
    print(f"{'mask':<22} {'kept':>10} {'%all':>7}  " +
          "  ".join(f"cls{c}%" for c in range(N_CLS)))

    def row(name, arr):
        pct = 100 * arr.sum() / max(tot.sum(), 1)
        per = "  ".join(f"{100*arr[c]/max(tot[c],1):5.1f}" for c in range(N_CLS))
        print(f"{name:<22} {int(arr.sum()):>10,} {pct:6.1f}%  {per}")

    row("agree (current)", agree)
    for t in ths:
        row(f"agree & maxp<{t}", unc[t])

    print(f"\nsurviving 'agree' rows: mean max-prob {conf_sum/max(n_agree,1):.4f}, "
          f"mean runner-up {runner_sum/max(n_agree,1):.4f}")
    print("runner-up mass is the entire advantage of a soft target over a hard "
          "label; if it is near zero the KL is a relabelled cross-entropy.")


if __name__ == "__main__":
    main()
