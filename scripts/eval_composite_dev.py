"""Score a checkpoint's composite DEV metric: mean(AMI-dev, DipCo-dev, vox_sel)
macro-F1, all three scored the SAME way -- SpeakerCountDataset, native
model frame rate, no upsampling -- via the exact per-recording-summarise
logic in scripts/eval_per_recording.py, so this reports numbers directly
comparable to that script's per-corpus output.

Used for the partial-unfreeze/LoRA backbone-adaptation triage (see
zipcount-wavlm-loss-unfreeze-fixes memory): a DIAGNOSTIC checkpoint-
selection signal, never scored against the 4 held-out corpora
(ami_test/dipco_eval/msdwild_manyval/voxconverse_voxlock).

DipCo-dev (data/dipco_dev/dipco_dev_manifest.json, DiPCo's own official dev
split S02/S04/S05/S09/S10) is built by scripts/prepare_dipco_dev.py and is
disjoint from dipco_eval (S01/S03/S06/S07/S08) -- verified zero session
overlap. Reusing dipco_eval here would spend that held-out corpus's
independence on tuning.

Usage:
  PYTHONPATH=. python scripts/eval_composite_dev.py \
      --checkpoint logs/wavlm_partial_unfreeze_last2/step400.pt \
      --config artifacts/wavlm_offline/partial_unfreeze/unfreeze_last2.yaml \
      --ami-dev data/meetings/val_manifest.json \
      --dipco-dev data/dipco_dev/dipco_dev_manifest.json \
      --vox-sel "/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json"
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import yaml

from src.data.dataset import SpeakerCountDataset, collate_fn
from src.data.feature_extractor import feature_extractor_for_config
from src.data.label_utils import align_batch_labels
from src.models.zipcount_v1 import build_model

N_CLS = 4


def f1_from_conf(c, k):
    tp = c[k, k]; fp = c[:, k].sum() - tp; fn = c[k, :].sum() - tp
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return float(2 * p * r / max(p + r, 1e-9))


def macro_f1_from_conf(conf):
    return float(np.mean([f1_from_conf(conf, k) for k in range(N_CLS)]))


def score_manifest(model, device, config, manifest, batch_size, use_amp):
    ds = SpeakerCountDataset(manifest, feature_extractor=feature_extractor_for_config(config))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_fn, num_workers=4)
    conf = np.zeros((N_CLS, N_CLS), dtype=np.int64)
    label_downsample_method = config["data"].get("label_downsample", "majority_vote")
    with torch.no_grad():
        for batch in loader:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, h_lens = model(batch["features"].to(device), batch["feature_lens"].to(device))
            aligned = align_batch_labels(batch["labels"].to(device), batch["label_lens"].to(device),
                                          h_lens, method=label_downsample_method)
            pred = logits[0].float().argmax(-1)
            for i in range(len(batch["ids"])):
                n = int(h_lens[i])
                p = pred[i, :n].cpu().numpy(); l = aligned[i, :n].cpu().numpy()
                m = l >= 0
                np.add.at(conf, (l[m], p[m]), 1)
    return conf, len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ami-dev", default="data/meetings/val_manifest.json")
    ap.add_argument("--dipco-dev", default="data/dipco_dev/dipco_dev_manifest.json")
    ap.add_argument("--vox-sel", default="/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config["training"].get("use_amp", True)) and device.type == "cuda"

    model = build_model(config)
    lora_cfg = config["model"]["encoder"].get("lora", {})
    if bool(lora_cfg.get("enabled", False)):
        # Mirror train.py's build_model -> apply_lora -> load_state_dict
        # order exactly. build_model()'s WavLMBackbone freezes the encoder
        # by default (freeze=True), so apply_lora()'s own "backbone must
        # already be frozen" invariant holds here too. Skipping this step
        # left the plain (non-LoRA) module topology in place, so
        # load_state_dict below choked on the checkpoint's extra
        # lora_A/lora_B keys and renamed base_layer.weight/bias paths.
        model.backbone.apply_lora(
            target_modules=lora_cfg["target_modules"],
            layer_indices=lora_cfg.get("layer_indices"),
            rank=int(lora_cfg.get("rank", 8)),
            alpha=int(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.0)),
        )
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model = model.to(device).eval()

    corpora = {"ami_dev": args.ami_dev, "dipco_dev": args.dipco_dev, "vox_sel": args.vox_sel}
    per_corpus = {}
    for name, manifest in corpora.items():
        if not os.path.exists(manifest):
            raise SystemExit(f"{name}: manifest not found: {manifest!r}")
        conf, n = score_manifest(model, device, config, manifest, args.batch_size, use_amp)
        macro_f1 = macro_f1_from_conf(conf)
        per_corpus[name] = {"macro_f1": macro_f1, "n_recordings": n,
                             "f1_per_class": [f1_from_conf(conf, k) for k in range(N_CLS)]}
        print(f"  {name}: macro_f1={macro_f1:.4f} ({n} windows/recordings)")

    composite = float(np.mean([v["macro_f1"] for v in per_corpus.values()]))
    print(f"composite DEV (mean of {list(corpora)}) = {composite:.4f}")

    out = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "per_corpus": per_corpus,
        "composite_dev_macro_f1": composite,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"-> {args.out}")
    return out


if __name__ == "__main__":
    main()
