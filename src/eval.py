import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SpeakerCountDataset, collate_fn
from src.data.feature_extractor import feature_extractor_for_config
from src.data.label_utils import align_batch_labels
from src.models.zipcount_v1 import build_model
from src.utils.metrics import compute_metrics, aggregate_metrics, format_confusion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    val_manifest = config["data"].get("val_manifest", "")
    if not val_manifest or not os.path.exists(val_manifest):
        raise FileNotFoundError(f"Validation manifest not found: {val_manifest}")
        
    feature_extractor = feature_extractor_for_config(config)
    dataset = SpeakerCountDataset(val_manifest, feature_extractor=feature_extractor)
    loader = DataLoader(
        dataset,
        batch_size=config["training"].get("batch_size", 16),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    model = build_model(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
        
    model = model.to(device)
    model.eval()
    
    label_downsample_method = config["data"].get("label_downsample", "majority_vote")
    all_metrics = []
    
    print("Evaluating...")
    with torch.no_grad():
        for batch in tqdm(loader):
            features = batch["features"].to(device)
            feature_lens = batch["feature_lens"].to(device)
            labels = batch["labels"].to(device)
            label_lens = batch["label_lens"].to(device)
            
            with torch.cuda.amp.autocast(enabled=config["training"].get("use_amp", True)):
                logits, h_lens = model(features, feature_lens)
                aligned_labels = align_batch_labels(labels, label_lens, h_lens, method=label_downsample_method)
                
            m = compute_metrics(logits[0], logits[1], logits[2], aligned_labels, h_lens)
            if m:
                all_metrics.append(m)
                
    agg_metrics = aggregate_metrics(all_metrics)
    
    print("\n--- Evaluation Results ---")
    for k, v in agg_metrics.items():
        if k.startswith("cm_"):
            continue
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print()
    print(format_confusion(agg_metrics))
            
if __name__ == "__main__":
    main()
