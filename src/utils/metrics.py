import torch
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

def compute_metrics(
    count_logits: torch.Tensor,
    vad_logits: torch.Tensor,
    overlap_logits: torch.Tensor,
    labels: torch.Tensor,
    h_lens: torch.Tensor
) -> Dict[str, float]:
    """Compute raw confusion counts for metrics calculation.
    
    Returns raw true positives, false positives, false negatives, etc.
    so they can be aggregated over the whole dataset.
    """
    B, T, _ = count_logits.shape
    
    if labels.shape[1] > T:
        labels = labels[:, :T]
    elif labels.shape[1] < T:
        pad = torch.full((B, T - labels.shape[1]), -100, dtype=labels.dtype, device=labels.device)
        labels = torch.cat([labels, pad], dim=1)
        
    pred_counts = torch.argmax(count_logits, dim=-1)
    pred_vad = (torch.sigmoid(vad_logits.squeeze(-1)) >= 0.5).long()
    pred_overlap = (torch.sigmoid(overlap_logits.squeeze(-1)) >= 0.5).long()
    
    mask = torch.arange(T, device=count_logits.device).unsqueeze(0) < h_lens.unsqueeze(1)
    valid_label_mask = (labels >= 0)
    final_mask = mask & valid_label_mask
    
    if final_mask.sum() == 0:
        return {}
        
    pred_counts_flat = pred_counts[final_mask]
    pred_vad_flat = pred_vad[final_mask]
    pred_overlap_flat = pred_overlap[final_mask]
    labels_flat = labels[final_mask].clamp(min=0, max=3)
    
    metrics = {
        "count_correct": (pred_counts_flat == labels_flat).sum().item(),
        "count_total": labels_flat.size(0),
        "count_abs_err": torch.abs(pred_counts_flat - labels_flat).float().sum().item(),
    }
    
    for c in range(4):
        true_c = (labels_flat == c)
        pred_c = (pred_counts_flat == c)
        metrics[f"tp_{c}"] = (true_c & pred_c).sum().item()
        metrics[f"fp_{c}"] = (~true_c & pred_c).sum().item()
        metrics[f"fn_{c}"] = (true_c & ~pred_c).sum().item()

    # Raw 4x4 confusion counts: cm_<true>_<pred>
    for i in range(4):
        true_i = (labels_flat == i)
        for j in range(4):
            metrics[f"cm_{i}_{j}"] = (true_i & (pred_counts_flat == j)).sum().item()
        
    # Overlap from count head
    true_ov = (labels_flat >= 2)
    pred_ov_count = (pred_counts_flat >= 2)
    metrics["tp_ov_count"] = (true_ov & pred_ov_count).sum().item()
    metrics["fp_ov_count"] = (~true_ov & pred_ov_count).sum().item()
    metrics["fn_ov_count"] = (true_ov & ~pred_ov_count).sum().item()
    
    # Overlap from overlap head
    metrics["tp_ov_head"] = (true_ov & (pred_overlap_flat == 1)).sum().item()
    metrics["fp_ov_head"] = (~true_ov & (pred_overlap_flat == 1)).sum().item()
    metrics["fn_ov_head"] = (true_ov & (pred_overlap_flat != 1)).sum().item()
    
    # VAD from VAD head
    true_vad = (labels_flat >= 1)
    metrics["tp_vad"] = (true_vad & (pred_vad_flat == 1)).sum().item()
    metrics["fp_vad"] = (~true_vad & (pred_vad_flat == 1)).sum().item()
    metrics["fn_vad"] = (true_vad & (pred_vad_flat != 1)).sum().item()
    
    return metrics

def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate raw counts and compute final F1/Precision/Recall."""
    if not metrics_list:
        return {}
        
    agg = Counter()
    for m in metrics_list:
        agg.update(m)
        
    def calc_prf(tp, fp, fn):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        return p, r, f1
        
    res = {}
    res["acc"] = agg["count_correct"] / max(agg["count_total"], 1)
    res["mae"] = agg["count_abs_err"] / max(agg["count_total"], 1)
    
    f1s = []
    for c in range(4):
        p, r, f1 = calc_prf(agg[f"tp_{c}"], agg[f"fp_{c}"], agg[f"fn_{c}"])
        res[f"p_{c}"] = p
        res[f"r_{c}"] = r
        res[f"f1_{c}"] = f1
        f1s.append(f1)
        
    res["macro_f1"] = sum(f1s) / 4.0
    
    p, r, f1 = calc_prf(agg["tp_ov_count"], agg["fp_ov_count"], agg["fn_ov_count"])
    res["p_ov_count"] = p
    res["r_ov_count"] = r
    res["f1_ov_count"] = f1
    
    p, r, f1 = calc_prf(agg["tp_ov_head"], agg["fp_ov_head"], agg["fn_ov_head"])
    res["p_ov_head"] = p
    res["r_ov_head"] = r
    res["f1_ov_head"] = f1
    
    p, r, f1 = calc_prf(agg["tp_vad"], agg["fp_vad"], agg["fn_vad"])
    res["p_vad"] = p
    res["r_vad"] = r
    res["f1_vad"] = f1

    # Row-normalized confusion matrix: cm_<true>_<pred> = P(pred=j | true=i)
    for i in range(4):
        row = sum(agg[f"cm_{i}_{j}"] for j in range(4))
        for j in range(4):
            res[f"cm_{i}_{j}"] = agg[f"cm_{i}_{j}"] / max(row, 1)

    return res


def format_confusion(res: Dict[str, float]) -> str:
    """Pretty row-normalized confusion matrix from aggregate_metrics output."""
    lines = ["confusion P(pred|true):  pred0   pred1   pred2   pred3"]
    for i in range(4):
        row = "  ".join(f"{res.get(f'cm_{i}_{j}', 0.0):6.3f}" for j in range(4))
        lines.append(f"  true{i}:              {row}")
    return "\n".join(lines)
