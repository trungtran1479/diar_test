import numpy as np
from typing import List, Tuple

def delays_durations_to_frame_counts(
    delays: List[float],
    durations: List[float],
    total_duration: float,
    frame_shift: float = 0.01,
) -> np.ndarray:
    """Returns count[t] in {0,1,2,3} for each frame t.
    CLIPS to 3 for >= 3 simultaneous speakers.
    
    Args:
        delays: per-speaker start times in seconds
        durations: per-speaker durations in seconds
        total_duration: total mixture duration in seconds
        frame_shift: time per frame in seconds (default 0.01s for 100Hz)
        
    Returns:
        np.ndarray of shape (num_frames,) with integer counts
    """
    num_frames = int(round(total_duration / frame_shift))
    counts = np.zeros(num_frames, dtype=np.int32)
    
    for delay, duration in zip(delays, durations):
        start_frame = int(round(delay / frame_shift))
        end_frame = int(round((delay + duration) / frame_shift))
        
        # Ensure we don't go out of bounds
        start_frame = max(0, min(start_frame, num_frames))
        end_frame = max(0, min(end_frame, num_frames))
        
        if start_frame < end_frame:
            counts[start_frame:end_frame] += 1
            
    # Clip to max 3
    counts = np.clip(counts, 0, 3)
    return counts

def align_labels_to_output_len(
    labels_100hz: np.ndarray,
    output_len: int,
    method: str = "majority_vote"
) -> np.ndarray:
    """Downsample labels to match actual encoder output length.
    
    Args:
        labels_100hz: 1D array of labels at 100Hz
        output_len: target length
        method: "majority_vote" or "center_frame"
        
    Returns:
        1D array of shape (output_len,)
    """
    if len(labels_100hz) == output_len:
        return labels_100hz
        
    if output_len == 0:
        return np.zeros(0, dtype=labels_100hz.dtype)
        
    if method == "center_frame":
        # Sample the center of each output frame
        indices = np.linspace(0, len(labels_100hz) - 1, output_len).astype(int)
        return labels_100hz[indices]
        
    elif method == "majority_vote":
        # Group frames and take majority vote
        splits = np.array_split(labels_100hz, output_len)
        aligned = np.zeros(output_len, dtype=labels_100hz.dtype)
        for i, split in enumerate(splits):
            if len(split) > 0:
                counts = np.bincount(split)
                aligned[i] = np.argmax(counts)
        return aligned
    else:
        raise ValueError(f"Unknown alignment method: {method}")

import torch
def align_batch_labels(labels, label_lens, h_lens, method="majority_vote"):
    """
    labels: [B, T_label] (tensor)
    label_lens: [B] (tensor)
    h_lens: [B] (tensor)
    """
    B = labels.size(0)
    max_h_len = h_lens.max().item()
    aligned_labels = torch.full((B, max_h_len), -100, dtype=labels.dtype, device=labels.device)
    
    for i in range(B):
        l_len = label_lens[i].item()
        h_len = h_lens[i].item()
        if l_len == 0 or h_len == 0:
            continue
            
        valid_labels = labels[i, :l_len].cpu().numpy()
        aligned = align_labels_to_output_len(valid_labels, h_len, method=method)
        aligned_labels[i, :h_len] = torch.from_numpy(aligned).to(labels.device)
        
    return aligned_labels

def align_batch_posteriors(post, label_lens, h_lens):
    """Downsample teacher count-posteriors [B, T_label, C] @100 Hz onto the
    encoder's output grid [B, max_h_len, C].

    Unlike hard labels (majority vote), a probability distribution must be
    AVERAGED over each group — taking a mode would throw away exactly the
    soft information KD exists to transfer.
    """
    B, _, C = post.shape
    max_h = int(h_lens.max().item())
    out = torch.zeros(B, max_h, C, device=post.device, dtype=post.dtype)
    for i in range(B):
        l_len = int(label_lens[i].item())
        h_len = int(h_lens[i].item())
        if l_len == 0 or h_len == 0:
            continue
        src = post[i, :l_len]                       # [l_len, C]
        # Group EXACTLY like align_labels_to_output_len's np.array_split, so a
        # posterior frame lands in the same bucket as its hard label. Any other
        # grouping shifts teacher and label against each other at bucket
        # boundaries and silently corrupts the agreement mask.
        base, rem = divmod(l_len, h_len)
        sizes = torch.full((h_len,), base, device=post.device, dtype=torch.long)
        sizes[:rem] += 1
        idx = torch.repeat_interleave(torch.arange(h_len, device=post.device), sizes)
        acc = torch.zeros(h_len, C, device=post.device, dtype=post.dtype)
        cnt = torch.zeros(h_len, 1, device=post.device, dtype=post.dtype)
        acc.index_add_(0, idx, src)
        cnt.index_add_(0, idx, torch.ones(l_len, 1, device=post.device, dtype=post.dtype))
        out[i, :h_len] = acc / cnt.clamp_min(1e-8)
    return out


def sup_speech_intervals(sup: dict, gap_s: float = 0.3):
    """Speech intervals (absolute seconds) for one lhotse supervision dict.
    Prefers word-level alignment (gaps <= gap_s bridged); falls back to the
    full segment span."""
    al = (sup.get("alignment") or {}).get("word") or []
    words = [(float(it[1]), float(it[1]) + float(it[2])) for it in al if float(it[2]) > 0]
    if not words:
        return [(sup["start"], sup["start"] + sup["duration"])]
    words.sort()
    merged = [list(words[0])]
    for a, b in words[1:]:
        if a - merged[-1][1] <= gap_s:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def sups_to_frame_counts(sups, duration: float, frame_hz: int = 100) -> np.ndarray:
    """lhotse supervisions -> frame counts @frame_hz, clipped to 3.
    Per-speaker activity union first, so overlapping supervisions from the
    SAME speaker are not double counted."""
    nf = int(round(duration * frame_hz))
    per_spk = {}
    for sup in sups:
        m = per_spk.setdefault(sup["speaker"], np.zeros(nf, dtype=bool))
        for a, b in sup_speech_intervals(sup):
            i, j = max(0, int(round(a * frame_hz))), min(nf, int(round(b * frame_hz)))
            if i < j:
                m[i:j] = True
    counts = np.zeros(nf, dtype=np.int32)
    for m in per_spk.values():
        counts += m.astype(np.int32)
    return np.clip(counts, 0, 3)


def rttm_to_frame_counts(
    rttm_path: str,
    total_duration: float,
    frame_shift: float = 0.01
) -> np.ndarray:
    """Parse RTTM -> frame-level count.
    
    Format: SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
    """
    num_frames = int(round(total_duration / frame_shift))
    counts = np.zeros(num_frames, dtype=np.int32)
    
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == "SPEAKER":
                start_time = float(parts[3])
                duration = float(parts[4])
                
                start_frame = int(round(start_time / frame_shift))
                end_frame = int(round((start_time + duration) / frame_shift))
                
                start_frame = max(0, min(start_frame, num_frames))
                end_frame = max(0, min(end_frame, num_frames))
                
                if start_frame < end_frame:
                    counts[start_frame:end_frame] += 1
                    
    counts = np.clip(counts, 0, 3)
    return counts
