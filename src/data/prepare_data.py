import json
import os
import random
import gzip
import argparse
import numpy as np
import torch
import torchaudio
from collections import Counter
from typing import Tuple, List

from src.data.label_utils import delays_durations_to_frame_counts

def get_total_duration(manifest_entry: dict, audio_path: str) -> float:
    try:
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except Exception:
        pass
    if "duration" in manifest_entry:
        return manifest_entry["duration"]
    return max(d + dl for d, dl in zip(
        manifest_entry.get("delays", [0.0]), manifest_entry.get("durations", [0.0])))

def generate_silence_noise_samples(
    musan_manifest: str,
    num_samples: int,
    output_dir: str,
    duration_range: Tuple[float, float] = (2.0, 15.0)
) -> List[dict]:
    os.makedirs(output_dir, exist_ok=True)
    samples = []
    
    noise_files = []
    if os.path.exists(musan_manifest):
        try:
            open_func = gzip.open if musan_manifest.endswith(".gz") else open
            with open_func(musan_manifest, "rt") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if "audio_filepath" in entry:
                            noise_files.append(entry["audio_filepath"])
                    except:
                        pass
        except Exception as e:
            print(f"Warning: Could not read MUSAN manifest: {e}")
            
    for i in range(num_samples):
        duration = random.uniform(duration_range[0], duration_range[1])
        item_id = f"noise_silence_{i:06d}"
        
        is_noise = (len(noise_files) > 0) and (random.random() < 0.7)
        audio_path = os.path.join(output_dir, f"{item_id}.wav")
        num_frames = int(duration * 16000)
        
        if is_noise:
            # Extract random chunk from random noise file
            noise_file = random.choice(noise_files)
            try:
                info = torchaudio.info(noise_file)
                if info.num_frames > num_frames:
                    start_frame = random.randint(0, info.num_frames - num_frames)
                    waveform, sr = torchaudio.load(noise_file, frame_offset=start_frame, num_frames=num_frames)
                else:
                    waveform, sr = torchaudio.load(noise_file)
                    # pad if too short
                    if waveform.shape[1] < num_frames:
                        pad = torch.zeros(waveform.shape[0], num_frames - waveform.shape[1])
                        waveform = torch.cat([waveform, pad], dim=1)
                
                if sr != 16000:
                    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                torchaudio.save(audio_path, waveform, 16000)
            except Exception as e:
                print(f"Failed to load noise {noise_file}: {e}")
                silence = torch.zeros(1, num_frames)
                torchaudio.save(audio_path, silence, 16000)
        else:
            silence = torch.zeros(1, num_frames)
            torchaudio.save(audio_path, silence, 16000)
            
        labels = np.zeros(int(round(duration * 100)), dtype=np.int32)
        label_path = os.path.join(output_dir, f"{item_id}_label.npy")
        np.save(label_path, labels)
        
        samples.append({
            "id": item_id,
            "audio_filepath": audio_path,
            "duration": duration,
            "label_filepath": label_path,
            "n_spk": 0,
            "source": "silence_noise",
            "sample_rate": 16000
        })
        
    return samples

def compute_class_weights(manifest_path: str, stats_file: str) -> dict:
    c = Counter()
    total_frames = 0
    
    with open(manifest_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if "label_filepath" in item and os.path.exists(item["label_filepath"]):
                labels = np.load(item["label_filepath"])
                counts = np.bincount(labels, minlength=4)
                for i in range(4):
                    c[i] += counts[i]
                total_frames += len(labels)
                
    weights = [1.0] * 4
    if total_frames > 0:
        inv_freqs = [total_frames / max(c[i], 1) for i in range(4)]
        mean_inv_freq = sum(inv_freqs) / 4.0
        # Clamp maximum weight to 20.0
        weights = [min(f / mean_inv_freq, 20.0) for f in inv_freqs]
        
    for i in range(4):
        if c[i] == 0:
            print(f"WARNING: Class {i} has 0 frames in the dataset! Model cannot learn this class.")
            
    stats = {
        "frame_counts": {0: int(c[0]), 1: int(c[1]), 2: int(c[2]), 3: int(c[3])},
        "class_weights": weights,
        "total_frames": total_frames
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
        
    print(f"Stats computed. Class weights: {weights}")
    print(f"Frame distribution: 0:{c[0]}, 1:{c[1]}, 2:{c[2]}, 3+:{c[3]}")
    return stats

def prepare_lsmix_dataset(
    lsmix_manifest: str, 
    labels_dir: str,
    lsmix_dir: str
) -> List[dict]:
    os.makedirs(labels_dir, exist_ok=True)
    out_items = []
    
    with open(lsmix_manifest, 'r') as f:
        for line in f:
            item = json.loads(line)
            audio_rel = item.get("mixed_wav", "")
            audio_abs = os.path.join(lsmix_dir, audio_rel)
            
            if not os.path.exists(audio_abs):
                continue
                
            total_dur = get_total_duration(item, audio_abs)
            
            labels = delays_durations_to_frame_counts(
                item.get("delays", []),
                item.get("durations", []),
                total_dur
            )
            
            item_id = item.get("id", "unknown")
            label_path = os.path.join(labels_dir, f"{item_id}.npy")
            np.save(label_path, labels)
            
            out_items.append({
                "id": item_id,
                "audio_filepath": audio_abs,
                "duration": total_dur,
                "label_filepath": label_path,
                "n_spk": item.get("n_spk", 0),
                "source": "librispeechmix_300h",
                "sample_rate": 16000
            })
            
    return out_items

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets and labels for ZipCount")
    parser.add_argument("--lsmix-manifest", type=str, required=True, help="Input LibriSpeechMix manifest")
    parser.add_argument("--lsmix-dir", type=str, required=True, help="Base directory for LibriSpeechMix audio")
    parser.add_argument("--musan-manifest", type=str, default="", help="MUSAN noise manifest for class 0")
    parser.add_argument("--out-train", type=str, required=True, help="Output train manifest")
    parser.add_argument("--out-val", type=str, required=True, help="Output val manifest")
    parser.add_argument("--out-labels-dir", type=str, required=True, help="Directory to save .npy labels")
    parser.add_argument("--stats-file", type=str, required=True, help="Output stats.json")
    parser.add_argument("--num-noise", type=int, default=5000, help="Number of pure noise/silence samples")
    parser.add_argument("--val-split", type=float, default=0.05, help="Fraction of data for validation")
    args = parser.parse_args()

    print("Processing LSMix...")
    items = prepare_lsmix_dataset(args.lsmix_manifest, args.out_labels_dir, args.lsmix_dir)
    print(f"Processed {len(items)} mixtures.")

    if args.num_noise > 0:
        print(f"Generating {args.num_noise} silence/noise samples...")
        noise_items = generate_silence_noise_samples(
            args.musan_manifest, args.num_noise, os.path.join(args.out_labels_dir, "noise")
        )
        items.extend(noise_items)

    # Split (Random for now, ideally speaker-disjoint)
    print("WARNING: Splitting randomly. This is not speaker-disjoint and may leak speakers between train and val!")
    random.shuffle(items)
    val_count = int(len(items) * args.val_split)
    val_items = items[:val_count]
    train_items = items[val_count:]
    
    print(f"Saving manifests (Train: {len(train_items)}, Val: {len(val_items)})...")
    with open(args.out_train, 'w') as f:
        for it in train_items:
            f.write(json.dumps(it) + "\n")
    with open(args.out_val, 'w') as f:
        for it in val_items:
            f.write(json.dumps(it) + "\n")

    print("Computing class weights from training set...")
    compute_class_weights(args.out_train, args.stats_file)
    print("Done.")

if __name__ == "__main__":
    main()
