import argparse
import yaml
import torch
import torchaudio
from src.models.zipcount_v1 import build_model
from src.data.feature_extractor import feature_extractor_for_config, load_audio_mono_16k

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--audio", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build Model
    model = build_model(config)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(device)
    model.eval()
    
    # Load audio (soundfile-based; torchaudio.load is broken in this venv)
    waveform = load_audio_mono_16k(args.audio)  # [1, T]


    extract_features = feature_extractor_for_config(config)
    features = extract_features(waveform, sample_rate=16000).unsqueeze(0).to(device)
    feature_lens = torch.tensor([features.size(1)], dtype=torch.long).to(device)

    audio_duration = waveform.shape[1] / 16000
    print(f"Processing audio: {args.audio} ({audio_duration:.2f}s)")

    use_amp = config["training"].get("use_amp", True) and device.type == "cuda"
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp):
            h, h_lens = model.backbone.forward_features(features, feature_lens)
            head_out = model.head(h)
            count_logits, vad_logits, overlap_logits = head_out[:3]
            
            probs = torch.softmax(count_logits, dim=-1)[0] # [T, 4]
            preds = torch.argmax(probs, dim=-1) # [T]
            
    print("\n--- Predictions ---")
    print("time(s)\tp0\tp1\tp2\tp3+\tpred")
    
    # T_enc is roughly T_fbank / 4, so each output frame is ~40ms
    # Downsample labels if needed
    h_lens = h_lens.cpu()
    
    # Calculate exact frame shift from audio duration and output length
    frame_shift_s = audio_duration / h_lens[0].item()
    print(f"Calculated frame shift: {frame_shift_s:.4f}s") 
    
    for t in range(h_lens[0].item()):
        time_s = t * frame_shift_s
        p0, p1, p2, p3 = probs[t].tolist()
        pred = preds[t].item()
        print(f"{time_s:.2f}\t{p0:.2f}\t{p1:.2f}\t{p2:.2f}\t{p3:.2f}\t{pred}")

if __name__ == "__main__":
    main()
