"""Chunk-by-chunk streaming inference with cached Zipformer states.

Mirrors icefall zipformer/streaming_decode.py: each step consumes
2*chunk_size fbank frames (100 Hz) plus a pad_length lookahead tail, and
emits chunk_size/2 encoder frames (25 Hz) through the count/vad/overlap
heads. States (attention KV caches, conv caches, ConvNeXt left pad) are
carried across chunks, so memory and latency are bounded regardless of
audio length.

Usage:
    PYTHONPATH=. python src/infer_streaming.py \
        --config configs/zipcount_v1.yaml \
        --checkpoint logs/zipcount/best_macro_f1.pt \
        --audio path/to.wav [--compare-offline]
"""
import argparse
import math
import yaml
import torch
import torchaudio

from src.models.zipcount_v1 import build_model
from src.data.feature_extractor import extract_fbank_features, load_audio_mono_16k

LOG_EPS = math.log(1e-10)  # padding value icefall uses for fbank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None, help="ZipCount checkpoint (head weights)")
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--compare-offline", action="store_true",
                        help="Also run the offline forward and report frame agreement")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if config["model"]["encoder"].get("type") != "zipformer":
        raise ValueError("infer_streaming.py requires encoder type 'zipformer' (real backbone with states).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(device)
    model.eval()

    backbone = model.backbone
    if not backbone.is_causal:
        raise ValueError("Encoder was built with causal=False; set model.encoder.causal: true in the config.")

    # Load audio -> mono 16 kHz (soundfile-based)
    waveform = load_audio_mono_16k(args.audio)  # [1, T]
    audio_duration = waveform.shape[1] / 16000

    features = extract_fbank_features(waveform, sample_rate=16000)  # [T, 80]
    T = features.size(0)

    chunk_size = backbone.chunk_size          # encoder-embed frames (50 Hz)
    pad_length = backbone.pad_length          # 13 fbank lookahead frames
    hop = 2 * chunk_size                      # fbank frames (100 Hz) per step
    win = hop + pad_length                    # fbank frames fed per step

    print(f"Audio: {args.audio} ({audio_duration:.2f}s, {T} fbank frames)")
    print(f"Streaming: chunk={chunk_size} (50Hz) -> {hop} fbank frames/step, "
          f"lookahead tail={pad_length} frames ({pad_length * 10} ms), "
          f"left_context={backbone.left_context_len} frames")

    # Tail padding so the last (partial) chunk still yields chunk_size embed frames
    features = torch.nn.functional.pad(
        features, (0, 0, 0, pad_length + hop), mode="constant", value=LOG_EPS
    ).to(device)  # [T + pad_length + hop, 80]

    states = backbone.get_init_states(batch_size=1, device=device)
    head_caches = None  # opaque temporal/count state for any stateful head

    all_probs = []   # each [chunk_size//2, 4]
    num_steps = math.ceil(T / hop)

    with torch.no_grad():
        for step in range(num_steps):
            start = step * hop
            chunk = features[start:start + win].unsqueeze(0)  # [1, win, 80]

            h, h_lens, states = backbone.forward_streaming(chunk, states)  # h: [1, chunk//2, D]
            if hasattr(model.head, "forward_streaming"):
                head_out, head_caches = model.head.forward_streaming(h, head_caches)
                count_logits, vad_logits, overlap_logits = head_out[:3]
            else:
                head_out = model.head(h)
                count_logits, vad_logits, overlap_logits = head_out[:3]
            all_probs.append(torch.softmax(count_logits[0], dim=-1))       # [chunk//2, 4]

    probs = torch.cat(all_probs, dim=0)  # [num_steps * chunk//2, 4] at 25 Hz
    # Trim padding tail: true number of 25 Hz frames for T fbank frames
    n_out = max(1, T // 4)
    probs = probs[:n_out]
    preds = probs.argmax(dim=-1)  # [n_out]

    # Print merged constant-count segments
    print("\n--- Streaming predictions (25 Hz frames, 40 ms each) ---")
    print("start(s)\tend(s)\tcount\tmean_conf")
    seg_start = 0
    for t in range(1, n_out + 1):
        if t == n_out or preds[t] != preds[seg_start]:
            seg = probs[seg_start:t]
            conf = seg.max(dim=-1).values.mean().item()
            print(f"{seg_start * 0.04:.2f}\t{t * 0.04:.2f}\t{preds[seg_start].item()}\t{conf:.2f}")
            seg_start = t

    if args.compare_offline:
        feats_off = extract_fbank_features(waveform, sample_rate=16000).unsqueeze(0).to(device)
        lens_off = torch.tensor([feats_off.size(1)], dtype=torch.long, device=device)
        with torch.no_grad():
            h_off, h_lens_off = backbone.forward_features(feats_off, lens_off)
            logits_off = model.head(h_off)[0]
            preds_off = logits_off[0, :h_lens_off[0]].argmax(dim=-1)
        n = min(preds_off.size(0), preds.size(0))
        agree = (preds_off[:n] == preds[:n]).float().mean().item()
        print(f"\nOffline-vs-streaming frame agreement: {agree:.1%} over {n} frames")


if __name__ == "__main__":
    main()
