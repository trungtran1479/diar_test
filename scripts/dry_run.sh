#!/bin/bash
set -e

echo "=== ZipCount v0.5 Dry-Run End-to-End Test ==="

export PYTHONPATH=.

# 1. Create mock data
echo "Creating mock data..."
mkdir -p data/labels_mock data/mock_wav

${PYTHON:-python} -c "
import numpy as np
import soundfile as sf
import os

# Create 14.42s audio (16kHz)
duration = 14.42
sr = 16000
num_samples = int(duration * sr)
audio = np.zeros(num_samples, dtype=np.float32)
sf.write('data/mock_wav/train-2mix-0.wav', audio, sr)

# Create 14.42s label (1442 frames at 100Hz)
labels = np.zeros(int(duration * 100), dtype=np.int32)
np.save('data/labels_mock/train-2mix-0.npy', labels)
"

# Create mock manifests
echo '{"id": "train-2mix-0", "audio_filepath": "data/mock_wav/train-2mix-0.wav", "duration": 14.42, "label_filepath": "data/labels_mock/train-2mix-0.npy"}' > data/train_manifest.json
cp data/train_manifest.json data/val_manifest.json

# Create mock stats
echo '{"frame_counts": {"0": 1000, "1": 2000, "2": 1500, "3": 500}, "class_weights": [1.25, 0.625, 0.833, 2.5], "total_frames": 5000}' > data/stats.json

# Create mock config
${PYTHON:-python} -c "
import yaml
with open('configs/zipcount_v1.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['training']['batch_size'] = 1
cfg['model']['encoder']['checkpoint'] = ''
cfg['model']['encoder']['type'] = 'mock'
# Dry-run must not depend on the real librimix data
cfg['data']['train_manifest'] = 'data/train_manifest.json'
cfg['data']['val_manifest'] = 'data/val_manifest.json'
cfg['data']['stats_file'] = 'data/stats.json'
with open('configs/zipcount_v1_mock.yaml', 'w') as f:
    yaml.dump(cfg, f)
"

# 2. Run Train (mock encoder)
echo "Running train (Mock Encoder)..."
${PYTHON:-python} src/train.py --config configs/zipcount_v1_mock.yaml --max-steps 2 --encoder-type mock

# 3. Check checkpoints
echo "Verifying checkpoints..."
if [ ! -f "logs/zipcount/last.pt" ]; then
    echo "ERROR: last.pt not found!"
    exit 1
fi
if [ ! -f "logs/zipcount/best_macro_f1.pt" ]; then
    echo "ERROR: best_macro_f1.pt not found!"
    exit 1
fi

# 4. Run Eval
echo "Running eval..."
${PYTHON:-python} src/eval.py --config configs/zipcount_v1_mock.yaml --checkpoint logs/zipcount/best_macro_f1.pt

# 5. Optional: Test Zipformer Wrapper load if real checkpoint exists
ZIP_DIR="/home/pc/icefall/egs/librispeech/ASR/pruned_transducer_stateless7_streaming"
if [ -d "$ZIP_DIR" ]; then
    echo "Testing real Zipformer load..."
    ${PYTHON:-python} -c "
import yaml
from src.models.zipformer_wrapper import StreamingZipformerEncoder
with open('configs/zipcount_v1_mock.yaml', 'r') as f:
    cfg = yaml.safe_load(f)['model']['encoder']
try:
    enc = StreamingZipformerEncoder(zipformer_dir=cfg['zipformer_dir'], checkpoint_path=cfg['checkpoint'], strict_load=False, min_load_ratio=0.90)
    print('SUCCESS: Real Zipformer loaded properly.')
except Exception as e:
    print('WARNING: Real Zipformer load failed:', e)
"
else
    echo "Skipping real Zipformer test (directory not found). Ensure config has correct zipformer_dir."
fi

echo "=== Dry-Run Completed Successfully ==="
