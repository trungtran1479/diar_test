# ZipCount — Streaming Frame-Level Active Speaker Counting

> **Tổng hợp kiến trúc và kết quả toàn dự án:** xem [README_MODELS_RESULTS.md](README_MODELS_RESULTS.md), bao gồm v1–v4, TCN, event-state, pyramid, FastConformer, WavLM/LoRA, các ablation và kết quả từng model. Phần dưới giữ nội dung hướng dẫn và báo cáo lịch sử ban đầu.

Đếm số người đang nói tại **mỗi frame** (40ms) trong audio streaming 1 kênh:

```
0 = non-speech / silence
1 = 1 người nói
2 = 2 người nói
3 = 3+ người nói
```

Không phải diarization (không speaker identity), không phải ASR. Backbone là
**Zipformer2 streaming ASR encoder (frozen)** của icefall; chỉ train phần head.
Mục tiêu cuối: tín hiệu VAD/overlap/count độ trễ thấp để route multi-talker ASR.

---

## 1. Trạng thái hiện tại (đã verify 2026-07-07)

| Hạng mục | Trạng thái |
|---|---|
| Mock dry-run end-to-end (train → checkpoint → eval) | ✅ pass |
| Load backbone streaming thật | ✅ **100.0%** params, causal=True, chunk 32, left-context 128 |
| Train frozen backbone (chỉ head có gradient) | ✅ v1: 532K params, v2: 1.32M params |
| Streaming vs offline (v1 head) | ✅ 99.5% frame agreement |
| Streaming vs offline (v2: hypercolumn + deformable cache) | ✅ 99.2% frame agreement |
| Hồi quy loss v1 sau khi thêm loss v2 | ✅ identical bit-for-bit |
| **Dữ liệu train thật (LibriMix đã làm sạch)** | ✅ **125,900 train / 6,399 val items** (`data/librimix/`, mục 4) |
| Train v2 trên data thật (backbone thật, mix on-the-fly) | ✅ pass (2026-07-11) |

---

## 2. Kiến trúc & các variant

### Variant matrix (đây chính là bảng ablation của paper)

| Variant | Config | Backbone | Head | Ý nghĩa |
|---|---|---|---|---|
| **A** | v1 + `--encoder-type crnn` | CRNN causal (train từ đầu) | linear | Baseline sàn |
| **B** | v1, `head.type: linear` | Zipformer frozen | linear | Chứng minh hidden ASR đủ tốt |
| **C** | v1, `head.type: temporal_adaptive` | Zipformer frozen | causal conv adapter k=5 | Baseline adapter cố định |
| **D** | v1, `stage2` + `unfreeze_last_n: 2` | Zipformer mở 2 stack cuối | như C | Fine-tune một phần |
| **E** | **v2** (`zipcount_v2.yaml`) | Zipformer frozen + **hypercolumn** | **deformable + ordinal** | **Contribution chính** |

### Variant E (v2) — CV-inspired

```
Audio 16kHz → fbank 80 chiều (100Hz)
  ↓
Zipformer2 streaming FROZEN (U-Net 6 stack, downsampling 1,2,4,8,4,2)
  ↓ lấy output CẢ 6 STACK (hypercolumn, dim 192+256+384+512+384+256 = 1984, 25Hz)
StackGatedProjection  ← FPN-lite: gate softmax học được cho từng scale
  ↓ [B, T, 256]
CausalDeformableTemporalBlock ×2  ← InternImage/DCNv3 chuyển sang 1D:
  ↓                                  mỗi frame TỰ dự đoán nhìn về đâu trong quá khứ
  ↓                                  (K=4 điểm × G=4 nhóm, offset ∈ (-640ms, 0], nội suy tuyến tính)
OrdinalConsistentHead
  ├── count logits [B,T,4]   (softmax, kiểu powerset của pyannote)
  └── cumulative [B,T,2]     (P(≥1)=VAD, P(≥2)=overlap — kiểu CORAL ordinal)
```

Căn cứ nghiên cứu: powerset > multilabel cho overlap ([Interspeech 2023](https://arxiv.org/html/2310.13025));
count là biến ordinal ([CORN](https://arxiv.org/abs/2111.08851)); dice cho lớp hiếm
([EEND-M2F](https://arxiv.org/abs/2401.12600)); deformable 1D đã có ở SED/separation
([1D-DETR](https://arxiv.org/pdf/2605.03934), [DTCN](https://arxiv.org/pdf/2210.15305))
nhưng **chưa ai làm causal-streaming cho speaker counting** — đây là ngách novelty.

### Loss v2 (mọi thành phần bật/tắt qua config, default-off = giữ nguyên v1)

| Key trong `loss:` | Ý nghĩa | Giá trị v2 |
|---|---|---|
| `count_loss_type` | `ce` / `focal` / `sord` (soft ordinal: sai 1↔3 phạt nặng hơn 1↔2) | `sord` |
| `sord_alpha` | Độ sắc của soft target (nhỏ = mềm hơn) | 1.5 |
| `lambda_vad` / `lambda_overlap` | BCE trên 2 kênh cumulative | 0.3 / 0.5 |
| `lambda_emae` | \|E[count] − y\| — regression ordinal-aware | 0.2 |
| `lambda_consistency` | Buộc softmax-derived P(≥1),P(≥2) khớp 2 kênh cumulative | 0.2 |
| `lambda_monotonic` | Phạt nếu P(≥2) > P(≥1) | 0.1 |
| `lambda_dice` | Dice trên kênh overlap (lớp hiếm) | 0.5 |
| `lambda_smooth` | KL smoothness giữa frame liền kề | 0.02 |

---

## 3. Môi trường & backbone

```bash
source /home/pc/venv/bin/activate
cd /home/pc/diar_new
export PYTHONPATH=.
```

**Backbone — 3 điều đã verify, ĐỪNG đổi nếu không hiểu:**

1. `zipformer_dir` **phải** là recipe icefall chuẩn: `/home/pc/icefall/egs/librispeech/ASR/zipformer`.
   KHÔNG dùng `/home/pc/v2t/zipformer` (recipe hybrid HuBERT nhận waveform, không tương thích).
2. Checkpoint **phải** là Zipformer2 train `--causal=True`:
   `/home/pc/v2t/models/icefall-asr-librispeech-streaming-zipformer-2023-05-17/exp/pretrained.pt` (load 100%).
   - `epoch-35-avg-6.pt`: Zipformer2 nhưng **non-causal** → thiếu 160 tensor causal-conv (98.8%), cấm dùng cho streaming.
   - `stateless7-streaming/pretrained.pt`: Zipformer **v1** → không load được.
3. `causal: true`, `chunk_size: "32"`, `left_context_frames: "128"` trong config **phải giữ** —
   thiếu chúng recipe build encoder full-context (default `causal=False`), kết quả offline đẹp giả tạo.

---

## 4. Dữ liệu — LibriMix lhotse cutsets (ĐÃ CÓ SCRIPT LÀM SẠCH)

### 4.1. Nguồn: 4 cutset lhotse MixedCut (mix on-the-fly, không có wav trộn sẵn)

`/home/pc/v2t/data/en/mtasr_prep/manifests/librimix/`:
libri2mix/libri3mix × train-clean-100/360 (`*_30s.jsonl.gz`), val = `*_dev-clean.jsonl.gz`.

Kết quả kiểm tra (2026-07-10): **107.9k cuts / ~443h**, 0 duplicate ID, 0 anomaly,
0 file thiếu, **speaker train↔dev disjoint hoàn toàn** (val chuẩn paper).
Nhưng label thô rất bẩn: **0% frame class 0**, overlap áp đảo (2mix: 77% class 2;
3mix: 66% class 3), khoảng lặng trong câu bị dán nhãn "đang nói".

### 4.2. Làm sạch + sinh label (một lệnh, chạy 1 lần)

```bash
PYTHONPATH=. python src/data/prepare_from_lhotse.py \
    --out-dir data/librimix \
    --musan-manifest /home/pc/v2t/data/musan_lhotse/musan_recordings_noise.jsonl.gz \
    --num-solo 12000 --num-noise 6000 --workers 16
# thêm --limit 40 để smoke test nhanh
```

Script làm gì:
1. **Energy VAD trên TỪNG NGUỒN** (mỗi track là 1 flac riêng → VAD chính xác tuyệt đối,
   không cần separation): rel −35dB so peak, floor −55dBFS, nối gap <200ms, bỏ đảo <100ms.
   Kết quả cache ở `data/librimix/vad_cache.json.gz` — chạy lại là tức thì.
2. Label = tổng mask VAD các track theo offset, clip 3, lưu `.npy` 100Hz.
3. Cân bằng class: `--num-solo` item 1-speaker (lấy từ chính các source),
   `--num-noise` item MUSAN/silence (class 0). Val thêm `--val-noise` item noise.
4. Train = 4 cutset train; **val = dev-clean cutsets (speaker-disjoint sẵn)**.
5. Tính class weights → `data/librimix/stats.json`.

**Kết quả full run (2026-07-11, ~500h label / 180M frames):**
- VAD 131,927 nguồn duy nhất, coverage mean 0.863 (p5 0.69, p95 1.0) — đúng kỳ vọng LibriSpeech.
- Train: 107,900 mixtures + 12,000 solo + 6,000 noise = **125,900 items**,
  dist `{0: 8.5%, 1: 33.6%, 2: 42.5%, 3+: 15.5%}` (trước làm sạch: `{0: 0%, 1: 23%, 2: 77%}`).
- Val (dev-clean, speaker-disjoint): 5,999 mixtures + 400 noise = **6,399 items**,
  dist `{0: 13.9%, 1: 37.8%, 2: 32.7%, 3+: 15.7%}`.
- Class weights: `[2.00, 0.50, 0.40, 1.09]` → `data/librimix/stats.json`.
- Config v1/v2 đã trỏ sẵn vào data này — chạy train được ngay (mục 5.2).

### 4.3. Trỏ config vào data mới

Sửa `data:` trong config (v1 và v2):
```yaml
data:
  train_manifest: "/home/pc/diar_new/data/librimix/train_manifest.json"
  val_manifest:   "/home/pc/diar_new/data/librimix/val_manifest.json"
  stats_file:     "/home/pc/diar_new/data/librimix/stats.json"
```

### 4.4. Format manifest (tự cắm data khác cũng được)

Mỗi dòng 1 JSON. Hai kiểu item:
```json
{"id": "x", "duration": 15.5, "n_spk": 2, "label_filepath": "/abs/label.npy",
 "tracks": [{"source": "/abs/spk1.flac", "volume": 0.65, "offset": 0.0, "duration": 15.5}, ...]}
```
→ `SpeakerCountDataset` tự mix on-the-fly (đọc từng source, nhân volume, cộng theo offset,
chống clip bằng peak-normalize). Hoặc kiểu cũ có sẵn wav:
```json
{"id": "x", "audio_filepath": "/abs/path.wav", "duration": 12.3, "label_filepath": "/abs/label.npy"}
```
`label.npy`: mảng int 1 chiều, 100Hz, giá trị 0–3.

**Ghi chú:** volume factor trong cutset trải 0.07→9.5 (~39dB) — track bị nén sâu vẫn
được dán nhãn "đang nói"; đây là thiết kế của data mtasr, chấp nhận (làm task khó hơn
một cách trung thực). Script prep cũ `prepare_data.py` (format delays/durations) vẫn giữ
cho tương thích nhưng **không dùng** cho LibriMix.

---

## 5. Train

### 5.1. Dry-run (kiểm tra pipeline còn sống, ~1 phút, không cần data)

```bash
bash scripts/dry_run.sh
```

### 5.2. Các lệnh train theo variant

```bash
# Variant A — CRNN baseline (sửa training.log_dir hoặc để mặc định logs/zipcount)
PYTHONPATH=. python src/train.py --config configs/zipcount_v1.yaml --encoder-type crnn

# Variant B — frozen Zipformer + linear head
#   (sửa configs/zipcount_v1.yaml: head.type: "linear")
PYTHONPATH=. python src/train.py --config configs/zipcount_v1.yaml

# Variant C — frozen Zipformer + temporal adaptive head
#   (head.type: "temporal_adaptive" — hiện là mặc định trong v1 yaml)
PYTHONPATH=. python src/train.py --config configs/zipcount_v1.yaml

# Variant D — mở 2 stack cuối (chạy SAU khi có checkpoint C tốt)
#   sửa v1 yaml: stage: "stage2_unfreeze_last_stacks", unfreeze_last_n: 2, lr: 1e-4
PYTHONPATH=. python src/train.py --config configs/zipcount_v1.yaml

# Variant E — v2 CV-inspired (contribution chính)
PYTHONPATH=. python src/train.py --config configs/zipcount_v2.yaml
```

Mẹo: `--max-steps N` override nhanh để chạy thử. Mỗi experiment nên đặt
`training.log_dir` riêng (v2 đã đặt sẵn `logs/zipcount_v2_E`) — checkpoint
`last.pt` / `best_macro_f1.pt` / `best_overlap_f1.pt` lưu vào đó.

### 5.3. Theo dõi

```bash
tensorboard --logdir logs/
```
- `Loss/*`: tất cả thành phần loss (count, vad, overlap, emae, consistency, monotonic, dice, smooth).
- `Val/*`: acc, mae, macro_f1, f1 từng class, f1_ov_count, f1_ov_head, f1_vad.
- `FusionGate/stack0..5` (chỉ v2): **gate của scale nào cao = thông tin speaker nằm ở tầng
  nén đó của U-Net** — đây là finding cho paper, để ý xem nó dồn về stack nào.

### 5.3b. Loạt run chẩn đoán class-2 (kết quả run đầu 2026-07-11)

Run đầu v1-C (weights auto) cho: VAD F1 0.965, overlap F1 0.82 nhưng **recall class 2
= 0.015** — model đoán 3 cho mọi frame overlap (r_3=0.963, p_3=0.297). Nguyên nhân
chính nghi là weights auto `[2.0, 0.5, 0.4, 1.09]` phạt sai class 2 rẻ hơn class 3
2.7 lần. Config đã đổi mặc định sang weights đều. Loạt run kiểm chứng:

```bash
# Run 1 — v1, weights đều, focal (config mặc định hiện tại)
PYTHONPATH=. python src/train.py --config configs/zipcount_v1.yaml --log-dir logs/r1_v1_uniform_focal
# Run 2 — v1, weights đều, CE thuần (tách tác dụng focal)
PYTHONPATH=. python src/train.py --config configs/zipcount_v1.yaml --count-loss-type ce --log-dir logs/r2_v1_uniform_ce
# Run 3 — v2, weights đều, sord+emae (đóng góp ordinal thật)
PYTHONPATH=. python src/train.py --config configs/zipcount_v2.yaml --log-dir logs/r3_v2_uniform_sord
# Quay lại weights auto nếu cần đối chứng:  --class-weights auto
```

Mỗi lần validation giờ in kèm **confusion matrix 4×4** (P(pred|true)) ra log +
TensorBoard (`Val/cm_i_j`) — nhìn hàng true2 để biết class 2 đi đâu.

**KẾT QUẢ Run 1 (50k steps, 2026-07-11, checkpoint: `r1_v1_uniform_focal/`):**
✅ **Giả thuyết weights xác nhận** — recall class 2: 0.015 → **0.694**.

| Metric | weights auto (run cũ @4k) | weights đều (Run 1 @50k) |
|---|---|---|
| acc / MAE | 0.469 / 0.661 | **0.687 / 0.321** |
| macro F1 | 0.441 | **0.688** |
| f1 class 0/1/2/3 | 0.69/0.59/0.03/0.45 | **0.79/0.74/0.64/0.58** |
| overlap F1 / VAD F1 | 0.823 / 0.965 | **0.851 / 0.970** |

Confusion @50k: sai số CHỈ còn ở class liền kề (true1→2: 0.230, true2→1/3: 0.163/0.140,
**true3→2: 0.442**) — cấu trúc lỗi thuần ordinal, đúng mục tiêu thiết kế của v2 (Run 3).
Lưu ý: một phần true3→2 có thể là trần label-noise (track gain 0.07 không nghe được
vẫn nhãn 3) → mục label refinement.

Demo streaming (`--smooth 11`, majority vote causal 440ms, không thêm latency):
silence sạch tuyệt đối (conf 0.85–1.0), overlap ổn định, nhưng vùng solo dao động 1↔2
với conf ~0.48 — nhất quán với true1→2=0.23. Hướng sửa: v2 consistency loss (VAD head
đang rất chuẩn 0.97, kéo count head theo), tăng `--num-solo`, random gain cho solo items.

**KẾT QUẢ Run 3 — v2/variant E (50k steps, 2026-07-11, checkpoint: `result/r3_v2_uniform_sord/`):**

| Metric | Run 1 (v1-C uniform) | **Run 3 (v2-E)** | Δ |
|---|---|---|---|
| acc / MAE | 0.687 / 0.321 | **0.887 / 0.114** | +20.0 điểm / −64% |
| macro F1 | 0.688 | **0.886** | +19.8 điểm |
| f1 class 0/1/2/3 | 0.79/0.74/0.64/0.58 | **0.92/0.92/0.86/0.85** | mọi class +13–27 |
| overlap F1 / VAD F1 | 0.851 / 0.970 | **0.958 / 0.987** | +10.7 / +1.7 |
| true3→pred2 (undercount) | 0.442 | **0.090** | −80% |
| true1→pred2 (overcount) | 0.230 | **0.054** | −77% |

Cả 2 mục tiêu đặt trước (<0.30 và <0.15) đều vượt xa. Demo streaming: solo ổn định
conf 0.70 (hết nhấp nháy 1↔2), segment dài sạch.

**Finding cho paper — fusion gates học được (đọc từ checkpoint):**
stack0 (50Hz nông nhất): **0.560**, stack1 (25Hz): **0.247** → 81% trọng số nằm ở
2 tầng acoustic nông; stack3 bottleneck 512-dim (linguistic): chỉ 0.037.
→ Thông tin speaker-count nằm ở tầng nông tốc độ cao mà output chuẩn của ASR encoder
(thiên về tầng sâu) làm loãng — giải thích một phần nhảy vọt của hypercolumn, và
khớp với văn liệu probing layer-wise (speaker info ở early layers).

**Caveat trung thực cho paper:** v2 khác v1 đồng thời 3 thứ (hypercolumn + deformable
head + ordinal loss, head 1.32M vs 0.53M params). Cần 2–3 run attribution để tách
đóng góp (xem 5.4b).

### 5.4b. Loạt run attribution (tách đóng góp từng khối — bảng ablation paper)

```bash
# A1: hypercolumn + head v1 (đo riêng multiscale)   — sửa v2 yaml: head.type temporal_adaptive
# A2: v2 head nhưng focal thường (đo riêng ordinal) — --count-loss-type focal (giữ nguyên phần còn lại v2)
# A3: v1 (output chuẩn) + sord (đo riêng loss)      — v1 yaml + --count-loss-type sord
```

### 5.4. Thứ tự chạy + gate quyết định

```
1. B (rẻ nhất)      → macro_f1 tệ (<0.5)?  → hidden ASR không đủ, cân nhắc lại backbone
2. C                → C ≤ B?               → adapter cố định vô dụng, càng có lý do cho deformable
3. E                → E ≤ C?               → contribution yếu, tune (sord_alpha, max_offset, num_layers) trước khi kết luận
4. D (đắt)          → chỉ chạy khi E đã chốt, để có hàng "partial fine-tune" trong bảng
5. A                → chạy song song lúc nào cũng được (GPU rảnh)
```

### 5.5. Tuning v2 khi kết quả chưa tốt

| Triệu chứng | Chỉnh |
|---|---|
| Overlap recall thấp | tăng `lambda_dice` (0.5→1.0) trước khi đụng class weights |
| Count hay nhảy 1↔3 | tăng `sord_alpha` hoặc `lambda_emae` |
| VAD và count mâu thuẫn | tăng `lambda_consistency` |
| Biên chuyển đổi trễ | giảm `max_offset` (16→8) hoặc tăng `num_points` |
| Prediction rung | tăng `lambda_smooth` (0.02→0.05, đừng quá — nó làm mờ biên thật) |

---

## 6. Eval

```bash
PYTHONPATH=. python src/eval.py \
    --config configs/zipcount_v2.yaml \
    --checkpoint logs/zipcount_v2_E/best_macro_f1.pt
```
In: acc, MAE, precision/recall/F1 từng class, macro-F1, overlap-F1 (từ count head và
từ overlap head riêng), VAD-F1. So sánh các variant trên **cùng val manifest**.

---

## 7. Inference

```bash
# Offline (cả file một lần, backbone vẫn causal)
PYTHONPATH=. python src/infer_offline.py \
    --config configs/zipcount_v2.yaml \
    --checkpoint logs/zipcount_v2_E/best_macro_f1.pt \
    --audio /path/to.wav

# Streaming thật (chunk 640ms, cached states, latency & memory không phụ thuộc độ dài audio)
PYTHONPATH=. python src/infer_streaming.py \
    --config configs/zipcount_v2.yaml \
    --checkpoint logs/zipcount_v2_E/best_macro_f1.pt \
    --audio /path/to.wav \
    --compare-offline     # in % agreement streaming-vs-offline (sanity, kỳ vọng >99%)
```
`infer_streaming.py` in các segment count không đổi (start, end, count, confidence);
tự mang cache deformable qua chunk nếu head là v2.

---

## 8. Cấu trúc project

```
diar_new/
├── configs/
│   ├── zipcount_v1.yaml           # Variant B/C/D (đổi head.type / stage)
│   └── zipcount_v2.yaml           # Variant E — CV-inspired
├── scripts/dry_run.sh             # Smoke test không cần data
├── data/                          # manifest + labels (mock hiện tại, thay bằng data thật)
├── logs/                          # TensorBoard + checkpoints (logs/_mock_archive = đồ cũ vô nghĩa)
└── src/
    ├── data/
    │   ├── prepare_data.py        # LSMix+MUSAN → manifest + label npy + class weights
    │   ├── dataset.py             # SpeakerCountDataset (soundfile loader)
    │   ├── feature_extractor.py   # fbank 80d + load_audio_mono_16k
    │   └── label_utils.py         # delays/durations → count 100Hz; align về 25Hz
    ├── models/
    │   ├── zipformer_wrapper.py   # Load/freeze Zipformer2; forward_features (+hypercolumn);
    │   │                          #   get_init_states/forward_streaming (cached states)
    │   ├── deformable.py          # CausalDeformableTemporalLayer/Block (v2 core)
    │   ├── heads.py               # Linear / TemporalAdaptive / StackGatedProjection
    │   │                          #   / OrdinalConsistentHead / DeformableCountHead
    │   ├── losses.py              # ZipCountLoss (focal/sord + emae/consistency/monotonic/dice)
    │   ├── crnn_baseline.py       # Variant A
    │   └── zipcount_v1.py         # build_model từ yaml
    ├── train.py                   # Loop train (freeze giữ backbone ở eval mode, warmup, AMP)
    ├── eval.py
    ├── infer_offline.py
    └── infer_streaming.py
```

## 9. Ghi chú kỹ thuật (đỡ dẫm lại vết xe đổ)

- `torchaudio.load` hỏng trong venv này (torchcodec) → mọi chỗ load audio dùng
  `soundfile` qua `load_audio_mono_16k`.
- Backbone frozen được giữ **eval mode** trong lúc train (Zipformer có dropout/whitening
  nội bộ, để train mode sẽ nhiễu feature) — `train.py` tự xử lý.
- Frame rate: fbank 100Hz → encoder output 25Hz (1 frame = 40ms). Label 100Hz được
  majority-vote về 25Hz trong `label_utils.align_batch_labels`.
- Wrapper tự vứt decoder/joiner/CTC của checkpoint ASR — chỉ giữ encoder_embed + encoder.
- Deformable offset bị chặn `≤ 0` (chỉ nhìn quá khứ) → streaming chỉ cần cache
  `max_offset` frame; đã verify agreement 99.2%.
- Checkpoint cũ trong `logs/_mock_archive/` train bằng mock encoder — số liệu vô nghĩa, đừng dùng.

## 10. Baseline & dữ liệu của các model khác (khảo sát 2026-07-11)

| Model | Train data | Có set TRẢ PHÍ? | Weights public? |
|---|---|---|---|
| **ZipCount (mình)** | LibriSpeech (backbone) + LibriMix + MUSAN | ❌ Không — **100% free public** | (của mình) |
| pyannote segmentation-3.0 | AISHELL-4, AliMeeting, AMI, AVA-AVD, **DIHARD** 💰, Ego4D (ký thỏa thuận), MSDWild, **REPERE** 💰(ELDA), VoxConverse | ✅ DIHARD (LDC), REPERE | ✅ MIT |
| Streaming Sortformer (NVIDIA) | **Fisher** 💰, **NIST SRE 2000/04-10** 💰(LDC), LibriSpeech, AMI, VoxConverse, ICSI, AISHELL-4, **DIHARD-3 dev** 💰, AliMeeting, DiPCo (+NOTSOFAR-1 ở v2.1) + 5150h mô phỏng NeMo | ✅ Fisher/SRE/DIHARD | ✅ HF |
| DiariZen (BUT) | AMI-SDM, AISHELL-4, AliMeeting, NOTSOFAR-1, MSDWild, **DIHARD-3** 💰, RAMC, VoxConverse | ✅ DIHARD-3 | ✅ CC BY-NC 4.0 |
| EEND-M2F | Pretrain: 400k mixture mô phỏng từ LibriSpeech (public recipe); finetune: AISHELL-4, AliMeeting, AMI, **CALLHOME** 💰, **DIHARD-III** 💰, RAMC, VoxConverse | ✅ CALLHOME/DIHARD | ❌ không thấy |
| CountNet (count 0–10) | LibriCount (từ LibriSpeech test) | ❌ public (Zenodo) | ✅ GitHub |

**Tính khả dụng từng dataset:** free public: LibriSpeech/LibriMix/MUSAN/AMI/ICSI/AISHELL-4/
AliMeeting (OpenSLR, CC BY-SA)/VoxConverse/AVA-AVD/DiPCo/NOTSOFAR-1/LibriCount;
free nhưng ký thỏa thuận non-commercial: MSDWild, RAMC, Ego4D;
**trả phí**: DIHARD-2/3 (LDC), Fisher (LDC), NIST SRE + CALLHOME (LDC), REPERE (ELDA).

**Kho data local (kiểm kê 2026-07-11, đĩa còn 575GB):**

| Dataset | Vị trí | Dung lượng | Số giờ | Ghi chú |
|---|---|---|---|---|
| AMI-SDM | `mtasr_prep/data/ami/sdm` (symlink → english_benchmark) | 11GB | train 79.4h / dev 9.7h / test 9.1h | ✅ có supervisions (speaker+time) → suy count label được ngay |
| NOTSOFAR-1 SDM | `mtasr_prep/data/nsf` | 50GB | snapshot mới nhất 240825.1: 35.6h (240501.1: 54.3h) | ✅ có supervisions; các snapshot chồng lấn |
| LibriSpeech | `data/en/LibriSpeech` | 60GB | 960h | nguồn backbone + LibriMix |
| LibriMix (data) | `mtasr_prep/data/LibriMix` | 38GB | — | nguồn cutsets đã dùng |
| WHAM noise | `mtasr_prep/data/wham_noise` | 53GB | — | augmentation sẵn có |
| VoxConverse dev/test | `data/en/eval_sets/voxconverse` | ~3.5GB | ~64h | tải 2026-07-11 (wav + RTTM) |
| LibriCount | `data/en/eval_sets/libricount` | ~0.6GB | 8.3h test | benchmark count 0–10, tải 2026-07-11 |

Chưa có (free, chưa tải vì lớn/tiếng Trung): AliMeeting (OpenSLR-119), AISHELL-4 (OpenSLR-111);
cần ký thỏa thuận: MSDWild; trả phí LDC: DIHARD-3, CALLHOME, Fisher.

**Hệ quả cho paper:**
1. **Không baseline lớn nào train được lại bằng data hoàn toàn miễn phí** — mọi hệ đều dính
   ít nhất 1 set LDC. Stack của mình 100% free public → claim **fully reproducible** là
   điểm cộng thật, ghi rõ trong paper.
2. So sánh công bằng = dùng **weights public của họ** (pyannote, Sortformer, DiariZen) chạy
   trên eval set chung miễn phí: LibriMix test/dev (mình có), **AMI** (mọi baseline đều báo
   số), AliMeeting, AISHELL-4, LibriCount (zero-shot count). Suy count[t] từ output diar
   của họ = số speaker active tại frame t.
3. EEND-M2F không có weights công khai → chỉ trích dẫn số trong paper họ (khác data,
   ghi chú rõ không so trực tiếp).
4. Tránh DIHARD-3/CALLHOME làm eval chính trừ khi mua LDC.

## 11. Chuyển sang máy khác

Bundle đã đóng gói sẵn: `zipcount_bundle_20260711.tar.gz` (~1.2GB) gồm:
toàn bộ code + `data/librimix` (manifests + labels + VAD cache — khỏi chạy lại prep 40 phút)
+ `third_party/icefall` (bản icefall chính xác đã verify) + `data/cutsets` (6 cutset gốc,
phòng khi cần sinh lại label với tham số khác).

Trên máy mới (cần ~75GB đĩa cho data + GPU NVIDIA driver mới):

```bash
scp user@may-cu:/home/pc/diar_new/zipcount_bundle_20260711.tar.gz .
tar xzf zipcount_bundle_20260711.tar.gz && cd diar_new
bash scripts/setup_new_machine.sh          # hoặc DATA_ROOT=/mnt/big bash scripts/...
```

Script tự làm 5 bước: (1) venv + torch 2.10 cu128 + k2 wheel khớp + icefall editable;
(2) tải LibriSpeech train-clean-100/360 + dev-clean (~30GB tar) + MUSAN (11GB) +
checkpoint streaming Zipformer2 từ HF (265MB) — wget -c nên đứt mạng chạy lại được;
(3) **rewrite toàn bộ đường dẫn tuyệt đối** trong manifests + configs (idempotent,
có marker `.paths_rewritten`); (4) sample 400 items kiểm tra path tồn tại;
(5) smoke test build backbone + forward 1 item thật.

Nếu máy mới khác CUDA: sửa `--index-url` của torch và chọn k2 wheel tương ứng tại
https://k2-fsa.github.io/k2/cuda.html (ghi chú sẵn trong script).
Muốn sinh lại label từ cutsets trên máy mới:
`LIBRIMIX_CUTSET_DIR=data/cutsets PYTHONPATH=. python src/data/prepare_from_lhotse.py ...`

## 11b. STAGE-2: domain adaptation trên meeting thật (chuẩn bị xong 2026-07-11)

Đánh trực diện 2 điểm yếu đo được ở benchmark (gain mismatch + far-field reverb):

- **Data**: `data/combined/train_manifest.json` = LibriMix 125.9k + **13,946 windows
  AMI-SDM train + NOTSOFAR-1 train** (114.9h, nhãn word-aligned 95–100%, cùng định
  nghĩa label với eval_bench). Val = 1,165 windows AMI-SDM dev (9.7h, đúng target domain).
  Item meeting = 1 "track" có `source_offset` vào wav gốc — không copy audio.
  Sinh bằng `src/data/prepare_meeting.py` (phân bố thật: overlap chỉ ~13%, khác hẳn LibriMix).
- **Augmentation** (train-only, config `data.augment`): random gain −28…+3 dB
  (giết lệch mức close-talk/far-field — thứ bắt phải dùng `--norm rms` khi eval)
  + random RIR OpenSLR-28 (60k RIRs tại `/home/pc/v2t/data/rir/RIRS_NOISES`), prob 0.35.
- **Train tiếp từ checkpoint cũ**: `train.py` giờ có `--init-from` (chỉ load weights,
  optimizer mới — dùng cho stage-2) và `--resume-from` (weights + optimizer + step —
  nối lại run đứt). Đã verify: init từ r3 load 0 missing/0 unexpected.

```bash
# Máy train:
PYTHONPATH=. python src/train.py --config configs/zipcount_v2_stage2.yaml \
    --init-from <path>/r3_v2_uniform_sord/best_macro_f1.pt
# (lr 3e-4, 20k steps ≈ 2.3 epochs combined, val mỗi 1000 — ~2-3h GPU)
```

**Baseline phải vượt** (chính checkpoint r3 zero-shot trên đúng val này, KHÔNG norm):
acc 0.556 / macroF1 0.418 / ovF1 0.427 / vadF1 0.725. Sau stage-2 kỳ vọng vadF1 >0.92,
ovF1 vào vùng 0.55–0.70 (các hệ in-domain: 0.75–0.83).

**Máy mới lấy data stage-2**: `OLD=pc@<ip-máy-cũ> bash scripts/download_more_data.sh`
(tự tải RIRs, rsync AMI ~11G + NSF snapshot ~6G + labels 183M, rewrite paths, rebuild
combined manifest, sanity check). Thêm `WITH_LS_OTHER=1` nếu muốn train-other-500 —
**không bắt buộc**: training hiện chỉ tham chiếu train-clean-100/360 + dev-clean (460h),
960h đầy đủ chỉ là data gốc của backbone.

Trên máy này có thể rebuild local:
```bash
mkdir -p data/manifests/ami data/combined
PYTHONPATH=. python -c "from lhotse.recipes.ami import prepare_ami; prepare_ami('data_ext/ami_sdm', output_dir='data/manifests/ami', mic='sdm', partition='full-corpus')"
PYTHONPATH=. python src/data/prepare_meeting.py \
    --out-dir data/meetings \
    --ami-manifest-dir data/manifests/ami \
    --nsf-manifest-dir data/manifests/notsofar1
cat data/librimix/train_manifest.json data/meetings/train_manifest.json \
    > data/combined/train_manifest.json
```

## 12. Benchmark chuẩn (2026-07-11, `src/eval_bench.py`, log: `logs/bench_suite.log`)

Checkpoint v2 = `result/r3_v2_uniform_sord/best_macro_f1.pt`. AMI: nhãn word-aligned
từ supervisions (100% coverage), cửa sổ 30s, **bắt buộc `--norm rms`** (AMI SDM nhỏ
tiếng hơn LibriSpeech nhiều; không norm thì VAD recall sập còn 0.14).

| Bench | Model | acc | MAE | VAD F1 | OSD F1 (P/R) |
|---|---|---|---|---|---|
| LibriCount 5720 clips (zero-shot, 0–10→{0,1,2,3+}) | v2 | **0.975** | 0.025 | — | — |
| AMI-SDM dev (zero-shot) | v2 | 0.653 | 0.369 | 0.880 | **0.476** (0.36/0.69) |
| AMI-SDM test (zero-shot) | v2 | 0.638 | 0.400 | 0.853 | 0.439 (0.37/0.53) |
| AMI-SDM dev (zero-shot) | v1 | 0.342 | 0.755 | 0.919 | 0.268 (0.16/0.85) |

- LibriCount: hoàn hảo mọi k trừ k=2 (0.73, nhầm sang 3+) — kể cả k=4..10 đều 100% về "3+".
- **v2 tổng quát hóa gấp ~1.8× v1 ngoài domain** (OSD 0.476 vs 0.268) — kiến trúc không
  chỉ thắng in-domain. v1 sập kiểu overcount (true1→pred2: 0.602 trên AMI).
- So với published **in-domain** trên AMI: pyannote OSD F1 75.3, WavLM-OSD 82.8,
  BeamTransformer (đa kênh) 87.3 → zero-shot 47.6 chưa cạnh tranh trực tiếp; khoảng
  cách = domain adaptation, không phải kiến trúc. Caveat so sánh: các hệ kia train
  trên AMI và thường đo trên headset-mix; mình đo SDM (khó nhất) nhãn word-aligned.
- Điểm yếu cụ thể: OSD precision 0.36 (over-trigger overlap trên reverb/noise far-field).

**Đường lên đã rõ (theo thứ tự rẻ→đắt):**
1. Gain augmentation khi train (random -30..0 dB mỗi item) — bỏ được cả `--norm rms` lúc infer.
2. Stage-2 finetune trên AMI train 79.4h (local, word-aligned labels build được bằng
   logic trong `eval_bench.py`) — các baseline đều hưởng lợi thế in-domain này rồi.
3. Reverb/RIR augmentation (OpenSLR-28) cho far-field.

## 13. Metric cần thêm trước khi viết paper (TODO)

- **Overlap onset detection latency**: sau khi người thứ 2 vào, bao nhiêu ms model phát hiện —
  metric thuyết phục nhất cho câu chuyện streaming routing, hiện chưa có trong `metrics.py`.
- RTF / latency đo thực tế của `infer_streaming.py` theo chunk size (32/16/8) — bảng
  accuracy-latency Pareto.
- Downstream: WER của multi-talker ASR khi dùng count để route (mục tiêu cuối).
