# Checkpoint / training-run reference (viết trước khi dọn, 2026-08-04)

Toàn bộ checkpoint `.pt` dưới `logs/` sẽ bị xoá sau file này, **chỉ giữ lại
một checkpoint tốt nhất** (`logs/stackgrad_pre012_s3456/step3000.pt` — xem
mục 0). File này ghi lại: mỗi thư mục `logs/<name>/` tương ứng config nào,
mục đích thí nghiệm là gì, và lệnh để train lại nếu cần.

Lệnh chung cho mọi run trong dự án (khác nhau đúng `--config`):

```bash
cd /home/edabk/hoangbpm/diar/diar_new
source ~/miniconda3/etc/profile.d/conda.sh && conda activate .zipformer
PYTHONPATH=. python src/train.py --config <path/to/config>.yaml
```

---

## 0. Checkpoint được giữ lại

| Thư mục | Config | Vì sao giữ |
|---|---|---|
| `logs/stackgrad_pre012_s3456/step3000.pt` | `artifacts/stack_mask_graduation/stackgrad_pre012_s3456.yaml` | Hệ thống chính (ZipCount, mask `[1,1,1,0,0,0]`), seed có mean macro-F1 cao nhất trong 3 seed (0.5766 vs 0.5762/0.5728) trên unified 4-corpus protocol. |

---

## 1. Thí nghiệm CÒN config (train lại được ngay bằng lệnh ở trên)

### `artifacts/stack_mask_graduation/` — hệ thống chính + các ablation mask
- `stackgrad_pre012_s{1234,2345,3456}` — **hệ thống chính**, mask giữ 3 stack
  nông nhất `[1,1,1,0,0,0]`. Số liệu paper Table chính = mean 3 seed này.
- `stackgrad_all6_s{1234,2345,3456}` — ablation "all stacks" (không mask, so
  sánh trade-off smoothness vs accuracy, Table ablation trong paper).
- `stackgrad_only1_s{1234,2345,3456}` — ablation chỉ giữ 1 stack (biến thể
  khác `stackonly5`, cùng họ thí nghiệm mask-graduation).
- `stackonly5_s{1234,2345,3456}` — baseline "final-stack only", mask
  `[0,0,0,0,0,1]`. Kết quả: 0.399 ± 0.007, đã ghi vào paper.

### `artifacts/logmel_baseline/` — baseline không pretrain
- `logmel_tcn_s{1234,2345,3456}` — backbone = 1 linear layer train từ đầu
  trên log-mel thô (không dùng Zipformer pretrain). Kết quả: 0.426 ± 0.003.

### `artifacts/loss_ablation/` — ablation từng nhóm loss
- `lossabl_countonly_s{1234,2345,3456}` — chỉ loss đếm (SORD), mọi λ khác=0.
- `lossabl_auxvad_s{1234,2345,3456}` — + VAD/OSD auxiliary heads.
- `lossabl_ordinal_s{1234,2345,3456}` — + ràng buộc ordinal (emae/consistency/monotonic).
- (nhóm "full" = chính là `stackgrad_pre012`, không cần train riêng.)

### `artifacts/renorm_gate/` — ablation chuẩn hoá gate
- `renormgate_s{1234,2345,3456}` — gate chỉ softmax trên stack active thay
  vì cả 6 stack. Kết quả: 0.544 ± 0.004 (kém hơn thiết kế gốc).

### `artifacts/context_oracle/` — tách rời chi phí causal vs capacity
- `oracle_causal_s*`, `oracle_headnc_s*`, `oracle_fullnc_s*` — nới ràng buộc
  causal ở encoder/decoder/cả hai để đo phần nào của gap là do streaming.
  max_steps=800 (ngắn hơn 3000 vì chỉ cần so sánh tương đối).
  Kết luận trong paper: causality chỉ giải thích ~20% gap, phần lớn là
  capacity/data/objective.

### `artifacts/phase7_event_state/` — biến thể head event-state
- `p7_c0/c1/c2_s{1234,2345,3456}` — 3 candidate cho một cơ chế head khác
  (event/state formulation), max_steps=800. Thuộc nhóm "eight rejected
  components" trong Negative Results — không đạt ngưỡng nên bị loại.

### `artifacts/phase8a_alignment/` — chọn cách align hypercolumn
- `p8a_average_s*`, `p8a_learned_s*` — so sánh alignment trung bình (đang
  dùng) vs alignment học được cho hypercolumn. Kết luận: learned alignment
  bị bác bỏ (memory: "P8A alignment null").

### `artifacts/phase8_chain/` — chuỗi thí nghiệm kiến trúc lớn nhất (30 config)
Chuỗi "rev-8" thử nhiều biến thể liên tiếp (residual, pyramid, transfer,
auxiliary structured losses...) trước khi chốt kiến trúc hiện tại
(`final_promoted_config.yaml`). Từng bước: `p8a0_pyr`, `p8a1_ctl`/`p8a1_trt`,
`p8a2_trt`, `p8b_cum`/`p8b_sord`, `p8c*aux_trt`, `p8d1/d2aux_trt`. Đa số bị
loại (memory: "rev-8 chain closed... all null past TCN+mask"). Giữ config
để tra cứu, không khuyến khích train lại trừ khi cần audit lại quyết định
kiến trúc.

### `artifacts/pyramid_mask_transfer/` — thử pyramid head + mask
- `pyrmask_all6_s*`, `pyrmask_pre012_s*` — kết hợp pyramid head với mask.
  Không vượt ngưỡng graduation (memory: "pyramid+mask candidate... final
  gate did not pass").

### `artifacts/stack_probe/` — probe từng stack riêng lẻ
- `probe_stack{0..5}_s*` — dùng đúng 1 stack (không phải hypercolumn) để đo
  riêng từng mức độ phân giải mang thông tin gì.

---

## 2. Thí nghiệm KHÔNG còn config (chỉ còn log text, không train lại y hệt được)

Các family sau có log `.log` trong `logs/` nhưng **không có file config
`.yaml` tương ứng nữa** — đã bị dọn ở một đợt cleanup trước đây trong dự án.
Muốn train lại đúng hệt phải đọc lại phần đầu file `.log` (thường có in ra
config/hyperparameter lúc khởi động) rồi dựng lại YAML thủ công. Số liệu
kết quả của các family này đã nằm trong text paper (Negative Results) rồi,
nên chỉ mất khả năng *tái tạo checkpoint*, không mất kết quả.

| Family | Log liên quan | Suy đoán mục đích (từ tên + paper) |
|---|---|---|
| `p2_ctrl_s*`, `p2_kd_s*` | `logs/p2_ctrl_s*.log`, `logs/p2_kd_s*.log` | Phase 2: control vs knowledge-distillation. Paper/memory: "KD gain unproven, seed noise ~0.02". |
| `p3_unc025_s*`, `p3_unc100_s*` | `logs/p3_unc*.log` | Phase 3: sweep một hệ số uncertainty/temperature (0.25 vs 1.00). |
| `p4_smooth_s*` | `logs/p4_smooth_s*.log` | Phase 4: tuning hệ số temporal-smoothness trước khi chốt `lambda_smooth=0.02`. |
| `p5c_deform_s*`, `p5c_tcn192_s*`, `p5c_tcn256_s*` | `logs/p5c_*.log` | Phase 5: so sánh decoder deformable vs TCN ở các d_model khác nhau — tiền thân của "Decoder choice" trong paper. |
| `p6_tcnft_s1234`, `p6c_tcnft_s*`, `p6c_deformft_s*` | `logs/p6*.log` | Phase 6: fine-tune decoder (TCN/deformable) trên backbone đã chọn. |
| `p8ft_s*`, `p8ftlr1e5_s*`, `p8ftlr15e5_s*`, `p8ftlr3e5_s*` | `logs/p8ft*.log` | Sweep learning-rate khi mở khoá fine-tune backbone (1e-5/1.5e-5/3e-5, tới 5 seed). Đây chính là "Fine-tuning is seed-unstable" trong paper (2/5 seed tệ đi ở mọi LR). |

---

## 3. Không phải checkpoint ZipCount — không đụng tới khi dọn

Các thư mục sau chứa **RTTM/kết quả dự đoán của hệ thống baseline khác**
(DiariZen, pyannote, Sortformer NVIDIA), không phải checkpoint train của dự
án này — cần giữ nguyên vì là input cho việc scoring/bootstrap:
`diarizen_*`, `pyannote_*`, `sortformer_official_*`, `sortformer_streaming*`,
`_restrict_voxlock`, `_restrict_voxsel`.
