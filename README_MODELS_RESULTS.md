# ZipCount — Toàn bộ kiến trúc, thí nghiệm và kết quả

Tổng hợp ngày **18/09/2026**, từ báo cáo, cấu hình và artifact đang có trong `diar_new`; các kết quả mới nhất tìm thấy thuộc loạt WavLM/LoRA tháng 08/2026. Đây là tổng hợp và tính lại metric từ kết quả đã lưu, **không phải một lượt train hoặc benchmark mới**.

Tài liệu bao quát lịch sử v1 → deformable v2/v3/v4 → TCN → event-state → pyramid, các ablation, backbone FastConformer/WavLM, partial unfreeze/LoRA, hậu xử lý và baseline diarization. Phân biệt rõ **đã chạy có kết quả**, **run cũ bị nhiễu thiết kế**, và **chỉ có code/config/đề xuất**.

- [Bảng kết quả chính](#3-benchmark-thống-nhất-trên-4-corpus)
- [V1–V4 và domain adaptation](#4-v1v4-và-domain-adaptation)
- [TCN, event-state, pyramid và stack mask](#6-deformable-so-với-tcn-phase-5-và-6)
- [FastConformer và WavLM](#11-thay-backbone-fastconformer-và-wavlm)
- [Partial unfreeze, LoRA và decoder](#12-wavlm-partial-unfreeze-và-lora)
- [Những điểm chưa khớp trong báo cáo cũ](#15-các-điểm-cần-đọc-đúng-khi-sử-dụng-báo-cáo)
- [Lộ trình nâng lên SOTA và cạnh tranh DiariZen](SOTA_ROADMAP.md)
- [CSV kết quả từng artifact](reports/model_results.csv), [CSV validation từng run](reports/training_validation.csv), [danh mục nguồn và SHA-256](reports/source_inventory.csv).

## 1. Bài toán và cách đọc các con số

Đầu vào là audio một kênh 16 kHz. Đầu ra ở 25 Hz (40 ms/frame) là số người đang nói đồng thời: `0 / 1 / 2 / 3+`; kèm VAD và overlap/OSD (`count >= 2`). Đây là **speaker counting**, không dự đoán speaker identity, nên DER không áp dụng cho ZipCount.

**Quy ước thống kê trong README này:**

- Macro-F1: trung bình F1 của 4 lớp. Với JSON `per_recording`, bảng chính lấy `pooled.macro_f1`: gộp frame trong corpus trước khi tính F1, rồi trung bình các seed.
- `recording_mean`: trung bình F1 từng recording, là đại lượng khác; giữ trong CSV để đối chiếu bootstrap.
- Bảng unified: tính lại confusion 4×4 trong từng seed/corpus, tính macro-F1, rồi lấy trung bình đều các corpus và seed. `SD` là độ lệch chuẩn mẫu (`ddof=1`) của điểm trung bình 4 corpus theo seed.
- Không gộp confusion của các seed thành một confusion chung để gọi đó là “mean seed”: F1 là hàm phi tuyến, hai cách có thể khác ở chữ số cuối. Báo cáo gốc/bootstrap có sử dụng cách gộp; README ghi rõ khi trích số từ đó.
- `—` nghĩa là không có kết quả xác nhận trong nguồn đã kiểm tra, không phải điểm 0. Run một seed chỉ có giá trị thăm dò; không có SD.
- Checkpoint **best validation** và checkpoint **locked step** được ghi riêng. Không lấy điểm tốt nhất sau khi nhìn test để thay endpoint của thí nghiệm.

| Nhóm đánh giá | Tập / cách đánh giá | Dùng để làm gì |
| --- | --- | --- |
| LibriMix validation | 6.399 item; label nguồn 100 Hz, align về output 25 Hz | V1/v2 trên dữ liệu tổng hợp |
| AMI-dev training validation | Meeting manifest trong từng giai đoạn; cửa sổ ngắn | Chọn checkpoint lịch sử r5–r9; không phải bảng unified |
| VoxConverse vox_sel | 116 recording dùng chọn thiết kế; JSON cửa sổ, output 25 Hz | Phase 2–8, oracle, stack probe/mask |
| bench / bench_paper | AMI/DipCo theo eval_bench; Vox/MSDWild theo window JSON | Benchmark lịch sử và ablation; khác unified |
| Unified 4 corpus | 90 s assets, reference 100 Hz, cùng window ID: AMI 570; DipCo 107; vox_lock 964; MSDWild 268 cửa sổ; frame support có sai khác (mục 3.1) | So sánh hệ thống trong mục 3 |
| Composite DEV WavLM | Mean AMI-dev + DipCo-dev + vox_sel | Partial unfreeze và LoRA; không phải test 4 corpus |
| Decoder heldout | Cache 25 Hz, tôn trọng valid spans/gaps, 4 corpus × 3 seed | Hysteresis/argmax; khác rasterizer unified 100 Hz |

AMI, VoxConverse và MSDWild có training split trong dữ liệu ZipCount; DipCo là corpus chưa có trong training mix. `vox_sel` là development thực tế dù lấy từ một nửa VoxConverse test; bảng unified chỉ dùng nửa còn lại `vox_lock`. Không so trực tiếp 0,890 trên LibriMix với 0,575 trên unified.

Nguồn protocol: [paper TASLP](paper/main_taslp.tex), [evaluator per-recording](scripts/eval_per_recording.py), [unified evaluator](scripts/eval_zipcount_nemo.py), [composite dev](scripts/eval_composite_dev.py).

## 2. Bản đồ kiến trúc và lịch sử khởi tạo

```text
Audio 16 kHz
├─ Kaldi Fbank 80D / 100 Hz
│  ├─ Zipformer 6 stack → hypercolumn 1984D / 25 Hz
│  │  ├─ v1: final output → Linear hoặc temporal adapter k=5
│  │  ├─ v2/v3/v4: gated projection → causal deformable ×2 → count/VAD/OSD
│  │  ├─ TCN: gated projection → dual-dilation TCN ×6 → count/VAD/OSD
│  │  │  └─ event-state: emission + DOWN/STAY/UP → Markov filter
│  │  └─ pyramid: per-stack projection + gate → TCN → count/boundary/direction
│  │     └─ CORN / refinement stage 2: có implementation, chưa có benchmark hoàn chỉnh
│  ├─ Mock projection 80→192 + stride 4 → TCN (baseline từ đầu)
│  └─ CRNN: Conv2D causal → GRU/BiGRU (có code, chưa thấy benchmark)
├─ NeMo Fbank → FastConformer → 6 layer taps → TCN causal
└─ Raw waveform chuẩn hóa → WavLM-Large → 6 layer taps → TCN
   ├─ head non-causal hoặc head causal; toàn hệ vẫn offline
   ├─ mở 2/4 transformer layer cuối
   └─ LoRA q_proj/v_proj trên 24 layer
```

**Lịch sử khởi tạo quan trọng:** nhiều thí nghiệm TCN, mask và pyramid load **backbone từ `r8_v3_diverse/best_macro_f1.pt`**, sau đó đóng băng backbone và train lại head. Backbone v3 trước đó đã được full fine-tune. “Frozen backbone” mô tả giai đoạn head-only này; không có nghĩa toàn bộ lịch sử học chỉ dùng ASR checkpoint nguyên bản.

Bằng chứng: [preregister stack mask](STACK_MASK_GRADUATION_PREREG.md), [runner](scripts/run_stack_mask_graduation.py) và [log pre012 seed 1234](logs/stackgrad_pre012_s1234.log) ghi `Init weights ... r8_v3_diverse ... (step 1000)`, `init_backbone_only`.

| Họ model | Chi tiết kiến trúc | Tham số được train / trạng thái |
| --- | --- | --- |
| V1 temporal adapter | Output cuối Zipformer; causal depthwise conv k=5, projection, gated residual; 3 output head | 531.974 khi backbone frozen |
| Deformable v2/v3/v4 | 6 stack widths 192/256/384/512/384/256; concat 1984; gate; width 256; 2 block, 4 points × 4 groups, look-back tối đa 16 output frame | Head 1.323.212; full model 65.879.235 |
| TCN-192 | Concat/gate; 6 block dual-dilation kernel 3, dilation 1/2/4/8/16/32, GLU; receptive field 127 frame ≈5,08 s | Head 1.284.812; tổng backbone+head 65.840.835 |
| TCN-256 | Cùng họ TCN, width 256 | Khoảng 2,10M head; capacity probe 1 seed |
| TCN event-state | TCN emission + signed event + causal sticky Markov posterior 4 trạng thái | C0/C1/C2; primary step 400 |
| Gated pyramid | LN→Linear→SiLU từng stack; global/framewise gate; optional final residual; boundary/direction; optional CORN và 2-stage | Target đầy đủ: head 2.381.116 theo smoke audit; không gán số này cho mọi arm |
| Log-mel scratch | MockZipformerEncoder: linear 80→192, subsample ×4; TCN-192 | 952.710 tổng trainable; không parameter-matched với ZipCount |
| FastConformer HC6 | 17 layer ×512; 6 taps concat 3072; native 12,5 Hz lặp ×2 →25 Hz; TCN-192 | Head 1.495.884; backbone frozen |
| WavLM HC6 | 24 layer ×1024; taps [0,5,9,14,18,23], concat 6144; 50 Hz average /2 →25 Hz; TCN-192 | Head 2.091.852; backbone frozen |
| WavLM + LoRA | q_proj và v_proj, 24 layer, rank 8, alpha 16, dropout 0 | LoRA 786.432 + head 2.091.852 = 2.878.284 |

Code: [model factory](src/models/zipcount_v1.py), [heads](src/models/heads.py), [pyramid](src/models/pyramid_head.py), [losses](src/models/losses.py), [structured losses](src/models/structured_losses.py). `stack_input_mask` chỉ chặn đặc trưng, **không cắt backbone hoặc giảm compute**.

## 3. Benchmark thống nhất trên 4 corpus

Điểm dưới đây **tính lại từ confusion JSON hiện có**, lấy mean score từng seed. Cột VoxC là `vox_lock`. Baseline ngoài dự án dùng một checkpoint phát hành, không có 3 lần train lại. WavLM/FastConformer là các thí nghiệm bổ sung sau bảng paper ngày 04/08/2026.

**Giới hạn phát hiện khi đối chiếu artifact:** “unified” là tên protocol trong báo cáo. Các file có cùng window ID, nhưng số frame thực sự được chấm chưa hoàn toàn giống nhau giữa mọi hệ thống do evaluator cắt theo độ dài prediction. Xem kiểm tra cụ thể ở cuối mục này; bảng phản ánh kết quả đã lưu, chưa phải benchmark mới đã sửa vấn đề đó.

| Hệ thống | Seed | AMI | DipCo | VoxC | MSDWild | Mean | SD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DiariZen WavLM + clustering | 1 | 0.7654 | 0.4953 | 0.6564 | 0.6421 | **0.6398** | — |
| WavLM HC6 + TCN non-causal | 3 | 0.6903 | 0.5702 | 0.5632 | 0.5872 | **0.6027** | 0.0032 |
| WavLM HC6 + TCN causal | 3 | 0.6828 | 0.5602 | 0.5654 | 0.5694 | **0.5945** | 0.0078 |
| Sortformer streaming mặc định | 1 | 0.5635 | 0.5182 | 0.6796 | 0.5538 | **0.5788** | — |
| FastConformer hc6 | 3 | 0.6568 | 0.5248 | 0.5664 | 0.5568 | **0.5762** | 0.0033 |
| ZipCount TCN-192 + mask pre012 | 3 | 0.6541 | 0.5515 | 0.5420 | 0.5533 | **0.5753** | 0.0020 |
| FastConformer frontloaded | 1 | 0.6547 | 0.5230 | 0.5605 | 0.5447 | **0.5707** | — |
| Sortformer offline | 1 | 0.6061 | 0.5008 | 0.6242 | 0.5125 | **0.5609** | — |
| Sortformer streaming chunk 8 | 1 | 0.5392 | 0.4950 | 0.6156 | 0.5369 | **0.5467** | — |
| pyannote.audio 3.1 | 1 | 0.5940 | 0.4638 | 0.5742 | 0.5091 | **0.5353** | — |
| Gate renorm | 3 | 0.5987 | 0.5098 | 0.5331 | 0.5323 | **0.5435** | 0.0040 |
| Logmel scratch | 3 | 0.4937 | 0.4419 | 0.3847 | 0.3843 | **0.4262** | 0.0027 |
| Final stack only | 3 | 0.4686 | 0.3737 | 0.3858 | 0.3652 | **0.3983** | 0.0069 |

Nguồn: [confusion baseline/ZipCount/ablation](results/bootstrap/), [FastConformer summary](logs/fastconformer_unified_eval_summary.txt), [WavLM rerun summary](logs/wavlm_unified_eval_summary.txt), [WavLM causal-head summary](logs/wavlm_causalhead_eval_summary.txt). CSV ghi đường dẫn từng JSON; FastConformer seed 1234 dùng tên `ami_conf/dipco_conf/vox_conf/msdwild_conf`, khác tên của 2 seed sau.

Diễn giải:

- **ZipCount TCN + pre012** là hệ thống streaming được chốt trong paper; mean 0,5753. **WavLM + TCN non-causal** là model counting nội bộ có mean unified cao nhất trong các kết quả đã tìm thấy: 0,6027, nhưng là offline. DiariZen baseline vẫn cao hơn: 0,6398.
- FastConformer HC6 đạt 0,5762, gần ZipCount 0,5753; chưa có kiểm định paired cho chênh lệch này trong nguồn đã đọc, không kết luận thắng.
- WavLM head causal đạt 0,5945; giữ head causal không làm WavLM thành streaming vì encoder vẫn bidirectional.
- Các backbone khác nhau cả pretraining, context, input width và số tham số head. Đây là **so sánh hệ thống**, không cô lập riêng hiệu quả backbone.
- Paper ghi 0,399 final-stack, 0,544 renorm, 0,546 ordinal; bảng tính lại mean từng seed cho lần lượt 0,3983, 0,5435, 0,5454. Chênh nhỏ liên quan cách aggregate/làm tròn; không thay số tính lại bằng số đã làm tròn trong paper.

Kiểm định đã báo cáo trong [TASLP](paper/main_taslp.tex), dùng 10.000 paired bootstrap, confusion gộp 3 seed trước bootstrap; Δ dương là ZipCount tốt hơn:

| So với ZipCount | Δ macro-F1 báo cáo | 95% CI báo cáo | Kết luận trong báo cáo |
| --- | --- | --- | --- |
| DiariZen | −0,065 | [−0,074; −0,054] | ZipCount thấp hơn |
| Sortformer streaming mặc định | −0,004 | [−0,016; +0,008] | Chưa phân biệt được |
| Sortformer offline | +0,014 | [+0,004; +0,024] | Lợi thế hẹp |
| Sortformer streaming chunk 8 | +0,028 | [+0,017; +0,039] | Có lợi thế |
| pyannote 3.1 | +0,040 | [+0,029; +0,047] | Có lợi thế |

Không áp dụng các CI này cho WavLM/FastConformer mới. Chi tiết latency ở mục 14; con số “640 ms” của ZipCount là cadence và cần cộng frontend context khi bàn về end-to-end.

### 3.1. Kiểm tra số frame được chấm

Đối chiếu key và tổng hàng ground-truth của confusion cho thấy các hệ có cùng tập cửa sổ, nhưng ZipCount chấm ít hơn baseline RTTM khoảng 6–10 frame 100 Hz/cửa sổ; WavLM ít hơn 0–4 frame. FastConformer HC6 khớp số frame reference. Nguyên nhân phù hợp với code [eval_zipcount_nemo.py](scripts/eval_zipcount_nemo.py): sau khi lặp prediction về 100 Hz, chỉ chấm `n = min(len(pred), len(gt))`.

| Corpus | Frame reference RTTM | Frame ZipCount mask | Frame WavLM HC6 | Frame FastConformer HC6 |
| --- | --- | --- | --- | --- |
| AMI | 5.066.043 | 5.061.468 | 5.063.796 | 5.066.043 |
| DipCo | 936.979 | 936.120 | 936.560 | 936.979 |
| vox_lock | 8.072.450 | 8.064.608 | 8.068.792 | 8.072.450 |
| MSDWild | 1.474.592 | 1.472.312 | 1.473.824 | 1.474.592 |

Các số trên lấy seed 1234; audit toàn bộ file dùng trong bảng chính lưu ở [unified_support_audit.csv](reports/unified_support_audit.csv). Phần thiếu chiếm khoảng 0,090–0,155% frame với ZipCount và 0,044–0,052% với WavLM. Chưa đo tác động lên F1, đặc biệt lớp hiếm, nên không khẳng định sai khác vô hại. Các CI phía trên là **kết quả báo cáo lịch sử**, chưa được chạy lại trên frame support hoàn toàn đồng nhất.

## 4. V1–V4 và domain adaptation

### 4.1. V1 và deformable V2 trên LibriMix

V1 dùng temporal adapter trên output cuối. V2 đồng thời thêm hypercolumn, deformable và ordinal loss; chênh V1→V2 không thể quy hết cho một thành phần. Các run `r1/r2/r3` có 50.000 bước, một run mỗi cấu hình.

| Run | Loss / model | Best val F1 (step) | F1 @50k | OSD @50k | VAD @50k | Nguồn |
| --- | --- | --- | --- | --- | --- | --- |
| r1 | V1 uniform focal | 0,693 @36.000 | 0,688 | 0,851 | 0,970 | [r1_v1_uniform_focal.log](logs/r1_v1_uniform_focal.log) |
| r2 | V1 uniform CE | 0,695 @46.000 | 0,687 | 0,848 | 0,970 | [r2_v1_uniform_ce.log](logs/r2_v1_uniform_ce.log) |
| r3 | V2 deformable + SORD | 0,890 @38.000 | 0,886 | 0,958 | 0,987 | [r3_v2_uniform_sord.log](logs/r3_v2_uniform_sord.log) |

Run v1 với auto class weight cũ được README ghi macro-F1 0,441 @4k và recall class 2 chỉ 0,015. Uniform focal @50k có recall class 2 0,694; do số bước cũng khác, không coi đây là ablation chỉ thay weight đã được kiểm soát hoàn toàn.

Đánh giá zero-shot lịch sử từ [README gốc](README.md):

| Model / benchmark | Accuracy | MAE | OSD F1 | Ghi chú |
| --- | --- | --- | --- | --- |
| V2 / LibriCount | 0,975 | 0,025 | — | 5.720 clip; clip-level 0–10 cap về 3+ |
| V2 / AMI-dev | 0,653 | 0,369 | 0,476 | Zero-shot, RMS normalization |
| V2 / AMI-test | 0,638 | 0,400 | 0,439 | Zero-shot, RMS normalization |
| V1 / AMI-dev | 0,342 | 0,755 | 0,268 | Zero-shot, RMS normalization |

### 4.2. Chuyển sang meeting và tăng đa dạng dữ liệu

Giữ deformable head, thay dữ liệu, augmentation và mức mở backbone:

| Run | Thay đổi | Trainable | Best AMI-dev F1 (step) | F1 cuối (step) | Nguồn |
| --- | --- | --- | --- | --- | --- |
| r5 stage2 | LibriMix + AMI/NSF; gain/RIR; backbone frozen | 1,323M | 0,600 @8k | 0,582 @20k | [r5_v2_stage2_train.log](logs/r5_v2_stage2_train.log) |
| r6 stage2b | Mở 2 stack cuối; SORD alpha 2,5; weight [1,1,1.5,3] | 16,984M | 0,627 @14k | 0,618 @25k | [r6_v2_stage2b_train.log](logs/r6_v2_stage2b_train.log) |
| r7 stage3 | Full fine-tune; RIR/noise mạnh hơn; tăng mẫu 3+ | 65,879M | 0,649 @4k | 0,624 @16k | [r7_v2_stage3_train.log](logs/r7_v2_stage3_train.log) |
| r8 v3 diverse | Thêm AliMeeting, AISHELL-4, MSDWild, VoxConverse dev, RAMC | 65,879M | 0,651 @1k | 0,628 @20k | [r8_v3_diverse_train.log](logs/r8_v3_diverse_train.log) |
| r9 v4 distill | V3 + teacher DiariZen; KD chỉ frame teacher đúng; λ=1, T=2 | 65,879M | 0,652 @1.4k | 0,641 @10k | [r9_v4_distill_train.log](logs/r9_v4_distill_train.log) |

Train v3 được báo cáo có 253.452 window từ LibriMix, AMI, NOTSOFAR-1, AliMeeting, AISHELL-4, MSDWild, RAMC, VoxConverse, solo và noise. Label LibriMix đến từ energy VAD trên từng nguồn; meeting dùng annotation/RTTM. Augmentation code bỏ RIR/noise bổ sung trên các corpus thật được liệt kê; **gain jitter vẫn áp dụng chung**.

r7 có AMI-test 0,649; DipCo-eval 0,533 và LibriCount clip accuracy 0,987 theo `logs/r7_eval_*`. V3 base có AMI-test 0,653, DipCo-eval 0,546 trong benchmark lịch sử bên dưới. Các kết quả này không phải protocol unified 90 s/100 Hz.

`configs/zipcount_v2_stage4.yaml` và `scripts/run_stage4.sh` còn tồn tại, nhưng chưa thấy `r8_v2_stage4_train.log` tương ứng trong snapshot. Không đồng nhất stage4 dự kiến đó với run đã thực hiện `r8_v3_diverse` hoặc `stage4_lora` của WavLM.

## 5. KD, learning rate, smoothing và interpolation

### 5.1. Các sweep ban đầu trên AMI-dev

Từ V3, các run một seed được so ở step 400/1.600. Đây là continue-training, không phải train head từ đầu.

| Run | Thiết lập | AMI-dev F1 @400 | F1 @1600 | Nguồn |
| --- | --- | --- | --- | --- |
| KD off | λ=0 | 0,651 | 0,644 | [kd_abl_lam0.log](logs/kd_abl_lam0.log) |
| KD 0.25 T1 | λ=0,25; T=1 | 0,651 | 0,643 | [kd_abl_lam025_T1.log](logs/kd_abl_lam025_T1.log) |
| KD 0.25 T2 | λ=0,25; T=2 | 0,652 | 0,643 | [kd_abl_lam025_T2.log](logs/kd_abl_lam025_T2.log) |
| KD 1 T1 | λ=1; T=1 | 0,649 | 0,641 | [kd_abl_lam1_T1.log](logs/kd_abl_lam1_T1.log) |
| LR uniform | backbone=head=3e−5 | 0,651 | 0,645 | [lr_abl_uniform.log](logs/lr_abl_uniform.log) |
| LR bb /10 | backbone=3e−6; head=3e−5 | 0,651 | 0,647 | [lr_abl_bb_div10.log](logs/lr_abl_bb_div10.log) |
| LR bb /30 | backbone=1e−6; head=3e−5 | 0,651 | 0,647 | [lr_abl_bb_div30.log](logs/lr_abl_bb_div30.log) |
| Head only | backbone frozen; head=3e−5 | 0,651 | 0,647 | [lr_abl_head_only.log](logs/lr_abl_head_only.log) |

### 5.2. Phase 2–4: kiểm tra bằng 3 seed, vox_sel, locked step 400

KD dùng KL teacher→student; teacher DiariZen powerset được marginalize về count. Dữ liệu synthetic bị loại khỏi KD. `agree_uncertain` giữ frame teacher vừa đúng vừa chưa bão hòa (`max posterior <0,9`), T=2.

| Model / biến thể | Seed | Macro-F1 | SD seed | OSD F1 | Nguồn mẫu; CSV có đủ seed |
| --- | --- | --- | --- | --- | --- |
| Control KD off | 3 | 0.5551 | 0.0108 | 0.5018 | [p2_ctrl](results/p2_ctrl_s1234_voxsel.json) |
| KD agree, λ=0,25, T=2 | 3 | 0.5568 | 0.0153 | 0.5077 | [p2_kd](results/p2_kd_s1234_voxsel.json) |
| KD agree_uncertain λ=0,25 | 3 | 0.5538 | 0.0141 | 0.5013 | [p3_unc025](results/p3_unc025_s1234_voxsel.json) |
| KD agree_uncertain λ=1 | 3 | 0.5515 | 0.0099 | 0.4930 | [p3_unc100](results/p3_unc100_s1234_voxsel.json) |
| Smoothing lịch sử; bị confound | 3 | 0.5557 | 0.0103 | 0.5028 | [p4_smooth](results/p4_smooth_s1234_voxsel.json) |

KD agree tăng mean pooled khoảng +0,0017, nhưng dấu delta không đồng nhất giữa seed và CI recording chứa 0: chưa chứng minh lợi ích. KD uncertain không cải thiện; λ=1 giảm kết quả ở cả 3 seed.

**Phase 4 là run có confound:** thời điểm chạy, `lambda_smooth` đồng thời thay KL 0,02→0,15 và bật boundary T-MSE 0,15. Không diễn giải +0,0006 pooled là hiệu quả riêng T-MSE. Script hiện tại đã tách key; chạy lại hôm nay không tái tạo đúng artifact cũ. Nguồn: [run_phase4.sh](scripts/run_phase4.sh), [Phase 2](logs/phase2_resume.log), [Phase 3](logs/phase3_all.log).

### 5.3. V3 base so với fine-tune không KD

Benchmark lịch sử, mean 3 seed `p2_ctrl` @400; không dùng bảng này thay unified:

| Corpus | V3 base | Fine-tune mean | Δ |
| --- | --- | --- | --- |
| AMI-dev | 0,6510 | 0,6470 | −0,0040 |
| AMI-test | 0,6530 | 0,6360 | −0,0170 |
| DipCo-dev | 0,4790 | 0,4610 | −0,0180 |
| DipCo-eval | 0,5460 | 0,5373 | −0,0087 |
| VoxConverse full | 0,5167 | 0,5367 | +0,0201 |
| vox_lock | 0,4987 | 0,5204 | +0,0218 |
| MSDWild | 0,5213 | 0,5484 | +0,0271 |

Nguồn: [results/bench](results/bench/), tính lại bằng [summarise_benchmark.py](scripts/summarise_benchmark.py). Fine-tune cải thiện một số domain và làm giảm các domain khác.

Interpolation trọng số `θ=(1−α)θ_v3+αθ_ft`, mean 3 seed; composite = mean AMI-dev/DipCo-dev/vox_sel:

| α | AMI-dev | DipCo-dev | vox_sel | Composite dev |
| --- | --- | --- | --- | --- |
| 0 | 0,6510 | 0,4790 | 0,5370 | 0,5557 |
| 0,25 | 0,6517 | 0,4763 | 0,5432 | 0,5571 |
| 0,50 | 0,6510 | 0,4720 | 0,5481 | 0,5570 |
| 0,75 | 0,6493 | 0,4667 | 0,5521 | 0,5560 |
| 1 | 0,6470 | 0,4610 | 0,5551 | 0,5544 |

α=0,25 được chọn theo composite dev của sweep, tăng khoảng +0,0014 so với endpoint tốt hơn. Không thống trị cả hai endpoint trên mọi corpus. Nguồn: [results/interp](results/interp/), [summarise_interp.py](scripts/summarise_interp.py).

## 6. Deformable so với TCN: Phase 5 và 6

Phase 5c giữ backbone V3 frozen, khởi tạo lại head, train 3.000 bước. TCN-192 gần bằng kích thước deformable; TCN-256 lớn hơn, chỉ chạy một seed.

| Model / biến thể | Seed | Macro-F1 | SD seed | OSD F1 | Nguồn mẫu; CSV có đủ seed |
| --- | --- | --- | --- | --- | --- |
| P5c deformable-256, ~1,32M | 3 | 0.5594 | 0.0043 | 0.4971 | [p5c_deform](results/p5c_deform_s1234_voxsel.json) |
| P5c TCN-192, ~1,28M | 3 | 0.5643 | 0.0024 | 0.5096 | [p5c_tcn192](results/p5c_tcn192_s1234_voxsel.json) |
| P5c TCN-256, ~2,10M | 1 | 0.5671 | — | 0.5195 | [p5c_tcn256](results/p5c_tcn256_s1234_voxsel.json) |
| P6c deformable full FT @400 | 3 | 0.5585 | 0.0201 | 0.5004 | [p6c_deformft](results/p6c_deformft_s1234_voxsel.json) |
| P6c TCN-192 full FT @400 | 3 | 0.5607 | 0.0208 | 0.5034 | [p6c_tcnft](results/p6c_tcnft_s1234_voxsel.json) |

P5c TCN-192 − deformable: **+0,004889 pooled macro-F1**; mean delta từng recording +0,0021, CI [0,0006; 0,0038], cùng dấu ở 3 seed. Đây là bằng chứng sạch hơn cho chọn TCN. P6c sau full fine-tune: +0,002216 pooled, nhưng CI recording [−0,0005; +0,0033] chứa 0.

Có một Phase 5 trước đó (`p5_*`) dùng double-smoothing ngoài ý định; giữ để audit, không trộn vào P5c:

| Model / biến thể | Seed | Macro-F1 | SD seed | OSD F1 | Nguồn mẫu; CSV có đủ seed |
| --- | --- | --- | --- | --- | --- |
| P5 cũ deformable | 3 | 0.5594 | 0.0040 | 0.4970 | [p5_deform](results/p5_deform_s1234_voxsel.json) |
| P5 cũ TCN-192 | 3 | 0.5645 | 0.0022 | 0.5102 | [p5_tcn192](results/p5_tcn192_s1234_voxsel.json) |
| P5 cũ TCN-256 | 1 | 0.5670 | — | 0.5203 | [p5_tcn256](results/p5_tcn256_s1234_voxsel.json) |

Paper trích +0,0022, CI [+0,0007; +0,0039] và fragmentation 2,53→2,36. Log Phase 5 cũ có boundary F1 0,341→0,350 và bootstrap +0,0022; P5c có pooled +0,0049, bootstrap +0,0021, fragmentation 2,570→2,406 cho seed 1234. Các dòng này không phải một phép đo thống nhất trên 4 corpus. README ưu tiên tách rõ P5/P5c/P6c.

Nguồn: [runner P5c](scripts/run_phase5.sh), [runner P6c](scripts/run_phase6.sh), [log sạch](logs/phase5c_6c_all.log), [log P5 cũ](logs/phase5_all.log). `p6_tcnft_s1234` là run tiền nhiệm một seed còn log; không thay thế P6c đối xứng.

## 7. Context oracle và event-state

### 7.1. Nới context ở decoder/encoder

Ba seed, full fine-tune từ TCN, locked step 400. `fullnc` giữ topology causal-conv nhưng mở chunk lên 4096 và left context −1 để cả cửa sổ có context; không đơn giản đổi encoder thành topology non-causal.

| Model / biến thể | Seed | Macro-F1 | SD seed | OSD F1 | Nguồn mẫu; CSV có đủ seed |
| --- | --- | --- | --- | --- | --- |
| O0 encoder causal + head causal | 3 | 0.5605 | 0.0206 | 0.5033 | [oracle_causal](results/oracle_causal_s1234_voxsel.json) |
| O1 encoder causal + head non-causal | 3 | 0.5550 | 0.0179 | 0.4920 | [oracle_headnc](results/oracle_headnc_s1234_voxsel.json) |
| O2 encoder full-context + head non-causal | 3 | 0.5807 | 0.0124 | 0.5468 | [oracle_fullnc](results/oracle_fullnc_s1234_voxsel.json) |

O1−O0 = −0,0055; O2−O1 = +0,0257; O2−O0 = +0,0202 pooled macro-F1. Chỉ là diagnostic cho cấu hình/thời lượng train này; không phải chứng minh upper bound tuyệt đối của mọi hệ streaming. Nguồn: [context configs](artifacts/context_oracle/), [runner](scripts/run_context_oracle.sh).

### 7.2. Oracle dùng ground-truth boundary

Một seed của P6c TCN @400; dùng biên thật nên **không triển khai được**:

| Decoder | Pooled F1 | OSD F1 | Recording-mean F1 |
| --- | --- | --- | --- |
| Argmax | 0,577388 | 0,540389 | 0,509013 |
| GT boundary + causal accumulation | 0,618850 | 0,591297 | 0,555045 |
| GT boundary + whole-segment pooling | 0,667576 | 0,708371 | 0,596269 |

Nguồn: [EVENT_STATE_ROADMAP](EVENT_STATE_ROADMAP.md). Oracle gợi ý lỗi biên/fragmentation có dư địa, không cho phép coi 0,6676 là kết quả model đã học.

### 7.3. Event-state C0/C1/C2

C0 = TCN emission; C1 = thêm signed event CE nhưng output vẫn raw; C2 = output qua Markov filter + raw anchor. Loss event 0,2; C2 raw anchor 0,25. Mỗi arm chạy 800 bước nhưng **primary là step 400**.

| Model / biến thể | Seed | Macro-F1 | SD seed | OSD F1 | Nguồn mẫu; CSV có đủ seed |
| --- | --- | --- | --- | --- | --- |
| C0 TCN control @400 | 3 | 0.5611 | 0.0192 | 0.5043 | [p7_c0_step400](results/p7_c0_s1234_step400_voxsel.json) |
| C1 event auxiliary @400 | 3 | 0.5600 | 0.0188 | 0.5020 | [p7_c1_step400](results/p7_c1_s1234_step400_voxsel.json) |
| C2 event filter @400 | 3 | 0.5337 | 0.0147 | 0.4715 | [p7_c2_step400](results/p7_c2_s1234_step400_voxsel.json) |

C2−C0 = −0,0275 pooled; CI recording [−0,0273; −0,0174]. Event-state bị loại. Step 200/800 chỉ là learning curve:

| Arm | F1 @200 | F1 @400 primary | F1 @800 |
| --- | --- | --- | --- |
| C0 | 0.5599 | 0.5611 | 0.5698 |
| C1 | 0.5593 | 0.5600 | 0.5693 |
| C2 | 0.5365 | 0.5337 | 0.5579 |

Nguồn: [báo cáo so sánh @400](artifacts/phase7_event_state/step400_comparisons.txt). Roadmap gọi 0,5611/0,5600/0,5337 là “recording macro-F1”, nhưng JSON xác nhận đó là **pooled**; trung bình từng recording thấp hơn, đã giữ riêng trong CSV.

## 8. Stack probe, stack mask và gated pyramid

### 8.1. Probe từng stack

Mỗi probe chỉ giữ một stack thật trong đầu vào, làm thay đổi input width, số tham số và initialization. Một seed @3000; chỉ dùng khám phá, không phải ablation parameter-matched.

| Model / biến thể | Seed | Macro-F1 | SD seed | OSD F1 | Nguồn mẫu; CSV có đủ seed |
| --- | --- | --- | --- | --- | --- |
| Stack 0 | 1 | 0.5430 | — | 0.4930 | [probe_stack0](results/probe_stack0_s1234_voxsel.json) |
| Stack 1 | 1 | 0.5626 | — | 0.5165 | [probe_stack1](results/probe_stack1_s1234_voxsel.json) |
| Stack 2 | 1 | 0.5460 | — | 0.4762 | [probe_stack2](results/probe_stack2_s1234_voxsel.json) |
| Stack 3 | 1 | 0.5301 | — | 0.4474 | [probe_stack3](results/probe_stack3_s1234_voxsel.json) |
| Stack 4 | 1 | 0.5062 | — | 0.3541 | [probe_stack4](results/probe_stack4_s1234_voxsel.json) |
| Stack 5 | 1 | 0.5050 | — | 0.3653 | [probe_stack5](results/probe_stack5_s1234_voxsel.json) |

Stack 1 tốt nhất trong probe. `only1` ở thí nghiệm mask sau đây có nghĩa **stack index 1**, không phải stack cuối.

### 8.2. Mask trên cùng graph TCN-192

Mask sau LayerNorm, trước gated projection; toàn bộ graph/parameter/init giữ nguyên. Backbone V3 frozen, 3 seed @3000.

| Model / biến thể | Seed | Macro-F1 | SD seed | OSD F1 | Nguồn mẫu; CSV có đủ seed |
| --- | --- | --- | --- | --- | --- |
| all6 [1,1,1,1,1,1] | 3 | 0.5642 | 0.0024 | 0.5096 | [stackgrad_all6](results/stackgrad_all6_s1234_voxsel.json) |
| only1 [0,1,0,0,0,0] | 3 | 0.5630 | 0.0038 | 0.5221 | [stackgrad_only1](results/stackgrad_only1_s1234_voxsel.json) |
| pre012 [1,1,1,0,0,0] | 3 | 0.5685 | 0.0033 | 0.5346 | [stackgrad_pre012](results/stackgrad_pre012_s1234_voxsel.json) |

`pre012` tăng pooled +0,004260 và OSD +0,025050 so với all6. Bootstrap recording mean +0,001435, CI [−0,002515; +0,005206]: bằng chứng macro-F1 được ghi **SUGGESTIVE**, chưa phải superiority chắc chắn. Mask được chọn theo tiêu chí thực dụng R3 (OSD/class-2), không đạt strict non-inferiority để suy ra có thể cắt backbone.

So với only1, pre012 có CI recording [+0,002470; +0,006613], bằng chứng `REAL`. Nguồn: [final report](artifacts/stack_mask_graduation/final_report.json), [preregister](STACK_MASK_GRADUATION_PREREG.md).

### 8.3. Chuyển mask sang pyramid-minimal

Pyramid đổi concat projection sang lateral projection từng stack + tổng có trọng số. Chạy global gate, softmax output, một TCN stage; **chưa phải CORN/two-stage đầy đủ**.

| Model / biến thể | Seed | Macro-F1 | SD seed | OSD F1 | Nguồn mẫu; CSV có đủ seed |
| --- | --- | --- | --- | --- | --- |
| Pyramid minimal all6 | 3 | 0.5576 | 0.0024 | 0.4920 | [pyrmask_all6](results/pyrmask_all6_s1234_voxsel.json) |
| Pyramid minimal pre012 | 3 | 0.5649 | 0.0022 | 0.5244 | [pyrmask_pre012](results/pyrmask_pre012_s1234_voxsel.json) |

Mask cải thiện +0,007303 pooled và +0,032438 OSD trong chính họ pyramid. CI recording [−0,000149; +0,007069] vẫn chạm 0; strict NI đạt theo margin của protocol. Điều này hỗ trợ mang mask vào chain, không có nghĩa pyramid đã thắng TCN. Nguồn: [pyramid mask final report](artifacts/pyramid_mask_transfer/final_report.json).

### 8.4. Alignment và chain Phase 8 rev-8

Alignment learned tái sử dụng downsampler đã học của encoder, không phải thêm một head count khác:

| Model / biến thể | Seed | Macro-F1 | SD seed | OSD F1 | Nguồn mẫu; CSV có đủ seed |
| --- | --- | --- | --- | --- | --- |
| TCN fixed average alignment | 3 | 0.5642 | 0.0024 | 0.5096 | [p8a_average](results/p8a_average_s1234_voxsel.json) |
| TCN encoder-learned alignment | 3 | 0.5632 | 0.0017 | 0.5061 | [p8a_learned](results/p8a_learned_s1234_voxsel.json) |

Learned alignment giảm −0,001058 pooled; CI recording [−0,0011; +0,0003], không được chọn. Nguồn: [alignment comparison](artifacts/phase8a_alignment/comparison.txt).

Chain rev-8: các dòng dưới lấy **đúng 3 seed 1234/2345/3456**, checkpoint @3000, không trộn 2 seed bổ sung cho nghiên cứu fine-tune.

| Model / biến thể | Seed | Macro-F1 | SD seed | OSD F1 | Nguồn mẫu; CSV có đủ seed |
| --- | --- | --- | --- | --- | --- |
| A0 pyramid minimal + pre012 | 3 | 0.5649 | 0.0022 | 0.5244 | [p8a0_pyr](results/p8a0_pyr_s1234_voxsel.json) |
| A1 gate theo frame | 3 | 0.5658 | 0.0013 | 0.5260 | [p8a1_trt](results/p8a1_trt_s1234_voxsel.json) |
| A2 thêm final-output residual | 3 | 0.5512 | 0.0025 | 0.4741 | [p8a2_trt](results/p8a2_trt_s1234_voxsel.json) |
| B SORD qua structured-loss bridge | 3 | 0.5562 | 0.0007 | 0.5013 | [p8b_sord](results/p8b_sord_s1234_voxsel.json) |
| B cumulative loss trên softmax | 3 | 0.5527 | 0.0010 | 0.5018 | [p8b_cum](results/p8b_cum_s1234_voxsel.json) |
| C boundary auxiliary 0,1 | 3 | 0.5646 | 0.0018 | 0.5252 | [p8caux_trt](results/p8caux_trt_s1234_voxsel.json) |
| C2 direction auxiliary 0,05 | 3 | 0.5647 | 0.0019 | 0.5248 | [p8c2aux_trt](results/p8c2aux_trt_s1234_voxsel.json) |
| D1 temporal-MSE auxiliary 0,05 | 3 | 0.5646 | 0.0026 | 0.5230 | [p8d1aux_trt](results/p8d1aux_trt_s1234_voxsel.json) |
| D2 segment auxiliary 0,05 | 3 | 0.5649 | 0.0022 | 0.5244 | [p8d2aux_trt](results/p8d2aux_trt_s1234_voxsel.json) |

- A0 được giữ làm bridge do đạt mean-only non-inferiority. A1 có hiệu ứng dương nhỏ được gắn `REAL` nhưng vẫn dưới ngưỡng adoption; A2 có hại.
- B SORD/cumulative thay objective, mất các auxiliary legacy nên phải đọc đúng control/bridge. `p8b_cum` có `ordinal_mode: softmax`, `count_loss_type: cumulative`: **không phải native CORN head**.
- C/D là từng loss auxiliary riêng trên baseline legacy; không cộng dồn các treatment đã bị loại. Boundary AP tăng lên mean 0,0560, recall tại fixed FP-budget tăng +0,0728, nhưng macro-F1 không tăng đủ.
- Final candidate A0 chỉ tăng **+0,000627 pooled** so với immutable TCN root; complexity gate yêu cầu +0,010 nên `final_gate_passed=false`, `promoted=root`.
- `final_promoted_config.yaml` là TCN all6 root của chain. Hệ thống paper **TCN pre012** đến từ quyết định mask-graduation riêng; hai tên “promoted” và “main paper system” không đồng nhất.

Nguồn: [rev-8 log](logs/phase8_chain_rev8_run.log), [final state](artifacts/phase8_chain/final_state.json), [chain configs](artifacts/phase8_chain/).

### 8.5. Những kiến trúc có code nhưng chưa có benchmark hoàn chỉnh

Native CORN (`q1=c1, q2=c1c2, q3=c1c2c3`), hai-stage refinement, capacity-matched refinement control và delta consistency được mô tả trong [pyramid roadmap](PYRAMID_ORDINAL_ROADMAP.md). Target config có 2 stage, framewise gate, learned alignment, final residual và CORN. Roadmap ghi smoke/unit/gradient audit và head 2,381M; **không tìm thấy kết quả long-run tương ứng đầy đủ để gán điểm F1 cho target đó**. Không diễn giải kết quả âm tính cumulative-softmax là đã bác bỏ native CORN hoặc mọi kiến trúc 2-stage.

Ngày 25/09/2026, factory được mở thêm ba temporal head streaming để chuẩn bị một bake-off không khóa vào TCN: `gru_ordinal`, `ssm_ordinal` (selective diagonal state-space) và `attention_ordinal` (local causal attention có relative-time bias). Cùng với TCN và deformable, runner giữ mask/fusion/output/loss giống nhau và cân bằng head quanh 1,28M tham số. Các head mới đã có test causality, cache và gradient nhưng **chưa được train/evaluate**, nên không có F1 để đưa vào các bảng kết quả ở trên. Xem [runner bake-off](scripts/run_decoder_bakeoff.py) và [roadmap](SOTA_ROADMAP.md#12-không-khóa-vào-tcn-decoder-bake-off).

Các nhánh acoustic energy/flux/flatness, multi-harmonic/F0/F1 và multichannel IPD/GCC-PHAT mới là đề xuất; chưa có kết quả model trong snapshot.

## 9. Fine-tune pyramid và độ nhạy seed

Từ A0 head-only 3.000 bước, mở toàn bộ backbone, primary FT @400. Bảng đủ 5 seed; LR 3e−5 lấy 3 seed đầu từ `p8ft_*`, 2 seed sau từ `p8ftlr3e5_*`.

| Seed | Head-only | FT 1e−5 (Δ) | FT 1,5e−5 (Δ) | FT 3e−5 (Δ) |
| --- | --- | --- | --- | --- |
| 1234 | 0.5671 | 0.5734 (+0.0064) | 0.5748 (+0.0078) | 0.5708 (+0.0037) |
| 2345 | 0.5626 | 0.5472 (-0.0154) | 0.5411 (-0.0216) | 0.5311 (-0.0315) |
| 3456 | 0.5649 | 0.5689 (+0.0040) | 0.5697 (+0.0048) | 0.5751 (+0.0103) |
| 4567 | 0.5661 | 0.5526 (-0.0134) | 0.5507 (-0.0153) | 0.5475 (-0.0186) |
| 5678 | 0.5670 | 0.5721 (+0.0050) | 0.5730 (+0.0060) | 0.5876 (+0.0206) |

Mean delta lần lượt −0,0027 / −0,0037 / −0,0031. Hai seed 2345 và 4567 giảm ở mọi LR; không thể chọn seed 5678 đạt 0,5876 rồi kết luận full fine-tune luôn tốt. Nguồn: [LR sweep report](logs/phase8_lr_sweep_report.log), `results/p8ft*` và `results/p8a0_pyr*`. Đây là fine-tune **pyramid A0**, không phải bằng chứng riêng cho TCN pre012 trong paper.

## 10. Ablation loss, gate và chất lượng đoạn

### 10.1. Các nhóm loss trên TCN pre012

3 seed × 4 corpus unified; graph giữ nguyên, bật dần các loss:

| Loss | AMI | DipCo | vox_lock | MSDWild | Mean | SD |
| --- | --- | --- | --- | --- | --- | --- |
| SORD; mọi auxiliary λ=0 | 0.6010 | 0.5129 | 0.5290 | 0.5273 | 0.5425 | 0.0042 |
| + VAD 0,3, overlap 0,5, Dice 0,5 | 0.6050 | 0.5103 | 0.5317 | 0.5327 | 0.5449 | 0.0053 |
| + expected-MAE 0,2, consistency 0,2, monotonic 0,1 | 0.6023 | 0.5093 | 0.5357 | 0.5344 | 0.5454 | 0.0053 |
| + symmetric-KL temporal smoothing 0,02 | 0.6541 | 0.5515 | 0.5420 | 0.5533 | 0.5753 | 0.0020 |

Temporal smoothing tăng khoảng **+0,0298**; chiếm khoảng 91% chênh từ count-only đến full objective. Các mức tăng còn lại nhỏ hơn nhiều. Đây là ablation sạch của KL legacy; tách biệt Phase 4 confounded và boundary-aware T-MSE của pyramid.

Nguồn: [loss configs](artifacts/loss_ablation/), [confusion JSON](results/bootstrap/), [TASLP loss-ablation](paper/main_taslp.tex). CI trong paper cho ba bước cộng loss: [+0,0015; +0,0033], [+0,0002; +0,0010], [+0,0241; +0,0361].

### 10.2. Gate và pretrained representation

- Final stack mask `[0,0,0,0,0,1]`: unified mean **0,3983**. Không nhầm với `only1=[0,1,0,0,0,0]`.
- Log-mel scratch + TCN: **0,4262**. Tốt hơn final-stack nhưng thấp hơn nhiều so với multi-stack. Baseline này **952.710 trainable**, nên không gọi parameter-matched với head 1.284.812.
- Renormalize gate chỉ trên active stack: **0,5435**, thấp hơn gate softmax cả 6 stack 0,5753. Nguồn báo cáo bootstrap ủng hộ thiết kế không renormalize; nguyên nhân tối ưu hóa chỉ là diễn giải, không được đo trực tiếp.

### 10.3. Stack mask tăng F1 nhưng làm đoạn dự đoán kém mượt

Benchmark lịch sử `bench_paper`, mean 3 seed, khác unified:

| Metric / corpus | TCN mask | TCN all6 |
| --- | --- | --- |
| Macro-F1 AMI-test | 0,6567 | 0,6217 |
| Macro-F1 DipCo-eval | 0,5397 | 0,5320 |
| Macro-F1 Vox full | 0,5544 | 0,5457 |
| Macro-F1 vox_lock | 0,5413 | 0,5291 |
| Macro-F1 MSDWild | 0,5515 | 0,5476 |
| OSD F1 vox_lock | 0,4605 | 0,4358 |
| Edit score Vox full ↑ | 38,6903 | 40,7006 |
| Edit score MSDWild ↑ | 53,3304 | 55,8216 |
| Fragmentation Vox full (gần 1 tốt hơn) | 3,2069 | 3,0639 |
| LibriCount clip accuracy | 0,9840 | 0,9843 |

Nguồn: [bench_paper](results/bench_paper/), [summarise_paper_benchmark.py](scripts/summarise_paper_benchmark.py). Plain TCN không có `boundary_logits`, nên các log boundary AP trong benchmark này không tạo được phép đo tương ứng; không coi đó là AP=0. Boundary AP của pyramid là thí nghiệm khác.

## 11. Thay backbone: FastConformer và WavLM

### 11.1. FastConformer

Dùng `stt_en_fastconformer_hybrid_large_streaming_480ms`, 17 layer, 512 chiều. Fbank phải dùng frontend NeMo đúng checkpoint; không dùng Kaldi fbank của Zipformer. Native 12,5 Hz được repeat ×2 về 25 Hz. TCN causal width 192; train head 3.000 bước, LR 1e−3.

- HC6 taps đều: `[0,3,6,10,13,16]`, 3 seed, mean unified **0,5762 ±0,0033**.
- Frontloaded taps: `[0,2,4,6,9,13]`, một seed 1234, **0,5707**; so cùng seed HC6 = **0,5724**, chưa có lợi ích.
- Wrapper kiểm tra `chunked_limited` attention `[70,6]` và causal convolution. **Chưa triển khai `forward_streaming` có cache trong wrapper này**; không có RTF/cadence end-to-end tương đương Zipformer để báo cáo. “480 ms” là tên/checkpoint context, không phải độ trễ hệ thống đã đo ở đây.

Nguồn: [wrapper](src/models/fastconformer_wrapper.py), [configs](artifacts/fastconformer_baseline/), [summary](logs/fastconformer_unified_eval_summary.txt).

### 11.2. WavLM-Large

Dùng waveform chuẩn hóa zero-mean/unit-variance, CNN frontend của WavLM; không dùng fbank. Taps `[0,5,9,14,18,23]`; 50 Hz average-pool về 25 Hz. Cùng head width 192 nhưng input 6144 nên head lớn hơn ZipCount.

- Head non-causal: **0,6027 ±0,0032** unified, 3 seed.
- Head causal: **0,5945 ±0,0078**, 3 seed. Encoder vẫn nhìn hai chiều, nên cả hai là offline.
- Mean per-seed 4 corpus của head non-causal: **0,6005 / 0,6064 / 0,6014**. Head causal: **0,5901 / 0,6035 / 0,5898**.
- Chênh non-causal−causal head khoảng +0,0083, nhưng chưa có CI paired độc lập trong báo cáo để gắn kết luận significance.

Logs eval ban đầu có `FAILED`; sau đó có gated rerun thành công và confusion JSON. README lấy **JSON của rerun**, không lấy dòng failed hoặc điểm làm tròn trong summary. Nguồn: [WavLM configs](artifacts/wavlm_offline/), [wrapper](src/models/wavlm_wrapper.py), [rerun summary](logs/wavlm_unified_eval_summary.txt).

## 12. WavLM partial unfreeze và LoRA

Các bảng sau là **composite DEV ba corpus**, một seed 1234, tiếp tục từ checkpoint WavLM đã có. Không so trực tiếp với 0,6027 unified test ở trên.

### 12.1. Mở 2 hoặc 4 transformer layer cuối

Head luôn có 2.091.852 trainable. Mở 2 layer thêm 25.193.520 tham số; mở 4 layer thêm 50.387.040. Microbatch 4 × accumulation 6 = effective batch 24. Nguồn: [configs](artifacts/wavlm_offline/partial_unfreeze/), [manual rerun summary](logs/wavlm_partial_unfreeze_summary_manual.txt).

| Arm | Step 100 | Step 200 | Step 400 |
| --- | --- | --- | --- |
| Head only | 0.609823 | 0.608172 | 0.605123 |
| Mở 2 layer | 0.609778 | 0.608540 | 0.604577 |
| Mở 4 layer | 0.609011 | 0.608365 | 0.603464 |

Chưa thấy lợi ích rõ: ở step 400, cả hai partial-unfreeze thấp hơn head-only. Log summary tự động từng báo thiếu checkpoint; manual rescore và JSON đã bổ sung đủ các mốc, nên bảng dùng JSON.

### 12.2. LoRA q/v, rank 8, 24 layer

48 Linear được bọc LoRA; 786.432 adapter params. Base backbone frozen; train head + adapter, LR head=LoRA=1e−4, constant schedule, warmup 100, effective batch 24, endpoint 800. Control head-only được chạy riêng cùng schedule, không trộn với control 400-step ở bảng trước.

| Step | Head-only composite | LoRA composite | Δ LoRA−control |
| --- | --- | --- | --- |
| 100 | 0.611018 | 0.611049 | +0.000031 |
| 200 | 0.609052 | 0.611127 | +0.002074 |
| 400 | 0.605586 | 0.605920 | +0.000333 |
| 800 | 0.605397 | 0.609540 | +0.004143 |

LoRA tốt nhất theo composite trong các mốc đã lưu tại step 200: **0,611127**, so control cùng step 0,609052. Step 800: **0,609540** so 0,605397, Δ=+0,004143. Tuy nhiên best observed head-only @100 là 0,611018, rất sát best LoRA; một seed và chỉ DEV chưa đủ kết luận LoRA cải thiện generalization.

Chi tiết LoRA @800: AMI-dev 0,719385; DipCo-dev 0,496812; vox_sel 0,612424. Nguồn: [LoRA config](artifacts/wavlm_offline/partial_unfreeze/lora_qv_all24_r8_800.yaml), [JSON @800](results/lora_diag_lora_qv_all24_r8_800_step800_composite_dev.json), [summary lịch sử](logs/wavlm_lora_diag_summary.txt). Summary cũ chứa lỗi load key trước khi rescore; các JSON hoàn chỉnh là căn cứ cho bảng.

## 13. Hậu xử lý chuỗi và baseline ngoài dự án

### 13.1. Duration decoder trên WavLM

Code có hysteresis + minimum-duration, Viterbi và semi-Markov. Artifact lựa chọn hiện tại chỉ lưu **hysteresis** trong `dev_all_candidates`; không đủ căn cứ ghi điểm riêng cho Viterbi/semi-Markov.

AMI-dev seed 1234: argmax 0,712273; hysteresis margin 0,05, min-duration 3 frame đạt 0,712445. Cấu hình được khóa rồi đánh giá heldout 4 corpus ×3 seed ở 25 Hz:

| Corpus | Argmax | Hysteresis | Δ |
| --- | --- | --- | --- |
| ami_test | 0.692565 | 0.693240 | +0.000675 |
| dipco_eval | 0.563470 | 0.561591 | -0.001880 |
| voxconverse_voxlock | 0.567095 | 0.567830 | +0.000734 |
| msdwild_manyval | 0.588048 | 0.587951 | -0.000098 |
| Mean | 0.602795 | 0.602653 | -0.000142 |

Verdict lưu: **`adopted: false`**, mean delta −0,000142. Đây là decode-cache protocol riêng, không ghi đè bảng unified WavLM 100 Hz.

Nguồn: [locked decoder](artifacts/wavlm_offline/locked_decoder_configs.json), [heldout report](results/decode_heldout_report.json), [decoder implementations](src/utils/decoders.py).

### 13.2. Baseline diarization

- **DiariZen:** WavLM-Large + pipeline diarization/clustering offline; count được suy từ speaker activity. Khác model WavLM+TCN do dự án train.
- **Sortformer offline:** checkpoint `diar_sortformer_4spk-v1`.
- **Sortformer streaming:** `diar_streaming_sortformer_4spk-v2.1`; default chunk 188/right context 1 ≈15,12 s; thử chunk 8/right context 0 =640 ms là cấu hình ép latency ngoài operating point train.
- **pyannote.audio 3.1:** dùng output giữ overlap, không dùng exclusive diarization.
- **Sortformer + postprocessing DIHARD3:** thử lịch sử AMI-test macro-F1 0,609 default vs 0,564 postprocessing; LibriCount accuracy 0,923. Không trộn với unified. Nguồn: [default](logs/sortformer_eval_ami_test.log), [postprocessing](logs/sortformer_eval_ami_test_pp.log), [LibriCount](logs/sortformer_eval_libricount.log).
- **CRNN, Zipformer linear-only:** có implementation/config option; chưa tìm thấy benchmark riêng đủ căn cứ trong snapshot. **Mock dry-run** chỉ kiểm tra pipeline; phân biệt với baseline log-mel scratch train thật có kết quả.
- EEND-M2F/CountNet và các model nhắc trong phần khảo sát của README cũ: chưa thấy kết quả chạy local tương ứng, không đưa số từ tài liệu ngoài vào bảng kết quả dự án.

DER (%) trích từ paper, collar 0,25 s, overlap included; chưa tính lại RTTM trong lượt tổng hợp:

| Model | AMI | DipCo | vox_lock | MSDWild | Mean |
| --- | --- | --- | --- | --- | --- |
| DiariZen | 11,09 | 35,12 | 7,10 | 28,89 | 20,55 |
| Sortformer offline | 21,99 | 28,68 | 5,53 | 34,10 | 22,57 |
| Sortformer streaming default | 26,22 | 31,76 | 4,63 | 34,22 | 24,21 |
| pyannote 3.1 | 20,83 | 34,90 | 9,91 | 37,72 | 25,84 |
| Sortformer streaming 640 ms | 29,66 | 36,63 | 7,75 | 39,07 | 28,28 |

## 14. Streaming, RTF và checkpoint hiện còn

Zipformer nhận mỗi bước 64 fbank frame mới (640 ms), cộng **13 frame tail/lookahead**; phát 16 output frame 25 Hz và giữ cache encoder/head. Temporal head là causal, nhưng toàn frontend có thêm context khoảng 130 ms. Vì vậy câu “zero lookahead, total 660,7 ms” trong paper chưa khớp code. Cần đo lại timestamp frontend/incremental extraction để chốt latency end-to-end; không đổi 640 ms cadence thành một phép đo end-to-end đã được xác nhận.

RTF đã lưu dùng streaming cached model compute, **không bao gồm feature extraction/I/O**, loại warmup:

| Thiết bị trong báo cáo | Mean/chunk | p95/chunk | RTF model-only | Nguồn |
| --- | --- | --- | --- | --- |
| RTX 3090 GPU | 20,75 ms | 21,94 ms | 0,0320 | [rtf_gpu.log](logs/rtf_gpu.log) |
| i7-12700 CPU 1 thread | 51,44 ms | 52,99 ms | 0,0795 | [rtf_cpu1.log](logs/rtf_cpu1.log) |

[Pyramid roadmap](PYRAMID_ORDINAL_ROADMAP.md) đã nhắc frontend 130 ms; [wrapper](src/models/zipformer_wrapper.py) và [inference](src/infer_streaming.py) xác nhận `pad_length=13`. FastConformer chưa có cached streaming path trong wrapper này; WavLM offline không có latency streaming để so cùng mức.

Checkpoint ZipCount paper hiện còn: [`logs/stackgrad_pre012_s3456/step3000.pt`](logs/stackgrad_pre012_s3456/step3000.pt), với [config tương ứng](artifacts/stack_mask_graduation/stackgrad_pre012_s3456.yaml). Nó được giữ theo lựa chọn lưu trữ sau benchmark; điểm đánh giá model vẫn báo mean 3 seed, không thay bằng seed tốt nhất.

Snapshot hiện còn các checkpoint FastConformer/WavLM sau này, nên [CHECKPOINTS_README cũ](logs/CHECKPOINTS_README.md) nói “chỉ giữ một checkpoint” không còn mô tả đủ trạng thái hiện tại:

**FastConformer:** `logs/fastconformer_frontloaded_s1234/step3000.pt`, `logs/fastconformer_hc6_s1234/step3000.pt`, `logs/fastconformer_hc6_s2345/step3000.pt`, `logs/fastconformer_hc6_s3456/step3000.pt`.

**WavLM frozen:** `logs/wavlm_hc6_causalhead_s1234/step3000.pt`, `logs/wavlm_hc6_causalhead_s2345/step3000.pt`, `logs/wavlm_hc6_causalhead_s3456/step3000.pt`, `logs/wavlm_hc6_s1234/step3000.pt`, `logs/wavlm_hc6_s2345/step3000.pt`, `logs/wavlm_hc6_s3456/step3000.pt`.

**WavLM LoRA/control:** `logs/wavlm_lora_head_only_800/step100.pt`, `logs/wavlm_lora_head_only_800/step200.pt`, `logs/wavlm_lora_head_only_800/step400.pt`, `logs/wavlm_lora_head_only_800/step800.pt`, `logs/wavlm_lora_qv_all24_r8_800/step100.pt`, `logs/wavlm_lora_qv_all24_r8_800/step200.pt`, `logs/wavlm_lora_qv_all24_r8_800/step400.pt`, `logs/wavlm_lora_qv_all24_r8_800/step800.pt`.

Manifest train trong nhiều config vẫn trỏ `/media/edabk/500GB Hard Disk/data_diar/manifests/train_manifest_v3.json`, không tồn tại tại thời điểm tổng hợp. Checkpoint initialization V3 cũ cũng không còn trong `logs/`. Do đó config + source hiện tại chưa đủ bảo đảm tái tạo nguyên vẹn run lịch sử; phải phục hồi đúng dữ liệu, checkpoint và revision.

## 15. Các điểm cần đọc đúng khi sử dụng báo cáo

1. **Backbone lineage:** hệ thống chính head-only load backbone đã fine-tune V3. Không viết “ASR weights giữ nguyên từ đầu đến cuối toàn bộ dự án”.
2. **Latency:** 640 ms cadence khác end-to-end; code có 130 ms frontend lookahead. RTF model-only không bao gồm extraction.
3. **Protocol:** unified 90 s/100 Hz, window 30 s/25 Hz, composite DEV và clip-level accuracy là các phép đo khác nhau. Pooled F1 và recording-mean cũng khác. Ngay trong artifact unified, cùng window ID chưa bảo đảm cùng frame support: chênh độ dài prediction làm thiếu một đoạn cuối cửa sổ (mục 3.1).
4. **Paper vs raw artifact:** dùng mean score từng seed nhất quán ở đây; không đồng nhất với pooled-confusion qua seed trong bootstrap. Sai khác nhỏ được giữ minh bạch.
5. **P5/P5c/P6c:** paper ghép một số con số từ run cũ double-smoothing và các regime khác. P5c là ablation head sạch; P6c là full fine-tune.
6. **CORN/refinement:** `p8b_cum` là cumulative loss trên softmax. Native CORN và two-stage có code/smoke, chưa có bảng long-run hoàn chỉnh để kết luận.
7. **Parameter matching:** mask giữ graph nên matched; log-mel và swap backbone thay width/count tham số nên không matched. Mask không cắt compute.
8. **Negative result:** không vượt adoption threshold khác với hoàn toàn không có hiệu ứng. A1 framewise gate có dấu dương ổn nhưng quá nhỏ; boundary AP cải thiện dù F1 gần như không đổi.
9. **Selection:** `vox_sel` đã dùng chọn model. Fine-tune 5 seed ở mục 9 là pyramid A0; WavLM LoRA mới chỉ một seed DEV, chưa xác nhận trên heldout.
10. **Một số câu trong paper cần sửa trước khi trích:** “OSD recall 0,695 cao nhất” mâu thuẫn ngay với 0,752 được ghi cho DiariZen; “real corpora unaugmented” không đúng với gain jitter trong code; “Adam” trong paper khác AdamW trong train/preregister. Không dùng các câu này làm kết luận thực nghiệm.

## 16. Kết luận theo từng hướng nghiên cứu

| Hướng | Điều kết quả hiện có hỗ trợ |
| --- | --- |
| V1→deformable V2 | Tăng lớn trên LibriMix; thay đồng thời nhiều yếu tố, chưa attribution từng thành phần |
| Domain adaptation r5→r8 | Nâng AMI rõ rệt; train lâu hơn thường làm giảm validation |
| KD DiariZen | Chưa chứng minh gain ổn định qua 3 seed; uncertain KD không giúp |
| TCN-192 | Lựa chọn head sạch tốt hơn deformable trên vox_sel khi backbone frozen |
| Stack pre012 | Được chọn vì OSD/class-2 và benchmark; frame F1 tăng nhưng đoạn kém mượt |
| Event-state filter | Có hại ở locked endpoint; bị loại |
| Pyramid chain | Có implementation và nhiều ablation; chưa vượt TCN root theo complexity gate |
| Full FT sau head-only | Seed-unstable; không chọn best seed để kết luận |
| Temporal KL | Thành phần loss đóng góp lớn nhất trong ablation unified |
| FastConformer | Mean gần ZipCount; chưa có lợi thế thống kê/latency cached được xác nhận |
| WavLM frozen + non-causal TCN | Model counting nội bộ có unified mean cao nhất đã thấy; offline |
| WavLM partial unfreeze | Chưa có lợi ích rõ trên một-seed composite DEV |
| WavLM LoRA | Có gain tại một số step DEV, nhỏ và chưa kiểm định heldout/multi-seed |
| Hysteresis decoder | DEV tăng rất nhỏ; heldout giảm mean, không được adopt |

## 17. Phụ lục nguồn và kết quả từng run

- [model_results.csv](reports/model_results.csv): kết quả từng artifact JSON, gồm pooled F1, recording-mean, OSD, class F1, composite DEV hoặc sequence metrics tùy schema. `kind` phân biệt protocol/schema; không lấy mean toàn file CSV.
- [training_validation.csv](reports/training_validation.csv): mọi dòng `[val@step]` có macro-F1 trích từ training/driver log trong snapshot. Driver log có thể lặp lại run log; phải lọc `source` trước khi aggregate. Số trong log đã làm tròn, không có độ chính xác như confusion JSON.
- [unified_support_audit.csv](reports/unified_support_audit.csv): đối chiếu tập window ID, số frame và histogram ground-truth của từng confusion JSON trong bảng unified với baseline RTTM DiariZen.
- [source_inventory.csv](reports/source_inventory.csv): đường dẫn, kích thước, SHA-256 của tài liệu/report/config/script đã quét; `.stale.*`, binary checkpoint, audio, raw labels và log build LaTeX không dùng làm metric hiện hành.
- Nguồn nền: [README lịch sử](README.md), [báo cáo 04/08](paper/RESULTS_2026-08-04.md), [ICASSP](paper/main_icassp.tex), [TASLP](paper/main_taslp.tex), [event roadmap](EVENT_STATE_ROADMAP.md), [pyramid roadmap](PYRAMID_ORDINAL_ROADMAP.md), [stack preregister](STACK_MASK_GRADUATION_PREREG.md), [pyramid mask preregister](PYRAMID_MASK_TRANSFER_PREREG.md), [checkpoint reference](logs/CHECKPOINTS_README.md).

`chatgpt.txt` và `promt.txt` là gợi ý/spec lịch sử, không phải báo cáo kết quả chạy. Các trích dẫn khoa học ngoài repo không được kiểm chứng lại trên web trong lượt tổng hợp này.

Phạm vi trích xuất: **405 artifact JSON có metric**, **1574 dòng validation**, **1183 file trong danh mục nguồn**. Các số này mô tả coverage file, không phải số model độc lập; một model có nhiều corpus/seed/checkpoint. Không train lại, không chạy suy luận, không kiểm chứng lại significance ngoài các báo cáo đã lưu.

## 18. Phạm vi bản lưu trên GitHub

[.gitignore](.gitignore) giữ mã nguồn, config, báo cáo, bảng kết quả JSON/CSV và log thí nghiệm; bỏ dataset, checkpoint, cache prediction, TensorBoard, download log và repository phụ thuộc. Các đường dẫn checkpoint trong tài liệu là tham chiếu đến snapshot local, không phải file được đính kèm trên GitHub. Danh mục SHA-256 mô tả nguồn đã đọc ở máy hiện tại; một số nguồn bị loại khỏi bản Git theo quy tắc này.

Icefall local dùng remote `https://github.com/k2-fsa/icefall.git`, commit `0904e490c5fb424dc5cb4d14ae468e4d32a07dc4`. Thay đổi local trong `export-onnx.py` được lưu tại [icefall_local.patch](reports/icefall_local.patch). Sau khi clone dự án trên máy khác, khôi phục dependency từ thư mục gốc dự án bằng:

```bash
git clone https://github.com/k2-fsa/icefall.git third_party/icefall
git -C third_party/icefall checkout 0904e490c5fb424dc5cb4d14ae468e4d32a07dc4
git -C third_party/icefall apply ../../reports/icefall_local.patch
```

Cần phục hồi dataset/checkpoint riêng và sửa đường dẫn môi trường trước khi chạy. `scripts/setup_new_machine.sh` hiện có phần cài Python dependency bị comment; hướng dẫn bundle lịch sử trong README gốc không đồng nghĩa clone GitHub là đủ để tái tạo toàn bộ thí nghiệm.
