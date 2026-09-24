# Hướng nâng ZipCount lên mức SOTA

Ngày rà soát: **24/09/2026**. Tài liệu này được viết sau khi đọc mã nguồn, cấu hình, test, báo cáo, log và các artifact trong project. Các điểm số bên dưới là kết quả đã lưu trong snapshot; không phải kết quả của một lượt train mới.

## 1. Kết luận quan trọng nhất

ZipCount hiện giải quyết **speaker counting**: mỗi frame trả về `0 / 1 / 2 / 3+` người đang nói. DiariZen giải quyết **speaker diarization**: trả lời ai nói khi nào, sau đó lấy embedding và clustering để giữ speaker identity. Vì vậy ZipCount hiện không thể được gọi là hệ diarization SOTA hoặc so trực tiếp bằng DER. Nó là một front-end đếm người/OSD có latency thấp.

Nếu mục tiêu là router cho multi-talker ASR, nên giữ bài toán counting và cạnh tranh bằng macro-F1/OSD/latency. Nếu mục tiêu là đánh bại DiariZen trên diarization, cần thêm speaker-aware segmentation, embedding và global clustering; tăng TCN hoặc đổi loss đếm sẽ không đủ.

Kết quả hiện có cho thấy ba sự thật:

1. Hệ causal Zipformer + TCN + mask đạt mean unified **0,5753**, với RTF model-only **0,032 GPU** và **0,0795 CPU một thread**. Đây là điểm mạnh về latency và chi phí.
2. DiariZen trên cùng bảng count đạt **0,6398**; khoảng cách khoảng **0,0645**. WavLM-Large + TCN offline của project đạt **0,6027**, cao hơn ZipCount **0,0275**. Điều này cho thấy biểu diễn WavLM còn nhiều thông tin chưa được head causal hiện tại khai thác.
3. Các ablation head nhỏ (mask, gate, event-state, pyramid) không tạo ra bước nhảy lớn. Hướng có khả năng thay đổi trần kết quả là dữ liệu real, fine-tuning/teacher đúng cách, Conformer hoặc state-space temporal modeling, và speaker-aware objective.

Phân tích context oracle trong paper cho thấy full context chỉ thêm khoảng `+0,0202` trên một protocol. Causality là chi phí thật, nhưng phần gap còn lại đến từ representation, domain, capacity và objective; chỉ tăng lookahead sẽ không đủ để vượt DiariZen.

## 2. Project đang có gì

### Kiến trúc

- **Zipformer2 streaming**: 6 stack, causal, output 25 Hz; encoder public được dùng làm backbone ASR.
- **Hypercolumn**: ghép sáu stack `192/256/384/512/384/256` thành 1984 chiều.
- **TCN-192**: sáu block depthwise dual-dilation, dilation `1..32`, receptive field xấp xỉ 5,08 giây ở 25 Hz.
- **Stack mask**: giữ graph và số tham số, chỉ chặn input của một số stack. Arm `pre012` là lựa chọn tốt nhất trong graduation trên `vox_sel`.
- **Event-state**: emission đếm + DOWN/STAY/UP + sticky Markov filter; locked endpoint không tốt nên đã loại.
- **Gated pyramid**: projection từng stack, gate global/framewise, boundary/direction, CORN và stage 2. Nhiều phần mới dừng ở smoke test hoặc ablation ngắn.
- **FastConformer**: 17 layer, 6 taps, native 12,5 Hz lặp lên 25 Hz. Wrapper hiện chưa có `forward_streaming` cached thực tế.
- **WavLM-Large**: 24 layer, 6 taps, raw waveform 50 Hz average-pool xuống 25 Hz. WavLM là bidirectional nên mọi run đều offline, dù TCN head causal.
- **LoRA/partial unfreeze**: có hạ tầng LoRA q/v cho 24 layer và mở 2/4 layer cuối, nhưng kết quả hiện chỉ là một seed trên composite DEV.

### Dữ liệu và train

- LibriMix được làm sạch bằng energy VAD từng source, trộn on-the-fly, thêm solo/noise để tạo class 0/1.
- Các giai đoạn sau bổ sung AMI, NOTSOFAR-1 và các real corpus khác; tỷ lệ real được thay đổi qua stage. Manifest chính của r8 hiện trỏ tới ổ ngoài đã không còn trong snapshot, nên không thể tái tạo nguyên vẹn run lịch sử.
- Augmentation có gain, RIR, noise; real far-field được bỏ qua RIR/noise để tránh double-reverb.
- Loss có CE/focal/SORD, VAD/OSD, expected-count MAE, consistency, monotonicity, Dice, smoothness, boundary và KD.
- Train có kiểm tra gradient/AMP, seed, checkpoint lineage, paired ablation và test causal cache. Đây là nền tảng tốt để làm nghiên cứu có kiểm soát.

## 3. Điểm mạnh có thể biến thành lợi thế SOTA

### 3.1. Latency thực sự thấp

ZipCount có streaming cache cho Zipformer và TCN, chạy CPU một thread, trong khi DiariZen là pipeline offline theo cửa sổ và clustering toàn recording. Đây là trục cạnh tranh thực tế: **độ chính xác gần DiariZen trong ngân sách 160–640 ms**.

Tuy nhiên cần ghi latency thành phần: cadence 640 ms, frontend còn khoảng 130 ms context theo code, và RTF hiện tại không bao gồm feature extraction/I/O. Chỉ sau khi đo end-to-end mới dùng con số latency triển khai.

### 3.2. Counting đơn giản hơn diarization

Không có permutation problem, không cần speaker inventory, không cần clustering và không cần speaker identity nếu downstream chỉ cần biết có nên route sang multi-talker ASR. Đây là use case mà DiariZen phải chạy thêm embedding/VBx trong khi ZipCount có thể phát tín hiệu ngay.

### 3.3. Quy trình thực nghiệm tốt hơn mức thông thường

Project có preregistration, seed matching, source hash, lineage sidecar, test full-vs-chunked, test mask không rò gradient và bảng unified nhiều corpus. Những cơ chế này nên được giữ khi đổi model; chúng giúp tránh một điểm số cao do leakage hoặc protocol khác.

### 3.4. WavLM là tín hiệu rõ nhất về trần mới

WavLM HC6 offline đạt **0,6027**, cao hơn ZipCount causal **0,5753**. Khoảng cách này đáng giá hơn việc tiếp tục thêm một loss nhỏ vào TCN. Cần đưa WavLM representation vào một backend mạnh hơn, rồi distill sang student streaming.

## 4. Điểm yếu và nguyên nhân của khoảng cách với DiariZen

### 4.1. Sai bài toán nếu dùng DER làm mục tiêu

ZipCount không có identity. DiariZen dùng WavLM-Large + Conformer EEND/powerset để dự đoán local speaker activity, overlap-add các cửa sổ, lấy embedding loại overlap và dùng VBx/PLDA để ghép identity toàn recording. Project hiện chỉ có count head, nên không thể sửa sai speaker A/B hoặc đo DER.

### 4.2. Backbone/head hiện chưa tương đương DiariZen

WavLM experiment của project ghép sáu hidden state rồi đưa qua TCN. DiariZen dùng learned layer weighting và Conformer backend được fine-tune trên nhiều real corpus. TCN-192 là decoder nhẹ, hợp streaming nhưng không có self-attention/cross-frame modeling linh hoạt như Conformer.

KD hiện chỉ marginalize teacher powerset thành bốn count classes và chỉ học các frame teacher đúng với GT. Nó truyền count posterior, không truyền local speaker streams, embedding, boundary hay clustering signal. Đây là một teacher yếu cho mục tiêu diarization.

### 4.3. Dữ liệu synthetic vẫn chi phối thiết kế

LibriMix giúp scale và tạo overlap có kiểm soát, nhưng không tái hiện đầy đủ reverberation, microphone, turn-taking, channel mismatch, speech style và annotation policy của meeting corpus. Class `3+` vẫn khó; các log r7/r8 cho F1 class 3+ khoảng `0,22–0,30` trên validation.

DiariZen được huấn luyện trên tập real đa dạng gồm AMI, AISHELL-4, AliMeeting, NOTSOFAR-1, MSDWild, DIHARD3, RAMC và VoxConverse. Project cần bảo đảm split speaker/session disjoint, tỷ lệ sampling và chất lượng nhãn của từng corpus trước khi kết luận kiến trúc.

Class-wise cho thấy bottleneck chính nằm ở overlap/in-the-wild, không phải class 0/1. Trên `vox_lock`, ZipCount mask có F1 `[0,693; 0,948; 0,464; 0,063]` cho `[0,1,2,3+]`, còn DiariZen là `[0,782; 0,964; 0,644; 0,236]`; OSD lần lượt `0,472` và `0,659`. Vì vậy ưu tiên phải là recall/precision của class 2/3+, overlap event và boundary, thay vì tiếp tục tối ưu class 0/1 vốn đã cao.

### 4.4. Fine-tuning bị overfit/forgetting

R7 full fine-tune đạt peak rất sớm rồi giảm; r8 cũng tốt nhất ở step đầu. Giảm learning rate không giải quyết hoàn toàn. Đây là dấu hiệu cần progressive unfreezing, differential LR, replay real+synthetic, EMA/SWA hoặc teacher regularization; không nên tiếp tục full fine-tune 66M tham số với một LR chung.

### 4.5. Một số kết quả chưa đủ để kết luận

- WavLM/LoRA mới có ít seed và chủ yếu là composite DEV.
- FastConformer chưa đo cached streaming RTF.
- Unified evaluator có cùng window ID nhưng một số model bị cắt ít frame hơn sau `min(len(pred), len(gt))`.
- `vox_sel` đã dùng để chọn thiết kế; `vox_lock` mới phù hợp cho xác nhận.
- Paper có một số con số ghép từ run cũ khác regime; README tổng hợp đã đánh dấu các trường hợp này.
- `src/infer_streaming.py` còn tính `n_out=T//4` theo floor; cần dùng `h_lens` thực tế và flush cuối stream để không mất tail.
- `ZipCountLoss.forward` có nhánh legacy cắt `labels[:, :T]` khi length lệch, có thể che misalignment thay vì fail sớm. `label_utils.py` dùng majority trên `np.array_split`, dễ làm mất overlap 40 ms hoặc boundary ngắn.
- CRNN baseline chưa dùng `pack_padded_sequence`; padded tail có thể đi vào GRU. Dataset trộn waveform/RIR/noise/fbank trên CPU cho từng item, làm bottleneck throughput khi scale data.

## 5. Hai mục tiêu SOTA cần tách riêng

### Mục tiêu A: SOTA counting/OSD dưới latency thấp — khuyến nghị trước

Sản phẩm nên là:

```text
causal audio
  → causal Zipformer hoặc streaming Conformer
  → count 0/1/2/3+, VAD, OSD, boundary, uncertainty
  → multi-talker ASR router
```

So sánh với DiariZen bằng count-F1/OSD-F1 trên cùng frame support, và báo cáo thêm latency, RTF, RAM, params. Không gọi kết quả này là DER SOTA.

### Mục tiêu B: SOTA diarization cạnh tranh trực tiếp DiariZen

Cần mở rộng thành:

```text
WavLM/pruned WavLM
  → learned layer weighting
  → Conformer/EEND powerset local activity
  → overlap-add
  → overlap-excluded speaker embedding
  → VBx/PLDA hoặc online speaker cache
  → RTTM + count/OSD auxiliary head
```

Count head của ZipCount có thể giữ làm auxiliary task và làm tín hiệu router, nhưng không còn là output duy nhất.

## 6. Lộ trình có xác suất thành công cao

### Phase 0 — khóa benchmark trước khi đổi model

Mục tiêu là loại bỏ mọi lợi thế do protocol.

1. Chốt một evaluator duy nhất: cùng window ID, cùng số frame, cùng 100 Hz reference, padding/gap rõ ràng, không dùng `min` để silently bỏ tail.
2. Re-score bốn hệ trên cùng support: ZipCount pre012, WavLM HC6, DiariZen checkpoint đang dùng, và DiariZen upstream hiện tại.
3. Tách `vox_sel` khỏi `vox_lock`; chỉ `vox_lock` và test corpus mới được dùng cho kết luận cuối.
4. Lưu macro-F1, OSD-F1, F1 class 2/3+, onset/offset F1, ECE/Brier, DER/JER nếu có identity, RTF end-to-end và peak RAM.

Điều kiện qua phase: mọi hệ có cùng support và kết quả có thể chạy lại từ manifest/checkpoint đã hash. Evaluator phải fail-closed khi output length/reference length không khớp.

Sửa các lỗi alignment trước khi đọc một gain nhỏ hơn khoảng `0,005`; mức này có thể do số frame được chấm hoặc cách aggregate.

### Phase 1 — xây teacher mạnh cho counting

Đây là thí nghiệm có giá trị thông tin cao nhất. Bản đầu tiên nên chạy frozen WavLM → scalar mix → Conformer, sau đó mới mở last-4/full; như vậy phân biệt được bottleneck ở head với bottleneck ở representation.

```text
WavLM-Large (hoặc pruned WavLM)
  → learned scalar layer mix / gated layer mix
  → 4–6 layer Conformer, d_model 512–768
  → powerset/local-count + VAD/OSD/boundary heads
```

Thiết lập:

- Fine-tune toàn bộ WavLM với LR backbone nhỏ hơn head 10–100 lần; thử progressive unfreeze.
- Thử learned scalar/gated layer mix trước khi concat sáu layer cố định; concat làm input head WavLM thành 6144 chiều và không tự thể hiện layer nào hữu ích cho từng miền.
- Huấn luyện real corpus theo sampling cân bằng session/corpus; synthetic chỉ làm replay và tăng overlap hiếm.
- Giữ soft powerset posterior và count marginalization. Không chỉ giữ argmax count.
- Dùng boundary loss có tolerance, duration/segment consistency và uncertainty calibration; không tăng smoothness đồng loạt vì dễ làm mất onset/offset.
- Đánh giá 3 seed trên held-out, không dùng LoRA một seed để tuyên bố gain.

Điều kiện dừng: nếu teacher không vượt WavLM HC6 `0,6027` trên unified strict, không tiếp tục tuning head nhỏ; quay lại data/label hoặc checkpoint/pretraining.

### Phase 2 — distill teacher thành student causal

Teacher offline cung cấp:

- hidden representation ở nhiều layer;
- powerset/count posterior với temperature;
- boundary và duration targets;
- confidence/uncertainty mask;
- nếu làm diarization: local speaker activity và embedding target.

Nếu vẫn ưu tiên counting, nên thêm auxiliary PIT speaker-activity branch với tối đa `K=4` streams từ RTTM/track identity; count được suy ra bằng tổng activity. Nhánh này đưa identity cues của DiariZen vào student nhưng vẫn cho phép output chính là count. Với synthetic data có speaker track sẵn, đây là supervision rẻ; với real data dùng RTTM và PIT.

Student nên là Zipformer hiện tại hoặc streaming Conformer/SSM nhẹ. Loss đề xuất:

```text
L = L_GT_count
  + λ1 KL(p_teacher || p_student)
  + λ2 hidden_projection_MSE
  + λ3 boundary/duration
  + λ4 OSD focal/Dice
  + λ5 calibration/consistency
```

KD không nên chỉ học frame teacher “đúng tuyệt đối”; nên dùng confidence-weighted soft target, tách non-overlap/overlap và không áp teacher trên synthetic mà teacher chưa được kiểm định.

Thử latency 160, 320 và 640 ms với cùng student. Kết quả mong muốn là đường Pareto accuracy–latency, không chỉ một operating point.

### Phase 3 — nếu cần đánh bại DiariZen bằng DER

Thêm ba thành phần theo thứ tự:

1. Powerset 16 lớp cho tối đa bốn local speakers, với overlap-aware loss.
2. Speaker embedding head được train bằng supervised contrastive/AAM-softmax/GE2E trên speaker labels; loại hoặc giảm trọng số vùng overlap khi lấy embedding.
3. Backend VBx/PLDA và overlap-add theo recording. Sau đó mới thử online speaker cache để giữ latency.

Khởi đầu từ checkpoint DiariZen/teacher nếu license cho phép; so sánh với upstream bằng đúng dataset và protocol. Distill count từ DiariZen là bước phụ, không thay thế identity distillation.

### Phase 4 — compression và triển khai

Sau khi accuracy đã ổn định:

- structured prune WavLM/Conformer theo FLOPs hoặc latency, rồi KD recovery;
- FP16/INT8 và export ONNX/TensorRT/torch.compile;
- cache-aware inference cho FastConformer/WavLM student;
- benchmark CPU/GPU batch 1 với feature extraction, I/O và timestamp thực tế.

Không prune trước khi khóa teacher và protocol; nếu không sẽ khó biết mất điểm do model hay do data/evaluator.

## 7. Các thí nghiệm nên chạy theo thứ tự

| Ưu tiên | Thí nghiệm | Câu hỏi | Tiêu chí giữ |
|---|---|---|---|
| P0 | Strict evaluator + re-score DiariZen/WavLM/ZipCount | Khoảng cách thật là bao nhiêu? | Support giống nhau, JSON/hash đầy đủ |
| P1 | WavLM scalar mix + Conformer, frozen rồi last-4 rồi full FT | Head hiện tại có phải bottleneck? | Vượt 0,6027 trên 3 seed held-out |
| P2 | Real-balanced curriculum + soft/forced-aligned labels | Data hay model là bottleneck? | Tăng class 2/3+ mà không giảm class 0/1 |
| P3 | Teacher-to-causal KD tại 160/320/640 ms | Có giữ được offline gain khi streaming? | Pareto tốt hơn ZipCount ở cùng latency |
| P4 | Powerset + speaker embedding + VBx | Có chuyển được sang DER? | DER/JER giảm trên AMI/Vox/DIHARD |
| P5 | Structured pruning/quantization | Có triển khai được? | Không giảm đáng kể score, đo end-to-end |

## 8. Những hướng không nên ưu tiên lúc này

- Thêm một loss nhỏ hoặc một biến thể gate khi teacher WavLM + TCN còn chưa được fine-tune tương đương.
- Dùng `final stack only`, pyramid/CORN smoke hoặc LoRA một seed làm bằng chứng SOTA.
- Tối ưu trên `vox_sel` rồi báo cáo như test.
- So sánh DER của DiariZen với ZipCount count-F1.
- Gọi WavLM causal-head là streaming; encoder WavLM vẫn bidirectional.
- Dùng synthetic LibriMix để distill teacher DiariZen khi teacher sai mạnh trên fully-overlapped mixtures.
- Tuyên bố latency end-to-end từ RTF model-only.
- Dùng stack mask như tối ưu compute: mask hiện chỉ chặn đặc trưng sau khi backbone đã chạy đủ sáu stack, nên chưa giảm FLOPs.

## 9. Rủi ro và cách giảm

| Rủi ro | Cách giảm |
|---|---|
| Leakage qua session/speaker hoặc `vox_sel` | speaker/session-disjoint split, locked test, hash manifest |
| Frame support khác nhau | evaluator fail-closed khi length/reference không khớp |
| Class 3+ hiếm và annotation không chắc | sampling theo frame, soft labels, report per-class/CI |
| Full FT quên ASR/overfit real dev | differential LR, replay, EMA, early dense validation |
| Teacher offline không truyền được causal timing | hidden/logit/boundary KD + causal student benchmark |
| WavLM/DiariZen license và dataset access | ghi rõ model license, provenance, allowed use |
| Không tái tạo được r8 | phục hồi manifest/checkpoint hoặc chạy lại pipeline với version/hash mới |

## 10. Quyết định khuyến nghị

Trong ngắn hạn, chọn **Mục tiêu A**: xây một `ZipCount-SSL` teacher WavLM+Conformer và distill thành student causal. Đây là hướng tận dụng kết quả mạnh nhất đã có (`WavLM HC6 0,6027`), vẫn giữ lợi thế latency và phù hợp với use case multi-talker ASR.

Chỉ chuyển sang **Mục tiêu B** khi sản phẩm thực sự cần “who spoke when”. Khi đó hãy coi ZipCount hiện tại là count/OSD auxiliary module và xây speaker embedding + clustering pipeline theo kiểu DiariZen. Một head TCN dự đoán bốn số đếm không thể tự tạo ra speaker identity.

## 11. Ý tưởng mới rút ra trực tiếp từ lỗi của TCN

Các lỗi hiện tại đủ cụ thể để thiết kế một decoder có giả thuyết rõ ràng, thay vì tiếp tục ghép thêm một block. Trong diagnostic `phase8_chain`, ZipCount có fragmentation overlap **3,307**, boundary F1 **0,386** với precision chỉ **0,260** và recall **0,746**. Strict overlap onset/offset recall lần lượt là **0,639/0,576**; trong các overlap thật, **19,4%** bị bỏ sót hoàn toàn và **42,6%** chỉ được bắt một phần. Như vậy TCN không chỉ thiếu context. Nó đang có hai lỗi đối nghịch: tạo nhiều transition giả và không duy trì đủ coverage cho overlap thật.

### 11.1. Thay smoothness chung bằng decoder trạng thái đoạn

TCN hiện dự đoán từng frame rồi dùng symmetric KL để kéo posterior của hai frame kề nhau lại gần. Loss này không biết một đoạn overlap nên dài bao lâu, transition hợp lệ nằm ở đâu, hoặc một đoạn dương ngắn là lỗi hay một overlap thật. Event-state filter trước đó cũng không giải quyết được vì nó áp một prior sticky cố định và chỉ cho DOWN/STAY/UP; prior đó làm hại các transition nhanh, multi-jump và các đoạn có confidence thấp.

Đề xuất decoder gồm ba đầu ra từ cùng representation:

```text
TCN / temporal backend
  ├─ emission z_t          : count 0/1/2/3+
  ├─ transition hazard h_t : xác suất bắt đầu/kết thúc một trạng thái
  └─ duration state d_t    : tuổi hoặc thời lượng còn lại của đoạn
```

Thay vì lọc bằng một ma trận chuyển trạng thái cố định, dùng hazard phụ thuộc vào state, duration và context:

```text
p_t(s, d) = normalize( emission_t(s)
                        × transition_t(s, d | s_prev, d_prev) )
```

Hazard phải cho phép `stay`, `up1`, `up2+`, `down1` và `down2+`; không ép mọi thay đổi thành một bước. Duration có thể bắt đầu bằng vài bucket (`1–2`, `3–5`, `6–12`, `13+` frame) để giữ decoder nhỏ. Đây là một semi-Markov filter nhân quả học được, không phải một hậu xử lý cố định. Nó tạo ra persistence thích nghi: mạnh ở đoạn ổn định, nhưng có thể chuyển nhanh khi có evidence rõ.

### 11.2. Loss phải đo đúng lỗi mà diagnostic đã chỉ ra

Giữ CE/count và OSD, nhưng thay smoothness toàn cục bằng các thành phần có ý nghĩa segment:

```text
L = L_count
  + λ_h   L_hazard
  + λ_occ L_occupancy
  + λ_run L_transition_count
  + λ_cal L_calibration
  + λ_hard L_overlap_boundary
```

- `L_hazard`: target onset/offset của mỗi đoạn với tolerance 1–2 frame; focal weighting cho các transition hiếm.
- `L_transition_count`: so sánh tổng hazard dự kiến với số onset/offset thật. Ngoài vùng boundary, phạt hazard dương để trực tiếp tăng boundary precision và giảm fragmentation.
- `L_occupancy`: với từng overlap run, buộc trung bình `p(count>=2)` trên toàn đoạn phải cao và ngoài đoạn phải thấp. Chuẩn hóa theo từng run để một đoạn dài không lấn át nhiều đoạn ngắn.
- `L_overlap_boundary`: Tversky/Focal-Tversky bất đối xứng cho class `2/3+`, trong đó false positive và false negative có trọng số khác nhau. Có thể tăng trọng số false positive ở negative interval để sửa precision `0,260`, rồi tăng false negative ở các run bị missed/partial để sửa coverage.
- `L_calibration`: Brier hoặc soft-ECE cho emission/hazard; decoder chỉ được chuyển trạng thái khi confidence tương xứng.

Một surrogate đơn giản để kiểm tra ý tưởng trước khi viết semi-Markov đầy đủ là:

```text
L_false_boundary = Σ_{t không gần boundary thật} h_t
L_run_coverage   = Σ_{run thật} | mean(p_overlap[run]) - target_coverage |
L_run_count      = | Σ_t h_t - số transition thật |
```

Ba term này có thể thêm vào TCN hiện tại mà không đổi backbone. Nếu chúng làm fragmentation giảm và boundary precision tăng mà macro-F1 không giảm, đó là bằng chứng loss mới đang sửa đúng cơ chế lỗi.

### 11.3. Dùng bằng chứng layer để phân vai, không concat tất cả vào một head

Nếu probe layer của project xác nhận final layer ổn định nhất, hãy coi đó là bằng chứng về **vai trò** chứ không phải bằng chứng rằng các layer trước vô ích:

- final layer làm `slow state branch`: count ổn định và duration dài;
- early/middle layer làm `fast event branch`: onset, offset và thay đổi overlap;
- một router theo state/hazard học trọng số hai nhánh theo từng frame.

Có thể triển khai trước bằng hai projection nhỏ và một gate:

```text
s_t = P_final(h_final)
e_t = P_event([h_early - h_early_prev, h_mid - h_mid_prev])
g_t = sigmoid(MLP([s_t, e_t, entropy(z_t)]))
u_t = g_t * e_t + (1 - g_t) * s_t
```

Boundary head dùng `e_t`, emission/duration head dùng `u_t`. Cách này biến bằng chứng “final tốt hơn” thành một kiến trúc có vai trò rõ ràng; nó khác với việc ghép sáu stack rồi hy vọng TCN tự phân biệt thông tin.

Một biến thể có giá trị nghiên cứu là **residual error refiner**: head thứ hai chỉ được phép sinh `Δz_t` khi entropy emission cao, hai nhánh temporal bất đồng hoặc đang gần boundary. Loss của head này chỉ tập trung vào frame overlap, missed/partial và boundary. Nó khai thác bản đồ lỗi của TCN thay vì thêm capacity đều trên mọi frame.

### 11.4. Ablation tối thiểu để biết ý tưởng có thật sự mới và có ích

Không cần chạy lại toàn bộ không gian kiến trúc. Giữ checkpoint/backbone và dùng cùng `vox_lock`, strict evaluator, 3 seed:

| Run | Thay đổi | Giả thuyết cần kiểm tra |
|---|---|---|
| A0 | TCN hiện tại | Control và số liệu lỗi gốc |
| A1 | Thêm hazard + `L_hazard` + `L_false_boundary` | Precision biên tăng, fragmentation giảm |
| A2 | A1 + occupancy/transition-count loss | Missed/partial overlap giảm, coverage tăng |
| A3 | A2 + learned duration/semi-Markov filter | Giữ đoạn đúng mà không làm trễ boundary |
| A4 | A3 + event/state layer router | Early/mid chỉ hỗ trợ event, final giữ count ổn định |

Báo cáo cùng lúc macro-F1, F1 class 2/3+, OSD F1, boundary precision/recall, fragmentation, missed/partial overlap, onset/offset bias và Brier/ECE. Go/no-go ban đầu nên là: boundary precision tăng rõ từ `0,260`, fragmentation giảm từ `3,307`, trong khi OSD recall không giảm quá mức và macro-F1 giữ được khoảng hiện tại. Các ngưỡng này là tiêu chí nghiên cứu cần xác nhận, không phải kết quả đã đạt.

### 11.5. Thứ tự triển khai thực tế

1. Sửa frame-support mismatch và tắt nhánh `labels[:, :T]` silent slice trước khi train; nếu không, loss mới có thể chỉ học cách bù lỗi alignment.
2. Thêm hazard head và ba surrogate loss vào TCN hiện tại, chưa thêm semi-Markov. Đây là test rẻ nhất để xác minh cơ chế.
3. Nếu A1/A2 cải thiện đúng diagnostic, thay hậu xử lý bằng learned duration filter và chạy latency/cache test.
4. Sau đó mới thử layer router và distill từ WavLM/Conformer teacher. Teacher vẫn là trục để nâng trần accuracy; decoder mới giải quyết lỗi segment và latency của student.

Điểm có thể viết thành contribution không phải “thêm một TCN”, mà là: **một causal count decoder học trạng thái và thời lượng đoạn, với objective được suy ra từ precision/fragmentation/missed-overlap diagnostics và có layer routing theo vai trò temporal**. Cần giữ cách diễn đạt đây là giả thuyết kiến trúc cho đến khi có ablation độc lập trên `vox_lock` và các corpus còn lại.

## 12. Không khóa vào TCN: decoder bake-off

TCN không phải temporal decoder duy nhất của project. Những hướng đã có code hoặc đã chạy gồm:

| Họ decoder | Trạng thái bằng chứng | Kết quả/giới hạn |
|---|---|---|
| Linear | Có code và variant lịch sử | Có dùng làm baseline, nhưng chưa có bảng unified sạch tương đương chain hiện tại |
| Causal temporal adapter | Đã chạy trong V1 | Depthwise conv `k=5` + gated residual; kết quả cũ bị trộn với regime dữ liệu/loss khác |
| Causal deformable | Đã chạy 3 seed | `0,5594` so với TCN `0,5643` trên `vox_sel`; là đối chứng sạch nhất hiện có |
| Dual-dilation TCN | Đã chạy 3 seed (192), 1 seed (256) | Decoder streaming mạnh nhất đã xác nhận, chưa chứng minh tối ưu toàn cục |
| Event-state/Markov | Đã chạy 3 seed | Filter làm giảm điểm; prior DOWN/STAY/UP cố định quá cứng |
| Gated pyramid + TCN | Đã chạy nhiều ablation | Có frame gate, boundary/direction/segment loss; chưa vượt TCN root theo complexity gate |
| Hysteresis/duration hậu xử lý | Đã đánh giá held-out | Mean delta `−0,000142`, không adopt; không nên dùng kết quả này để bác bỏ learned duration decoder |
| CRNN backbone / GRU head | CRNN lịch sử và GRU head mới có code/config | Chưa có benchmark đáng tin cậy; không gán điểm trước khi train |

Các thí nghiệm WavLM và FastConformer chủ yếu thay **backbone** rồi vẫn dùng TCN làm temporal head. Vì vậy project đã thử nhiều biểu diễn, nhưng chưa có một cuộc so sánh rộng giữa các temporal decoder hiện đại. Tuyên bố chính xác hiện tại là: **TCN thắng deformable trong một phép thử sạch**, không phải “TCN là decoder tốt nhất”.

Các họ cần mở trong cùng framework:

1. causal Conformer hoặc local/chunk attention;
2. GRU/LSTM/QRNN;
3. diagonal/selective state-space có cache chính xác;
4. MS-TCN++/ASFormer nhiều stage;
5. decoder segmental semi-Markov với emission, transition hazard và duration;
6. boundary refiner cục bộ sau một decoder coarse;
7. mixture-of-experts/router để chọn temporal scale theo state.

### Trạng thái implementation ngày 25/09/2026

Ba decoder độc lập với TCN đã được thêm vào model factory; đây mới là code đã kiểm thử, **chưa phải kết quả accuracy**:

| Arm bake-off | `head.type` | Cấu hình matched | Số tham số head |
|---|---|---|---:|
| TCN control | `tcn_ordinal` | 6 block, dilation `1..32`, width 192 | 1.284.812 |
| Deformable control | `deformable_ordinal` | 4 block, width 192 | 1.305.676 |
| GRU | `gru_ordinal` | 4 layer, hidden 192 | 1.275.596 |
| Selective state-space | `ssm_ordinal` | 4 layer, state 384 | 1.280.972 |
| Local causal attention | `attention_ordinal` | 3 block, 4 head, context 128, relative-time bias | 1.278.860 |

Mọi head dùng cùng `StackGatedProjection`, mask `pre012`, `OrdinalConsistentHead` và có `forward_streaming` với cache. Unit test kiểm tra causality, full-vs-arbitrary-chunk equivalence, bounded attention cache, gradient và factory wiring. Runner [run_decoder_bakeoff.py](scripts/run_decoder_bakeoff.py) sinh config cho 5 arm × 3 seed và từ chối chạy nếu parameter budget lệch quá ±5%.

```bash
# Chỉ sinh config và in lệnh train; không tự chạy job dài.
/home/edabk/miniconda3/envs/.zipformer/bin/python \
  scripts/run_decoder_bakeoff.py

# Chạy sau khi phục hồi đúng historical train manifest/data.
/home/edabk/miniconda3/envs/.zipformer/bin/python \
  scripts/run_decoder_bakeoff.py --execute
```

Checkpoint paper còn lại chứa đúng backbone r8 đã bị đóng băng, nên runner dùng nó qua `init_backbone_only` và bỏ toàn bộ TCN head cũ. Historical manifest vẫn trỏ tới ổ ngoài không có trong snapshot; runner fail-closed thay vì âm thầm đổi data. Vì vậy bước tiếp theo về thực nghiệm là phục hồi data hoặc đặt tên một protocol dữ liệu mới rồi chạy đủ seed.

### Giao thức so sánh bắt buộc

Để tránh biến bake-off thành một tập module chắp vá, mọi decoder phải dùng cùng:

```text
frozen r8_v3 backbone
→ cùng hypercolumn 1984D
→ cùng fusion 1984→192
→ temporal decoder duy nhất
→ cùng OrdinalCount/VAD/OSD output
→ cùng loss, seed, frame support và evaluator
```

Bộ control nên gồm `linear`, `temporal_adaptive`, `deformable`, `tcn`, `gru`, `causal_attention`, `state_space` và `semi_markov`. Mỗi họ được giữ trong ngân sách tham số gần nhau (khoảng 1–2M head), chạy ba seed trên `vox_lock` và ít nhất một corpus ngoài miền. Chỉ sau khi chọn được hai decoder tốt nhất mới mở fine-tune backbone.

Ngoài macro-F1, bảng quyết định phải có OSD F1, F1 class 2/3+, boundary precision/recall, fragmentation, missed/partial overlap, onset/offset bias, RTF, memory và cache size. Một decoder chỉ tăng macro-F1 nhờ dự đoán class 0/1 nhưng làm overlap hoặc boundary tệ hơn không được gọi là winner.

### Thứ tự biến đổi ngay

1. Chuẩn hóa interface `forward(x)` và `forward_streaming(x, cache)` cho mọi decoder; test offline/chunked bằng cùng input.
2. Chạy bake-off head-only trên một backbone frozen. Đây là phép thử rẻ nhất để biết giới hạn nằm ở temporal inductive bias hay representation.
3. Với hai decoder đứng đầu, thêm objective hazard/occupancy và layer routing; giữ một control không có các term mới.
4. Chỉ sau đó mới thử full fine-tuning, KD từ WavLM/Conformer và thay backbone.

Nếu decoder mới không vượt TCN ở cùng latency và ngân sách, TCN được giữ vì có bằng chứng. Nếu GRU, attention, state-space hoặc semi-Markov thắng, toàn bộ thiết kế sẽ chuyển sang họ đó; TCN chỉ còn là baseline. Quy trình này cho phép biến đổi mọi phần có thể thay mà vẫn biết chính xác gain đến từ đâu.

## 13. Tài liệu nguồn cần đối chiếu

- [README tổng hợp kết quả](README_MODELS_RESULTS.md)
- [Mã factory và các head](src/models/zipcount_v1.py), [TCN/pyramid heads](src/models/heads.py), [pyramid](src/models/pyramid_head.py)
- [Dataset và label generation](src/data/dataset.py), [prepare LibriMix](src/data/prepare_from_lhotse.py), [evaluator unified](scripts/eval_zipcount_nemo.py)
- [Train loop](src/train.py), [WavLM wrapper](src/models/wavlm_wrapper.py), [FastConformer wrapper](src/models/fastconformer_wrapper.py)
- [Paper TASLP](paper/main_taslp.tex), [reports CSV](reports/model_results.csv)
- [DiariZen official repository](https://github.com/BUTSpeechFIT/DiariZen)
- [DiariZen ICASSP paper](https://arxiv.org/abs/2409.09408)
- [DiariZen structured pruning](https://arxiv.org/abs/2505.24111), [generalizable pruning study](https://arxiv.org/abs/2506.18623)
- [NVIDIA Sortformer model/evaluation documentation](https://github.com/NVIDIA/NeMo/blob/main/examples/speaker_tasks/diarization/README.md)
- [Sortformer paper](https://proceedings.mlr.press/v267/park25h.html)

Các nguồn bên ngoài mô tả DiariZen/Sortformer và không thay thế kết quả đã đo trong project. Khi viết paper mới, cần ghi rõ checkpoint, license, protocol, latency definition và ngày truy cập.
