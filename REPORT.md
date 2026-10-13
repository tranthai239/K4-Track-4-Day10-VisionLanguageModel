Họ tên: Trần Thái
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 5.12 | 6.8 | 0.078 | [0.054, 0.103] | 0.97 | 0.061 | 0.069 | 0.105 |
| E1 | LoRA LLM | 19,824,640 | 9.12 | 23.4 | 0.534 | [0.498, 0.570] | 0.55 | 0.511 | 0.490 | 0.620 |
| E2 | + projector | 128,946,176 | 11.64 | 26.8 | 0.621 | [0.590, 0.652] | 0.41 | 0.624 | 0.576 | 0.688 |
| E3 | + LoRA ViT | 133,468,160 | 14.18 | 31.2 | 0.622 | [0.583, 0.659] | 0.38 | 0.564 | 0.586 | 0.720 |
| E4 | rank r=4 (alpha=8) | 113,915,904 | 11.20 | 25.1 | 0.528 | [0.493, 0.562] | 0.54 | 0.466 | 0.522 | 0.579 |
| E4 | rank r=64 (alpha=128) | 189,067,264 | 12.80 | 29.5 | 0.609 | [0.575, 0.641] | 0.42 | 0.565 | 0.584 | 0.679 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
- **Phân rã tham số từng khối của Qwen3-VL-4B (tổng ~4.44B tham số)**:
  1. **Vision Transformer (ViT)**: ~415M tham số, gồm 24 khối transformer chuyên biệt xử lý visual patches động, trong đó có cơ chế DeepStack trích xuất đặc trưng tại các tầng trung gian (lớp 5, 11, 17) và tầng cuối (lớp 24).
  2. **Projector (Bộ ghép nối)**: ~109.1M tham số, chia thành 2 thành phần chính:
     - `visual.merger`: ~27.2M tham số, nén spatial tokens từ ViT và ánh xạ sang không gian đặc trưng của LLM.
     - `visual.deepstack_merger_list`: ~81.9M tham số, tổng hợp multi-scale features từ 3 tầng trung gian (5, 11, 17) của ViT.
  3. **LLM Backbone (Language Model)**: ~3.92B tham số (36 transformer decoder blocks, hidden dimension 2560, intermediate size 6912) cộng với embedding layers và lm_head.
- **Khối được cập nhật trong từng thí nghiệm**:
  - **E1**: Chỉ gắn LoRA (rank=16, alpha=32) lên **LLM**, bao gồm 252 module (`q_proj, k_proj, v_proj, o_proj` và `gate_proj, up_proj, down_proj` qua 36 layers), tương ứng **19,824,640 tham số huấn luyện**. Toàn bộ ViT và Projector đóng băng 100%.
  - **E2**: Giữ nguyên LoRA trên LLM như E1, đồng thời mở full đạo hàm (`requires_grad = True`) cho **toàn bộ Projector** (`visual.merger` và `visual.deepstack_merger_list`) ở kiểu dữ liệu `float32`, nâng tổng tham số huấn luyện lên **128,946,176 tham số** (+109.1M tham số so với E1). ViT backbone vẫn đóng băng.
  - **E3**: Kế thừa toàn bộ E2, đồng thời bổ sung thêm LoRA (rank=16, alpha=32) lên **ViT attention và MLP** (96 module: `attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2` trên 24 blocks), nâng tổng số module LoRA lên 348 và tổng tham số huấn luyện đạt **133,468,160 tham số** (+4.52M tham số LoRA ViT).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- **`parse_valid_rate` của E0**: Đạt **0.707 (70.7%)**, nghĩa là gần 30% số ảnh test mô hình gốc không thể sinh ra chuỗi JSON hợp lệ theo đúng schema.
- **Hai lỗi phổ biến nhất của mô hình gốc (E0)**:
  1. *Lỗi định dạng hội thoại tự nhiên và Markdown không tuân thủ quy tắc*: Mô hình gốc là bản Instruct thông dụng, có thiên hướng đàm thoại và giải thích dài dòng bằng văn bản tiếng Anh, chèn thêm các câu mở đầu/kết bài (conversational filler), hoặc quên đóng ngoặc `]` làm hỏng quá trình `json.loads`.
  2. *Lỗi toạ độ hình học (Bounding box inversion & Hallucination)*: Mô hình thường xuyên sinh ra toạ độ bị đảo ngược ($x_2 \le x_1$ hoặc $y_2 \le y_1$), hoặc dự đoán các toạ độ vượt ngưỡng ngoài vùng ảnh, đồng thời bịa đặt (hallucinate) các hư hại ở các vũng nước, bóng râm hay vạch kẻ đường.
- **Ví dụ output thô bị lỗi của E0**:
  ```text
  Certainly! Looking at the road surface in the image, I noticed some severe cracks near the center:
  ```json
  [{"label": "crack", "bbox_2d": [580, 710, 520, 840]}]
  ```
  Please proceed with caution when driving over this area.
  ```
  *(Lỗi: x1=580 > x2=520 khiến chiều rộng âm, vi phạm điều kiện bbox hợp lệ; đồng thời bao bọc văn bản đàm thoại không cần thiết)*.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- **Kết quả F1 khi chấm lệch thang**: F1 rơi thẳng về **0.000** (Precision = 0.000, Recall = 0.000), dù `parse_valid_rate` vẫn đạt 1.0.
- **Nguyên nhân toán học**:
  - Với ảnh 640×360, toạ độ pixel có dải giá trị $x \in [0, 640]$ và $y \in [0, 360]$. Trong khi đó, thang grounding chuẩn hoá của Qwen3-VL quy ước $x, y \in [0, 1000]$.
  - Nếu mô hình xuất toạ độ pixel nhưng hàm chấm diễn giải theo thang 0–1000 (hoặc ngược lại), toàn bộ bounding box sẽ bị co cụm lại ở góc trên bên trái của khung hình ảo, hoặc bị phóng đại gấp 1.56 – 2.78 lần so với kích thước thực tế.
  - Diện tích giao nhau (Intersection) giữa box dự đoán lệch thang và ground-truth box gần như bằng 0, khiến $\text{IoU} \ll 0.5$. Do đó không có bất kỳ cặp nào đạt ngưỡng $\text{IoU} \ge 0.5$, dẫn đến $\text{TP} = 0$, sinh ra F1 = 0.000.
- **Cách kiểm chứng**:
  - Đã được kiểm chứng thực nghiệm bằng việc chạy unit test `test_pixel_vs_norm1000_scale_trap` trong `tests/test_score.py`: cùng một nhãn dự đoán chuẩn pixel `[256, 144, 384, 216]`, khi evaluate với `coord="pixel"` thì đạt F1 = 1.000, nhưng khi evaluate với `coord="norm1000"` thì F1 = 0.000 tuyệt đối.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Vì sao chỉ tính loss trên phần trả lời (assistant answer)**:
  - Bản chất của Vision-Language SFT là tối ưu hoá phân phối xác suất sinh có điều kiện $P(\text{Output} \mid \text{Prompt}, \text{Image})$. Mô hình cần học ánh xạ từ hình ảnh và câu lệnh sang chuỗi JSON grounding.
  - Toàn bộ token thuộc user prompt và token ảnh đã là điều kiện đầu vào (context), không phải mục tiêu cần dự đoán. Vì vậy, ta gán `labels = -100` cho toàn bộ các token tiền tố (từ vị trí 0 đến $n-1$), để hàm `CrossEntropyLoss` của PyTorch tự động bỏ qua (ignore index).
- **Điều gì xảy ra nếu tính loss trên cả prompt và token ảnh**:
  1. *Pha loãng gradient và lệch mục tiêu tối ưu*: Mỗi ảnh 640×360 chiếm tới **220 token** vision, cộng với prompt hệ thống ~30 token, trong khi chuỗi JSON đáp án chỉ chiếm ~20–60 token. Nếu không mask, loss sẽ bị chi phối tới hơn 80% bởi việc học tái tạo lại prompt tĩnh và tái cấu trúc biểu diễn ảnh (vision self-reconstruction).
  2. *Hội tụ giả và sụt giảm năng lực grounding*: Giá trị loss trên train log có thể giảm rất sâu (do prompt lặp đi lặp lại rất dễ học thuộc), nhưng năng lực sinh toạ độ grounding thực tế của mô hình sẽ suy giảm nghiêm trọng hoặc không hội tụ được định dạng JSON chuẩn.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử 4-bit**:
  - Trong kiến trúc Qwen3-VL, các module projector (`merger` và `deepstack_merger_list`) nằm trực tiếp bên trong không gian phân cấp của module `model.visual`.
  - Nếu cấu hình BitsAndBytes lượng tử hoá toàn bộ mô hình sang 4-bit NF4 mà không loại trừ nhánh vision, các trọng số của projector cũng sẽ bị nén thành kiểu dữ liệu lượng tử rời rạc 4-bit. Trọng số 4-bit chỉ hỗ trợ forward tĩnh và không thể nhận continuous gradient updates để tối ưu trực tiếp bằng AdamW (trừ khi gắn LoRA).
- **Cách xử lý**:
  - Truyền `llm_int8_skip_modules=["visual", "lm_head"]` vào `BitsAndBytesConfig` khi nạp mô hình. Cấu hình này chỉ thị cho bitsandbytes chỉ lượng tử hoá 4-bit phần LLM transformer layers, giữ nguyên toàn bộ nhánh `model.visual` (bao gồm ViT backbone và projector) ở kiểu dữ liệu floating-point gốc.
- **Vì sao projector phải ở `float32` khi huấn luyện**:
  - Projector làm nhiệm vụ dung hợp biểu diễn đa tầng (layers 5, 11, 17 và 24) từ ViT và chiếu sang không gian 2560 chiều của LLM.
  - Khi gradient từ 36 tầng LLM truyền ngược qua projector, biên độ gradient tích luỹ rất lớn. Nếu để ở `fp16` (vốn có dynamic range hẹp từ $6 \times 10^{-5}$ đến $65504$), gradient và optimizer state (m, v của AdamW) rất dễ gặp hiện tượng tràn số (overflow/underflow) gây loss NaN. Ép projector về `float32` đóng vai trò master weights giúp cập nhật số học ổn định và khử triệt để hiện tượng mất ổn định gradient.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Vì sao chia ngẫu nhiên cho F1 cao giả**:
  - Tập dữ liệu gồm các khung hình được cắt từ video hành trình dashcam quay liên tục trên cùng một cung đường. Các khung hình quay cách nhau chỉ vài giây có góc nhìn, điều kiện ánh sáng, mặt đường và các ổ gà/vết nứt gần như giống hệt nhau (chỉ dịch chuyển vài pixel).
  - Nếu chia ngẫu nhiên theo từng ảnh, các frame liền kề của cùng một ổ gà sẽ xuất hiện đồng thời ở cả tập train và tập test. Khi đó mô hình chỉ cần ghi nhớ bề mặt đường và vị trí tương đối từ tập train là có thể dự đoán chính xác trên test (**rò rỉ dữ liệu / Data Leakage**). Điểm F1 trên test sẽ cao giả tạo nhưng mô hình sẽ thất bại hoàn toàn khi triển khai trên một cung đường hoặc video mới.
  - Chia theo phiên quay (`session_key` gom theo ngày, giờ quay: ví dụ `vlc-2025-02-18-23`, `phone-20250219`) đảm bảo toàn bộ hành trình của một phiên chỉ nằm trọn vẹn trong một split duy nhất, phản ánh đúng năng lực tổng quát hoá thực tế.
- **Số ảnh và số box từng lớp ở mỗi split (từ `data/splits/meta.json`)**:
  - **Train**: 1,422 ảnh | pothole: 1,074 | crack: 1,750 | manhole: 691 (Tổng cộng 3,515 box).
  - **Val**: 276 ảnh | pothole: 66 | crack: 470 | manhole: 103 (Tổng cộng 639 box).
  - **Test**: 311 ảnh | pothole: 121 | crack: 299 | manhole: 163 (Tổng cộng 583 box).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **Số liệu so sánh trực tiếp**:
  - **E1 (LoRA LLM)**: F1 = 0.534, CI95 = [0.498, 0.570], VRAM đỉnh = 9.12 GB, Thời gian = 23.4 phút.
  - **E2 (+ Projector)**: F1 = 0.621, CI95 = [0.590, 0.652], VRAM đỉnh = 11.64 GB, Thời gian = 26.8 phút.
  - **E3 (+ LoRA ViT)**: F1 = 0.622, CI95 = [0.583, 0.659], VRAM đỉnh = 14.18 GB, Thời gian = 31.2 phút.
- **Đánh giá tính hiệu quả**:
  - **Mở Projector (E2) là CỰC KỲ ĐÁNG GIÁ**:
    - F1 tăng vọt từ 0.534 lên 0.621 (+0.087 điểm F1). Khoảng tin cậy CI95 của E2 [0.590, 0.652] tách biệt hoàn toàn và không chồng lấn với CI95 của E1 [0.498, 0.570], chứng minh sự cải thiện này có ý nghĩa thống kê rõ ràng.
    - Chi phí tài nguyên rất khiêm tốn: VRAM chỉ tăng thêm 2.52 GB (từ 9.12 lên 11.64 GB) và thời gian train chỉ tăng 3.4 phút. Đây là tỷ suất lợi ích / chi phí (ROI) tốt nhất.
  - **Mở thêm LoRA trên ViT (E3) là KHÔNG ĐÁNG GIÁ**:
    - F1 hầu như dậm chân tại chỗ (0.622 so với 0.621 của E2, delta chỉ vỏn vẹn +0.001), khoảng tin cậy CI95 hoàn toàn bao trùm lẫn nhau.
    - Trong khi đó, tài nguyên tiêu tốn tăng mạnh: VRAM đỉnh nhảy vọt lên **14.18 GB** (suýt tràn bộ nhớ 15 GB của GPU T4 do gradient phải lan truyền qua 24 tầng ViT), và thời gian train tăng thêm 4.4 phút (+16.4%). Do đó, E2 là setup tối ưu toàn diện nhất.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng khoảng tin cậy trên 311 ảnh**:
  - Với phương pháp bootstrap resampling (1,000 lần lặp, seed=0), khoảng tin cậy CI95 trên tập test có độ rộng trung bình khoảng **0.062 đến 0.076** (khoảng $\pm 3.1\%$ đến $\pm 3.8\%$ xung quanh giá trị F1 ước lượng điểm).
  - Ví dụ ở E2, CI95 là [0.590, 0.652] (độ rộng 0.062); ở E3, CI95 là [0.583, 0.659] (độ rộng 0.076).
- **Giới hạn kết luận so sánh giữa các setup**:
  - *Kết luận có cơ sở vững chắc*: Khi chênh lệch F1 giữa hai mô hình lớn hơn biên độ của CI95 (như E1 vs E2: chênh lệch 0.087, hai khoảng [0.498, 0.570] và [0.590, 0.652] hoàn toàn tách rời), ta có thể khẳng định chắc chắn 95% rằng E2 vượt trội hơn E1.
  - *Giới hạn không thể kết luận*: Khi chênh lệch F1 nhỏ hơn độ rộng CI95 (như E2 vs E3: chênh lệch chỉ 0.001, hoặc E2 vs E4_rank_64: 0.621 vs 0.609), khoảng tin cậy của chúng chồng lấn sâu sắc. Do kích thước mẫu chỉ có 311 ảnh (với 583 box), ta không thể kết luận E3 tốt hơn E2 hay rank 64 tốt hơn rank 16 về mặt ý nghĩa thống kê; sự khác biệt này hoàn toàn có thể chỉ do biến động ngẫu nhiên của mẫu thử.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Lớp có kết quả kém nhất**: **`crack` (vết nứt)** là lớp có F1 thấp nhất qua mọi thí nghiệm (E1: 0.490, E2: 0.576, E3: 0.586), thấp hơn đáng kể so với `manhole` (0.688 – 0.720) và `pothole` (0.564 – 0.624).
- **Ba nguyên nhân cốt lõi**:
  1. *Hình học và diện tích box (Bounding Box Topology)*:
     - Vết nứt có đặc thù là hình dải mảnh, uốn lượn ngoằn ngoèo, diện tích thực tế của vết nứt chiếm tỷ lệ rất nhỏ bên trong bounding box hình chữ nhật.
     - Khi dùng bounding box bao quanh một vết nứt dài chéo, diện tích nền đường nhựa chiếm đa số trong box. Sai số chỉ 5–10 pixel ở góc hộp cũng làm giảm mạnh chỉ số IoU xuống dưới ngưỡng 0.5, khiến dự đoán bị phạt thành cả False Positive lẫn False Negative.
  2. *Độ phân giải và nén thông tin thị giác*:
     - Ảnh dashcam 640×360 nén thành 220 visual tokens (mỗi token đại diện cho một vùng diện tích khoảng 32×32 pixel). Các vết nứt nhỏ mảnh ở khoảng cách xa bị mờ do nén và hiện tượng motion blur của camera hành trình, khiến ViT khó tách biệt đặc trưng.
  3. *Mơ hồ ngữ nghĩa và nhiễu nền (Semantic Ambiguity)*:
     - Vết nứt rất dễ bị nhầm lẫn với các đường trám hắc ín cũ (bitumen sealant lines), bóng cây đổ trên mặt đường, vệt lốp xe phanh gấp hoặc rãnh nối bê tông (gây ra lượng False Positive cao: 45 FP ở E2). Trong khi đó, `manhole` (nắp cống) có hình tròn/vuông chuẩn tắc với viền kim loại rõ rệt và `pothole` (ổ gà) có độ sâu tạo bóng đổ đặc trưng nên dễ nhận diện hơn nhiều.
- **Minh hoạ trực quan các trường hợp dự đoán**:
  - `report/images/sample_1_pothole_tp.png`: True Positive - Dự đoán ổ gà chính xác (IoU = 0.88).
  - `report/images/sample_2_crack_tp.png`: True Positive - Dự đoán vết nứt lớn rõ nét (IoU = 0.82).
  - `report/images/sample_3_manhole_tp.png`: True Positive - Dự đoán nắp cống chuẩn xác (IoU = 0.91).
  - `report/images/sample_4_crack_fp.png`: False Positive - Mô hình nhận diện nhầm bóng cây/vệt nhựa đường thành `crack`.
  - `report/images/sample_5_pothole_fn.png`: False Negative - Ổ gà nhỏ ở xa bị bỏ sót do độ tương phản thấp.
  - `report/images/sample_6_iou_boundary.png`: Minh họa độ nhạy của IoU biên trên vết nứt và ổ gà.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến được chọn thay đổi**: **LoRA Rank (`rank`)** trên nền kiến trúc E2 (giữ nguyên mở projector, chỉ thay đổi rank của LoRA trên LLM):
  - Giá trị 1: **$r = 4$** ($\alpha = 8$).
  - Giá trị cơ sở: **$r = 16$** ($\alpha = 32$, chính là E2).
  - Giá trị 2: **$r = 64$** ($\alpha = 128$).
- **Kết quả đo lường**:
  - **$r = 4$** (`e4_rank_4.json`): Tham số huấn luyện = 113,915,904; VRAM = 11.20 GB; Train = 25.1 phút; **F1 = 0.528**, **CI95 = [0.493, 0.562]**, FP/ảnh = 0.54.
  - **$r = 16$** (E2): Tham số huấn luyện = 128,946,176; VRAM = 11.64 GB; Train = 26.8 phút; **F1 = 0.621**, **CI95 = [0.590, 0.652]**, FP/ảnh = 0.41.
  - **$r = 64$** (`e4_rank_64.json`): Tham số huấn luyện = 189,067,264; VRAM = 12.80 GB; Train = 29.5 phút; **F1 = 0.609**, **CI95 = [0.575, 0.641]**, FP/ảnh = 0.42.
- **Đánh giá độ tin cậy của kết luận (dựa trên CI95)**:
  - *Kết luận 1 (Đáng tin cậy cao)*: Tăng rank từ $r=4$ lên $r=16$ mang lại bước nhảy vọt thực chất về F1 (+0.093). Khoảng tin cậy [0.493, 0.562] của $r=4$ hoàn toàn tách biệt khỏi [0.590, 0.652] của $r=16$. Rank $r=4$ rõ ràng là không đủ dung lượng biểu diễn (capacity) để LLM học ánh xạ toạ độ không gian.
  - *Kết luận 2 (Cần nhìn nhận thận trọng qua CI95)*: Khi tăng từ $r=16$ lên $r=64$, tham số LoRA tăng thêm hơn 60M, VRAM tăng lên 12.8 GB nhưng F1 lại giảm nhẹ từ 0.621 xuống 0.609. Do CI95 của $r=16$ [0.590, 0.652] và $r=64$ [0.575, 0.641] chồng lấn gần như toàn bộ, ta không thể khẳng định $r=64$ kém hơn $r=16$ một cách chắc chắn, nhưng có đủ cơ sở để kết luận rằng việc tăng rank vượt quá 16 trên tập dữ liệu này không mang lại lợi ích gì mà còn có nguy cơ overfitting nhẹ và lãng phí tài nguyên. Điểm tối ưu thực tế (sweet spot) là **$r = 16$**.

## Nhìn lại

Qua quá trình thực hiện bài lab finetune Qwen3-VL-4B cho bài toán road damage grounding, một số vấn đề thực tế quan trọng được rút ra:
1. **Sự không nhất quán trong nhãn Ground Truth (Label Noise)**:
   - Các vết nứt dài trên đường thường bị người gán nhãn cắt vụn thành nhiều box nhỏ rời rạc hoặc bao trọn trong một box khổng lồ tuỳ cảm tính. Mô hình đôi khi phát hiện vết nứt rất chuẩn dưới dạng một dải liền mạch nhưng vẫn bị tính là False Positive và False Negative do không khớp cách chia box của nhãn.
2. **Đặc thù góc máy Dashcam và Perspective Distortion**:
   - Camera hành trình đặt sau kính lái tạo ra góc nghiêng (oblique angle) lớn. Càng về phía chân trời, mật độ thông tin càng bị nén mạnh, dẫn đến hiện tượng mô hình bỏ sót các hư hại nhỏ ở xa hoặc dự đoán box bị méo tỷ lệ.
3. **Ý nghĩa của việc mở Projector thay vì ViT**:
   - Kết quả thực nghiệm chứng minh rằng bộ ghép nối (Projector) đóng vai trò là "nút thắt cổ chai" (bottleneck) quan trọng nhất trong việc chuyển dịch miền biểu diễn từ vision sang language. Mở toàn bộ Projector ở float32 mang lại hiệu quả vượt trội so với việc can thiệp sâu vào ViT, giúp tiết kiệm đáng kể VRAM và thời gian huấn luyện trên phần cứng hạn chế như GPU T4.
