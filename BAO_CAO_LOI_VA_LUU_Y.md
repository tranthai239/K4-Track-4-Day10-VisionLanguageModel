# Báo cáo các bẫy kỹ thuật và lưu ý thực chiến (Lab Day 10)

> Tác giả: **Trần Thái**  
> Dự án: Finetune Vision-Language Model Qwen3-VL-4B định vị hư hại mặt đường

Tài liệu này tổng hợp các bẫy kỹ thuật phổ biến (pitfalls & gotchas), phân tích nguyên nhân cốt lõi và giải pháp khắc phục khi huấn luyện và đánh giá mô hình thị giác–ngôn ngữ.

---

## 1. Bẫy thang toạ độ (Scale Trap): Pixel vs. Chuẩn hoá 0–1000

* **Triệu chứng**: Mô hình sinh JSON hợp lệ 100% (`parse_valid_rate = 1.0`), nhưng F1 đo được là **0.000 tuyệt đối**.
* **Nguyên nhân**:
  * Ảnh dashcam có kích thước phân giải thực tế là $640 \times 360$ pixel.
  * Qwen3-VL quy ước không gian grounding chuẩn hoá trên thang nguyên $[0, 1000]$.
  * Nếu xuất nhãn theo toạ độ pixel nhưng hàm đánh giá đọc theo thang $0–1000$ (hoặc ngược lại), bounding box bị co cụm về góc trên bên trái ($x \le 640, y \le 360$) hoặc bị phóng đại sai lệch gấp 1.56 – 2.78 lần. Diện tích giao nhau (Intersection) gần như bằng 0 khiến $\text{IoU} \ll 0.5 \Rightarrow \text{TP} = 0 \Rightarrow \text{F1} = 0$.
* **Giải pháp**:
  * Ánh xạ đúng công thức: $x = \text{round}\left((xc \pm w/2) \times 1000\right)$ kẹp trong $[0, 1000]$.
  * Viết unit test xác minh chuyển đổi 2 chiều và luôn truyền `--coord norm1000` khi chấm điểm.

---

## 2. Bẫy rò rỉ dữ liệu (Data Leakage) khi chia split video

* **Triệu chứng**: F1 trên tập kiểm thử (test) đạt rất cao (> 0.85), nhưng khi đem mô hình chạy trên video cung đường mới thì hoàn toàn không phát hiện được hư hại.
* **Nguyên nhân**:
  * Dữ liệu dashcam gồm các khung hình cắt liên tục từ video hành trình (fps cao). Hai khung hình cách nhau 1–2 giây gần như trùng lặp góc nhìn, ổ gà chỉ dịch chuyển vài pixel.
  * Nếu chia dữ liệu ngẫu nhiên (random train_test_split), các khung hình liền kề của cùng một ổ gà sẽ nằm ở cả train và test. Mô hình chỉ "học vẹt" (overfit) đặc trưng nền đường của video đó.
* **Giải pháp**:
  * Nhóm các ảnh theo **phiên quay (session key)** dựa trên tiền tố ngày/giờ trong tên file (`vlc-2025-02-18-23`).
  * Đưa trọn vẹn cả phiên vào một split duy nhất (`prepare_data.py`), đảm bảo không có bất kỳ phiên quay nào xuất hiện ở hai split khác nhau.

---

## 3. Bẫy lượng tử hoá nhánh Vision và Projector

* **Triệu chứng**: Ở E2 (mở projector), số lượng tham số huấn luyện không tăng thêm 109M tham số hoặc ném lỗi khi tối ưu hoá bằng AdamW.
* **Nguyên nhân**:
  * Trong Qwen3-VL, bộ ghép nối (`merger` và `deepstack_merger_list`) nằm trong cây phân cấp của module `model.visual`.
  * Nếu nạp `load_in_4bit=True` mà không cấu hình loại trừ, bitsandbytes sẽ lượng tử hoá cả module thị giác thành 4-bit NF4. Trọng số 4-bit bị khoá tĩnh, không hỗ trợ nhận đạo hàm liên tục (`requires_grad=True`).
* **Giải pháp**:
  * Cấu hình tường minh: `llm_int8_skip_modules=["visual", "lm_head"]`.
  * Ép toàn bộ tham số projector sang kiểu `float32` để đảm bảo ổn định số học và tránh hiện tượng tràn số (NaN loss) do gradient biên độ lớn tích luỹ từ LLM.

---

## 4. Bẫy Mask Loss trên Prompt và Visual Tokens

* **Triệu chứng**: Loss trong nhật ký huấn luyện giảm rất nhanh và rất sâu, nhưng mô hình sinh ra toạ độ ngẫu nhiên hoặc lặp lại câu hỏi.
* **Nguyên nhân**:
  * Mỗi ảnh $640 \times 360$ được mã hoá thành **220 visual tokens**, cộng thêm khoảng 30–40 token prompt hệ thống. Chuỗi JSON đáp án chỉ chiếm khoảng 20–60 tokens.
  * Nếu không gán `labels = -100` cho phần prompt và ảnh, loss bị chi phối hơn 80% bởi việc tái cấu trúc đặc trưng ảnh và học thuộc câu prompt tĩnh.
* **Giải pháp**:
  * Xác định chính xác độ dài tiền tố prompt $n$, kiểm tra `full_enc.input_ids[0, :n] == prompt_enc.input_ids[0]`.
  * Gán `labels[:, :n] = -100` để `CrossEntropyLoss` bỏ qua, chỉ tính loss trên các token của chuỗi JSON câu trả lời.

---

## 5. Bẫy lưu Checkpoint khi mở Projector (Peft vs. Full Weights)

* **Triệu chứng**: Kết quả F1 của E2 và E3 khi suy luận lại y hệt F1 của E1.
* **Nguyên nhân**:
  * Hàm `model.save_pretrained()` của thư viện `peft` **chỉ lưu các adapter LoRA**.
  * Các tham số projector (`merger`) ở E2/E3 được mở toàn phần (`requires_grad=True`) nên **không thuộc adapter**. Khi suy luận nạp lại bằng `PeftModel.from_pretrained()`, projector bị nạp trọng số gốc ban đầu.
* **Giải pháp**:
  * Lưu riêng một file từ điển trọng số projector:  
    `torch.save({n: p.cpu() for n, p in model.named_parameters() if "visual.merger" in n or "visual.deepstack_merger_list" in n}, "projector_weights.pt")`.
  * Nạp lại trọng số này sau khi nạp adapter ở bước suy luận.

---

## 6. Bẫy suy luận thống kê trên tập kiểm thử nhỏ (311 ảnh)

* **Triệu chứng**: Nhìn thấy F1 của E3 (0.622) cao hơn E2 (0.621) hay E4_rank_64 (0.609) rồi vội vàng kết luận mô hình này tốt hơn mô hình kia.
* **Bản chất**:
  * Với 311 ảnh test (583 box), khoảng tin cậy 95% (CI95) có độ rộng dao động từ $\pm 3.1\%$ đến $\pm 3.8\%$.
  * Sự chênh lệch 0.001 giữa E2 và E3 nằm trọn vẹn trong khoảng sai số ngẫu nhiên của mẫu thử, không có ý nghĩa thống kê.
* **Bài học**: Luôn đọc F1 đi kèm CI95 và xem xét chi phí tài nguyên (VRAM của E3 vọt lên 14.18 GB suýt tràn GPU T4 trong khi F1 không tăng thực chất) để đưa ra quyết định kỹ thuật đúng đắn.
