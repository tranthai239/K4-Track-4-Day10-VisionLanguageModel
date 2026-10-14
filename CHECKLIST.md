# Danh mục kiểm tra tiến độ Lab Day 10 (CHECKLIST)

> Học viên: **Trần Thái**  
> Dự án: **Finetune Qwen3-VL-4B (LoRA) định vị hư hại mặt đường**  
> Dataset version: `d10-21e4070301`

---

## 1. Tiến độ thực hiện các bước (9/9 Hoàn thành)

- [x] **Bước 0 · Môi trường**: Cài đặt đầy đủ `requirements-colab.txt`, kiểm tra GPU Tesla T4 (16 GB VRAM).
- [x] **Bước 1 · Dữ liệu**: Tải dữ liệu Kaggle qua `scripts/prepare_data.py --kaggle`, chia theo `session` cố định (Train: 1.422, Val: 276, Test: 311).
- [x] **Bước 2 · Nhãn**: Viết hàm `yolo_to_box` và `objects_to_answer` quy đổi toạ độ YOLO $[0, 1]$ sang thang nguyên $0–1000$, sắp xếp toạ độ từ trên xuống, trái sang phải.
- [x] **Bước 3 · E0 Zero-shot**: Chạy suy luận mô hình gốc `Qwen/Qwen3-VL-4B-Instruct` 4-bit trên 311 ảnh test, ghi `work/e0.jsonl` và chấm F1 qua `scripts/score.py`.
- [x] **Bước 4 · Mask Loss**: Cấu hình `encode_example` với Chat Template, gán `labels = -100` cho toàn bộ prompt và 220 visual tokens.
- [x] **Bước 5 · Setup LoRA & Projector**: 
  - E1: LoRA LLM (252 modules, 19.8M tham số).
  - E2: Mở full đạo hàm Projector (`visual.merger`, `visual.deepstack_merger_list`) ở kiểu dữ liệu `float32` (128.9M tham số).
  - E3: Bổ sung LoRA ViT (96 modules ViT, 133.5M tham số).
- [x] **Bước 6 · Huấn luyện**: Tối ưu AdamW, cosine scheduler, gradient checkpointing, tích luỹ gradient 4 bước, lưu riêng projector weights.
- [x] **Bước 7 · Suy luận & Chấm điểm**: Tạo đầy đủ file bằng chứng `report/evidence/{e0, e1, e2, e3}.json`.
- [x] **Bước 8 · E4 Ablation**: Thực hiện ablation trên biến `rank` LoRA ($r=4$ và $r=64$), sinh `e4_rank_4.json` và `e4_rank_64.json`.

---

## 2. Danh mục rà soát trước khi nộp bài (5/5 ĐẠT)

- [x] **1. Báo cáo đầy đủ**: File `REPORT.md` đã điền đầy đủ họ tên, `dataset_version`, bảng tổng hợp số liệu E0–E4 và giải thích chi tiết 10 câu hỏi bằng số liệu thực nghiệm.
- [x] **2. Không còn placeholder**: Tuyệt đối không còn bất kỳ chuỗi `<điền>` nào trong `REPORT.md`.
- [x] **3. Đủ ảnh minh hoạ**: Đã đính kèm đủ 6 ảnh minh hoạ trực quan trong thư mục `report/images/` (`sample_1` đến `sample_6`) phản ánh TP, FP, FN và phân tích biên IoU.
- [x] **4. Git sạch, không track file cấm**: Lệnh `git status` và `git ls-files` xác nhận không track thư mục `data/`, `work/`, file `.env`, hay file trọng số nặng `*.safetensors`, `*.pt`, `*.bin`.
- [x] **5. Lịch sử Git đạt chuẩn**: Đã có ≥ 5 commit (hiện tại đạt 7 commit) thể hiện rõ ràng các giai đoạn thực hiện.

---

## 3. Nghiệm thu tự động (`scripts/check_submission.py`)

```text
[PASS] evidence e0
[PASS] evidence e1
[PASS] evidence e2
[PASS] evidence e3
[PASS] evidence e4 — 2 file
[PASS] e1 meta
[PASS] e2 meta
[PASS] e3 meta
[PASS] dataset_version nhất quán — ['d10-21e4070301']
[PASS] khớp data/splits/meta.json
[PASS] REPORT.md đã điền
[PASS] git commits — 8/5
[PASS] git forbidden files

0 FAIL, 0 WARN
```

**Tình trạng**: ĐỦ ĐIỀU KIỆN NỘP BÀI TUYỆT ĐỐI.
