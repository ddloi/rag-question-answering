# RAG cho hỏi-đáp — hướng dẫn chạy theo trình tự

Cấu hình tham khảo: RTX 3050 6GB, 24GB RAM (đủ chạy toàn bộ pipeline này ở mức đồ án).

## 0. Cài đặt môi trường

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Kiểm tra CUDA hoạt động:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 1. Tải và chuẩn bị dataset

Bạn **phải tự chọn và tải dataset** — đây là phần đề bài yêu cầu bạn mô tả nguồn/license/thống kê, không thể để AI tự chọn thay.

```bash
python 00_download_dataset.py
```
Mặc định tải subset 800 câu hỏi từ TriviaQA (`rc.wikipedia`, split validation). Đổi `DATASET_CHOICE = "natural_questions"` trong file nếu muốn dùng NQ thay thế. In ra thống kê (số doc, số QA, license) — copy phần này vào báo cáo.

→ tạo `raw_documents.json`.

## 2. Chunking

```bash
python 01_prepare_and_chunk.py
```
→ tạo `data/corpus_chunks.jsonl` và `data/qa_pairs.jsonl`.

## 3. Tách train/test + tạo dữ liệu fine-tune

```bash
python 01b_split_train_test.py
```
→ tạo `data/test_qa.jsonl` (KHÔNG dùng để fine-tune) và `data/training_pairs.jsonl` (câu hỏi + gold passage + câu trả lời, dùng ở bước 5). Split cố định seed=42, tỉ lệ test 15%.

## 4. Xây FAISS index

```bash
python 02_build_index.py
```
→ tạo `data/faiss.index` và `data/chunk_metadata.jsonl`. Chạy 1 lần, không cần chạy lại trừ khi corpus/chunk_size thay đổi.

## 5. Fine-tune generator

```bash
python 03_finetune_generator.py
```
→ model lưu tại `generator_finetuned/`. Với t5-base trên 3050 6GB, dùng batch_size=4 + fp16 + gradient_accumulation=4 như đã cấu hình sẵn.

Theo dõi VRAM khi train:
```bash
nvidia-smi -l 1
```
Nếu vẫn OOM: giảm `per_device_train_batch_size` xuống 2, tăng `gradient_accumulation_steps` lên 8.

## 6. Chạy thử pipeline RAG đầy đủ

```bash
python 04_rag_pipeline.py
```
(Đổi tên file thành `rag_pipeline.py` trước, vì `05_evaluate.py` import theo tên module này — hoặc sửa câu import cho khớp tên file bạn đặt.)

## 7. Đánh giá

```bash
python 05_evaluate.py
```
→ in ra Recall@5, EM/F1 của cả 3 hệ, và bảng phân rã lỗi retrieval vs generation.

## 8. Cải tiến 1 — Hybrid retrieval (BM25 + dense)

```bash
python 06_hybrid_retrieval.py
```
Dùng Reciprocal Rank Fusion (RRF) để kết hợp BM25 (sparse, bắt từ khoá/tên riêng chính xác) với dense retrieval (semantic). Không cần train gì thêm — chỉ thay đổi cách retrieve. Để so sánh với dense-only, dùng chung `test_qa.jsonl` và đo Recall@k bằng hàm tương tự trong `05_evaluate.py`.

## 9. Cải tiến 2 — Ablation chunk_size / top_k

```bash
python 07_ablation_chunk_topk.py
```
Thử 3 chunk_size (100/200/300) × 3 top_k (3/5/10), so sánh Recall@k giữa dense-only và hybrid. Kết quả lưu ra `ablation_results.csv` — dùng trực tiếp để vẽ bảng/biểu đồ trong báo cáo. Chạy trên subset 100 câu hỏi để nhanh (đổi số này trong file nếu muốn chính xác hơn, đánh đổi thời gian chạy).

## Ghi chú báo cáo (để đáp ứng yêu cầu chung của đề bài)

- Ghi rõ: seed=42 (đã set trong code), phiên bản thư viện (`pip freeze > requirements_lock.txt`), môi trường chạy (GPU, driver, CUDA version).
- Bắt buộc: error analysis dựa trên bảng phân rã lỗi ở bước 7 — chọn vài ví dụ cụ thể (retrieval sai, hoặc generation sai dù retrieval đúng) để phân tích trong báo cáo.
- Cải tiến điểm cộng gợi ý: hybrid BM25 + dense retrieval, hoặc thêm cross-encoder reranker sau bước retrieval.
