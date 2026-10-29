# Dữ liệu và chỉ mục

| Tệp | Nguồn tạo |
| --- | --- |
| `raw_documents.json` | `python -m experiments.download_dataset` |
| `corpus_chunks.jsonl`, `qa_pairs.jsonl` | `python -m experiments.prepare_chunks` |
| `training_pairs.jsonl`, `test_qa.jsonl` | `python -m experiments.split_train_test` |
| `faiss.index`, `chunk_metadata.jsonl` | `python -m experiments.build_index` |

Web demo chỉ đọc `chunk_metadata.jsonl` và các câu hỏi mẫu trong `test_qa.jsonl`. Phần thí nghiệm T5/FAISS đọc thêm các artifact tương ứng.

Các tệp lớn vẫn được bỏ qua trong Git theo `.gitignore`; clone repository không đồng nghĩa có đủ dữ liệu để demo. Trọng số mô hình nằm trong `generator_finetuned/`, kết quả đo nằm trong `results/`.

Chạy các bước tạo dữ liệu có thể ghi đè artifact hiện tại. Lần sắp xếp cấu trúc này giữ nguyên dữ liệu đã có, không tạo lại train/test hoặc chỉ mục.
