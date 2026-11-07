# Đọc mã thí nghiệm

Các module được giữ đúng đường dẫn để không phá source hashes, imports và các snapshot đã chốt. Không cần đọc hết thư mục khi học research_v1.

## Luồng hiện tại

Đọc `configs/research_v1.json` trước, sau đó theo thứ tự:

| Bước | Module | Cần hiểu gì? |
| --- | --- | --- |
| 1 | research_data.py | Chọn QA/split, thu corpus, chunking, embedding và FAISS |
| 2 | research_pipeline.py | Retrieval, context sau token budget, T5 generation và DistilBERT extraction |
| 3 | research_train.py | Train closed-book và RAG riêng với QA/budget chung |
| 4 | metrics.py | Normalization TriviaQA, official aliases/canonical, EM/F1 |
| 5 | research_evaluate.py | Chạy ba hệ, Hybrid, ablation và lưu prediction/evidence |
| 6 | research_ablation.py | Dev: ablation top-k/token budget và cross-encoder reranker |
| 7 | research_train_retrieved.py | Train RAG T5 với context Dense + reranker (cùng QA/siêu tham số v1) |
| Hỗ trợ | research_artifacts.py, verify_dataset.py | Hash, manifest, dữ liệu và kiểm tra provenance dùng chung |

Các điểm gọi chính:

```powershell
python -m experiments --list
python -m experiments.research_data --help
python -m experiments.research_train --help
python -m experiments.research_evaluate --help
python -m experiments.research_ablation --help
python -m experiments.research_train_retrieved --help
```

`--help` của module ML có thể chậm vì import thư viện. `--list` chỉ liệt kê, không khởi động thí nghiệm. Lệnh thật và điều kiện chạy nằm trong [hướng dẫn vận hành](../docs/HOC_VA_VAN_HANH.md).

## Module lịch sử

download_dataset, prepare_chunks, split_train_test, build_index, finetune_generator, rag_pipeline, evaluate, hybrid_retrieval, ablation và baseline_retrieval thuộc pipeline cũ. legacy_metrics giữ cách chấm lịch sử. Chúng vẫn có lệnh module và một số phần được code hiện tại dùng chung; không tự chuyển dữ liệu hoặc kết quả của chúng sang research_v1.

Các launcher đánh số ở root trước đây chỉ chuyển tiếp sang những module này; đã đưa ra bản lưu riêng để root chỉ còn điểm chạy web. [Tài liệu lịch sử](../docs/history/) giữ giải thích và số liệu cũ.
