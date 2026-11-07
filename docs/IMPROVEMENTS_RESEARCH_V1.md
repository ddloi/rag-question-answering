# Cải tiến sau research_v1: ablation, reranker, train với context truy hồi

Phần điểm cộng của đề bài. Mọi số liệu đo trên **200 câu dev** với dataset, FAISS index và closed-book T5 của research_v1. Tập test vẫn khóa, nên đây là kết quả chẩn đoán trên dev; bảng test chính thức trong README vẫn là research_v1.

## Tóm tắt

| Cấu hình RAG (dev, official) | EM | F1 |
| --- | ---: | ---: |
| Cấu hình đã chốt: Dense, 2 passage, T5 v1 | 51,5% | 57,8% |
| + Reranker | 57,5% | 63,3% |
| + T5 train với context truy hồi (Dense) | 59,0% | 64,7% |
| **+ Reranker + T5 train với context truy hồi** | **63,0%** | **67,6%** |

| Cải tiến | Bằng chứng | Mã |
| --- | --- | --- |
| Ablation top-k/token budget, reranker | [results/research_v1_ablation_dev/](../results/research_v1_ablation_dev/) | [research_ablation.py](../experiments/research_ablation.py) |
| T5 train với context truy hồi | [results/research_v1_ablation_dev_rag_retrieved/](../results/research_v1_ablation_dev_rag_retrieved/), [results/training/research_v1/rag_retrieved/](../results/training/research_v1/rag_retrieved/) | [research_train_retrieved.py](../experiments/research_train_retrieved.py), [compare_ablation_runs.py](../scripts/compare_ablation_runs.py) |

Mỗi run có `REPORT.md`, `metrics.json`, `predictions.jsonl.gz`, `manifest.json` (SHA của dataset, model, reranker, source) và `source_snapshot.zip`. Thống kê: paired bootstrap 2.000 lần theo câu hỏi, seed 42.

## 1. Ablation top-k và token budget

Mốc `dense_top2_512` là đúng cấu hình `rag_full` đã chốt; run mới cho **200/200 đáp án trùng** với run dev đã lưu.

| Cấu hình (T5 v1) | EM | F1 | ΔEM so với mốc [CI 95%] | Phần passage T5 thấy |
| --- | ---: | ---: | --- | ---: |
| Dense top-1, 512 | 51,0% | 56,7% | −0,5 [−5,5; +4,5] | 100% |
| **Dense top-2, 512 (mốc)** | **51,5%** | **57,8%** | — | 81% |
| Dense top-3, 512 | 51,5% | 57,8% | 0,0 (200/200 đáp án giống mốc) | 54% |
| Dense top-2, 1024 | 53,5% | 59,0% | +2,0 [−1,5; +5,5] | 100% |
| Dense top-3, 1024 | 53,0% | 59,3% | +1,5 [−3,5; +6,5] | 99% |

- **Thêm passage thứ 3 vô ích khi giới hạn 512 token.** Mỗi chunk khoảng 290 token T5, nên passage thứ 3 gần như bị cắt hết. Top-k chỉ có nghĩa khi đi cùng token budget.
- **Nâng budget lên 1024** giảm lỗi "bị cắt" từ 17 xuống 13 câu và tăng EM 2 điểm, nhưng CI vẫn chứa 0. Checkpoint chỉ được train ở 512 token.

## 2. Cross-encoder reranker

`cross-encoder/ms-marco-MiniLM-L6-v2` (revision `233902d2…`, dùng sẵn, không fine-tune) chấm lại 30 ứng viên Dense theo cặp (câu hỏi, tiêu đề + đoạn văn), rồi lấy top-n cho T5.

| Retrieval | Dense | Dense + reranker |
| --- | ---: | ---: |
| Gold doc hit@1 | 84,0% | 91,0% |
| Answer recall@1 (official) | 66,5% | 75,0% |
| Answer recall@2 (official) | 76,0% | 83,0% |
| Answer recall@5 (official) | 82,5% | 88,5% |

| Cấu hình (T5 v1) | EM | F1 | ΔEM so với mốc [CI 95%] |
| --- | ---: | ---: | --- |
| Rerank top-1, 512 | 56,5% | 62,9% | +5,0 [−1,0; +11,5] |
| **Rerank top-2, 512** | **57,5%** | **63,3%** | **+6,0 [+1,0; +11,0]** |
| Rerank top-3, 512 | 57,0% | 62,8% | +5,5 [+0,5; +11,0] |

Số câu retrieval trượt giảm từ 35 xuống 23. Reranker tốt hơn Hybrid RRF, vốn không cải thiện so với Dense.

## 3. Train lại T5 RAG với context truy hồi

T5 v1 được train với **một passage gold**, nhưng lúc chạy thật lại đọc **hai passage truy hồi**, thường có passage gây nhiễu. Ở mốc, 45/200 câu có chuỗi đáp án trong context mà T5 vẫn sinh sai; đây là nhóm lỗi lớn nhất.

Bản mới chỉ đổi **context lúc train**; giữ nguyên 1.865 ID câu train của v1, base `google-t5/t5-base` cùng revision, 3 epoch, micro-batch 2 × tích lũy 8, LR 3e-4, bf16, seed 42, prompt và budget 512 token, chọn checkpoint theo dev loss (cả hai bản đều chọn epoch 1). Context train:

1. Dense lấy 30 ứng viên, reranker giữ top-2, đúng như pipeline lúc chạy.
2. Đáp án hiện trong phần T5 đọc được: dùng nguyên 2 passage (**1.453 câu, 78%**).
3. Nếu không: ghép passage gold với passage truy hồi tốt nhất không phải gold, thứ tự ngẫu nhiên theo seed (**412 câu, 22%**). Không câu nào bị loại.
4. Dev dùng top-2 sau rerank, không chèn gold.

Train mất 28,5 phút trên RTX 3050 6GB (351 bước), dev loss tốt nhất 1,26.

So sánh từng câu với T5 v1, cùng câu hỏi và cùng retrieval:

| Retrieval cho T5 | EM v1 | EM mới | ΔEM [CI 95%] | Sửa đúng / làm sai | F1 v1 → mới |
| --- | ---: | ---: | --- | ---: | ---: |
| Dense, 2 passage (mốc) | 51,5% | **59,0%** | **+7,5 [+3,0; +12,5]** | 20 / 5 | 57,8% → 64,7% |
| Dense, 1 passage | 51,0% | 55,0% | +4,0 [+1,0; +7,0] | 10 / 2 | 56,7% → 60,0% |
| Dense, 2 passage, 1024 token | 53,5% | 56,5% | +3,0 [−1,5; +8,0] | 15 / 9 | 59,0% → 62,1% |
| Dense + reranker, 2 passage | 57,5% | **63,0%** | **+5,5 [+1,0; +10,0]** | 17 / 6 | 63,2% → 67,6% |

Canonical EM cho cùng xu hướng: 47,5% → 54,5% (Dense) và 51,5% → 56,0% (có reranker).

## Phân rã lỗi (official, Dense top-2 trừ khi ghi khác, số câu trên 200)

| Cấu hình | Đúng | Retrieval trượt | Context thiếu/bị cắt | Sinh sai dù context có đáp án |
| --- | ---: | ---: | ---: | ---: |
| T5 v1 (mốc) | 103 | 35 | 17 | 45 |
| T5 v1, 1024 token | 107 | 35 | 13 | 45 |
| T5 v1 + reranker | 115 | 23 | 12 | 50 |
| T5 mới | 118 | 34 | 17 | **31** |
| T5 mới + reranker | 126 | 23 | 12 | 39 |

Reranker giảm lỗi retrieval; T5 mới giảm lỗi sinh. Hai cải tiến tác động lên hai bước khác nhau của pipeline nên cộng dồn được.

## Ví dụ nên dùng khi trình bày

| Câu hỏi (dev) | T5 v1 | T5 mới |
| --- | --- | --- |
| In our solar system, the sun is composed of mostly which gas? | Helium | Hydrogen |
| Politician Lev Davidovich Bronshtein was better known by what name? | Bronstein | Leon Trotsky |
| Jack Nicholson played Randle McMurphy in which film? | Patrick McMurphy | One Flew Over the Cuckoo's Nest |

Ở cả ba câu, context của v1 đã chứa đáp án nhưng v1 chép nhầm một cụm khác. Câu đầu có sẵn trong danh sách tình huống dev của demo.

**Không dùng câu Martina Hingis làm ví dụ cải tiến.** Model mới trả "Switzerland", khớp nhãn TriviaQA, nhưng nhãn này đã bị đánh dấu nghi vấn ([audit](../results/research_v1_label_audit_dev.json)): bà sinh ở Košice, Tiệp Khắc (nay thuộc Slovakia). Model mới chọn quốc tịch thay vì nơi sinh, tức là khớp nhãn chứ không đúng hơn về sự thật.

## Chạy lại

```powershell
python -m experiments.research_ablation                                   # ablation + reranker, T5 v1, ~10 phút
python -m experiments.research_train_retrieved                             # train T5 mới, ~30 phút
python -m experiments.research_ablation --rag checkpoints/research_v1/rag_retrieved --run-dir results/<run_moi>
python -m scripts.compare_ablation_runs --base results/research_v1_ablation_dev --candidate results/<run_moi> --output results/<run_moi>/comparison_vs_v1.json
```

Reranker cần ở `checkpoints/rerankers/ms-marco-MiniLM-L6-v2/`; tải `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a` bằng `huggingface_hub.snapshot_download(..., local_dir=...)`. T5 mới lưu ở `checkpoints/research_v1/rag_retrieved/`. Cả hai nằm trong `checkpoints/` nên Git bỏ qua; khi bàn giao phải gửi kèm. Có hai model này, demo `/research` hiện thêm lựa chọn **Dense + Reranker** và **T5 train với context truy hồi**.

## Giới hạn

- Kết quả trên dev, 7 cấu hình so cùng một mốc, chưa hiệu chỉnh cho so sánh nhiều lần. Dev cũng được dùng để chọn checkpoint (theo dev loss) cho cả v1 lẫn model mới, nên con số tuyệt đối có thể lạc quan một chút; so sánh giữa hai model vẫn công bằng vì cùng quy trình.
- Một seed, một lần train; CI chỉ phản ánh biến thiên theo câu hỏi.
- Muốn có bằng chứng test, cần chốt trước một cấu hình duy nhất (ví dụ Dense + reranker + T5 mới), ghi vào biên bản khóa, rồi chạy test một lần.
- 22% câu train vẫn dùng passage gold, nên model chưa học trường hợp context không có đáp án (khi đó nó vẫn đoán thay vì từ chối).
- Phân rã lỗi dựa trên trùng chuỗi từ vựng, không thay cho chấm faithfulness của người. Closed-book T5 không được train lại.
