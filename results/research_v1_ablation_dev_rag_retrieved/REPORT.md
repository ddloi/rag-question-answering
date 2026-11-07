# Ablation top-k và reranker trên dev (research_v1)

n = 200 câu dev. Mốc so sánh: `dense_top2_512` (đúng cấu hình rag_full đã chốt). Không train lại; tập test vẫn khóa.

## EM/F1 của T5 RAG

| Cấu hình | Official EM | Official F1 | Canonical EM | Canonical F1 | Token vào | % passage T5 thấy | ΔEM official [CI 95%] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense_top1_512 | 55.00% | 59.98% | 51.00% | 56.37% | 323 | 100% | -4.0 [-7.5; -0.5] |
| dense_top2_512 | 59.00% | 64.73% | 54.50% | 60.54% | 509 | 81% | — |
| dense_top3_512 | 59.00% | 64.73% | 54.50% | 60.54% | 512 | 54% | +0.0 [+0.0; +0.0] |
| dense_top2_1024 | 56.50% | 62.15% | 52.00% | 59.19% | 624 | 100% | -2.5 [-6.0; +0.5] |
| dense_top3_1024 | 57.50% | 62.63% | 52.50% | 58.94% | 919 | 99% | -1.5 [-6.0; +2.5] |
| rerank_top1_512 | 60.00% | 65.00% | 53.50% | 60.29% | 328 | 100% | +1.0 [-5.0; +7.0] |
| rerank_top2_512 | 63.00% | 67.57% | 56.00% | 63.14% | 508 | 81% | +4.0 [-1.5; +9.5] |
| rerank_top3_512 | 63.00% | 67.57% | 56.00% | 63.14% | 512 | 54% | +4.0 [-1.5; +9.5] |

## Retrieval: Dense so với Dense + reranker (pool 30)

| Chỉ số | Dense | Rerank |
| --- | ---: | ---: |
| doc_hit_rate@1 | 84.0% | 91.0% |
| doc_hit_rate@2 | 88.0% | 93.5% |
| doc_hit_rate@3 | 91.0% | 94.0% |
| doc_hit_rate@5 | 93.0% | 95.0% |
| doc_hit_rate@10 | 94.5% | 95.0% |
| official_answer_recall@1 | 66.5% | 75.0% |
| official_answer_recall@2 | 76.0% | 83.0% |
| official_answer_recall@3 | 77.5% | 85.0% |
| official_answer_recall@5 | 82.5% | 88.5% |
| official_answer_recall@10 | 88.5% | 90.5% |
| canonical_answer_recall@1 | 59.5% | 66.0% |
| canonical_answer_recall@2 | 68.5% | 76.5% |
| canonical_answer_recall@3 | 71.5% | 79.5% |
| canonical_answer_recall@5 | 76.5% | 83.5% |
| canonical_answer_recall@10 | 83.5% | 86.0% |

## Phân rã lỗi (official)

| Cấu hình | context_selection_or_truncation | correct | generation_wrong_with_answer_string_in_context | retrieval_answer_string_missing |
| --- | ---: | ---: | ---: | ---: |
| dense_top1_512 | 28 | 110 | 28 | 34 |
| dense_top2_512 | 17 | 118 | 31 | 34 |
| dense_top3_512 | 17 | 118 | 31 | 34 |
| dense_top2_1024 | 13 | 113 | 40 | 34 |
| dense_top3_1024 | 8 | 115 | 42 | 35 |
| rerank_top1_512 | 24 | 120 | 33 | 23 |
| rerank_top2_512 | 12 | 126 | 39 | 23 |
| rerank_top3_512 | 12 | 126 | 39 | 23 |

Giới hạn: kết quả dev, một seed, không phải bằng chứng test độc lập. CI là paired bootstrap 2.000 lần theo câu hỏi. Phân rã lỗi là proxy từ vựng.
