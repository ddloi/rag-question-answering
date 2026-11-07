# Ablation top-k và reranker trên dev (research_v1)

n = 200 câu dev. Mốc so sánh: `dense_top2_512` (đúng cấu hình rag_full đã chốt). Không train lại; tập test vẫn khóa.

## EM/F1 của T5 RAG

| Cấu hình | Official EM | Official F1 | Canonical EM | Canonical F1 | Token vào | % passage T5 thấy | ΔEM official [CI 95%] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense_top1_512 | 51.00% | 56.72% | 47.50% | 53.23% | 323 | 100% | -0.5 [-5.5; +4.5] |
| dense_top2_512 | 51.50% | 57.80% | 47.50% | 54.42% | 509 | 81% | — |
| dense_top3_512 | 51.50% | 57.80% | 47.50% | 54.42% | 512 | 54% | +0.0 [+0.0; +0.0] |
| dense_top2_1024 | 53.50% | 58.96% | 49.00% | 56.17% | 624 | 100% | +2.0 [-1.5; +5.5] |
| dense_top3_1024 | 53.00% | 59.27% | 49.50% | 56.66% | 919 | 99% | +1.5 [-3.5; +6.5] |
| rerank_top1_512 | 56.50% | 62.91% | 51.00% | 58.31% | 328 | 100% | +5.0 [-1.0; +11.5] |
| rerank_top2_512 | 57.50% | 63.25% | 51.50% | 58.61% | 508 | 81% | +6.0 [+1.0; +11.0] |
| rerank_top3_512 | 57.00% | 62.75% | 51.00% | 58.11% | 512 | 54% | +5.5 [+0.5; +11.0] |

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
| dense_top1_512 | 28 | 102 | 35 | 35 |
| dense_top2_512 | 17 | 103 | 45 | 35 |
| dense_top3_512 | 17 | 103 | 45 | 35 |
| dense_top2_1024 | 13 | 107 | 45 | 35 |
| dense_top3_1024 | 9 | 106 | 50 | 35 |
| rerank_top1_512 | 25 | 113 | 39 | 23 |
| rerank_top2_512 | 12 | 115 | 50 | 23 |
| rerank_top3_512 | 12 | 114 | 51 | 23 |

Kiểm tra tái lập: `dense_top2_512` cho 200/200 đáp án trùng với run đã chốt `research_colab_dev`.

Giới hạn: kết quả dev, một seed, không phải bằng chứng test độc lập. CI là paired bootstrap 2.000 lần theo câu hỏi. Phân rã lỗi là proxy từ vựng.
