# Ablation top-k/token budget và reranker (dev, research_v1)

Phần điểm cộng của đề bài: **reranker** và **ablation top-k**. Mọi số liệu đo trên **200 câu dev**. Không train lại; dùng nguyên dataset, FAISS index và checkpoint RAG T5 của research_v1. Tập test vẫn khóa, nên đây là kết quả chẩn đoán trên dev, không phải bằng chứng test độc lập.

Bằng chứng: [results/research_v1_ablation_dev/](../results/research_v1_ablation_dev/), gồm `REPORT.md`, `metrics.json`, `predictions.jsonl.gz`, `manifest.json` (SHA của dataset, model, reranker, source) và `source_snapshot.zip`. Mã: [experiments/research_ablation.py](../experiments/research_ablation.py).

```powershell
python -m experiments.research_ablation            # 200 dev, khoảng 10 phút trên RTX 3050 6GB
python -m experiments.research_ablation --samples 5 # chạy thử
```

## Thiết kế

- **Mốc so sánh** `dense_top2_512`: Dense MiniLM, 2 passage, T5 nhận tối đa 512 token. Đây đúng là cấu hình `rag_full` đã chốt. Run mới cho **200/200 đáp án trùng** với run dev đã lưu, nên phần so sánh không bị lệch do môi trường.
- **Ablation top-k**: đưa 1, 2 hoặc 3 passage vào T5.
- **Ablation token budget**: nâng giới hạn đầu vào T5 từ 512 lên 1024 token. T5 dùng relative position nên đọc được chuỗi dài hơn, nhưng checkpoint chỉ được train ở 512.
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L6-v2` (revision `233902d2…`), chấm lại 30 ứng viên Dense theo cặp (câu hỏi, tiêu đề + đoạn văn), rồi lấy top-n cho T5. Reranker dùng sẵn, không fine-tune trên TriviaQA.
- Thống kê: paired bootstrap 2.000 lần theo câu hỏi, seed 42, so với mốc.

## Kết quả QA (official)

| Cấu hình | EM | F1 | ΔEM so với mốc [CI 95%] | Phần passage T5 thấy |
| --- | ---: | ---: | --- | ---: |
| Dense top-1, 512 | 51,0% | 56,7% | −0,5 [−5,5; +4,5] | 100% |
| **Dense top-2, 512 (mốc)** | **51,5%** | **57,8%** | — | 81% |
| Dense top-3, 512 | 51,5% | 57,8% | 0,0 (200/200 đáp án giống mốc) | 54% |
| Dense top-2, 1024 | 53,5% | 59,0% | +2,0 [−1,5; +5,5] | 100% |
| Dense top-3, 1024 | 53,0% | 59,3% | +1,5 [−3,5; +6,5] | 99% |
| Rerank top-1, 512 | 56,5% | 62,9% | +5,0 [−1,0; +11,5] | 100% |
| **Rerank top-2, 512** | **57,5%** | **63,3%** | **+6,0 [+1,0; +11,0]** | 81% |
| Rerank top-3, 512 | 57,0% | 62,8% | +5,5 [+0,5; +11,0] | 54% |

Canonical EM cho cùng xu hướng: mốc 47,5%, rerank top-2 51,5%. Số đầy đủ trong `REPORT.md`.

## Kết quả retrieval

| Chỉ số | Dense | Dense + rerank |
| --- | ---: | ---: |
| Gold doc hit@1 | 84,0% | 91,0% |
| Answer recall@1 (official) | 66,5% | 75,0% |
| Answer recall@2 (official) | 76,0% | 83,0% |
| Answer recall@5 (official) | 82,5% | 88,5% |

## Phân rã lỗi (official, số câu trên 200)

| Cấu hình | Đúng | Retrieval trượt | Context chọn thiếu/bị cắt | Sinh sai dù context có đáp án |
| --- | ---: | ---: | ---: | ---: |
| Dense top-2, 512 (mốc) | 103 | 35 | 17 | 45 |
| Dense top-2, 1024 | 107 | 35 | 13 | 45 |
| Rerank top-2, 512 | 115 | 23 | 12 | 50 |

## Đọc kết quả

1. **Thêm passage thứ 3 vô ích khi giới hạn 512 token.** Hai passage (~290 token/chunk) đã gần chạm 512, nên passage thứ 3 gần như bị cắt hết; T5 cho 200/200 đáp án giống hệt mốc. Top-k chỉ có nghĩa khi đi cùng token budget.
2. **Nâng budget lên 1024** làm giảm lỗi "bị cắt" (17 → 13) và tăng EM 2 điểm, nhưng CI vẫn chứa 0, nên chưa kết luận được.
3. **Reranker là cải tiến rõ nhất.** Số câu retrieval trượt giảm từ 35 xuống 23, EM tăng 6 điểm và CI 95% không chứa 0 trên dev. Reranker tốt hơn Hybrid RRF, vốn không cải thiện so với Dense.
4. **Nút thắt còn lại là bước sinh.** Sau rerank, 50/200 câu có chuỗi đáp án trong context nhưng T5 vẫn trả sai. Nguyên nhân khả dĩ: T5 chỉ được train với một passage gold, chưa quen với passage gây nhiễu. Hướng tiếp theo là train với context lấy từ retrieval; đã thực hiện trong [train lại với context truy hồi](RETRIEVAL_TRAINING_RESEARCH_V1.md), giảm nhóm lỗi này từ 45 xuống 31 câu ở cấu hình Dense.

## Giới hạn

- Chỉ đo trên dev, với 7 cấu hình so cùng một mốc và chưa hiệu chỉnh cho so sánh nhiều lần. Kết quả +6,0 EM của rerank top-2 nên trình bày là bằng chứng trên dev, chưa phải kết luận trên test.
- Một seed, một checkpoint, cùng 200 câu; CI chỉ phản ánh biến thiên theo câu hỏi.
- Phân rã lỗi dựa trên trùng chuỗi từ vựng, không thay cho chấm faithfulness của người.
- Muốn có bằng chứng test, cần chốt trước đúng một cấu hình (ví dụ rerank top-2, 512), ghi vào biên bản khóa, rồi chạy test một lần.
