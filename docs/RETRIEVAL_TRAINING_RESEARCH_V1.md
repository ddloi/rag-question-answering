# Train lại RAG T5 với context truy hồi (dev, research_v1)

## Vì sao làm

Trong research_v1, T5 RAG được train với **một passage gold** chứa đáp án. Khi chạy thật, nó lại đọc **hai passage do retrieval trả về**, thường có passage gây nhiễu. Ở cấu hình đã chốt, 45/200 câu dev có chuỗi đáp án trong context nhưng T5 vẫn sinh sai ([ablation](ABLATION_RERANKER_RESEARCH_V1.md)). Đây là nhóm lỗi lớn nhất.

## Thiết kế

Chỉ thay đổi **context lúc train**. Mọi thứ khác giữ giống research_v1:

| Giữ nguyên | Giá trị |
| --- | --- |
| QA train | Đúng 1.865 ID của run RAG v1 |
| Base model | `google-t5/t5-base`, cùng revision |
| Siêu tham số | 3 epoch, micro-batch 2 × tích lũy 8, LR 3e-4, weight decay 0,01, bf16, seed 42 |
| Prompt và budget | `question: … context: …`, tối đa 512 token |
| Chọn checkpoint | Dev loss thấp nhất (cả hai bản đều chọn epoch 1) |

Context lúc train được dựng như sau:

1. Dense MiniLM lấy 30 ứng viên, cross-encoder `ms-marco-MiniLM-L6-v2` xếp hạng lại, giữ top-2. Đây chính là pipeline lúc suy luận.
2. Nếu đáp án hiện trong phần T5 đọc được (sau khi cắt ở 512 token): dùng nguyên 2 passage đó. Có **1.453 câu (78%)** thuộc trường hợp này.
3. Nếu không: ghép passage gold với passage truy hồi tốt nhất không phải gold, thứ tự ngẫu nhiên theo seed, để đáp án luôn có căn cứ trong context. Có **412 câu (22%)**.
4. Không câu nào bị loại. Dev dùng top-2 sau rerank, không chèn gold, giống hệt lúc chạy thật.

Train mất 28,5 phút trên RTX 3050 6GB (351 bước). Dev loss tốt nhất: 1,26.

```powershell
python -m experiments.research_train_retrieved
python -m experiments.research_ablation --rag checkpoints/research_v1/rag_retrieved --run-dir results/<run_moi>
python -m scripts.compare_ablation_runs --base results/research_v1_ablation_dev --candidate results/<run_moi> --output results/<run_moi>/comparison_vs_v1.json
```

Model lưu ở `checkpoints/research_v1/rag_retrieved/` (Git bỏ qua). Bằng chứng train: [results/training/research_v1/rag_retrieved/](../results/training/research_v1/rag_retrieved/), gồm manifest, log, trạng thái trainer và context đã dùng cho từng câu. Bằng chứng đánh giá: [results/research_v1_ablation_dev_rag_retrieved/](../results/research_v1_ablation_dev_rag_retrieved/).

## Kết quả trên 200 câu dev (official)

So sánh từng câu (paired) với T5 v1, **cùng câu hỏi và cùng retrieval**, chỉ khác checkpoint T5:

| Retrieval cho T5 | EM v1 | EM mới | ΔEM [CI 95%] | Sửa đúng / làm sai | F1 v1 → mới |
| --- | ---: | ---: | --- | ---: | ---: |
| Dense, 2 passage (cấu hình đã chốt) | 51,5% | **59,0%** | **+7,5 [+3,0; +12,5]** | 20 / 5 | 57,8% → 64,7% |
| Dense, 1 passage | 51,0% | 55,0% | +4,0 [+1,0; +7,0] | 10 / 2 | 56,7% → 60,0% |
| Dense, 2 passage, 1024 token | 53,5% | 56,5% | +3,0 [−1,5; +8,0] | 15 / 9 | 59,0% → 62,1% |
| Dense + Reranker, 2 passage | 57,5% | **63,0%** | **+5,5 [+1,0; +10,0]** | 17 / 6 | 63,2% → 67,6% |

Canonical EM cho cùng xu hướng: 47,5% → 54,5% (Dense, 2 passage) và 51,5% → 56,0% (có reranker).

**Gộp cả hai cải tiến** (reranker + T5 train với context truy hồi): EM official trên dev đi từ **51,5% lên 63,0%** so với cấu hình đã chốt.

Phân rã lỗi (Dense, 2 passage, số câu trên 200):

| Model | Đúng | Retrieval trượt | Context thiếu/bị cắt | Sinh sai dù context có đáp án |
| --- | ---: | ---: | ---: | ---: |
| T5 v1 | 103 | 35 | 17 | 45 |
| T5 train với context truy hồi | 118 | 34 | 17 | **31** |

Phần cải thiện đến đúng từ nhóm lỗi sinh. Hai nhóm lỗi retrieval gần như không đổi, đúng như dự kiến vì retrieval không đổi.

## Ví dụ nên dùng khi trình bày

| Câu hỏi (dev) | T5 v1 | T5 mới |
| --- | --- | --- |
| In our solar system, the sun is composed of mostly which gas? | Helium | Hydrogen |
| Politician Lev Davidovich Bronshtein was better known by what name? | Bronstein | Leon Trotsky |
| Jack Nicholson played Randle McMurphy in which film? | Patrick McMurphy | One Flew Over the Cuckoo's Nest |

Ở cả ba câu, context của v1 đã chứa đáp án nhưng v1 chép nhầm một cụm khác. Câu đầu có sẵn trong danh sách tình huống dev của demo.

**Không dùng câu Martina Hingis làm ví dụ cải tiến.** Model mới trả "Switzerland", khớp nhãn TriviaQA, nhưng nhãn này đã bị đánh dấu nghi vấn ([audit](../results/research_v1_label_audit_dev.json)): bà sinh ở Košice, Tiệp Khắc (nay thuộc Slovakia). Model mới chọn quốc tịch thay vì nơi sinh, tức là khớp nhãn chứ không đúng hơn về sự thật.

## Giới hạn

- Kết quả trên dev. Dev được dùng để chọn checkpoint (theo dev loss) cho cả v1 lẫn model mới, nên con số tuyệt đối có thể lạc quan một chút; phép so sánh giữa hai model vẫn công bằng vì cùng quy trình.
- Một seed, một lần train. Chưa đo biến thiên giữa các seed.
- Tập test chưa được chạy với model mới; bảng test chính thức trong README vẫn là research_v1. Muốn có bằng chứng test, cần chốt trước một cấu hình duy nhất (ví dụ Dense + Reranker, 2 passage, T5 mới), ghi vào biên bản khóa, rồi chạy test một lần.
- 22% câu train vẫn dùng passage gold, nên model chưa học trường hợp context hoàn toàn không có đáp án (khi đó nó vẫn đoán thay vì từ chối).
- Closed-book T5 không được train lại, nên so sánh ba hệ chính thức vẫn dùng cặp model v1.
