# Quy trình thực nghiệm đã sửa

> Tài liệu này giữ protocol **v2 lịch sử**. Nghiên cứu mới dùng [RESEARCH_PROTOCOL_V1.md](../RESEARCH_PROTOCOL_V1.md), không trộn model/split/điểm giữa hai phiên bản. Normalizer mới trong metrics.py đã sửa; legacy_metrics.py bảo toàn phép chấm lịch sử.

> Rà soát 31/10/2026: protocol hiện tại còn các điểm cần chốt về evidence training, template/baseline, chuẩn hóa TriviaQA và holdout mới. Xem [SCIENTIFIC_READINESS.md](SCIENTIFIC_READINESS.md) trước khi dùng kết quả để viết báo. Canonical-only trong code hiện tại vẫn dùng normalizer tùy chỉnh, chưa phải full official TriviaQA evaluation.

## Dữ liệu và phạm vi

Giữ nguyên 800 QA và thứ tự split: 680 training pairs cũ, 120 test cũ, không loại mẫu theo kết quả. Dữ liệu gốc giữ tại `data/`; bản có bổ sung bằng chứng tại `data/verified_v2/`.

Nguồn phục hồi: `mandarjoshi/trivia_qa`, `rc.wikipedia`, `validation`, revision `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f`. Giữ question ID chính thức và answer aliases. Nếu excerpt 5.000 ký tự không chứa alias theo phép kiểm tra bảo thủ, dùng trang bằng chứng đầy đủ đầu tiên do TriviaQA cung cấp có alias. Nếu không tìm thấy, giữ nguyên và gắn cờ. Kiểm tra bằng chứng sau đó dùng ranh giới từ, nhận đúng số đứng cạnh ký hiệu đơn vị như `720°`.

Đây là **corpus subset có kiểm soát, có dùng nhãn để xây nguồn bằng chứng**, không phải benchmark tìm kiếm toàn Wikipedia. Việc sửa nguồn áp dụng cùng quy tắc cho cả 800 QA, trước khi đo lại; không chọn chunk size/top-k theo điểm test mới. Split theo câu hỏi trong cùng miền; có thể có trang Wikipedia chung giữa train và test, không tuyên bố khái quát sang tài liệu chưa gặp.

Checkpoint T5 giữ nguyên, đã fine-tune trên training pairs cũ; không fine-tune bằng câu hỏi/nhãn test sau khi phục hồi. Chất lượng checkpoint có thể bị ảnh hưởng bởi dữ liệu huấn luyện cũ; phiên bản trọng số được lưu SHA256.

## Ba hệ

- Generator-only: bỏ context khỏi cùng checkpoint T5. Đây là ablation không retrieval, không phải một mô hình closed-book được huấn luyện riêng.
- Retrieval-only: Dense + DistilBERT, chỉ trích span từ token context, không lấy question/special/padding tokens. Xét tối đa ba passage.
- RAG: Dense + T5, lấy tối đa hai passage và 512 token đầu vào. Lưu context thực sự còn lại theo offset tokenizer. Cả hai nhánh T5 dùng beam=4, max_new_tokens=32 và các cấu hình sinh còn lại giống nhau.

Thí nghiệm luôn chạy cả ba hệ, không dùng cosine gate để bỏ qua mẫu khó. Web vẫn có gate 0,25 nhằm hỗ trợ demo, vì vậy không dùng từng kết quả web để khẳng định tái lập hoàn toàn thí nghiệm.

## Chỉ số và phân rã lỗi

Mặc định mới: EM/F1 chỉ dùng canonical answer sau chuẩn hóa. Chỉ bật `--answer-policy official-aliases` khi cần tái lập cách chấm lịch sử; alias chưa được duyệt có thể sai loại đáp án. Run gốc dùng aliases, còn `canonical_rescore_v2` chấm lại cùng dự đoán với canonical-only. Cách chấm bảo thủ có thể bỏ sót biến thể đúng; dùng phiếu alias_review.csv để duyệt trước khi tạo bộ nhãn mới. Retrieval đo trực tiếp BM25, Dense và Hybrid RRF (k=60, pool=30), gồm doc Recall@3/5/10, alias evidence Recall@3/5/10 và doc MRR@30. Doc recall chỉ xét ID được gán nguồn; alias recall có thể khớp một đoạn không thật sự trả lời câu hỏi, nên không dùng thay đánh giá ngữ nghĩa.

Phân rã sau EM: `corpus_evidence_missing`, `retrieval_evidence_missing`, `context_selection_or_truncation`, `generation_wrong_with_lexical_evidence`. Kiểm tra bằng chứng bằng alias là proxy; tên nhóm mô tả đúng điều đó. Nếu bằng chứng nằm rank 3 nhưng T5 chỉ nhận rank 1–2, không tự gán thành lỗi generation. Nếu token bị cắt, xét context sau truncation.

`token_support_fraction` là độ trùng token, **không được gọi là semantic faithfulness**. `source_doc_hit_in_used_passages` và `used_context_has_gold_alias` là proxy nguồn/bằng chứng, không phải citation precision theo từng mệnh đề.

## Đánh giá trên mẫu

Chọn ngẫu nhiên 20/120 câu bằng seed 42, độc lập với điểm số. `review_sample.csv` để trống nhãn, giữ context thực sự đưa vào T5. Bản đã rà soát lưu riêng, ghi `reviewer_kind=agent_self_review` nếu do trợ lý kiểm tra, không giới thiệu là đánh giá người độc lập.

Faithfulness: `supported` nếu context hỗ trợ đáp án cho đúng câu hỏi; `unsupported` nếu không có bằng chứng hoặc mâu thuẫn; `unclear` nếu không đủ để kết luận. Citation correctness ở đây xét **nguồn được cung cấp kèm câu trả lời**, không giả định T5 sinh inline citations: `supports_answer`, `does_not_support` hoặc `unclear`. Mỗi nhãn có lý do và ID nguồn. Ground-truth EM khác với faithfulness: câu trả lời có thể đúng nhưng không được context hỗ trợ.

## Tái lập

Từ `RAG`, cài môi trường ML cùng model/cache local. Các lệnh phục hồi tạo file mới và dừng nếu đã tồn tại để tránh ghi đè artifact đã chốt.

```powershell
python -m scripts.recover_trivia_source
python -m experiments.verify_dataset --build-index
python -m experiments.evaluate --run-dir results/evaluation_new_run
python -m unittest discover -s tests -v
```

Mỗi run lưu manifest môi trường, seed, SHA256 dữ liệu/index/checkpoint/mã đánh giá, metrics tổng hợp và predictions từng câu. Các artifact lớn được Git bỏ qua nhưng manifest vẫn được lưu; cần phục hồi bằng lệnh hoặc sao chép kèm đúng checksum. Các bảng nghiên cứu cũ là lịch sử, không so sánh trực tiếp với corpus v2 có nhiều passage hơn.

## Demo v3 tách khỏi thực nghiệm

Web có Hybrid RRF và ưu tiên từ khóa hiếm, corpus demo_v3 thêm hai trang được chọn sau khi xem lỗi test. Các tùy chọn này không thay đổi mặc định Dense của RAGSystem trong evaluation. Không dùng kết quả demo để báo cải thiện held-out; xem [phạm vi bản cập nhật](IMPROVEMENTS_V3.md).
