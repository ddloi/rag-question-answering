# Chấm faithfulness và citation correctness

Mở `human_review_blinded.csv` của run nghiên cứu đã hoàn tất. Mẫu chọn cố định 50 QA bằng seed 42; mỗi QA có đáp án Dense RAG và Hybrid RAG, thứ tự điều kiện được che. Không đưa `review_key.json` hoặc bảng EM/F1 cho người chấm trước khi hoàn tất.

Người dùng đã chọn tự chấm. Phiếu [human_review_author.csv](../results/research_v1_test/human_review_author.csv) của test đã điền sẵn `reviewer_id=author_1` và `reviewer_kind=human_author`; ba cột nhãn/lý do vẫn trống. Phiếu có 100 dòng cho 50 QA, không phải 100 câu độc lập. Tệp gốc được giữ nguyên. Điền `faithfulness`, `citation_labels_json`, `rationale`; giữ nguyên các cột bằng chứng và ID. UTF-8 BOM giúp mở tiếng Việt trong Excel.

Chỉ dùng câu hỏi, đáp án dự đoán và context thực dùng trong phiếu để chấm **faithfulness**. Không xem đáp án chuẩn để suy ra rằng câu trả lời chắc chắn faithful. Một câu đúng nhãn vẫn có thể không được context hỗ trợ.

| Nhãn | Quy tắc |
| --- | --- |
| supported | Context hỗ trợ đáp án cho đúng quan hệ mà câu hỏi yêu cầu |
| unsupported | Context không hỗ trợ hoặc mâu thuẫn; có một từ xuất hiện chưa đủ |
| unclear | Context mơ hồ, thiếu thông tin hoặc không đủ để quyết định |

Với **citation correctness**, chấm từng chunk trong `source_chunk_ids`. Điền `citation_labels_json` là JSON ánh xạ chính xác mỗi chunk ID sang `supports_answer`, `does_not_support` hoặc `unclear`. Ví dụ:

```json
{"chunk_abc": "supports_answer", "chunk_xyz": "does_not_support"}
```

Đánh giá các nguồn do ứng dụng cung cấp từ passage thật sự còn sau truncation; không giới thiệu chúng là inline citations do T5 tự sinh. Khi một nguồn không hỗ trợ đáp án, không đổi nguồn để làm citation đúng hơn.

Mỗi dòng phải có `rationale`, `reviewer_id` và `reviewer_kind`:

- `human_independent`: người không triển khai/chọn kết quả, được che điều kiện.
- `human_author`: tác giả đồ án tự chấm.
- `agent_self_review`: trợ lý tự rà; không đổi thành human để hoàn tất checklist.

Nếu hai người chấm, giữ hai bộ phiếu hoặc hai dòng với cùng review_id và reviewer_id khác nhau; giữ nhãn ban đầu, ghi lý do xử lý bất đồng trong biên bản riêng. Không xóa các nhãn bất lợi.

```powershell
python -m scripts.summarize_research_review --reviews <phieu_da_cham.csv> --key <run/review_key.json> --output <ket_qua_moi.json>
```

Với phiếu tác giả đã chuẩn bị, sau khi chấm xong:

```powershell
python -m scripts.summarize_research_review --reviews results/research_v1_test/human_review_author.csv --key results/research_v1_test/review_key.json --output results/research_v1_test/human_author_final.json
```

`human_author_status.json` hiện ghi **NOT RUN** vì chưa có nhãn/lý do; việc điền sẵn danh tính người chấm không tạo ra kết quả đánh giá.

Phiếu trống trả `NOT RUN`; phiếu chưa đủ trả `PARTIAL`. Citation precision tính trên nguồn đã quyết định supports_answer/does_not_support; số nguồn unclear được báo riêng. Faithfulness báo cả `supported_fraction_all` (mẫu số gồm unclear) và `supported_fraction_assessed` (chỉ supported/unsupported), kèm số unclear. Khi báo tỷ lệ phải nói rõ mẫu số và số annotation, không dùng tỷ lệ đã loại unclear mà giấu độ bao phủ. Kết quả agent và human được tổng hợp riêng.
