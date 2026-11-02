# Bản cải tiến demo và chấm điểm — 31/10/2026

## Thay đổi dùng được ngay

- Web cho chọn **Hybrid** hoặc **Dense** để so sánh trên cùng corpus và checkpoint. Hybrid lấy pool 30 của Dense và BM25, kết hợp RRF với hằng số 60.
- Khi một từ khóa có df ≤ 1% corpus và IDF ≥ 1,5 lần từ đứng sau, Hybrid ưu tiên passage chứa nguyên từ đó trong pool. Đây là heuristic cho demo, chưa có bằng chứng tăng chất lượng trên tập độc lập. Không tra đáp án hoặc viết quy tắc riêng cho câu hỏi mẫu.
- Cosine vẫn tính từ vector MiniLM thật; RRF lưu trong trường riêng. Thứ hạng Hybrid không phải thứ tự giảm dần Cosine. Ngưỡng 0,25 được áp dụng lên Cosine của passage đứng đầu.
- Giữ đủ ba kết quả T5 không context, extractive và RAG. Cảnh báo từ khóa thiếu không ngăn suy luận. Đáp án tham chiếu gắn sau suy luận.
- Nguồn bổ sung có liên kết Wikipedia trong từng passage. Giao diện ghi rõ corpus demo không phải benchmark.

## Corpus demo riêng

`data/demo_v3` có **12.602 chunks**: 12.556 chunks của `verified_v2` và 46 chunks từ hai trang Knucklebones, The Winter's Tale. Chọn các trang sau khi quan sát hai lỗi Fivestones/Antigonus; **không dùng bộ này để báo điểm kiểm thử độc lập**. Không sửa nhãn test hoặc thay trọng số T5.

Nguồn JSON, revision, thời điểm lấy và checksum lưu tại `data/demo_sources` và `data/demo_v3/manifest.json`. Chạy `python -m scripts.build_demo_corpus` sau khi dựng verified_v2 và chuẩn bị cache MiniLM. Lệnh dừng khi thư mục đầu ra đã tồn tại. Web ưu tiên corpus đầy đủ theo thứ tự demo_v3 → verified_v2 → corpus gốc.

## Kết quả kiểm tra demo

43/43 tests đạt; preflight xác nhận 12.602 chunks, 120 test và API/static hoạt động. Chạy thật 8 mẫu đã chọn: Fivestones trả Jacks; Antigonus đổi nguồn từ William Shakespeare sang The Winter's Tale với Hybrid, nhưng T5 trả The Bear nên vẫn sai. Lưu phản hồi và manifest tại `results/demo_v3_smoke`; xem [bảng mẫu](HUONG_DAN_CASE_DEMO.md). Không dùng tỷ lệ đúng của 8 mẫu để báo chất lượng toàn tập.

Kiểm tra cú pháp JavaScript đạt. Lệnh chụp Edge trước đó bị từ chối, nhưng sau đó đã xác minh trực tiếp qua công cụ trình duyệt: luồng Fivestones, Antigonus Dense/Hybrid, bộ lọc, trạng thái, tab bàn phím và bố cục ở hai kích thước. Có [ảnh và biên bản xác minh UI](UI_VERIFICATION.md). Antigonus vẫn sai ở bước sinh; không tuyên bố hệ thống đạt chính xác toàn bộ.

## Chấm lại dự đoán đã lưu

Official aliases có mục sai loại đáp án, ví dụ “Mopsa” được nhận cho câu hỏi tên vở kịch. Mặc định evaluation mới dùng **canonical answer**, không chấp nhận alias chưa duyệt. Tệp `results/canonical_rescore_v2/alias_review.csv` để trống quyết định cho **1.626 alias ứng viên**, cần người đánh giá rà soát.

Điểm dưới đây chấm lại **cùng 120 dự đoán lịch sử trên verified_v2**, không chạy lại mô hình và không đo Hybrid mới:

| Hệ | EM canonical | F1 canonical |
| --- | ---: | ---: |
| T5 không context | 0,83% | 1,93% |
| Dense + DistilBERT extractive | 26,67% | 36,67% |
| Dense + T5 RAG | **34,17%** | **39,76%** |

Canonical-only là cách chấm bảo thủ: có thể từ chối biến thể đúng hoặc sửa chính tả của nhãn. Ví dụ nhãn gốc “Julius Ceasar” không giống “Julius Caesar”. Không gọi đây là nhãn đã được duyệt hoàn toàn.

Doc Recall@5 của Dense/Hybrid RRF lịch sử vẫn là 92,50%; chỉ số này chỉ xét tài liệu được gán nguồn. Canonical evidence Recall@5 lần lượt 75,00%/72,50%. Phân rã RAG: 41 đúng, 8 thiếu chuỗi canonical trong corpus, 21 thiếu trong top-k, 15 do chọn/cắt context, 35 sai dù có chuỗi canonical trong context. Các phép dò chuỗi là proxy, không thay thế đánh giá ngữ nghĩa.

Artifact mới có checksum dự đoán/dữ liệu và mã chấm điểm. Giữ nguyên run cũ để kiểm toán; không trộn điểm alias cũ 42,50% với điểm canonical mới.

```powershell
# Chấm lại, không suy luận; dùng một tên đầu ra chưa tồn tại
python -m scripts.rescore_canonical --output results/canonical_rescore_new
# Đánh giá mới trên corpus nghiên cứu, canonical là mặc định
python -m experiments.evaluate --answer-policy canonical --run-dir results/evaluation_new_run
python -m unittest discover -s tests -v
python -m scripts.check_demo
```

Phiếu faithfulness/citation 20 câu trước đây vẫn là **agent_self_review**, không phải đánh giá độc lập của con người. Bản nâng cấp chưa được chạy Colab và chưa đo đủ 120 câu với Hybrid mới. Nên dùng kết quả lịch sử đã chấm lại để báo cáo khoa học, còn các ví dụ v3 để minh họa cách xử lý lỗi.

[Metrics canonical](../../results/canonical_rescore_v2/metrics.json) · [Manifest](../../results/canonical_rescore_v2/manifest.json) · [Scores từng câu](../../results/canonical_rescore_v2/scores.jsonl) · [Phiếu duyệt alias](../../results/canonical_rescore_v2/alias_review.csv) · [Kịch bản demo](HUONG_DAN_CASE_DEMO.md)
