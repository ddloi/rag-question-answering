# Các trường hợp để báo cáo

Bản hiện tại: demo_v3, 12.602 chunks, Hybrid RRF + ưu tiên từ hiếm, cùng checkpoint T5. Chạy thật ngày 31/10/2026 qua API; toàn bộ phản hồi lưu trong [responses.jsonl](../results/demo_v3_smoke/responses.jsonl) và [manifest](../results/demo_v3_smoke/manifest.json).

Đây là **8 ví dụ đã chọn để trình bày**, không phải mẫu ngẫu nhiên để ước lượng độ chính xác. Nguồn v3 được bổ sung sau khi xem lỗi test. Các nhóm câu hỏi trên giao diện mô tả lần chạy lịch sử; kết quả mới dưới đây có thể khác.

| Câu hỏi / từ khóa | Đáp án tham chiếu | T5 RAG hiện trả | Dùng minh họa |
| --- | --- | --- | --- |
| Nullarbor Plain ở nước nào? | Australia | Australia | Trả lời đúng; mở nguồn để kiểm tra bằng chứng. |
| Melbourne Cup 1930 | Phar Lap | Phar Lap | Trả lời đúng, dù tiêu đề nguồn là Horse racing. |
| Tên cá mập máy trong Jaws | Bruce | Shark | Sai đáp án; đúng chủ đề không đồng nghĩa đúng thực thể. |
| Et Tu Brute | Julius Ceasar (nhãn gốc sai chính tả) | Jesus | Sai đáp án; khi duyệt alias cần kiểm tra biến thể Julius Caesar. |
| Fivestones / Knucklebones / Snobs | Jacks | Jacks | Bổ sung nguồn giúp khắc phục thiếu coverage. |
| Antigonus trong vở nào? | The Winter’s Tale | The Bear | Hybrid đã chọn đúng trang và đoạn có Antigonus, nhưng T5 vẫn sinh sai. |
| Robin Hood bản Disney là con gì? | Fox | Animal | Đáp án quá chung và không khớp tham chiếu. |
| Phim Beatles đầu tiên | A Hard Day’s Night | Magical Mystery Tour | Sai; cần kiểm tra đoạn chọn và phần context còn sau giới hạn token. |

## Đối chiếu Dense và Hybrid

Trên cùng corpus demo_v3, câu Antigonus: Dense lấy William Shakespeare và trả **Macbeth**; Hybrid lấy The Winter's Tale, khớp từ hiếm “antigonus”, nhưng trả **The Bear**. Đây là cải thiện việc chọn bằng chứng, **chưa sửa được lỗi sinh**. Không tuyên bố câu này đã trả lời đúng.

Câu Fivestones hiện trả **Jacks** với cả Dense và Hybrid nhờ trang Knucklebones mới. Không gán riêng cải thiện này cho Hybrid.

## Cách trình bày

1. Nạp RAG, chờ trạng thái Sẵn sàng. Giữ server chạy và tải lại trang sau khi cập nhật JS/CSS.
2. Chọn mẫu đúng, bấm chạy, mở nguồn và đáp án tham chiếu.
3. Chọn Antigonus, chạy Dense rồi Hybrid để chỉ ra khác biệt nguồn và lỗi generation còn tồn tại.
4. Chọn Fivestones để trình bày vai trò của corpus coverage.
5. Chọn Jaws hoặc Beatles để trình bày giới hạn và hướng cải tiến tiếp theo; không thay câu trả lời sinh bằng nhãn tham chiếu.

T5 dùng tối đa hai passage và 512 token đầu vào, nên đoạn nằm trong top-k chưa chắc được dùng đầy đủ. Cosine, độ phủ từ khóa và chuỗi đáp án có trong passage đều không chứng minh quan hệ trả lời đúng. Cảnh báo thiếu từ khóa vẫn cho mô hình sinh câu trả lời.

Ground truth trên web chỉ so chuỗi đơn giản. Báo cáo định lượng dùng [bản chấm lại canonical](IMPROVEMENTS_V3.md): RAG EM 34,17%, F1 39,76% trên 120 dự đoán verified_v2 cũ. Bộ alias cần người đánh giá duyệt; không trộn điểm demo v3 với điểm nghiên cứu.
