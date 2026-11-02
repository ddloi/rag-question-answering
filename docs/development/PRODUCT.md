# Product

## Register

product

## Users

Sinh viên trình bày đồ án hỏi–đáp và giảng viên xem demo trên máy tính, máy chiếu. Người xem cần hiểu câu hỏi, kết quả truy hồi và bằng chứng nguồn mà không phải đọc thuật ngữ triển khai trước.

## Product Purpose

Cho phép thử câu hỏi trên corpus TriviaQA tiếng Anh, theo dõi các bước xử lý và kiểm tra đoạn nguồn. Trang FastAPI so sánh BM25 + trích câu với FAISS + T5 fine-tuned khi mô hình local đã sẵn sàng. Web có Dense hoặc Hybrid RRF với ưu tiên từ hiếm; thí nghiệm khoa học trong experiments giữ cấu hình Dense riêng. Corpus demo v3 tách khỏi corpus đánh giá và được ghi nhãn rõ.

Trang nghiên cứu `/research` dùng đúng các lớp inference/artifacts research_v1 của evaluation: closed-book T5 được fine-tune riêng, Dense + DistilBERT extractive và Dense/Hybrid RRF + T5 RAG. Ba đầu ra là ba cột; ground truth là phần đối chiếu riêng sau suy luận. Câu mẫu lấy từ run dev đầy đủ, nhóm lỗi dựa trên dự đoán đã lưu; test không được dùng làm bộ câu hỏi dễ để demo. Khi đủ hai model, trang gốc mở giao diện nghiên cứu; `/legacy` giữ trải nghiệm v3.

## Brand Personality

Rõ ràng, điềm tĩnh, trung thực. Giao diện tiếng Việt sáng, gọn; câu hỏi và tài liệu giữ tiếng Anh theo dữ liệu. Hướng này đã được người dùng xác nhận cho demo giảng viên.

## Anti-references

Tránh dashboard dày đặc thuật ngữ, nhiều chữ in hoa, số liệu giả lập và lời hứa độ chính xác không được kiểm chứng. Không cần mô phỏng hội thoại khi thao tác chính là một lần truy vấn và kiểm tra nguồn.

## Design Principles

- Câu hỏi → đoạn trích → nguồn là luồng chính.
- Phân biệt trích dẫn nguyên văn với câu trả lời đúng.
- Đo số liệu thực tế và giải thích ý nghĩa của điểm số.
- Giữ tham số, phân tích kỹ thuật ở lớp thông tin mở rộng.
- Luôn có hướng dẫn tiếp theo khi rỗng, đang tải, không có kết quả hoặc lỗi.

## Accessibility & Inclusion

Chữ dễ đọc trên máy chiếu, tương phản tối thiểu WCAG AA, sử dụng bằng bàn phím, nhãn biểu mẫu rõ ràng, thông báo trạng thái cho công nghệ hỗ trợ, hỗ trợ màn hình nhỏ và giảm chuyển động.
