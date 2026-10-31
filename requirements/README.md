# Môi trường Python

Từ thư mục gốc repository:

| Nhu cầu | Lệnh |
| --- | --- |
| Demo BM25 trên web | `python -m pip install -r requirements/demo.txt` |
| Chạy kiểm thử web và đường dẫn | `python -m pip install -r requirements/test.txt` |
| Web RAG neural và thí nghiệm ML đầy đủ | `python -m pip install -r requirements/ml.txt` |
| Triển khai Gradio Space | Dùng `deployment/huggingface/requirements.txt` |

Các tệp `requirements*.txt` ở root chỉ chuyển tiếp để các lệnh cài cũ tiếp tục hoạt động. `test.txt` bao gồm môi trường demo và HTTPX, không cần tải mô hình ML.

`lock.txt` giữ nguyên snapshot môi trường thí nghiệm đã có, bao gồm các gói CUDA `+cu124`. Đây không phải lockfile đa nền tảng và không tự cấu hình nguồn tải CUDA. Dùng môi trường phù hợp với snapshot khi tái lập thí nghiệm; dùng `demo.txt` để chạy giao diện nhẹ. Không cập nhật phiên bản ML trong lần sắp xếp này.

Notebook có thêm các phụ thuộc phân tích như matplotlib, seaborn và scipy. Cài bằng `python -m pip install -r requirements/notebook.txt` nếu muốn chạy lại notebook; việc sắp xếp không thực thi notebook hoặc huấn luyện lại.

Web không tự nạp thư viện ML lúc khởi động. Khi bật RAG, cần checkpoint fine-tuned local và cache MiniLM; xem hướng dẫn trong docs/DEMO_READINESS.md. Các kiểm thử mặc định không tải mô hình.
