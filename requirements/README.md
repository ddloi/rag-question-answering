# Môi trường Python

Từ thư mục gốc repository:

| Nhu cầu | Lệnh |
| --- | --- |
| Demo BM25 trên web | `python -m pip install -r requirements/demo.txt` |
| Chạy kiểm thử web và đường dẫn | `python -m pip install -r requirements/test.txt` |
| Research_v1: hai T5, MiniLM, DistilBERT và FAISS | `python -m pip install -r requirements/research.txt` |
| Web RAG neural và thí nghiệm ML đầy đủ | `python -m pip install -r requirements/ml.txt` |
| Triển khai Gradio Space | Dùng `deployment/huggingface/requirements.txt` |

Các tệp requirements chuyển tiếp ở root đã đưa ra bản lưu riêng để tránh nhầm môi trường. Dùng trực tiếp các đường dẫn trong bảng. `test.txt` bao gồm môi trường demo và HTTPX, không cần tải mô hình ML.

`lock.txt` giữ nguyên snapshot môi trường thí nghiệm đã có, bao gồm các gói CUDA `+cu124`. Đây không phải lockfile đa nền tảng và không tự cấu hình nguồn tải CUDA. Dùng môi trường phù hợp với snapshot khi tái lập thí nghiệm; dùng `demo.txt` để chạy giao diện nhẹ. Không cập nhật phiên bản ML trong lần sắp xếp này.

Notebook có thêm các phụ thuộc phân tích như matplotlib, seaborn và scipy. Cài bằng `python -m pip install -r requirements/notebook.txt` nếu muốn chạy lại notebook; việc sắp xếp không thực thi notebook hoặc huấn luyện lại.

Web không tự nạp thư viện ML lúc khởi động. Research_v1 cần hai checkpoint local và cache model; xem [hướng dẫn vận hành](../docs/HOC_VA_VAN_HANH.md). Các kiểm thử mặc định không tải mô hình.
