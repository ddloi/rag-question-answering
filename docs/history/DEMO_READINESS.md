> **Bản hiện tại 31/10/2026:** 43/43 tests đạt. Web chọn demo_v3 (12.602 chunks), rồi verified_v2 (12.556), rồi corpus gốc (3.940) theo artifact đầy đủ. Có lựa chọn Hybrid/Dense. Corpus demo có nguồn được chọn sau khi xem lỗi test; số liệu nghiên cứu dùng verified_v2. Xem [bản cập nhật](IMPROVEMENTS_V3.md).

# Kiểm tra trước buổi demo

## Cách chạy

```powershell
# Mở PowerShell trong thư mục RAG sau khi clone repository
python -m pip install -r requirements/demo.txt
python app.py
```

Mở http://127.0.0.1:8001. Giữ tiến trình server trong lúc demo. Khi code Python thay đổi, cần khởi động lại server; khi đổi JS/CSS hãy tải lại trang. BM25 đọc corpus local và có thể chạy ngay. Các đường dẫn dữ liệu/static của bản chính độc lập thư mục làm việc.

## Chế độ BM25 và RAG

- **BM25:** tắt checkbox RAG nếu muốn chỉ xem xếp hạng từ khóa và câu trích nguyên văn.
- **RAG:** cài môi trường `requirements/ml.txt`; cần `data/faiss.index`, `data/chunk_metadata.jsonl`, checkpoint T5 trong `generator_finetuned/` và MiniLM trong cache Hugging Face. Bấm **Nạp mô hình RAG**, chờ trạng thái sẵn sàng rồi chạy câu hỏi. DistilBERT là đối chứng tùy chọn; thiếu cache của nó không chặn T5.
- Nạp mô hình chạy nền. Truy vấn gửi trong lúc nạp vẫn trả BM25; chạy lại sau khi RAG sẵn sàng. Khi nạp lỗi, giao diện có nút thử nạp lại; truy vấn thường không tự lặp lại lần nạp đã lỗi.
- Web chỉ đọc mô hình local/cache. Khi chuẩn bị máy mới, tải model trước buổi demo bằng lệnh dưới; không cần chạy nếu cache đã có.

```powershell
python -m pip install -r requirements/ml.txt
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
# Đối chứng DistilBERT tùy chọn
python -c "from transformers import AutoTokenizer, AutoModelForQuestionAnswering; name='distilbert-base-cased-distilled-squad'; AutoTokenizer.from_pretrained(name); AutoModelForQuestionAnswering.from_pretrained(name)"
```

Checkpoint T5 đã fine-tune cần được sao chép hoặc khôi phục riêng. Web không thay bằng T5 gốc khi thiếu checkpoint. Các artifact lớn bị Git bỏ qua, vì vậy clone mã nguồn chưa đủ để chạy neural.

## Lệnh kiểm tra

```powershell
python -m pip install -r requirements/test.txt
python -m unittest discover -s tests -v
python -m scripts.check_demo
python -m scripts.check_demo --neural
```

`check_demo` chỉ đọc dữ liệu, kiểm tra API/static, ID đoạn nguồn, nguồn tham chiếu, câu hỏi train/test trùng nhau và một truy vấn BM25. `--neural` nạp model local và chạy thêm một câu hỏi thật, không tải model hay ghi đè kết quả.

Kiểm tra ngày **31/10/2026**: **26/26 tests đạt** cho BM25, validation, đường dẫn, API và vòng đời RAG. Kiểm thử hồi quy neural dùng mô phỏng để chạy được trên môi trường nhẹ. Kiểm tra suy luận thật riêng trên máy này: CUDA RTX 3050 6GB, nạp khoảng 46,8 giây; câu hỏi Proteus/Nereid trả “Neptune”, tổng suy luận khoảng 1,46 giây trong lần chạy đầu. Lần nạp từ web có thể lâu hơn lần chạy trực tiếp; nên nạp trước buổi trình bày và chờ trạng thái sẵn sàng. Tốc độ thay đổi theo máy và trạng thái cache.

## Luồng trình bày gợi ý

1. Mở trang và nạp RAG trước khi bắt đầu thuyết trình.
2. Nhập **Proteus and Nereid are moons of which planet?** rồi chạy. T5 đã trả Neptune trong kiểm tra thực tế. Mở nguồn FAISS và BM25 để chỉ ra bằng chứng.
3. Tắt RAG, chạy **The river Danube flows into which sea?** để giải thích BM25, câu trích và bảng đóng góp điểm từ khóa.
4. Đổi Dense/Hybrid, Top-K hoặc k1/b: giao diện báo kết quả cũ. Chạy lại để cập nhật; k1/b tác động điểm thật của BM25.
5. Chạy một mẫu yếu như tên cá mập cơ khí trong Jaws để nêu giới hạn: truy hồi đúng tài liệu chưa bảo đảm chọn/sinh đúng đáp án.

8 câu hỏi mẫu được chọn từ test_qa.jsonl, chia nhóm theo lần chạy lịch sử để giải thích đúng, lỗi sinh, thiếu từ khóa và context. Bấm mẫu chỉ điền câu hỏi; không tự chạy. Câu hỏi trùng chính xác một câu trong toàn bộ 120 câu test vẫn có nhãn tham chiếu, kể cả khi không nằm trong 8 mẫu hiển thị. Nhãn được gắn sau suy luận, không đưa vào prompt T5.

## Cách đọc kết quả

- BM25 dùng IDF dương và chọn một câu nguyên văn; khác biến thể rank_bm25 dùng trong thí nghiệm baseline. Độ phủ từ khóa không phải xác suất đúng.
- Chọn Dense để chỉ dùng MiniLM/FAISS theo cosine; chọn Hybrid để kết hợp pool Dense/BM25 bằng RRF và ưu tiên từ khóa hiếm. Điểm passage hiển thị vẫn là Cosine thật, không phải điểm RRF. T5 dùng tối đa hai đoạn đầu, đầu vào cắt ở 512 token; không phải mọi đoạn Top-K đều được sinh từ. Web dùng beam=4 và ngưỡng cosine 0,25; script nghiên cứu có cấu hình riêng.
- Ngưỡng cosine chỉ loại một số truy vấn ít liên quan, không bảo đảm chống ảo giác. T5 không có context có thể đúng hoặc sai. Trong smoke test, câu này trả Jupiter còn T5 có context trả Neptune; DistilBERT trả sai “Saturn's moon Mimas”.
- Nhãn đối chiếu chỉ kiểm tra chuỗi đáp án sau trim/lowercase; không tính EM/F1 trên tập dữ liệu và không công nhận mọi chuỗi con là đúng.
- Các bước và độ trễ ở thanh quy trình là nhánh BM25. Tổng máy chủ tính cả neural; toàn yêu cầu trên trình duyệt tính thêm truyền nhận và đọc phản hồi. Thời gian sinh T5 không bao gồm các đối chứng.
- API giữ kết quả BM25 nếu neural lỗi hoặc đang bận. Mỗi tiến trình chỉ chạy một lượt suy luận neural cùng lúc.

Corpus gốc có **3.940 chunks**, **120 câu test**; bản demo_v3 hiện tại có **12.602 chunks**. Tiền kiểm không thấy câu hỏi train/test trùng nhau. Đây không phải kiểm định rò rỉ dữ liệu đầy đủ. Số liệu nghiên cứu cũ trong README cần được tái lập bằng môi trường/tham số tương ứng trước khi khẳng định chất lượng toàn tập.

## Xử lý nhanh khi demo

| Hiện tượng | Thao tác |
| --- | --- |
| Mất kết nối | Kiểm tra server còn chạy, sau đó bấm Thử lại |
| RAG đang nạp | Chờ Sẵn sàng rồi chạy lại câu hỏi |
| Nạp model thất bại | Xem log server, môi trường ML, cache MiniLM và checkpoint; bấm thử nạp lại |
| Thiếu DistilBERT | Vẫn dùng T5 + BM25; nạp đối chứng trước buổi sau |
| RAG đang bận | Chờ truy vấn trước xong hoặc tắt RAG để dùng BM25 |
| Không có nguồn BM25 | Thử câu hỏi tiếng Anh, tên riêng hoặc mẫu trong corpus |

[Kiến trúc và vị trí mã nguồn](../ARCHITECTURE.md) · [Môi trường Python](../../requirements/README.md)
