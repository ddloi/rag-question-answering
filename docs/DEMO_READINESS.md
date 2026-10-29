# Kiểm tra trước buổi demo

## Phạm vi đã kiểm tra

Giao diện web tại `http://localhost:8001` chạy **BM25 retrieval + chọn một câu nguyên văn từ đoạn nguồn**. Đường chạy này nằm trong `ragqa/web.py` và `ragqa/engine.py` (qua launcher `app.py` ở root); **không gọi T5, FAISS hoặc Hybrid Retrieval** của các script thí nghiệm.

Có thể dùng giao diện này để trình bày việc chuẩn hóa câu hỏi, xếp hạng nguồn BM25, xem đoạn trích và kiểm tra đóng góp điểm của từng từ khóa. Chưa nên giới thiệu đây là demo đầy đủ của hệ RAG sinh câu trả lời trong README.

## Cách chạy

Mở PowerShell trong thư mục project, dùng môi trường Python đã có FastAPI, Uvicorn và Pydantic 2:

```powershell
Set-Location 'D:\Projects\Git_Projects\Rag-question-answering\RAG'
python -m pip install -r requirements/demo.txt
python app.py
```

Sau đó mở `http://localhost:8001`. Giữ cửa sổ chạy server trong lúc demo; dừng bằng `Ctrl+C`. Giao diện cần server hoạt động để truy vấn dữ liệu. Các đường dẫn dữ liệu và static được tính theo vị trí mã nguồn, không phụ thuộc thư mục làm việc khi import ứng dụng.

Corpus local được đọc từ `data/chunk_metadata.jsonl`; tại thời điểm kiểm tra có **3.940 chunks**. Danh sách câu hỏi mẫu đọc từ tám dòng đầu `data/test_qa.jsonl`. Số lượng trên giao diện cần lấy từ API hiện tại, không coi con số trong tài liệu là dữ liệu cố định.

## Kiểm thử backend

```powershell
python -m pip install -r requirements/test.txt
python -m unittest discover -s tests -v
```

Bộ kiểm thử theo cấu trúc mới chạy **12/12 tests đạt** (9 engine/API và 3 đường dẫn). Lệnh engine tương thích cũ cũng chạy **9/9 tests đạt**, bao gồm:

- `k1` thay đổi điểm thật và `b` thay đổi ranking thật trên corpus kiểm thử.
- Điểm xếp hạng dùng cùng công thức với bảng phân rã theo từ khóa.
- Query không có từ phù hợp hoặc chỉ có dấu câu trả về trạng thái không có nguồn.
- Câu được chọn là đoạn nguyên văn và kết quả xếp hạng ổn định giữa các lần chạy.
- Query và tham số vượt giới hạn bị từ chối.
- Import ứng dụng từ thư mục làm việc khác vẫn đọc đúng corpus và câu hỏi mẫu.
- API thông tin, mẫu, truy vấn và các tệp giao diện trả về thành công.

Các kiểm thử này xác nhận hành vi phần mềm, **không đo độ chính xác QA** và không thay thế đánh giá mô hình trên tập test.

## Những điểm đã sửa có ảnh hưởng trực tiếp đến demo

- Trước đây `k1/b` chỉ tác động vào bảng giải thích, không tác động vào điểm ranking thật. Hai tham số nay được áp dụng ngay trong tính BM25.
- Điểm BM25 và đóng góp của từng từ khóa nay dùng cùng công thức IDF dương. Biến thể này khác cách xử lý IDF/epsilon trong `rank_bm25` ở một số script nghiên cứu; không dùng các điểm hiện trên giao diện để diễn giải lại số liệu baseline đã lưu.
- API giới hạn query từ 1–2.000 ký tự, `top_k` từ 1–10, `k1` từ 0,5–3 và `b` từ 0–1.
- Nhãn hệ thống mô tả đúng chế độ retrieval. Không có cam kết rằng trích nguyên văn đồng nghĩa với đáp án đúng.
- Chạy `python app.py` tránh khởi tạo corpus hai lần.

## Hạn chế cần trình bày trung thực

Corpus và câu hỏi mẫu là tiếng Anh. Cơ chế phân tích từ hiện tại phù hợp với dữ liệu này; chưa được đánh giá cho câu hỏi tiếng Việt.

Chọn câu dựa vào mức khớp từ khóa có thể đưa ra câu không trả lời đúng câu hỏi, ngay cả khi tài liệu nguồn đúng. Ví dụ câu hỏi về tên cá mập cơ khí trong *Jaws* có thể lấy đúng tài liệu nhưng chọn câu giới thiệu bộ phim thay vì câu chứa “Bruce”. Các câu hỏi đầu về trò chơi Fivestones hoặc nhân vật Antigonus cũng có thể truy hồi sai tài liệu.

Danh sách tám mẫu đầu được giữ nguyên để người xem có thể thấy cả trường hợp yếu. Không nên kỳ vọng mọi câu hỏi mẫu đều cho câu trích chứa đáp án.

“Độ phủ từ khóa” là tỷ lệ từ khóa câu hỏi xuất hiện trong câu trích; đây **không phải** độ chính xác, xác suất đáp án đúng hay chỉ số faithfulness được đánh giá độc lập. Điểm `confidence` cũ trong API cũng chỉ là một heuristic từ vựng.

Thời gian xử lý đo tại server và thời gian toàn bộ yêu cầu trên trình duyệt có thể khác nhau. Không cam kết một ngưỡng độ trễ cố định cho mọi máy hoặc mọi truy vấn.

## Câu hỏi minh họa có thể dùng

Những câu dưới đây nằm trong tập test hiện tại và đã cho câu trích chứa đáp án tham chiếu khi chạy cấu hình mặc định. Đây là các ví dụ thuận tiện để giải thích giao diện, **không phải tập đánh giá ngẫu nhiên, không phải minh chứng cho accuracy và không bảo đảm mọi câu hỏi khác đạt chất lượng tương tự**.

| Câu hỏi nhập vào | Đáp án tham chiếu để đối chiếu |
| --- | --- |
| The Nullarbor Plain is in which country? | Australia |
| Proteus and Nereid are moons of which planet? | Neptune |
| Who was famous for popularising the 'little black dress' in the 1920s? | Coco Chanel |
| The river Danube flows into which sea? | Black Sea |
| In Greek mythology, Amphitrite, queen of the sea, was the wife of which god? | Poseidon |
| ‘Back to Bedlam’ was the debut album for which British singer? | James Blunt |
| What was the first Beatles film? | A Hard Day's Night |
| Of which African country is Niamey the capital? | Niger |

Trong buổi demo, có thể chạy một ví dụ tốt, mở đoạn nguồn và bảng điểm, thay `k1/b` để giải thích tác động, rồi chạy một mẫu yếu để nêu giới hạn. Nếu cần chứng minh chất lượng hệ RAG sinh câu trả lời, cần chạy và kiểm tra riêng đường T5/FAISS/Hybrid; phần đó chưa được xác nhận bằng các kiểm thử backend nói trên.

## Kiểm tra UX/UI đã thực hiện

- Giao diện tiếng Việt, nền sáng; câu hỏi mẫu chỉ điền biểu mẫu, không chạy truy vấn ngoài ý muốn.
- Chạy mẫu Jaws và mở đáp án tham chiếu: thấy rõ khác biệt giữa câu trích và đáp án “Bruce”.
- Chạy ví dụ Proteus/Nereid và Danube; mở nguồn, phân rã điểm, sao chép đoạn trích.
- Chạy chuỗi không có trong corpus: hiện hướng dẫn không có kết quả, không hiển thị nguồn giả.
- Tắt máy chủ thử nghiệm, xác nhận lỗi kết nối nội tuyến; khởi động lại và dùng “Thử lại” thành công.
- Thay Top-K: hiện cảnh báo kết quả cũ; Ctrl+Enter thực thi và Enter giữ chức năng xuống dòng.
- Kiểm tra bố cục 320px, 390px, 768px và desktop 1280px: không thấy tràn ngang. Mobile chuyển focus đến kết quả sau khi chạy.
- Kiểm tra tương phản chữ hiển thị ở trạng thái desktop có kết quả: không có mục dưới tỷ lệ 4,5:1 trong phép kiểm tra màu chữ/nền. Đây chưa phải kiểm định khả năng tiếp cận đầy đủ.
- Kiểm tra cú pháp JavaScript/Python và `git diff --check`.

Đã thêm `requirements/demo.txt` cho web demo nhẹ (lệnh cài cũ qua `requirements_demo.txt` vẫn hoạt động), không cần GPU hoặc tải mô hình. Các thư viện ML đầy đủ vẫn nằm trong `requirements/ml.txt`. Để chạy kiểm thử API, cài thêm `httpx` nếu môi trường chưa có.

## Vị trí mã nguồn sau khi sắp xếp

Mã web ở `ragqa/`, thí nghiệm ở `experiments/`, kiểm thử ở `tests/`. Notebook ở `notebooks/`, kết quả đo ở `results/`. `ragqa/paths.py` là nơi xác định đường dẫn dữ liệu. Xem `docs/ARCHITECTURE.md` để tra lệnh mới và các điểm gọi tương thích.
