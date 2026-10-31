# Cập nhật project lên GitHub

Repository Git nằm trong thư mục `RAG`. README trong thư mục cha và bản `RAG_Baseline_Submission` là các thư mục độc lập, không tự được đưa vào commit của repository này.

## Những gì được theo dõi

- Mã web, thí nghiệm, kiểm thử, tài liệu, câu hỏi test và kết quả đánh giá đầy đủ.
- Corpus/index gốc đã được Git theo dõi. `.gitignore` không tự bỏ theo dõi các file đã được commit.
- `generator_finetuned/model.safetensors` được theo dõi bằng **Git LFS**; file trên máy khoảng 892 MB, còn Git lưu pointer. Cần cài Git LFS và kiểm tra dung lượng/quyền đẩy LFS của repository.
- Dữ liệu phục hồi, corpus/index `verified_v2`, `demo_v3` và các smoke run được giữ trên máy nhưng Git bỏ qua. Manifest v2 được giữ; người clone cần tự khôi phục corpus hoặc nhận artifact riêng. Web chọn bản đầy đủ theo thứ tự demo_v3 → verified_v2 → corpus gốc. Nguồn JSON nhỏ và manifest demo được giữ trong Git; chạy `python -m scripts.build_demo_corpus` để tạo v3 sau khi dựng v2.
- Predictions của run đầy đủ khoảng 17 MB và được giữ để có thể kiểm tra từng kết quả. Chúng chứa đoạn văn từ corpus TriviaQA/Wikipedia; giữ provenance và tuân thủ giấy phép nguồn khi chia sẻ.

## Chuẩn bị trên máy trước khi commit

```powershell
git lfs install
git lfs status
python -m pip install -r requirements/test.txt
python -m unittest discover -s tests -v
python -m scripts.check_demo
git diff --check
git status --short
```

Rà lại file thay đổi, đặc biệt manifest, kết quả từng câu và các tệp mới chưa theo dõi. Chỉ stage các file muốn công bố; không dùng `git add -f` cho cache, artifact v2 hoặc trọng số bị bỏ qua ngoài chính sách đã chọn. Kiểm tra bản đã stage bằng `git diff --cached --stat` trước khi commit.

## Trên máy clone

```powershell
git lfs install
git clone <URL-repository>
cd <ten-repository>
git lfs pull
python -m pip install -r requirements/demo.txt
python app.py
```

Để chạy nhánh T5/FAISS, xem [hướng dẫn demo](DEMO_READINESS.md) và chuẩn bị thư viện ML cùng cache MiniLM/DistilBERT. Nếu thiếu dữ liệu v2, xem [quy trình thực nghiệm](EVALUATION_PROTOCOL.md); các bước tạo dữ liệu cần nguồn gốc và raw documents tương ứng, không giả định chỉ clone là có đủ file.

Run v2 gốc dùng aliases chưa duyệt; điểm canonical chấm lại nằm tại `results/canonical_rescore_v2` (RAG EM 34,17%, F1 39,76%). Giữ cả run gốc và bản chấm lại để kiểm toán. Bản v1 là đầu ra trung gian đã bị bỏ qua trong Git. Cần người đánh giá duyệt alias và phiếu faithfulness để hoàn tất chất lượng nhãn. Các đường dẫn và SHA256 trong manifest kết quả mô tả lần chạy lịch sử; mã đang chỉnh sửa có thể khác hash đã ghi.
