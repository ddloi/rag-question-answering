# Rà soát mã nguồn ngày 06/11/2026

Áp dụng clean-code cho mã ứng dụng, experiments, scripts, deployment và tests. Ưu tiên lỗi vận hành, tránh lặp logic và tác động ngoài ý muốn. Kiểm tra tĩnh trên toàn bộ Python của các thư mục này; kiểm tra cú pháp cả hai tệp JavaScript. Graph codebase không khả dụng trong phiên làm việc, nên rà soát cấu trúc bằng source và tìm nơi sử dụng.

## Những thay đổi đã thực hiện

| Vấn đề | Xử lý | Kiểm chứng |
| --- | --- | --- |
| Tệp câu hỏi mẫu dev thiếu hoặc hỏng làm API thông tin trả lỗi | Ghi warning máy chủ, trả danh sách mẫu rỗng; trang vẫn mở được | TestClient với tệp thiếu, JSON sai, gzip sai và dữ liệu thiếu trường |
| Chunk size/overlap không hợp lệ có thể làm vòng lặp không tiến | Kiểm tra tham số trước khi chia chunk | Test trường hợp size không dương, overlap âm hoặc lớn hơn/bằng size |
| Ablation sao chép thuật toán chunking | Dùng chung hàm của prepare_chunks | Kiểm tra overlap và chunk cuối giữ kết quả hợp lệ |
| Ablation cũ fallback sang qa_pairs khi thiếu test | Báo lỗi trước khi nạp model; từ chối tập test rỗng | Test có qa_pairs nhưng thiếu test, và test rỗng |
| Seed toàn cục gây tác dụng phụ và split khác giữa các lần gọi | Dùng random.Random(SEED) riêng | Hai lần main trong thư mục tạm cho cùng split; RNG toàn cục giữ nguyên; khớp thứ tự seed 42 cũ |
| Config baseline hỏng bị thay bằng mặc định | Báo lỗi có đường dẫn và nguyên nhân | Test config sai và độ ưu tiên config local |
| Import, biến và f-string thừa | Dọn các trường hợp đã kiểm tra nơi sử dụng; giữ re-export metric/model bằng alias tường minh | Ruff và toàn bộ unittest |

Ablation chỉ nạp thư viện ML khi chạy tác vụ tương ứng. Import prepare_chunks không tự tạo thư mục dữ liệu; thư mục được tạo trong main khi cần ghi.

## Phạm vi kiểm tra

Trước sửa: 97/97 unittest đạt. Sau sửa: 106/106 đạt trên môi trường local có đủ thư viện ML và corpus lịch sử. Chín test mới bao phủ lỗi nêu trên. Hai test baseline sẽ được bỏ qua nếu chưa cài môi trường ML; không ép bổ sung thư viện ML vào môi trường kiểm thử nhẹ.

Ruff 0.17.4 kiểm tra nhóm E9 và F, không ép định dạng toàn dự án. Cài bằng requirements/lint.txt, chạy:

```powershell
python -m ruff check app.py ragqa experiments scripts deployment tests
python -m unittest discover -s tests
node --check ragqa/static/app.js
node --check ragqa/static/research.js
python -m scripts.check_demo
```

Ruff đạt với ngoại lệ tường minh trong ruff.toml: hai import chưa dùng ở research_data được giữ để bảo toàn source; cảnh báo F821 của research_evaluate xuất phát từ biến closure bị xóa sau khi đã gọi support xong. Đã đọc thứ tự khởi tạo → gọi → xóa và đối chiếu evaluator đã chạy; đây không phải sửa evaluator. Ngoại lệ áp dụng theo file, vì vậy khi tạo phiên bản nghiên cứu mới cần xem lại chúng.

Hai JavaScript qua kiểm tra cú pháp. Preflight web/BM25 đạt với 12.602 chunks và 120 câu lịch sử; đây là kiểm tra vận hành, không phải đánh giá chất lượng mới. Các script lint/type/i18n ở đường dẫn ~/.claude được skill đề cập không tồn tại trên máy; không báo chúng đã chạy. Chưa thực hiện kiểm tra kiểu tĩnh hoặc audit i18n toàn diện.

## Truy vết nghiên cứu

Bốn file phương pháp trong manifest test research_v1 vẫn khớp SHA256: research_evaluate, research_pipeline, metrics và rag_engine. Không chạy lại training hay thay output/nhãn/kết quả đã khóa trong đợt này. Các source_snapshot.zip cũ vẫn giữ nguyên và khớp hash phương pháp tương ứng.

Đối chiếu thêm cho thấy research_train và research_pipeline hiện tại khác hash thời điểm huấn luyện hai checkpoint; khác biệt đã tồn tại trong HEAD trước đợt clean-code. Snapshot của cả hai lượt train vẫn khớp manifest. Khi cần tái lập chính xác mã huấn luyện, dùng snapshot của lượt train cùng môi trường và manifest tương ứng; không coi mã hiện tại là bằng chứng đã tái lập training.

Kết quả đối chiếu hash chi tiết lưu local tại outputs/clean_code_frozen_hash_check.json. Quy trình kiểm tra mã được bổ sung vào README; gói source export có cả ruff.toml và requirements/lint.txt. Các tệp khóa Word đang mở thuộc thao tác của người dùng không nằm trong phần dọn mã.

Rà soát này nâng chất lượng phần mềm và kiểm tra các lỗi có căn cứ; không chứng nhận mọi nhánh thực thi hoặc hiệu năng mô hình đều đã được kiểm thử. Các hàm nghiên cứu đã khóa chưa được chia nhỏ chỉ để đạt giới hạn số dòng của style guide.
