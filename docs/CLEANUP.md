# Dọn file cũ ngày 02/11/2026

File cũ được đưa ra thư mục lưu riêng bên cạnh project, không xóa vĩnh viễn: `../Rag-question-answering_archive/cleanup_20261102_003207/` tính từ thư mục cha `Rag-question-answering`. Manifest ở đó ghi đường dẫn gốc, đường dẫn lưu, lý do, kích thước và SHA256 của từng file để có thể phục hồi.

## Đã bỏ khỏi thư mục làm việc

| Nhóm | Lý do |
| --- | --- |
| Bản `RAG_Baseline_Submission/` ngoài repository | Bản nộp độc lập cũ; tránh nhầm với bản RAG chính |
| Thư mục docs/ ở ngoài và docs/pipeline_studio_demo.png trong RAG | Hai ảnh minh họa cũ giống hệt checksum, không còn được tham chiếu |
| docs/REVIEW_2026-10-31.md | Biên bản hoàn thiện cũ, thay bằng trạng thái research_v1 và kiểm chứng hiện tại |
| notebooks/preprocessing.ipynb | Notebook khám phá dữ liệu cũ; giữ hai notebook research/Colab hiện tại |
| results/canonical_rescore_v1/ | Bản trung gian; scores, metrics và phiếu alias giống bản v2 còn giữ; manifest gốc được lưu riêng |
| results/evaluation_v2_smoke/, results/research_v1_dev_smoke/ | Run thử 3/5 câu; các run đầy đủ 120/200 câu vẫn còn |
| outputs/research_colab_transfer/ và helper/import comparison tạm | Các phần truyền đã ghép thành ZIP evidence có checksum khớp; dữ liệu nhập và receipt vẫn giữ |
| Source candidate/preview ZIP, hai thư mục giải nén thử, notebook draft trong outputs/ | Các bản trung gian; frozen source gốc, notebook chính và gói delivery được giữ |

Tổng cộng **16 nhóm, 357 file, 65.920.870 bytes (khoảng 62,9 MiB)** đã chuyển ra ngoài project. Đây là dung lượng được đưa ra khỏi thư mục làm việc; bản lưu vẫn chiếm dung lượng trên ổ đĩa.

## Được giữ vì còn dùng

- Corpus/index/split, hai checkpoint research_v1, config và source phương pháp đã khóa.
- Kết quả dev/test đầy đủ ở local/Colab, logs train, các pilot, source snapshots, test lock và phiếu người chấm.
- Bốn ZIP đầu vào frozen và ZIP Colab evidence; không tạo lại hoặc đổi checksum.
- Kết quả v2 đầy đủ, canonical v2 và demo_v3_smoke: còn được tài liệu tham chiếu, là bằng chứng lịch sử.
- Code demo `/legacy` và deployment Hugging Face: còn import/entry point hoặc lệnh sử dụng; file cũ không đồng nghĩa với file thừa. Launcher chuyển tiếp ở root sau đó đã được đưa ra bản lưu theo mục dưới.
- Cache và trạng thái local đã bị Git bỏ qua; không cần xóa để cập nhật GitHub.

README ngoài, tài liệu cấu trúc, danh mục notebook và danh mục kết quả đã được cập nhật để không trỏ tới các file đã chuyển. Không sửa đáp án, nhãn, model, evaluator hoặc protocol để dọn project.

Kiểm tra sau dọn: 357 file trong bản lưu khớp hash, 107 file dữ liệu/model/bằng chứng được bảo vệ không đổi, 97/97 tests đạt trước và sau, preflight web/BM25 và danh sách lệnh experiments đạt. Logs nằm trong outputs/cleanup_*.log. Gói source/evidence sau dọn có tên research_clean_*; xem [hướng dẫn GitHub](GITHUB_UPDATE.md).

## Thu gọn cấu trúc sau lần dọn đầu

- Root chỉ giữ app.py, README.md và metadata Git; 18 launcher/requirements chuyển tiếp cũ nằm trong bản lưu `../Rag-question-answering_archive/structure_20261102_003853/` tính từ thư mục cha.
- Chín tài liệu cũ và nội dung README dài chuyển vào docs/history/; PRODUCT.md và DESIGN.md chuyển vào docs/development/.
- docs/README.md là điểm đọc tài liệu; experiments/README.md chỉ rõ các module của luồng hiện tại.
- Bản lưu có 45 file trước sửa cùng SHA256 và mapping đường dẫn. Mã triển khai ML, corpus, model và các run hoàn chỉnh giữ nguyên. Lệnh đánh số ở root được thay bằng lệnh module; xem bảng chuyển đổi trong [ARCHITECTURE.md](ARCHITECTURE.md).

Kiểm tra sau thu gọn: 97/97 tests đạt, 151 liên kết tương đối trong tài liệu hợp lệ, preflight và lệnh module đạt; 107 file artifact được bảo vệ vẫn khớp checksum. Logs của lượt này là outputs/structure_*.log. Gói bàn giao mới nhất mang tên research_organized_*.

Kiểm tra source-only: 54 tests nghiên cứu/TriviaQA và cold web đạt. Full suite source-only có 8 failures/6 errors ở test demo lịch sử do không có corpus local; không đổi các lỗi đó thành PASS. README và hướng dẫn GitHub ghi rõ điều kiện chạy từng nhóm.
