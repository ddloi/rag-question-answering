# Cập nhật project lên GitHub

## Phiên bản nghiên cứu research_v1

Đọc [protocol](RESEARCH_PROTOCOL_V1.md) và [tiến độ](RESEARCH_UPGRADE_STATUS.md). Commit source, config, manifest/replay descriptor, portable legacy exclusions, notebook, tests và kết quả nén/log nhỏ. Corpus JSONL/index và `checkpoints/` đã bị Git bỏ qua; không ép stage chúng. `.gitattributes` giữ LF cho manifest có byte checksum để clone Windows/Linux không làm mất liên kết data–model.

Tạo các gói riêng khi cần chuyển sang Colab/máy demo:

```powershell
python -m scripts.export_research_bundle --kind source --output outputs/research_source.zip
python -m scripts.export_research_bundle --kind data --output outputs/research_data.zip
python -m scripts.export_research_bundle --kind models --output outputs/research_models.zip
python -m scripts.export_research_bundle --kind evidence --output outputs/research_evidence.zip
```

Gói models yêu cầu hai run hoàn tất. Gói này chứa trọng số suy luận, logs và manifest, không chứa optimizer để resume train. Mọi gói có danh sách SHA256, không chứa `.git`, credentials hoặc cache model. Lệnh từ chối ghi đè: chọn tên mới khi cần xuất lại. Gói data bỏ raw source caches; có thể dựng lại nguồn theo revision pin. Không cần xóa dữ liệu local để giữ repository gọn.

Sau thu gọn cấu trúc ngày 02/11/2026, gói bàn giao mới nhất là `outputs/research_organized_source_final.zip` và `outputs/research_organized_evidence_final.zip`; receipt tại `outputs/research_organized_receipts.json`. Các gói organized chưa có hậu tố final, research_clean_* và research_delivery_* là snapshot trước đó. Giữ nguyên bốn gói gốc `research_source/data/models/evidence.zip` và checksum của lượt Colab; không thay chúng bằng gói organized/clean/delivery khi chạy notebook kiểm chứng frozen. Notebook `notebooks/research_verify_frozen_colab.ipynb` là source sạch chưa có output; evidence execution thật và liên kết notebook Drive nằm trong [hồ sơ Colab](COLAB_VERIFICATION_RESEARCH_V1.md).

Trên clone chỉ có source, app vẫn mở trang nghiên cứu và báo thiếu artifact rõ ràng. Dựng theo notebook hoặc giải nén source/data/models cùng thư mục gốc `RAG/`, cài `requirements/research.txt` và `requirements/demo.txt`, rồi chạy `python app.py`. Trang `/` chọn research khi đủ hai model; `/research` mở trực tiếp, `/legacy` giữ demo cũ nếu có corpus cũ. Nếu dùng corpus replay ở thư mục khác, đặt `RAG_RESEARCH_DATA_DIR` đúng đường dẫn; model manifest phải khớp dataset đó.

97 tests trên máy hiện tại đạt. Full suite có test demo cũ phụ thuộc corpus; lượt thử source-only ghi 8 failures/6 errors do thiếu các fixture này, giữ nguyên log outputs/structure_clean_source_tests.log. Hai nhóm research/TriviaQA chạy từ source giải nén đạt 49+5 tests; cold web mở /research và báo thiếu artifact đúng. Trên clone chỉ source, dùng hai lệnh test có pattern trong README; chuẩn bị corpus trước khi chạy full suite/preflight demo lịch sử.

Repository Git nằm trong thư mục `RAG`; README ở thư mục cha không tự được đưa vào commit của repository này. Bản nộp baseline độc lập và file trung gian cũ đã chuyển ra thư mục lưu riêng ngoài project; xem [biên bản dọn project](CLEANUP.md). Các bằng chứng nghiên cứu đầy đủ vẫn giữ tại results/.

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

Để chạy nhánh T5/FAISS, xem [hướng dẫn demo](history/DEMO_READINESS.md) và chuẩn bị thư viện ML cùng cache MiniLM/DistilBERT. Nếu thiếu dữ liệu v2, xem [quy trình thực nghiệm](history/EVALUATION_PROTOCOL.md); các bước tạo dữ liệu cần nguồn gốc và raw documents tương ứng, không giả định chỉ clone là có đủ file.

Run v2 gốc dùng aliases chưa duyệt; điểm canonical chấm lại nằm tại `results/canonical_rescore_v2` (RAG EM 34,17%, F1 39,76%). Giữ cả run gốc và bản chấm lại để kiểm toán. Bản v1 là đầu ra trung gian đã bị bỏ qua trong Git. Cần người đánh giá duyệt alias và phiếu faithfulness để hoàn tất chất lượng nhãn. Các đường dẫn và SHA256 trong manifest kết quả mô tả lần chạy lịch sử; mã đang chỉnh sửa có thể khác hash đã ghi.
