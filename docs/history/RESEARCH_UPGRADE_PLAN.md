# Kế hoạch hoàn thiện trong 3 ngày

Mốc lập kế hoạch: 31/10/2026; báo cáo dự kiến sau 3 ngày. Mục tiêu là đáp ứng đầy đủ đề bài và có bằng chứng nghiên cứu kiểm tra được. Điểm số và khả năng công bố do hội đồng/người phản biện quyết định; không đồng nhất hoàn thiện phương pháp với 100% câu trả lời đúng.

## Thiết kế được chọn

- Tiếp tục **TriviaQA rc.wikipedia**, revision `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f`.
- Tạo `research_v1` riêng; giữ nguyên corpus, checkpoint và run lịch sử. Chọn 2.000 QA ứng viên từ official train, 200 dev và 300 test từ official validation. Loại các question IDs/câu hỏi đã được dùng trong bộ 800 cũ khỏi dev/test mới. Đây là split nghiên cứu nội bộ, không gọi là official test.
- Chọn QA bằng quy tắc cố định, không dựa vào việc mô hình trả lời đúng. Dùng toàn bộ evidence pages được dataset cung cấp cho các QA đã chọn, không chọn trang theo đáp án test. Deduplicate nguồn/chunk và giữ ánh xạ nhiều nguồn cho passage trùng.
- Kiểm tra trùng question ID và câu hỏi chuẩn hóa giữa các split; báo riêng page overlap. Mục tiêu chính là QA chưa gặp trong cùng corpus; không tự tuyên bố khái quát sang tài liệu chưa gặp.
- Chỉ dùng positive training context có canonical answer xuất hiện theo kiểm tra ranh giới từ chuẩn hóa; ghi rõ những dòng thiếu evidence. Train và RAG dùng cùng tập QA được chấp nhận. Dev/test giữ toàn bộ mẫu đã chọn, kể cả trường hợp corpus không đủ bằng chứng.
- **Ba hệ chính:** T5 closed-book được fine-tune riêng; Dense + DistilBERT extractive; Dense + T5 RAG. Thêm ablation bỏ context của checkpoint RAG, có tên riêng.
- Hai T5 khởi tạo từ cùng base model; cùng template `question: ... context: ...`, giới hạn input 512, output 32, cùng cấu hình train/sinh. Closed-book có context rỗng. Không dùng cosine gate để loại mẫu trong evaluation.
- Dense index: MiniLM cố định + normalized vectors + FAISS IndexFlatIP. Pipeline ứng dụng ý tưởng RAG; không gọi là tái lập RAG-Sequence/RAG-Token end-to-end.
- Cải tiến chính: **Hybrid RRF Dense + BM25**. Ablation retrieval k và số passage thực đưa vào generator được phân biệt. Heuristic từ hiếm trên demo cũ không tự trở thành phương pháp chính.
- Đo cả điểm theo official normalized aliases và canonical-only; chỉ gọi curated khi có ledger duyệt nhãn. Không sửa lịch sử điểm số hoặc trả nhãn thay đầu ra mô hình.

## Các mốc và điều kiện nghiệm thu

| Mốc | Việc làm | Điều kiện nghiệm thu | Bằng chứng |
| --- | --- | --- | --- |
| 1. Evaluator | Chuẩn hóa theo TriviaQA, xử lý khoảng trắng/dấu câu/aliases; cô lập cách chấm legacy | Khớp fixture evaluator gốc, có regression Eugene O'Neill/Hard Day's Night; legacy corpus không thay | Unit tests và evaluator version |
| 2. Dữ liệu | Tải nguồn pin revision, đóng băng split, loại trùng, tìm positive training evidence | 0 question overlap; 0 positive train âm thầm fallback; test không bị lọc theo đáp án | Manifest, checksums, rejection audit, source IDs |
| 3. Pipeline | Cấu hình dùng chung cho chunk/index/retrieve/generate | Chunk IDs, source mapping, vector count/dimension/norm đúng; không lẫn corpus/model version | Preflight và pilot |
| 4. Train | Fine-tune closed-book và RAG; ghi log, seed, validation và checkpoint tốt nhất | Loss hữu hạn, model cập nhật, checkpoint có thể nạp; cùng QA và training budget đã công bố | Train logs/config/checkpoint hashes |
| 5. Evaluation | Chạy đủ ba hệ, lưu từng dự đoán và context thực dùng | Đủ số test, không bỏ mẫu khó; Recall@k/EM/F1 tính lại từ predictions được | Run manifest, predictions, metrics |
| 6. Cải tiến | Đo Hybrid trên cùng corpus/model/template; ablation dùng dev để chọn | Có Dense–Hybrid đối chiếu retrieval và answer, báo cả kết quả âm | Bảng ablation và config khóa trước test |
| 7. Lỗi | Phân biệt corpus/retrieval/chọn context/generation | Giữ tên proxy lexical; kiểm tra ngữ nghĩa trên mẫu | Error table và ví dụ truy vết |
| 8. Faithfulness/citation | Phiếu ngẫu nhiên 50 mẫu, hướng dẫn và source IDs | Nhãn người đánh giá và lý do có nguồn; agent review ghi riêng | Phiếu/mapping/annotation summary |
| 9. Colab | Notebook/lệnh chạy từ môi trường sạch, model/data có version | Có log chạy thực tế trên Colab và metrics đối chiếu; chưa chạy thì ghi NOT RUN | Notebook, environment, manifest |
| 10. Báo cáo/demo | Đồng bộ mô hình chạy và số liệu, hướng dẫn học lại, kịch bản trình bày | Giải thích được ba hệ, metrics và ít nhất một lỗi mỗi lớp; demo không gắn nhãn giả | Báo cáo, tài liệu vận hành, case table |

“Đạt 10 mục” là checklist nội bộ, không phải điểm học phần tự chấm. Mục 8 cần người đánh giá thực sự; mục 9 cần phiên Colab thực sự. Không điền PASS thay cho bằng chứng còn thiếu.

## Lịch 3 ngày

**Ngày 1:** sửa evaluator, cô lập legacy, dựng/freeze research_v1, kiểm tra training evidence, chạy pilot và giải thích lại sơ đồ hệ thống. Chỉ đi tiếp khi kiểm tra dữ liệu và cách chấm đạt.

**Ngày 2:** train hai T5, kiểm tra checkpoint, đánh giá ba hệ và Hybrid. Chuẩn bị notebook, chạy Colab khi có runtime phù hợp. Dùng dev để chọn cấu hình; khóa config trước test. Gửi phiếu đánh giá mẫu cho người chấm bằng thao tác của người dùng.

**Ngày 3:** tổng hợp annotation, bảng kết quả, phân rã lỗi, độ bất định và giới hạn; đồng bộ demo. Người dùng học tài liệu vận hành, trình bày thử và trả lời câu hỏi phản biện. Chỉ sửa lỗi ảnh hưởng tính đúng, không thêm tính năng giao diện ngoài mục tiêu.

## Giới hạn tài nguyên và điều kiện dừng

Ưu tiên GPU local RTX 3050 6GB, không dùng dịch vụ trả phí. Pilot phải đo thời gian/bộ nhớ trước full training. Giới hạn ban đầu: tối đa 3 epoch/model, không tìm kiếm hyperparameter rộng; validation chọn best checkpoint. Nếu thiếu VRAM, giảm micro-batch và giữ effective batch bằng gradient accumulation; ghi thay đổi. Không giảm test hoặc đổi ground truth để làm điểm cao hơn.

Nếu full training vượt tài nguyên hiện có, giữ run hoàn chỉnh nhỏ hơn dưới nhãn pilot và chuyển cấu hình đã kiểm tra sang Colab; không gọi pilot là kết quả cuối. Mọi giai đoạn lưu trạng thái PASS/FAIL/NOT RUN, thông số và artifact thực tế. Việc mở rộng nhiều seed, dataset lớn hoặc đóng góp mới cho bài báo được thực hiện sau khi bản báo cáo ổn định.

## Tài liệu học và vận hành cần bàn giao

- Bản đồ thư mục và luồng dữ liệu từ raw QA tới câu trả lời.
- Giải thích chunking, embedding, FAISS, BM25, T5, closed-book và extractive bằng một ví dụ xuyên suốt.
- Lệnh dựng dữ liệu, train, evaluate, chạy web; cách nhận biết model/corpus nào đang chạy.
- Cách đọc Recall@k, EM/F1, faithfulness/citation; phân biệt đo lexical và kiểm tra ngữ nghĩa.
- Kịch bản báo cáo 10–15 phút và câu hỏi giảng viên thường hỏi, gắn với kết quả thực tế.

Theo dõi tiến độ trong `docs/RESEARCH_UPGRADE_STATUS.md`; kế hoạch này không tự xác nhận bất kỳ thí nghiệm chưa chạy nào.
