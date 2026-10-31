# Kết quả kiểm tra yêu cầu đồ án — verified_v2

> **Báo cáo lịch sử dùng official aliases.** Đã có [bản chấm lại canonical-only](IMPROVEMENTS_V3.md): RAG EM 34,17%, F1 39,76% trên cùng 120 dự đoán, không chạy lại model. Các điểm alias dưới đây giữ để kiểm toán, không dùng làm kết quả cuối.

> **Cần rà soát lại:** Alias của câu Antigonus có các mục như “Mopsa” và “Queen Hermione”, không phải tên vở kịch. Vì evaluation nhận mọi alias, các điểm EM/F1 và số liệu kiểm tra bằng chứng bên dưới có thể bị sai lệch; không dùng làm kết quả cuối cùng cho đến khi rà alias và đánh giá lại. Hybrid và corpus demo_v3 trên web chưa được đánh giá trên tập độc lập.

Ngày 31/10/2026. Đã sửa pipeline, phục hồi ground truth/source có phiên bản và chạy đủ 120 câu trên CUDA RTX 3050 6GB. Không huấn luyện lại checkpoint và không dùng test để chọn tham số tối ưu.

## Dữ liệu

- Giữ đủ 120 câu test và 680 training pairs gốc; không loại mẫu khó.
- Khôi phục official question ID, answer aliases và evidence pages từ TriviaQA revision cố định.
- Corpus v2: **800 tài liệu, 12.556 chunks**, chunk_size=200, overlap=40. Cả 120 nguồn gán có chuỗi đáp án/alias sau kiểm tra ranh giới từ. Điều này chưa chứng minh mọi nguồn đều đủ hỗ trợ quan hệ mà câu hỏi yêu cầu.
- Tài liệu bị cắt được mở rộng; một số câu đổi sang trang bằng chứng khác do TriviaQA cung cấp. Việc chọn trang có sử dụng alias: đây là corpus có kiểm soát, không phải benchmark toàn Wikipedia.
- Dữ liệu gốc không bị thay thế; kiểm tra checksum raw, test, train và nguồn phục hồi đều đạt.

## Ba hệ thống — đo lại trên 120 câu

| Hệ | EM | F1 |
| --- | ---: | ---: |
| T5 không context | 6.67% | 11.00% |
| FAISS + DistilBERT extractive | 35.00% | 44.51% |
| FAISS + T5 RAG | 42.50% | 47.40% |

Generator-only là ablation bỏ context của cùng checkpoint đã fine-tune, chưa phải mô hình closed-book được train riêng. Hai nhánh T5 dùng beam=4 và cùng cấu hình sinh. Extractor chỉ lấy span thuộc context. EM/F1 xét official aliases.

## Retrieval — đo trực tiếp

| Phương pháp | Doc Recall@3 | Doc Recall@5 | Doc Recall@10 | Doc MRR@30 |
| --- | ---: | ---: | ---: | ---: |
| BM25 Okapi | 73.33% | 77.50% | 86.67% | 0.6705 |
| Dense MiniLM + FAISS | 90.83% | 92.50% | 97.50% | 0.8737 |
| Hybrid RRF | 87.50% | 92.50% | 97.50% | 0.8327 |

Hybrid bằng Dense ở Recall@5 và Recall@10, thấp hơn về MRR ở lần chạy này. Không kết luận hybrid luôn tốt hơn. Alias evidence Recall@k được lưu riêng trong metrics.json; doc recall không đảm bảo đoạn được lấy có chứa bằng chứng đáp án. Không so trực tiếp với bảng corpus 3.940 chunks cũ.

## Phân rã lỗi RAG

| Nhóm | Số câu |
| --- | ---: |
| Đúng đáp án theo EM/aliases | 51 |
| Top-5 thiếu alias bằng chứng | 18 |
| Có alias trong top-5 nhưng không còn trong context T5 | 14 |
| Sai dù context có alias | 37 |

Các nhóm dựa trên kiểm tra alias; không khẳng định chỉ có một nguyên nhân nhân quả. Context thật sau giới hạn 512 token được lưu từng câu. Bằng chứng ở rank 3 không còn bị tự coi là lỗi generation nếu T5 chỉ dùng top-2.

## Faithfulness/citation trên mẫu

Chọn ngẫu nhiên 20/120 câu bằng seed 42, độc lập điểm. Đã đọc context thực tế và ghi nhãn, lý do, source IDs trong `review_sample_agent.csv`:

- 11/20 được context hỗ trợ.
- 7/20 không được hỗ trợ hoặc mâu thuẫn.
- 2/20 chưa đủ rõ để kết luận.

Đây là **agent_self_review**, không phải đánh giá độc lập của con người và không dùng để khẳng định chất lượng toàn tập. Citation ở đây xét nguồn được cung cấp kèm câu trả lời, không giả định T5 sinh inline citations. Ví dụ trả lời White cho sọc dưới cờ Mỹ là sai dù White có trong context; nguồn nói rõ red ở trên và dưới. Ví dụ Windy Miller đúng nhãn nhưng context đưa vào chưa nhắc Colly's Mill nên nhãn faithfulness là unclear. Phiếu trống vẫn giữ để giảng viên/người đánh giá có thể chấm độc lập.

Độ trùng token tự động của RAG là 87,5% trên 120 câu; chỉ mang tên token_support_fraction, không diễn giải thành 87,5% không ảo giác.

## Bằng chứng tái lập và phần mềm

- 35/35 unit tests đạt, gồm alias, đơn vị 720°, phân rã rank-3/context, nguồn không tìm thấy, tính retrieval và các regression web.
- 120 predictions được đọc lại để tính EM/F1 và retrieval; khớp metrics đã lưu.
- Context thực tế không vượt 512 token; khớp phần text của các passage đã dùng.
- Tại thời điểm kiểm tra run, SHA256 đầu vào, index, checkpoint và mã đánh giá khớp manifest; mã có thể thay đổi sau đó, nên so hash trước khi tái lập.
- Script classical baseline không còn chèn Dense/Hybrid cố định; số đo mới nằm trong run evaluation_v2_full.
- Tại lần chạy v2, web chọn verified_v2 khi artifact đầy đủ. Bản web hiện tại ưu tiên demo_v3, rồi verified_v2 và corpus gốc. Lệnh preflight xác nhận 12.556 chunks, 120 test, endpoint/static hoạt động.

## Artifact và lệnh

- [Metrics](../results/evaluation_v2_full/metrics.json)
- [Manifest run](../results/evaluation_v2_full/manifest.json)
- [Validation](../results/evaluation_v2_full/validation.json)
- [Predictions từng câu](../results/evaluation_v2_full/predictions.jsonl)
- [Phiếu mẫu đã tự rà soát](../results/evaluation_v2_full/review_sample_agent.csv)
- [Phiếu mẫu trống](../results/evaluation_v2_full/review_sample.csv)
- [Manifest dữ liệu](../data/verified_v2/manifest.json)
- [Quy trình và cách tái lập](EVALUATION_PROTOCOL.md)

```powershell
python -m unittest discover -s tests -v
python -m scripts.check_demo
python -m experiments.evaluate --run-dir results/evaluation_new_run
```

Chạy lại evaluation tạo thư mục run mới, không ghi đè các run đã chốt. Môi trường Colab chưa được thực thi trong lần kiểm tra này; có hướng dẫn chuẩn bị model/cache và chạy cùng script. RAG vẫn có 69/120 câu chưa đạt EM: đây là giới hạn chất lượng cần trình bày, không che bằng số liệu lịch sử.
