# Hướng dẫn tự chấm faithfulness và citation

Phiếu cần chấm: [results/research_v1_test/human_review_author.csv](../results/research_v1_test/human_review_author.csv). Phiếu có 100 dòng, ứng với 50 câu hỏi test, mỗi câu có đáp án của Dense RAG và Hybrid RAG. Thứ tự hai hệ đã được xáo trộn và không ghi tên. Quy tắc gốc nằm trong [REVIEW_RUBRIC.md](REVIEW_RUBRIC.md); tài liệu này hướng dẫn từng bước thực hiện.

Thời gian dự kiến: khoảng 1–2 phút mỗi dòng, tổng 2–3 giờ. Có thể chia làm nhiều buổi vì nhãn được lưu tự động.

## Ba điều không được làm

1. **Không mở** `review_key.json`, `predictions.jsonl.gz`, `metrics.json` hay bảng EM/F1 trước khi chấm xong. Biết dòng nào thuộc hệ nào, hoặc đáp án chuẩn là gì, sẽ làm lệch kết quả.
2. **Không sửa** các cột `question`, `prediction`, `context_used`, `source_chunk_ids`. Script tổng hợp kiểm tra mã băm của bốn cột này và sẽ từ chối phiếu bị sửa.
3. **Không mở rồi lưu lại phiếu bằng Excel.** Excel có thể đổi dấu xuống dòng hoặc mã hóa ký tự và làm hỏng mã băm. Hãy dùng công cụ chấm bên dưới.

## Bước 1: Mở công cụ chấm

Nhấp đúp vào file `scripts/review_tool.html` để mở bằng Chrome hoặc Edge. Trang chạy hoàn toàn trên máy, không gửi dữ liệu đi đâu.

Bấm **Chọn tệp** rồi chọn `results/research_v1_test/human_review_author.csv`. Trang hiện từng dòng gồm câu hỏi, đáp án của hệ và các nguồn mà T5 thực sự đã đọc (sau khi cắt theo giới hạn token).

Nhãn được lưu trong trình duyệt sau mỗi lần bấm. Đóng trang rồi mở lại, chọn lại đúng file đó, nhãn cũ sẽ hiện lại. Không xóa dữ liệu duyệt web trước khi xuất nhãn.

## Bước 2: Chấm từng dòng

Mỗi dòng có ba phần.

### a. Citation: chấm từng nguồn

Với mỗi nguồn (mỗi chunk ID), chỉ hỏi: *đoạn này có chứng minh được đáp án của hệ cho câu hỏi này không?*

| Nút | Khi nào chọn |
| --- | --- |
| Hỗ trợ đáp án | Đoạn văn nói rõ quan hệ mà câu hỏi hỏi, và đáp án đúng là thực thể đó |
| Không hỗ trợ | Đoạn văn không nhắc tới, nói về chuyện khác, hoặc mâu thuẫn với đáp án |
| Không rõ | Đoạn văn liên quan nhưng mơ hồ, hoặc bị cắt đúng chỗ cần đọc |

Chấm từng nguồn riêng. Một dòng có thể có nguồn 1 "Hỗ trợ" và nguồn 2 "Không hỗ trợ"; đây là chuyện bình thường.

### b. Faithfulness: chấm cả câu trả lời

Hỏi: *gộp tất cả các nguồn lại, context có hỗ trợ đáp án cho đúng câu hỏi không?*

| Nút | Khi nào chọn |
| --- | --- |
| Hỗ trợ | Ít nhất một nguồn chứng minh đáp án cho đúng quan hệ câu hỏi hỏi |
| Không hỗ trợ | Không nguồn nào chứng minh, hoặc context nói điều khác |
| Không rõ | Không đủ thông tin để quyết định |

Phần tô vàng chỉ đánh dấu chỗ trùng chữ với đáp án. **Có chữ trùng chưa đủ để chọn "Hỗ trợ".** Phải đọc xem đoạn đó có trả lời đúng điều câu hỏi hỏi hay không.

Chấm theo context, không theo kiến thức của bạn. Nếu bạn biết đáp án đúng nhưng context không nói, chọn "Không hỗ trợ". Nếu đáp án sai sự thật nhưng context nói đúng như vậy, vẫn chọn "Hỗ trợ".

### c. Lý do

Viết 1–2 câu, bắt buộc. Ví dụ:

- *Nguồn 1 nói rõ "narrated by Denise van Outen".*
- *Amy Childs chỉ được nhắc là diễn viên trong pilot, không phải người dẫn chuyện.*
- *Nguồn 2 bị cắt trước đoạn nói về năm thành lập.*

## Ví dụ minh họa (giả định, không có trong phiếu)

Câu hỏi: *Who composed the opera Carmen?* Nguồn 1: "Carmen is an opera in four acts by Georges Bizet. The libretto was written by Henri Meilhac and Ludovic Halévy." Nguồn 2: một đoạn về các buổi diễn Carmen tại Paris.

| Đáp án của hệ | Nguồn 1 | Nguồn 2 | Faithfulness | Lý do |
| --- | --- | --- | --- | --- |
| Georges Bizet | Hỗ trợ đáp án | Không hỗ trợ | Hỗ trợ | Nguồn 1 nói rõ opera "by Georges Bizet". |
| Henri Meilhac | Không hỗ trợ | Không hỗ trợ | Không hỗ trợ | Meilhac có trong nguồn 1 nhưng là người viết lời, không phải người soạn nhạc. |

Dòng thứ hai cho thấy vì sao không được chọn "Hỗ trợ" chỉ vì có chữ trùng.

## Bước 3: Xuất và ghép nhãn

1. Khi thanh tiến độ báo **100/100 xong** (nút **Dòng chưa xong** giúp tìm dòng còn thiếu), bấm **Xuất nhãn**. Trình duyệt tải về file `review_labels.csv`.
2. Chuyển file đó vào thư mục `RAG`, rồi chạy trong PowerShell tại thư mục `RAG`:

```powershell
python -m scripts.merge_review_labels --sheet results/research_v1_test/human_review_author.csv --labels review_labels.csv --key results/research_v1_test/review_key.json
```

Script kiểm tra nhãn hợp lệ và các cột bằng chứng không bị sửa, rồi mới ghi nhãn vào phiếu. Kết quả in ra phải là `"status": "COMPLETE"`. Nếu còn thiếu, script báo `PARTIAL` cùng số dòng còn lại; khi đó có thể xuất lại và ghép lại bao nhiêu lần cũng được.

## Bước 4: Tổng hợp kết quả

```powershell
python -m scripts.summarize_research_review --reviews results/research_v1_test/human_review_author.csv --key results/research_v1_test/review_key.json --output results/research_v1_test/human_author_final.json
```

File `human_author_final.json` chứa, cho từng hệ:

- `faithfulness_supported_fraction_all`: số dòng "Hỗ trợ" chia cho cả 50 dòng, kể cả "Không rõ".
- `faithfulness_supported_fraction_assessed`: chỉ tính trên các dòng "Hỗ trợ" và "Không hỗ trợ".
- `citation_precision_assessed`: số nguồn "Hỗ trợ đáp án" chia cho các nguồn đã có kết luận.

Khi đưa vào báo cáo, ghi rõ mẫu số và số dòng "Không rõ". Ghi người chấm là **tác giả tự chấm (human_author)**, không phải người chấm độc lập. Nếu kết quả khác với bản agent tự chấm ([agent_review_20261102_summary.json](../results/research_v1_test/agent_review_20261102_summary.json)), báo cáo cả hai, không chọn bản có lợi hơn.

Sau khi tổng hợp xong mới được mở `review_key.json` để đối chiếu dòng nào thuộc hệ nào.
