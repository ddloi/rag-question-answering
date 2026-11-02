# Tập trình bày và trả lời phản biện

Đọc cùng [hướng dẫn vận hành](HOC_VA_VAN_HANH.md). Tập giải thích bằng lời trước khi mở code; chỉ trích số liệu từ run complete và REPORT.md tương ứng.

## Bài tập 30 phút

1. Trong 5 phút, tự vẽ hai luồng: train cập nhật T5; inference tìm passage rồi sinh. Đánh dấu vị trí ground truth: train/evaluation, không đi vào inference prompt.
2. Trong 5 phút, đọc `configs/research_v1.json`, manifest dữ liệu và hai model manifests. Nói được dataset revision, 2.000 train candidates, 1.865 QA dùng train, 200 dev/300 test, seed, số epoch và checkpoint nào đang dùng.
3. Trong 10 phút, mở `/research`, nạp ba hệ, chọn một câu dev đúng và một câu sai. Đọc context thực dùng, so với full passage và nhãn; phân biệt truy hồi thiếu với sinh sai. Đổi Dense/Hybrid và chạy lại; đừng giải thích theo điểm cosine/RRF như xác suất đúng.
4. Trong 5 phút, đọc bảng EM/F1, doc hit/recall và CI Hybrid–Dense trong báo cáo. Nói được đâu là phần trăm, đâu là chênh điểm phần trăm, đâu là proxy lexical.
5. Trong 5 phút, đóng tài liệu và trả lời các câu dưới. Nếu vướng, quay lại module phụ trách đúng bước đó.

## Các câu giảng viên có thể hỏi

**1. RAG là gì trong đồ án này?** T5 là bộ sinh parametric, lưu kiến thức trong trọng số; corpus với vector FAISS là bộ nhớ ngoài. MiniLM mã hóa query/passage, FAISS tìm passage, T5 nhận question + context để sinh đáp án. MiniLM cũng là mô hình có tham số, nhưng trọng số được giữ cố định trong thí nghiệm này.

**2. Có tái lập đúng Lewis et al. 2020 không?** Áp dụng ý tưởng seq2seq + dense external memory. Đồ án dùng MiniLM/T5 với context nối chuỗi; chưa triển khai latent-document marginalization và joint query-encoder training như RAG gốc. Không gọi đây là bản tái lập đầy đủ RAG-Sequence/RAG-Token.

**3. Tại sao cần cả ba hệ?** Closed-book cho thấy khả năng trả lời từ trọng số; extractive cho thấy khả năng tìm và chọn span mà không sinh mới; RAG cho thấy retrieval kết hợp generation. Nếu chỉ so RAG với BM25 trích câu trên giao diện lịch sử thì chưa tương đương phép so ba hệ nghiên cứu.

**4. Generator-only có phải chỉ tắt context của RAG không?** Baseline chính là T5 được fine-tune riêng bằng context rỗng. Bỏ context của checkpoint RAG là một ablation khác, được giữ ở dòng riêng.

**5. Tại sao từ 2.000 train chỉ dùng 1.865?** 134 QA không tìm thấy canonical answer trong supplied evidence theo phép kiểm tra lexical; một QA nữa không đạt ngân sách token/target. Cả hai T5 dùng cùng 1.865 QA. Các rejection có log. Dev/test không bị lọc theo mức dễ/khó hay output.

**6. Có rò rỉ test không?** Question IDs và câu hỏi chuẩn hóa không trùng giữa split; loại cả các câu đã xem ở bộ cũ khỏi dev/test mới. Corpus chung gồm supplied pages, có thể trùng tài liệu giữa split; đây là QA mới trên corpus biết trước. Chưa kiểm soát contamination của model pretraining. Dữ liệu demo_v3 bổ sung sau khi xem lỗi không dùng chấm research_v1.

**7. Vì sao Recall cao nhưng EM thấp?** Một document liên quan không bảo đảm passage chứa evidence. Evidence có thể nằm ngoài top-2, bị cắt bởi 512 token hoặc model sinh sai. Cũng có thể answer đúng nghĩa nhưng reference lexical thiếu biến thể. Cần đọc case và context để kết luận.

**8. Top-k=5 có nghĩa T5 thấy năm passage không?** Không. Lấy top-5 để đánh giá retrieval; cấu hình T5 chỉ nối tối đa hai passage rồi truncation tổng 512 token. Embedding có giới hạn riêng 256 token. Giao diện và predictions giữ phần context thực dùng.

**9. Hybrid có luôn tốt hơn Dense không?** Không. RRF cộng điểm từ thứ hạng, có thể đưa passage từ khóa sai lên cao. Bảng paired comparison, CI và case `hybrid_regression` cho thấy cả cải thiện lẫn giảm chất lượng. Không cộng trực tiếp BM25 với cosine.

**10. EM/F1 có đo sự thật không?** Chỉ so output với references sau normalization. EM đòi khớp toàn chuỗi; F1 đo trùng token. Ví dụ tên viết khác hoặc lỗi nhãn nguồn có thể ảnh hưởng điểm. Báo đồng thời official aliases và canonical, giữ nguyên đầu ra thật, không đổi nhãn để tăng điểm.

**11. Faithfulness và accuracy có giống nhau không?** Accuracy so với nhãn trả lời; faithfulness xem evidence có hỗ trợ câu trả lời cho đúng question không. Đáp án có thể đúng nhưng không được context hỗ trợ, hoặc context sai khiến đáp án được lặp lại mà vẫn không đúng thực tế.

**12. Citation correctness đang chấm loại citation nào?** Các passage ứng dụng gắn kèm output, đã thực sự tham gia context. Đây không phải inline citations do T5 tự sinh. Người chấm gán nhãn từng source và lý do; nguồn trùng từ với answer chưa đủ chứng minh nó trả lời question.

**13. Vì sao không tự dùng lexical overlap làm faithfulness?** Chuỗi có thể xuất hiện trong quan hệ khác, phủ định hoặc một danh sách. Overlap chỉ là proxy hỗ trợ phân nhóm để đọc; mẫu ngẫu nhiên cần người kiểm tra ngữ nghĩa. Agent self-review phải ghi riêng, không đổi thành human-independent.

**14. Có thể tái lập không?** Có mã/seed/revisions/config/data hashes/model hashes/logs/source snapshots và notebook. Phải phân biệt kiểm tra replay local với chạy Colab thật. Dùng đúng artifacts có thể đối chiếu predictions; train lại trên GPU/dtype khác có thể không bitwise giống. Chưa chạy Colab thì ghi NOT RUN.

**15. Đã đủ để viết bài báo chưa?** Có thể phát triển báo cáo phương pháp và thực nghiệm có kiểm toán. Khả năng công bố còn cần câu hỏi nghiên cứu/đóng góp rõ, nhiều training seed, dữ liệu rộng hơn, đánh giá người chấm và kết quả tái lập. Hybrid hoặc một UI đẹp tự nó chưa chứng minh đóng góp mới.

## Khi một câu demo sai

Giữ output và mở nguồn. Nói: “Theo canonical, output này chưa đúng. Tôi kiểm tra corpus, thứ hạng, context sau cắt token và bước generation để phân loại.” Nếu EM/F1 và ý nghĩa khác nhau, chỉ ra references và nêu giới hạn phép chấm. Không thay đầu ra model bằng ground truth; dùng case sai để giải thích điểm yếu của hệ thống.
