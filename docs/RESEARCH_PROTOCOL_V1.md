# Protocol research_v1

Phiên bản nghiên cứu mới được tách khỏi demo_v3 và các run v2. Xem [tiến độ có bằng chứng](RESEARCH_UPGRADE_STATUS.md) trước khi trích số liệu. Thiết kế này áp dụng ý tưởng retrieval-augmented generation của Lewis et al.; MiniLM được giữ cố định, T5 được fine-tune theo context nối chuỗi. Bài gốc fine-tune DPR query encoder cùng BART, giữ document encoder/index cố định và marginalize tài liệu ẩn theo RAG-Sequence/RAG-Token ([Lewis et al., §2](https://papers.nips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)). Đồ án này không tái lập cơ chế huấn luyện đó.

## Dữ liệu và phạm vi kết luận

TriviaQA `rc.wikipedia`, revision `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f`. Chọn 2.000 QA đủ trường từ official train; 500 QA từ official validation, loại question IDs/câu hỏi cũ và các câu trùng train. Thứ tự hash với seed 42 chia 200 dev/300 test. Đây là split nội bộ của subset có kiểm soát, không phải official hidden test. Cấu hình và nguồn được đóng băng trong manifest.

Corpus gồm **toàn bộ entity pages được cung cấp** cho QA đã chọn; không chọn passage corpus theo đáp án test. Các tài liệu và chunks trùng được gộp, giữ ánh xạ nhiều nguồn. Corpus chung chứa 3.157 documents/90.667 chunks. Có 80 documents chung train/dev, 85 chung train/test và 44 chung dev/test (không cộng ba số như số tài liệu duy nhất). Bài toán là QA chưa gặp trong một corpus đã biết. Kiến thức pretraining của T5/MiniLM không kiểm soát được; không kết luận không có contamination từ pretraining.

Chunk 200 từ, overlap 40 từ. MiniLM có giới hạn thực tế 256 token; một phần cuối chunk dài có thể không tham gia embedding. Passage cho T5 có ngân sách khác (512 token cả question/context). Hai giới hạn được công bố riêng, không coi toàn bộ chunk là text mà mọi model đã thấy.

Positive train được chọn theo canonical answer với ranh giới từ sau chuẩn hóa. 1.866 QA có positive, 134 bị từ chối và ghi lý do; kiểm tra token/target budget giữ 1.865 QA cho **cả hai T5**. Không dùng fallback chunk đầu. Đây là kiểm tra lexical, chưa phải xác nhận ngữ nghĩa thủ công cho mọi positive. Dev/test giữ nguyên cả câu khó hoặc thiếu chuỗi đáp án.

## Hệ so sánh và các biến được giữ cố định

| Nhánh | Huấn luyện/suy luận | Vai trò |
| --- | --- | --- |
| generator_only | T5-base fine-tune riêng, context rỗng | Baseline closed-book |
| retrieval_only | Dense + DistilBERT SQuAD, chọn span trong tối đa 3 passage | Baseline extractive; không fine-tune cùng QA |
| rag_full | T5-base fine-tune positive context; Dense lúc suy luận | RAG chính |
| rag_hybrid | Cùng checkpoint RAG, thay thứ hạng bằng Dense + BM25 RRF | Cải tiến có kiểm soát |
| rag_checkpoint_no_context | Checkpoint RAG, context rỗng | Ablation context; không gọi là closed-book baseline |

Hai T5 cùng initialization revision, QA IDs, seed, 3 epoch, effective batch 16, AdamW learning rate 0,0003, weight decay 0,01; checkpoint được chọn theo dev loss. RAG train dùng một positive passage; validation/inference dùng passage truy hồi (tối đa 2), là một khác biệt phân phối cần nêu. BF16 nếu GPU hỗ trợ; nếu không thì FP32. Không dùng FP16 vì pilot đã phát hiện overflow. Một seed chưa đo biến thiên huấn luyện.

Extractive xét tối đa 3 passage Dense, input 512 token/cặp question–context, span tối đa 20 token; chỉ giữ span có tổng start/end logit >2,0. Đây là ngưỡng logit cố định, không phải xác suất đã calibration. Question, padding và special tokens bị loại khỏi vị trí đáp án.

Prompt chung `question: {question} context: {context}`. Input 512 token, output 32 token, beam 4, length penalty 0,8, no-repeat-ngram 3. Trong suy luận dev/test, gold answer chỉ dùng để tính điểm và phân tích sau suy luận; không đưa vào prompt inference. Gold answer của train được dùng làm target và để chọn positive context như mô tả ở trên. Không có cosine gate loại câu khi đánh giá.

MiniLM encode query/passage cùng encoder, vector unit length, FAISS IndexFlatIP. Dense/BM25 lấy pool 30, Hybrid cộng `1/(60+rank)` theo chunk ID. Top-k đánh giá retrieval là 1/3/5/10; generator chỉ nhận tối đa 2 passage và có thể bị truncation. BM25 dùng cùng tokenizer từ cho query/corpus. Không có heuristic ưu tiên tên riêng của demo lịch sử trong protocol này.

## Cách chấm

EM/F1 dùng normalization TriviaQA: lower, `_` thành space, dấu câu thành space, bỏ articles `a/an/the`, gộp khoảng trắng. `official` chỉ dùng `normalized_aliases` từ nguồn; `canonical` dùng một `answer.value`. Hai policy được báo song song; không âm thầm thêm canonical vào official. Canonical-only có thể bỏ sót biến thể đúng; aliases nguồn có thể có lỗi ngữ nghĩa. Không tự sửa nhãn dựa vào model output.

Rà định tính dev cũng phát hiện canonical có thể sai quan hệ (ví dụ nơi sinh Martina Hingis); ledger đề xuất giữ riêng, chưa coi là nhãn đã được con người duyệt. Xem [casebook](CASES_RESEARCH_V1.md). Không sửa corpus/nhãn frozen hoặc thay điểm official sau khi quan sát lỗi; curated evaluation cần version/ledger riêng.

Retrieval báo **doc hit rate@k** (ít nhất một nguồn), **doc recall@k** (tỷ lệ nguồn gold được phủ), **answer recall@k** (chuỗi reference xuất hiện) và MRR@30. Qrels dựa vào supplied documents; không phải mọi passage trong document đều trả lời câu hỏi. Answer recall là proxy lexical.

Phân rã lỗi RAG sau EM: chuỗi thiếu trong corpus → thiếu trong top-k → có ở top-k nhưng không ở context thực dùng → sai dù chuỗi có trong context. Nhãn này không xác lập nhân quả/ngữ nghĩa. Đọc context sau truncation và nhờ người chấm kiểm tra mẫu.

Hybrid–Dense có paired bootstrap 2.000 resamples theo question, seed 42, CI 95% cho chênh EM/F1. CI này chỉ thể hiện biến thiên mẫu câu hỏi; không thay thế nhiều training seed hoặc kiểm định nhiều dataset. Các ablation trên dev không được trình bày như test độc lập.

Chọn ngẫu nhiên 50 QA, chấm cả Dense/Hybrid với thứ tự nhánh được che: 100 dòng annotation, **không phải 100 QA độc lập**. Faithfulness chấm đáp án có được context hỗ trợ cho đúng question không; citation correctness chấm từng nguồn ứng dụng gắn vào đáp án. T5 không tự sinh inline citations. Dùng [rubric](REVIEW_RUBRIC.md); agent/self-review tách khỏi người chấm độc lập.

## Khóa và tái lập

Chạy dev trước. Test yêu cầu `--split test --final`, đủ 300 câu, model non-pilot, đúng data/model hashes. Manifest và source snapshot được ghi trước khi đọc dự đoán test. Nếu sửa phương pháp sau khi xem test, gọi kết quả tiếp theo là exploratory; cần holdout mới để khẳng định độc lập.

Notebook [research_reproduce_colab.ipynb](../notebooks/research_reproduce_colab.ipynb) có hai chế độ: dựng/train từ revision pin hoặc evaluate frozen artifacts. Replay so sánh logical record hashes để bỏ khác biệt LF/CRLF, vẫn giữ SHA từng file gốc. Không đòi bitwise giống nhau giữa GPU/precision khác. Chỉ ghi Colab PASS khi có log thực chạy, cấu hình và metrics tương ứng.

Nguồn: [bài báo Lewis et al.](https://arxiv.org/abs/2005.11401), [TriviaQA](https://nlp.cs.washington.edu/triviaqa/), [evaluator gốc](https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py).
