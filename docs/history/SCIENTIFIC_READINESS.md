# Rà soát yêu cầu đồ án và khả năng phát triển bài báo

> Đây là audit lịch sử ngày 31/10 cho v2/v3. Từ 01/11, `research_v1` xử lý các phát hiện trong một phiên bản riêng: normalizer đã sửa, corpus/splits mới đã đóng băng, hai T5 đã train xong và có đủ 200 dev/300 test. Theo [tiến độ hiện tại](../RESEARCH_UPGRADE_STATUS.md) và [protocol mới](../RESEARCH_PROTOCOL_V1.md); các nhận xét “hiện tại” phía dưới mô tả trạng thái lúc audit, không phải tự động cập nhật kết quả nâng cấp.

Ngày 31/10/2026. Phạm vi: mã thực nghiệm, dữ liệu hiện có, checkpoint local và các dự đoán đã lưu; không huấn luyện lại, không sửa nhãn, không thay corpus và không chạy lại 120 lượt suy luận. Đây là **implementer self-review**, không phải đánh giá độc lập hoặc kết luận của hội đồng/tạp chí.

**Kết luận:** đã có đủ khung triển khai theo đề bài. Chưa nên coi phần thực nghiệm là hoàn thiện để chốt kết quả bài báo. Kiểm thử phần mềm và demo chạy được không chứng minh chất lượng nhãn, độ công bằng baseline hoặc khả năng tái lập trên Colab.

## Đối chiếu yêu cầu

| Yêu cầu | Tình trạng | Bằng chứng và giới hạn |
| --- | --- | --- |
| Natural Questions hoặc TriviaQA, subset kiểm soát | Đạt về loại dữ liệu; tái lập còn thiếu bằng chứng | TriviaQA rc.wikipedia; phục hồi nguồn tại revision cố định. 800 QA lấy từ validation gốc rồi chia nội bộ. Không gọi 120 câu này là official test split. Colab chưa chạy thực tế. |
| Seq2seq parametric + non-parametric dense index | Đạt ý tưởng kiến trúc | T5 và MiniLM + FAISS IndexFlatIP. Chưa phải tái lập RAG-Sequence/RAG-Token nguyên bản. |
| Generator không retrieval | Có, nhưng baseline còn yếu về thiết kế | Cùng checkpoint T5 được bỏ context lúc inference, chưa train một closed-book model tương ứng. |
| Retrieval-only/extractive | Có | Dense + DistilBERT, chỉ lấy span context. Có đủ 120 dự đoán. BM25 + trích câu trên web là một nhánh khác. |
| RAG | Có | Dense + T5, tối đa hai passage và 512 token. Có đủ 120 dự đoán và context thực dùng. |
| Chunking → embedding/index → retrieval → generation | Đạt phân tách mã | prepare_chunks, build_index, retrieve và generate. Đọc riêng từng bước được; không train end-to-end retriever. |
| Recall@k, EM/F1 | Đã có kết quả; chưa chốt quy tắc chấm | Recall tài liệu khác evidence recall. EM/F1 hiện dùng normalizer tùy chỉnh, khác TriviaQA gốc. Canonical-only khác official alias evaluation. |
| Faithfulness/citation correctness trên mẫu | Sơ bộ | 20 mẫu có agent_self_review; phiếu dành cho người đánh giá vẫn trống. Đánh giá nguồn được cung cấp, không phải độ chính xác inline citations do T5 sinh. |
| Phân rã lỗi retrieval/generation | Có proxy tự động | Tách thiếu corpus, thiếu top-k, chọn/cắt context và sinh sai. Dò chuỗi đáp án không xác lập nguyên nhân ngữ nghĩa; cần rà trên mẫu. |
| Ít nhất một cải tiến tùy chọn | Đã có Hybrid và ablation retrieval | Run v2 đo RRF retrieval; chưa đo answer EM/F1 với Hybrid mới trên tập độc lập. Ablation cũ dùng corpus gốc và 100 câu, không ghép với kết quả v2. |

## Những phát hiện cần xử lý trước khi chốt thực nghiệm

### 1. Chất lượng context đầu vào huấn luyện

`training_pairs.jsonl` có 680 dòng. Theo script train hiện tại, 10% được tách làm validation nội bộ, tức cấu hình tương ứng 612 train/68 validation; đây không phải 680 mẫu đều đi vào gradient update. Không có trainer_state/loss log trong thư mục checkpoint đang kiểm tra để xác nhận lại diễn biến lần huấn luyện lịch sử.

Kiểm tra toàn bộ 680 dòng:

| Cách dò chuỗi bằng chứng | Thiếu canonical | Thiếu cả canonical và official aliases |
| --- | ---: | ---: |
| Normalizer hiện tại của project | 238 | 206 |
| Chuẩn hóa dấu câu thành khoảng trắng theo TriviaQA, khớp ranh giới từ | 232 | 201 |

Hai hàng khác nhau vì chính normalizer cũng ảnh hưởng phép dò bằng chứng. Đây là **cảnh báo lexical**, không có nghĩa mọi dòng thiếu chuỗi đều không thể suy luận đáp án. Cần đọc/nghiệm thu quan hệ question–context–answer trước khi quyết định sửa hoặc loại một mẫu.

Có 223 dòng dùng quy tắc fallback lấy chunk đầu tiên khi không tìm thấy chuỗi canonical bằng phép tìm substring cũ. Context lưu hiện tại khớp hoàn toàn quy tắc chọn trong `experiments/split_train_test.py:29`. Cùng tài liệu không đảm bảo passage chứa câu trả lời. Backend tokenizer local ở giới hạn train 384 token không làm mất thêm canonical nào theo phép dò hiện tại: vấn đề chính đã có trước bước truncation trong kiểm tra này.

**Cần làm:** xây một phiên bản training data mới với evidence được kiểm tra; không đưa context thiếu hỗ trợ vào như positive một cách im lặng. Giữ dữ liệu cũ/checkpoint cũ để đối chiếu. Không lọc câu khó theo điểm test.

### 2. Cách tính điểm chưa tương thích hoàn toàn với TriviaQA

[Evaluator gốc TriviaQA](https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py) thay dấu câu bằng khoảng trắng. Project hiện xóa dấu câu, có thêm NFKC/casefold. Điều này làm một số span đúng bị chấm thấp, ví dụ `Eugene O ' Neill` và `A Hard Day ' s Night`.

Chấm chẩn đoán trên **cùng 120 dự đoán**, chỉ đổi normalizer và vẫn dùng canonical-only:

| Hệ | EM custom hiện tại | F1 custom hiện tại | EM kiểu chuẩn hóa TriviaQA | F1 kiểu chuẩn hóa TriviaQA |
| --- | ---: | ---: | ---: | ---: |
| Generator-only | 0,83% | 1,93% | 0,83% | 1,93% |
| Dense + DistilBERT | 26,67% | 36,67% | 29,17% | 38,72% |
| Dense + T5 | 34,17% | 39,76% | 34,17% | 39,98% |

Có 5 cặp system–question thay đổi điểm. **Bảng chẩn đoán không phải full official TriviaQA score**: official evaluation xét tập aliases, còn bảng này chỉ dùng canonical. Không thay thế nhãn đã duyệt bằng toàn bộ aliases một cách mù quáng; 1.626 alias ứng viên vẫn đang chờ duyệt. Canonical gốc của 800 QA đều khớp `answer.value` đã phục hồi, nhưng vẫn có lỗi nhãn nguồn như “Julius Ceasar”.

**Cần làm:** version hóa evaluator và answer policy; kiểm tra tương thích với evaluator gốc; rà alias/nhãn, rồi chấm lại toàn bộ dự đoán. Giữ nguyên điểm cũ dưới nhãn historical/custom; không diễn giải các cột này là những lần mô hình tiến bộ.

### 3. Baseline và template cần được kiểm soát

Train hiện dùng `question: ... context: ...`, input tối đa 384 token. RAG inference thêm tiền tố hướng dẫn khác và dùng 512 token; generator-only lại dùng tiền tố “without context” với 256 token. Những khác biệt này phải được mô tả và kiểm soát, vì kết quả không chỉ phản ánh có/không retrieval.

Generator-only hiện hợp lệ như **ablation bỏ context của checkpoint đã học có context**. Không dùng EM 0,83% để kết luận tổng quát RAG vượt mọi mô hình closed-book. Nếu nghiên cứu hiệu quả retrieval, nên thống nhất template và ngân sách inference; nếu tuyên bố so với closed-book được huấn luyện, cần baseline train tương ứng và ghi rõ chi phí/cấu hình.

### 4. Split và corpus chỉ hỗ trợ phạm vi tuyên bố hẹp

- Không có câu hỏi trùng chính xác sau chuẩn hóa giữa 680 dòng và 120 test.
- **12/120 câu test** có cùng tiêu đề trang nguồn với ít nhất một dòng đầu vào training. Đây không tự động là rò rỉ đáp án, nhưng không chứng minh khái quát trên tài liệu chưa gặp. Nếu câu hỏi nghiên cứu đòi hỏi tài liệu mới, cần group split theo canonical page/entity ID.
- 800 document IDs chỉ có **748 tiêu đề nguồn khác nhau**. Corpus v2 có **1.544 chunk trùng nội dung dư**; **11/120 truy vấn** có hai passage Dense đầu giống nhau. Cần chuẩn hóa page identity/deduplicate và ánh xạ lại gold sources trong một phiên bản corpus mới.
- verified_v2 đã dùng aliases để chọn/mở rộng evidence; demo_v3 thêm nguồn sau khi quan sát lỗi test. Không dùng chúng làm benchmark tìm kiếm toàn Wikipedia. Bộ 120 câu đã được xem nhiều lần nên phù hợp làm tập phát triển/lịch sử; cần một holdout mới để đánh giá cải tiến cuối.

### 5. Cải tiến chưa được đánh giá đầu-cuối trên một protocol chung

`experiments.evaluate` lưu BM25, Dense và Hybrid RRF rankings nhưng sinh câu trả lời bằng **Dense**. Do đó Recall@5 Hybrid bằng Dense 92,50% chưa chứng minh Hybrid cải thiện answer EM/F1. Hybrid trên web còn có ưu tiên từ hiếm và BM25/tokenization khác nhánh thí nghiệm; không gán điểm v2 cho cấu hình web này.

`experiments.ablation` dùng raw corpus gốc, 100 câu đầu của test cũ, chỉ đo retrieval recall, không đo generation EM/F1. Dense RAG luôn nhận tối đa hai passage: tăng danh sách top-k lên 5 hoặc 10 không có nghĩa T5 đọc tất cả các đoạn đó. Khi ablation số passage thực dùng, cần thay đúng context budget và đo lại chất lượng câu trả lời.

**Cần làm:** so Dense, BM25 và Hybrid với cùng corpus/QA/reader/generator/template; giữ fixed top-n hoặc chủ động ablate top-n, không lẫn với retrieval k. Báo kết quả âm nếu Hybrid không tốt hơn. Với bài báo, thêm độ bất định và so sánh ghép cặp phù hợp sau khi chốt protocol; việc triển khai Hybrid đơn thuần chưa tự tạo đóng góp khoa học mới.

### 6. Faithfulness, nguồn và tái lập còn phần chưa chạy

Mẫu tự rà soát 20 câu: 11 supported, 7 unsupported, 2 unclear. Phiếu human review hiện có **0/20 dòng được chấm**. Không gọi 11/20 là tỷ lệ đã được con người xác nhận; không gọi token overlap 87,5% là tỷ lệ không hallucination. Nên có người đánh giá độc lập theo rubric, ghi lý do và passage IDs, xử lý bất đồng nếu dùng nhiều người chấm.

FAISS v2 đã kiểm tra: IndexFlatIP, 12.556 vector, 384 chiều; hữu hạn, unit norm trong sai số tuyệt đối 0,001. Dữ liệu, index, checkpoint và training pairs vẫn khớp checksum run lịch sử. Một số mã evaluation/metrics/neural engine hiện khác checksum lịch sử: cần chốt source snapshot/commit và phiên bản evaluator cho run mới.

Chưa thực thi Colab trong lần rà soát này. Có hướng dẫn không đồng nghĩa có bằng chứng tái lập. Cần chạy trên một runtime Colab sạch, ghi phiên bản/thời gian/RAM-VRAM/manifest/metrics và pin model revisions cùng môi trường phù hợp. Không khẳng định bitwise reproducibility chỉ từ seed 42.

## Quan hệ với bài báo nền tảng

[Lewis et al.](https://arxiv.org/html/2005.11401v4) dùng DPR và BART, học query encoder cùng generator, marginalize passage trong RAG-Sequence/RAG-Token. Project hiện dùng MiniLM cố định, T5 và ghép hai passage. Cách gọi đúng là **pipeline thực nghiệm ứng dụng ý tưởng RAG**, không phải tái lập đầy đủ các công thức/huấn luyện của paper gốc.

[TriviaQA chính thức](https://nlp.cs.washington.edu/triviaqa/) cũng phân biệt RC và unfiltered IR-style QA. Project dùng rc.wikipedia có evidence được cung cấp/chọn; cần mô tả rõ phạm vi controlled-corpus QA thay vì so thẳng với điểm open-domain leaderboard.

## Thứ tự hoàn thiện trước khi chốt kết quả

1. **Dữ liệu và protocol:** duyệt nhãn/aliases, sửa evidence đầu vào, quyết định phạm vi split; tạo train/dev/holdout mới và đóng băng corpus bằng quy tắc không chọn theo đáp án test. Giữ các bộ lịch sử nguyên trạng.
2. **Evaluator và baseline:** thống nhất chuẩn hóa/answer policy/template; xác định generator-only là ablation hay model được train riêng. Ghi rõ lượng train thực tế và validation.
3. **Thí nghiệm mới:** train trên dữ liệu đã nghiệm thu, lưu training logs/checkpoint/config; chạy đủ ba hệ và Hybrid/ablation trên protocol chung. Chọn cấu hình bằng dev, chỉ chốt holdout sau khi khóa phương pháp.
4. **Đánh giá ngữ nghĩa và tái lập:** người chấm độc lập trên mẫu, lý do/source IDs; thực thi Colab sạch và lưu artifact. Không dùng demo chọn lọc thay cho các bước này.
5. **Viết báo:** chốt câu hỏi nghiên cứu/đóng góp, báo cả kết quả âm, giới hạn, độ bất định và phương pháp tái lập. Không thể bảo đảm khả năng được chấp nhận chỉ từ checklist phần mềm.

## Bằng chứng lần rà soát

- [Audit máy đọc](../../results/research_audit_20261031/audit.json)
- [Rà context của 680 dòng](../../results/research_audit_20261031/training_context_audit.csv)
- [Dò bằng chứng với chuẩn hóa TriviaQA](../../results/research_audit_20261031/training_triviaqa_style_check.json)
- [Những điểm thay đổi khi đổi normalizer](../../results/research_audit_20261031/normalization_differences.json)
- [Checksum đầu vào và kiểm tra fallback/phiếu mẫu](../../results/research_audit_20261031/evidence_manifest.json)

Lệnh kiểm tra cốt lõi đã chạy thành công, khoảng 7 giây trên môi trường local; dùng tên thư mục đầu ra chưa tồn tại nếu chạy lại:

```powershell
python -m scripts.audit_research --output results/research_audit_new --check-tokenizer --check-index
```

Các kiểm tra thuật toán/định dạng và artifact có bằng chứng PASS; tương thích normalizer với TriviaQA là FAIL; human review độc lập, Colab và holdout mới là NOT RUN. Không thay một mục NOT RUN bằng kết quả suy đoán. Kết quả unit tests 43/43 vẫn là bằng chứng phần mềm; không coi đó là chứng nhận tính hợp lệ khoa học.
