# Tiến độ nghiệm thu research_v1

Cập nhật 02/11/2026; lượt Colab hoàn tất ngày 01/11/2026. Kế hoạch gốc: [RESEARCH_UPGRADE_PLAN.md](history/RESEARCH_UPGRADE_PLAN.md). Tài liệu học cấu trúc/vận hành đã có. Không tự quy đổi checklist này thành điểm học phần hoặc chấp nhận công bố.

| Hạng mục | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Evaluator TriviaQA + policy rõ ràng | PASS | metrics.py; legacy_metrics.py giữ lịch sử; regression tests |
| Dataset/split/corpus mới | PASS kiểm tra triển khai | 3.157 docs, 90.667 chunks; manifest/replay descriptor; 0 question overlap; logical replay khớp 5/5 |
| Vector/index và pipeline | PASS | Toàn bộ vectors finite/unit; kiểm tra FAISS IP, dimension, encoder token budget và checksum |
| Hai T5 fine-tune riêng | COMPLETE, checkpoint preflight PASS | Cùng 1.865 train/200 dev, seed 42, budget 3 epoch, 351 steps/run; best dev loss 1,527 RAG và 2,590 closed-book |
| Đánh giá ba hệ + Hybrid/ablation | COMPLETE | 200 dev, 300 test; predictions, metrics, context thực dùng, source snapshot và model hashes |
| Tính lại điểm và khóa test | PASS | QA/retrieval recompute khớp 1e-12; source/data/model lock và đủ 300 test được xác minh |
| Phân rã lỗi và ví dụ báo cáo | COMPLETE ở mức proxy + agent case review | cases.csv, case_examples.json, CASES_RESEARCH_V1.md; không gọi proxy là kết luận ngữ nghĩa |
| Demo ba hệ | PASS kiểm tra UI có giới hạn | 4 requests/12 answers khớp dev saved outputs; desktop/mobile, filter, stale, Ctrl+Enter, cờ rà nhãn; xem UI_VERIFICATION_RESEARCH_V1.md |
| Faithfulness/citation do người chấm | NOT RUN | Người dùng chọn human_author; human_review_author.csv có 50 QA/100 dòng test, chưa điền nhãn/lý do; rubric và aggregator đã có |
| Suy luận trên Colab thật | PASS_EXACT_ANSWERS; một dev ranking diagnostic REVIEW_REQUIRED | T4/Python 3.13.16/Torch 2.11.0+cu130; đủ 200 dev/300 test, 2.500 raw answers và QA EM/F1 khớp local. Context thực dùng khớp; test ranking khớp. Evidence ZIP đã nhập và kiểm tra hash. Xem COLAB_VERIFICATION_RESEARCH_V1.md |

## Kết quả test cần dùng

Official EM/F1: closed-book 8,33/12,82%; extractive 53,33/60,17%; Dense RAG 57,67/62,68%; Hybrid 56,33/61,36%. Canonical Dense RAG 53,67/60,39%. Xem [bảng đầy đủ](RESEARCH_RESULTS_V1.md) và REPORT.md trong results/research_v1_test.

Hybrid–Dense test EM official = -1,33 điểm phần trăm, CI 95% [-6,00; +3,33], gồm 0. Dev tăng 3 điểm phần trăm không được dùng để khẳng định Hybrid cải thiện ổn định trên test.

## Giới hạn còn mở

- Cần người dùng hoàn tất [phiếu human_author](../results/research_v1_test/human_review_author.csv) và lưu lý do theo rubric; agent casebook không thay human evaluation. Tự chấm không phải human_independent.
- Colab chỉ replay suy luận với corpus/index/checkpoints đã chốt; chưa dựng lại corpus từ nguồn hoặc train lại trên Colab. Notebook có output đã lưu trên Drive; bản raw executed .ipynb chưa tải được về local. Evidence predictions/logs/metrics/environment đã lưu và xác minh đầy đủ.
- Một câu dev (qb_4397) đổi thứ tự Dense hạng 23/24 và Hybrid 28/29; top-1/3/5/10, context, đáp án và mọi retrieval metric đã báo không đổi. Chưa chứng minh nguyên nhân; giữ REVIEW_REQUIRED cho đối chiếu toàn bộ thứ hạng.
- Hai câu Antigonus/Fivestones vẫn sai trên research_v1 khi probe thủ công; được giữ nguyên trong manual_probes và casebook, chưa gọi đã khắc phục hoàn toàn.
- Có cờ rà canonical ở câu nơi sinh Martina Hingis; chỉ là ledger đề xuất đã đối chiếu WTA, chưa có human approval. Giữ nguyên dataset/điểm frozen.
- 85 documents chung train/test; đây là QA mới trên corpus chung. Một training seed, pretraining contamination chưa kiểm soát; không tự nhận unseen-document generalization hoặc đóng góp công bố mới.

## Lịch sử và provenance

Pilot FP16 FAIL vì gradient không hữu hạn; BF16 pilot PASS và đã kiểm tra weight update/tied weights/reload. RAG full được resume từ step 117 sau gián đoạn; elapsed_seconds chỉ tính lượt tiếp tục. Giữ incident record và log gốc.

Source train của hai hệ khớp nhau. Sau train chỉ bổ sung kiểm tra checksum/index, ghi nguồn title bị cắt và chặn resume run complete; prompt/context format không đổi. Có source snapshot và post-training audit record. Sau test, chỉ chỉnh thông báo/cache asset/metadata cờ nhãn của UI; bốn file phương pháp trong test lock giữ nguyên.

Corpus/checkpoint/số liệu v2/v3 là lịch sử riêng. Corpus demo_v3 bổ sung nguồn sau quan sát lỗi không được thay vào benchmark research_v1. Các gói source/data/models/evidence nằm trong outputs và bị Git bỏ qua; source/config/manifests/kết quả nén/log nhỏ được chuẩn bị để người dùng review trước khi commit/push.
