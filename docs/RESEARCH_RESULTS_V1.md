# Kết quả nghiên cứu research_v1

Đã chạy đủ 200 dev và **300 test nội bộ** của controlled TriviaQA subset. Protocol/source/data/model hashes được khóa trước test; test không được lấy ít câu hoặc bỏ mẫu khó. Đây không phải official hidden test hay full-Wikipedia benchmark. Kết quả v2/v3 lịch sử dùng dữ liệu/model khác nên không so trực tiếp như cùng một thí nghiệm.

## Ba hệ chính và các đối chiếu

| Hệ | Official EM | Official F1 | Canonical EM | Canonical F1 |
| --- | ---: | ---: | ---: | ---: |
| T5 closed-book | 8.33% | 12.82% | 6.33% | 9.41% |
| Dense + DistilBERT extractive | 53.33% | 60.17% | 47.33% | 55.47% |
| Dense + T5 RAG | 57.67% | 62.68% | 53.67% | 60.39% |
| Hybrid RRF + T5 RAG | 56.33% | 61.36% | 52.67% | 59.20% |
| Checkpoint RAG bỏ context (ablation) | 6.67% | 12.26% | 3.33% | 5.63% |

Hai T5 được fine-tune riêng trên cùng 1.865 QA, cùng base revision, seed 42 và budget 3 epoch; chọn best checkpoint theo dev loss. Dòng ablation bỏ context khác baseline closed-book. MiniLM cố định, FAISS IndexFlatIP; đây là ứng dụng ý tưởng RAG, chưa tái lập joint retriever training/latent-document marginalization của RAG gốc.

## Kết quả Hybrid cần báo trung thực

Trên dev, Hybrid EM official cao hơn Dense 3 điểm phần trăm; trên test, Hybrid thấp hơn **1,33 điểm phần trăm**. Paired bootstrap 95% CI của chênh test là **[-6,00; +3,33] điểm phần trăm**, gồm 0. Một seed và một controlled subset chưa chứng minh Hybrid cải thiện ổn định. Giữ nguyên cả kết quả âm; không đổi phương pháp sau khi xem test rồi tiếp tục gọi test này là holdout chưa dùng.

## Retrieval và phân rã lỗi

| Phương pháp | Doc hit@5 | Doc recall@5 | Canonical answer recall@5 |
| --- | ---: | ---: | ---: |
| dense | 94.00% | 79.33% | 84.00% |
| bm25 | 93.67% | 78.86% | 80.67% |
| hybrid | 95.00% | 80.97% | 84.67% |

Doc hit là tìm được ít nhất một nguồn; doc recall là phần nguồn liên quan được phủ. Answer recall dò chuỗi, không phải xác nhận bằng chứng ngữ nghĩa. Có đầy đủ k=1/3/5/10 và MRR@30 trong metrics.json.

Dense RAG theo canonical: 161 câu đúng EM; 3 thiếu chuỗi trong corpus, 42 thiếu chuỗi trong top-5, 34 rơi vào chọn/cắt context và 60 sai dù context có chuỗi reference. Các nhóm là proxy lexical, chưa xác lập nguyên nhân ngữ nghĩa. [Casebook dev](CASES_RESEARCH_V1.md) có ví dụ đáp án cụ thể hơn canonical, lỗi quan hệ, nhãn nguồn cần rà, Hybrid tốt/kém hơn và EM đúng nhưng context chưa đủ để khẳng định faithful.

## Bằng chứng và phần còn thiếu

- [Báo cáo test đầy đủ](../results/research_v1_test/REPORT.md), predictions.jsonl.gz, cases.csv, case_examples.json và score_recomputation.json cùng thư mục. Điểm QA/retrieval đã tính lại từ predictions, khớp ở sai số 1e-12.
- Mỗi split có 50 QA ngẫu nhiên cho Dense/Hybrid, tương ứng 100 dòng phiếu. **Faithfulness/citation người chấm: NOT RUN** vì phiếu chưa điền; agent casebook không thay thế metric người chấm.
- **Colab T4: PASS_EXACT_ANSWERS** cho đủ 200 dev/300 test và năm cấu hình: 2.500 raw answers, official/canonical EM/F1 và context thực dùng khớp local. Một câu dev đổi thứ tự Dense hạng 23/24 và Hybrid 28/29; retrieval metrics đã báo không đổi, diagnostic giữ REVIEW_REQUIRED. Test ranking khớp. Đây là replay suy luận với checkpoint có sẵn; chưa tái lập huấn luyện. [Bằng chứng Colab và giới hạn](COLAB_VERIFICATION_RESEARCH_V1.md).
- Source/data/model/evidence bundles được tạo riêng; xem [GitHub và đóng gói](GITHUB_UPDATE.md). Trọng số/corpus/cache không bị ép vào Git.
- Corpus dùng chung: 85 documents xuất hiện cả train/test; không trùng question IDs/câu hỏi chuẩn hóa. Kết luận giới hạn ở QA mới trên corpus biết trước, không phải tài liệu mới. Pretraining contamination chưa kiểm soát.

Hai probe Antigonus/Fivestones vẫn sai ở research_v1; xem manual_probes và casebook. Chưa coi độ chính xác ngoài subset là đã giải quyết hoàn toàn.

## Học và trình bày

Đọc [học/vận hành](HOC_VA_VAN_HANH.md), sau đó [casebook](CASES_RESEARCH_V1.md) và [15 câu phản biện](PHAN_BIEN_VA_TAP_TRINH_BAY.md). Mở `/research`, nạp ba hệ, chọn một case đúng và một case sai, rồi đối chiếu context thực dùng. T5 vẫn có thể sinh sai; không thay output bằng ground truth để demo đẹp hơn.
