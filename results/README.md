# Kết quả thí nghiệm

## Research_v1 hiện tại

- `research_v1_dev/`, `research_v1_test/`: đầy đủ 200 dev/300 test, predictions, metrics, source snapshots và các phiếu chấm.
- `research_colab_dev/`, `research_colab_test/`, `research_colab_commands/`: replay Colab thật và logs. Các JSON `research_colab_*` ghi environment, checksum, import và đối chiếu.
- `training/research_v1/`: hai run train và các pilot, gồm cả pilot thất bại để bảo toàn provenance.
- `research_v1_ablation_dev/`: ablation top-k/token budget và reranker trên 200 dev; xem [các cải tiến](../docs/IMPROVEMENTS_RESEARCH_V1.md).
- `research_v1_ablation_dev_rag_retrieved/`: cùng lưới cấu hình với T5 train trên context truy hồi, kèm `comparison_vs_v1.json`; bằng chứng train ở `training/research_v1/rag_retrieved/`. Xem [các cải tiến](../docs/IMPROVEMENTS_RESEARCH_V1.md).
- `research_v1_test_lock.json` và source ZIP: protocol/source/model identity đã khóa trước test.
- `research_v1_test/human_review_author.csv`: phiếu người dùng tự chấm, hiện chưa có nhãn. Không coi phiếu trống là đánh giá hoàn tất.

Xem [kết quả nghiên cứu](../docs/RESEARCH_RESULTS_V1.md) và [trạng thái nghiệm thu](../docs/RESEARCH_UPGRADE_STATUS.md). Các run thử 3/5 câu và bản chấm canonical trung gian đã đưa ra ngoài project, ghi trong [biên bản dọn](../docs/CLEANUP.md).

## Bằng chứng lịch sử được giữ

`research_audit_20261031/` là rà soát dữ liệu/normalizer/index và mức độ sẵn sàng nghiên cứu, không phải một lần train hay suy luận mới. Đọc [kết luận và việc cần xử lý](../docs/history/SCIENTIFIC_READINESS.md) trước khi trích các điểm cũ như kết quả cuối.

`canonical_rescore_v2/` chấm lại 120 dự đoán bằng canonical-only: RAG EM 34,17%, F1 39,76%; không chạy lại mô hình. Có metrics, scores, manifest/checksum và alias_review.csv để người đánh giá duyệt. Xem ../docs/IMPROVEMENTS_V3.md.

`evaluation_v2_full/` là run lịch sử dùng official aliases chưa được duyệt trên đủ 120 câu với corpus verified_v2: predictions, metrics, manifest, validation và phiếu mẫu. Run thử 3 câu đã được lưu riêng ngoài project. Xem ../docs/EVALUATION_RESULTS_V2.md.

`baseline_results.json` và `ablation_results.csv` là artifact lịch sử của corpus cũ, giữ nguyên. Script classical baseline nay ghi classical_baselines_measured.json và không chèn số tham chiếu Dense/Hybrid. Dense/Hybrid được đo trực tiếp trong evaluation. Không so trực tiếp hai phiên bản corpus.

`demo_v3_smoke/` lưu 8 phản hồi API từ các ví dụ được chọn trên corpus demo bổ sung, không phải kết quả benchmark. Manifest ghi cấu hình và hash để kiểm tra.
