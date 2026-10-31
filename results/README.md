# Kết quả thí nghiệm

`research_audit_20261031/` là rà soát dữ liệu/normalizer/index và mức độ sẵn sàng nghiên cứu, không phải một lần train hay suy luận mới. Đọc [kết luận và việc cần xử lý](../docs/SCIENTIFIC_READINESS.md) trước khi trích các điểm cũ như kết quả cuối.

`canonical_rescore_v2/` chấm lại 120 dự đoán bằng canonical-only: RAG EM 34,17%, F1 39,76%; không chạy lại mô hình. Có metrics, scores, manifest/checksum và alias_review.csv để người đánh giá duyệt. Xem ../docs/IMPROVEMENTS_V3.md.

`evaluation_v2_full/` là run lịch sử dùng official aliases chưa được duyệt trên đủ 120 câu với corpus verified_v2: predictions, metrics, manifest, validation và phiếu mẫu. `evaluation_v2_smoke/` chỉ kiểm tra nhanh 3 câu. Xem ../docs/EVALUATION_RESULTS_V2.md.

`baseline_results.json` và `ablation_results.csv` là artifact lịch sử của corpus cũ, giữ nguyên. Script classical baseline nay ghi classical_baselines_measured.json và không chèn số tham chiếu Dense/Hybrid. Dense/Hybrid được đo trực tiếp trong evaluation. Không so trực tiếp hai phiên bản corpus.

`demo_v3_smoke/` lưu 8 phản hồi API từ các ví dụ được chọn trên corpus demo bổ sung, không phải kết quả benchmark. Manifest ghi cấu hình và hash để kiểm tra.
