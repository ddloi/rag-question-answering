# Cấu hình retrieval baseline

`baseline_retrieval.json` là cấu hình mặc định có thể theo dõi trong Git, giữ các giá trị của cấu hình local hiện tại khi sắp xếp project.

`experiments.baseline_retrieval` ưu tiên `.pipeline/hyperparams.json` nếu tệp local này tồn tại; nếu không, dùng `configs/baseline_retrieval.json`. Các đường dẫn tương đối trong cấu hình được tính từ thư mục gốc repository, đường dẫn tuyệt đối được giữ nguyên. `.pipeline/` vẫn chứa trạng thái chạy local và không được đưa vào Git.
