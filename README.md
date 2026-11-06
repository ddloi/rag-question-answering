# RAG Question Answering

Đồ án hỏi–đáp trên controlled TriviaQA subset, so sánh **T5 closed-book**, **Dense + DistilBERT extractive** và **Dense/Hybrid + T5 RAG**. Phiên bản dùng để báo cáo là **research_v1**.

## Bắt đầu ở đâu?

1. [Học cấu trúc và cách vận hành](docs/HOC_VA_VAN_HANH.md).
2. [Bản đồ thư mục và lệnh chạy](docs/ARCHITECTURE.md).
3. [Protocol nghiên cứu](docs/RESEARCH_PROTOCOL_V1.md), [kết quả](docs/RESEARCH_RESULTS_V1.md) và [các case để báo cáo](docs/CASES_RESEARCH_V1.md).

[Danh mục tài liệu](docs/README.md) gom các hướng dẫn hiện tại. Nội dung v2/v3 và README dài cũ nằm trong [tài liệu lịch sử](docs/history/README_LEGACY.md).

## Chạy demo

Chạy PowerShell trong thư mục `RAG`:

```powershell
python -m pip install -r requirements/demo.txt -r requirements/research.txt
python app.py
```

Mở http://127.0.0.1:8001/research, bấm **Nạp ba hệ**, chờ sẵn sàng rồi nhập câu hỏi hoặc chọn tình huống dev. Cần corpus/index và hai checkpoint research_v1; clone source riêng chưa đủ để suy luận. Xem [chuẩn bị artifact và GitHub](docs/GITHUB_UPDATE.md).

## Cấu trúc để làm việc

```text
RAG/
├── app.py              # Mở web demo
├── README.md           # Bắt đầu tại đây
├── ragqa/              # API, giao diện và adapter suy luận
├── experiments/        # Dữ liệu, retrieval, train, evaluate
├── scripts/            # Kiểm tra, báo cáo, đóng gói
├── configs/            # Cấu hình nghiên cứu
├── requirements/       # Môi trường demo, research, test
├── notebooks/          # Hai workflow Colab
├── tests/              # Kiểm thử
├── docs/               # Hướng dẫn và hồ sơ nghiên cứu
│   ├── history/        # Tài liệu các phiên bản cũ
│   └── development/    # Đặc tả sản phẩm và thiết kế
├── deployment/         # Hugging Face Space tùy chọn
├── data/               # Corpus, index, split và provenance
├── checkpoints/        # Hai model research_v1
├── generator_finetuned/ # Model lịch sử cho /legacy
├── results/            # Kết quả và bằng chứng đã chốt
└── outputs/            # ZIP bàn giao và logs local; Git bỏ qua
```

Để học pipeline, bắt đầu với `configs/research_v1.json` rồi đọc [bản đồ experiments](experiments/README.md). Các script đánh số và requirements chuyển tiếp cũ ở root đã đưa ra bản lưu riêng; dùng lệnh module trong hướng dẫn cấu trúc. Các đường dẫn source/data/model/results đã chốt được giữ nguyên.

## Kết quả và giới hạn

Đã chạy đủ 200 dev/300 test nội bộ. Điểm official trên test:

| Hệ | EM | F1 |
| --- | ---: | ---: |
| T5 closed-book | 8.33% | 12.82% |
| Dense + DistilBERT extractive | 53.33% | 60.17% |
| Dense + T5 RAG | 57.67% | 62.68% |
| Hybrid RRF + T5 RAG | 56.33% | 61.36% |
| Checkpoint RAG bỏ context — ablation | 6.67% | 12.26% |

Hybrid–Dense test EM = -1,33 điểm phần trăm, CI 95% [-6,00; +3,33] gồm 0. Chưa chứng minh Hybrid tốt hơn ổn định. Bộ truy hồi MiniLM cố định; pipeline nối context vào T5 áp dụng ý tưởng Lewis et al. (2020), chưa tái lập joint retriever training/latent-document marginalization của bài gốc.

[Colab T4](docs/COLAB_VERIFICATION_RESEARCH_V1.md) đã replay đủ 500 QA × 5 cấu hình: **2.500 raw answers và QA EM/F1 khớp local**. Một câu dev đổi thứ tự passage ở hạng thấp; context và đáp án không đổi. Chưa train lại trên Colab.

**Faithfulness/citation do người chấm còn NOT RUN**. Người dùng tự chấm [phiếu 100 dòng/50 QA](results/research_v1_test/human_review_author.csv) theo [rubric](docs/REVIEW_RUBRIC.md), với loại reviewer `human_author`. Hai probe Antigonus/Fivestones vẫn sai; không thay output bằng ground truth để demo.

## Kiểm tra phần mềm

```powershell
python -m pip install -r requirements/test.txt
python -m unittest discover -s tests -v
python -m experiments --list
```

Bộ kiểm thử hiện có **106 tests**, gồm các test demo lịch sử cần corpus local; hai test cấu hình baseline cần môi trường ML và sẽ được bỏ qua nếu chưa cài các thư viện này. Trên clone chỉ có source, chạy hai nhóm nghiên cứu không phụ thuộc corpus đầy đủ:

```powershell
python -m unittest discover -s tests -p 'test_research*.py' -v
python -m unittest discover -s tests -p 'test_triviaqa_metrics.py' -v
```

Hai nhóm này có 56 tests. Chạy toàn bộ sau khi chuẩn bị corpus demo lịch sử. Dữ liệu/model lớn và cache không đưa vào Git; không ép stage chúng. Xem [biên bản dọn](docs/CLEANUP.md) và [trạng thái nghiên cứu](docs/RESEARCH_UPGRADE_STATUS.md) trước khi bàn giao.

Kiểm tra mã Python bằng `python -m pip install -r requirements/lint.txt`, sau đó chạy `python -m ruff check app.py ragqa experiments scripts deployment tests`. Phạm vi và ngoại lệ của mã nghiên cứu đã khóa được giải thích trong [biên bản clean code](docs/CLEAN_CODE_REVIEW.md).
