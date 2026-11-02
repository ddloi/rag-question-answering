# Chạy lại đánh giá trên Colab

> Phần bên dưới là hướng dẫn v2 lịch sử. **Research_v1** dùng [notebook mới](../../notebooks/research_reproduce_colab.ipynb), [protocol](../RESEARCH_PROTOCOL_V1.md) và [cách đóng gói source/data/models](../GITHUB_UPDATE.md). Notebook mới hỗ trợ dựng lại từ revision pin hoặc evaluate frozen artifacts; kiểm tra cú pháp local không được gọi là chạy Colab thành công.

Hướng dẫn này chưa được thực thi trong một phiên Colab ở lần hoàn thiện này. Run đã xác minh dùng Windows, Python 3.13.1, CUDA RTX 3050 6GB; phiên bản thư viện chính nằm trong manifest của run. Không khẳng định kết quả bitwise giống nhau trên GPU/hệ điều hành khác.

## Chuẩn bị

Chọn runtime GPU. Đưa thư mục RAG vào Google Drive, gồm mã nguồn, data gốc, checkpoint `generator_finetuned` và run `results/evaluation_v2_full`. Các file lớn bị Git bỏ qua nên chỉ clone mã nguồn sẽ chưa có trọng số/dữ liệu. Giữ nguyên manifest/checksum đã chốt để đối chiếu.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Đổi đường dẫn sau theo vị trí lưu thực tế. Chép project vào đĩa local của Colab trước khi chạy để tránh truy cập hàng nghìn file qua Drive.

```python
import shutil
source = '/content/drive/MyDrive/Rag-question-answering/RAG'
shutil.copytree(source, '/content/RAG', dirs_exist_ok=True)
%cd /content/RAG
%pip install -r requirements/ml.txt -r requirements/test.txt
```

Đối chiếu các phiên bản torch/transformers/sentence-transformers/faiss-cpu/rank-bm25 với `results/evaluation_v2_full/manifest.json`. CUDA wheel của Windows không dùng để khóa cứng môi trường Linux Colab. Nếu Colab yêu cầu restart runtime sau khi cài thư viện, restart rồi chuyển lại `/content/RAG`.

## Model cache và corpus

Checkpoint T5 đã có cần giữ đúng SHA256; MiniLM và DistilBERT cần cache trên runtime mới:

```python
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
name = 'distilbert-base-cased-distilled-squad'
AutoTokenizer.from_pretrained(name)
AutoModelForQuestionAnswering.from_pretrained(name)
```

Nếu đã chép verified_v2, kiểm tra checksum theo manifest và dùng nguyên corpus/index. Nếu chưa có, phục hồi từ revision đã pin; các lệnh tạo mới dừng khi artifact đã tồn tại:

```python
!python -m scripts.recover_trivia_source
!python -m experiments.verify_dataset --build-index
```

## Chạy lại

```python
!python -m unittest discover -s tests -v
!python -m experiments.evaluate --answer-policy canonical --run-dir results/evaluation_colab
```

Giữ `manifest.json`, `metrics.json`, `predictions.jsonl` và `review_sample.csv` của run Colab. Đối chiếu số lượng 120 câu, 12.556 chunks, chính sách canonical/aliases, checksum model và input trước khi so điểm. Không dùng số liệu corpus 3.940 chunks cũ làm baseline trực tiếp cho corpus v2.

```python
shutil.copytree('/content/RAG/results/evaluation_colab',
                source + '/results/evaluation_colab', dirs_exist_ok=False)
```

Không cần huấn luyện lại để tái lập phép đánh giá checkpoint hiện tại. Nếu train lại, đó là một thí nghiệm mới và cần ghi phiên bản training data/checkpoint riêng.
