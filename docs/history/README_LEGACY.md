# Nội dung README lịch sử

Tài liệu này giữ phần nội dung dài của các phiên bản cũ để đối chiếu. Các số liệu, cây thư mục và lệnh bên dưới là lịch sử; vận hành bản hiện tại theo [README chính](../../README.md) và [hướng dẫn học](../HOC_VA_VAN_HANH.md). Những launcher ở root đã được rút khỏi bản hiện tại; sử dụng lệnh module ghi trong hướng dẫn cấu trúc.

## Bản cập nhật 31/10/2026

**Lưu ý lịch sử:** [audit ngày 31/10](SCIENTIFIC_READINESS.md) phát hiện context training cần nghiệm thu, normalizer EM/F1 khác TriviaQA gốc và các baseline cần kiểm soát template. Các phát hiện đã được xử lý trong research_v1 theo tiến độ ở trên. Điểm dưới đây giữ policy custom/canonical lịch sử và không thay kết quả research_v1.

Web có **Hybrid (Dense + BM25 + ưu tiên từ hiếm)** và **Dense** để đối chiếu. Corpus demo_v3 thêm nguồn Wikipedia cho Fivestones và Antigonus, tổng **12.602 chunks**. Vì nguồn được chọn sau khi quan sát lỗi test, đây là dữ liệu demo, không dùng báo điểm benchmark. Khi thiếu artifact, web dùng verified_v2 hoặc corpus gốc. Xem [bản cải tiến và cách tái lập](IMPROVEMENTS_V3.md).

Đã chấm lại 120 dự đoán của run verified_v2 bằng **canonical answer**, loại alias chưa duyệt khỏi phép chấm mặc định:

| Hệ | EM | F1 |
| --- | ---: | ---: |
| T5 không context | 0,83% | 1,93% |
| Dense + DistilBERT extractive | 26,67% | 36,67% |
| Dense + T5 RAG | **34,17%** | **39,76%** |

Đây là chấm lại dự đoán đã lưu, không phải lần chạy mô hình mới. Canonical-only có thể bỏ sót biến thể đúng; phiếu duyệt 1.626 alias nằm trong [artifact canonical](../../results/canonical_rescore_v2). Điểm alias cũ 42,50% EM không dùng làm kết quả cuối. Doc Recall@5 lịch sử của Dense/Hybrid RRF vẫn là 92,50%; Hybrid mới trên web chưa được đánh giá toàn tập.

**Tại thời điểm 31/10/2026: 43/43 tests đạt**. Phiếu faithfulness/citation 20 câu vẫn ghi rõ agent_self_review. [Báo cáo run gốc](EVALUATION_RESULTS_V2.md) giữ để đối chiếu; các bảng 3.940 chunks bên dưới là kết quả lịch sử, không so sánh trực tiếp với v2/v3.

## 🎯 Câu hỏi Nghiên cứu (Research Questions)

Đồ án được thiết kế xoay quanh 3 câu hỏi nghiên cứu cốt lõi:
- **RQ1 (RAG vs Baselines & Faithfulness):** Mô hình RAG cải thiện Exact Match (EM) và F1 bao nhiêu so với Generator-only (không context) và Extractive QA? Tỷ lệ bám sát ngữ cảnh (Faithfulness) đạt mức nào trong việc kiểm soát ảo giác (hallucination)?
- **RQ2 (Sparse vs Dense vs Hybrid Retrieval):** Các phương pháp truy hồi truyền thống (BM25, TF-IDF, Word2Vec, FastText) đạt hiệu năng Recall@k và MRR ra sao so với Dense Retrieval (FAISS)? Kỹ thuật Hybrid Search (BM25 + FAISS via RRF) có giúp vượt trội hơn Dense-only không?
- **RQ3 (Ablation Trade-off):** Sự thay đổi về kích thước chunk (`chunk_size`) và số lượng đoạn truy hồi (`top_k`) ảnh hưởng như thế nào đến độ bao phủ thông tin (coverage) và độ nhiễu ngữ cảnh (context noise)?

---

## 📖 Tổng quan Công trình Liên quan (Related Work)

1. **Lewis et al. (NeurIPS 2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks***: Công trình nền tảng đề xuất kiến trúc RAG end-to-end kết hợp mô hình sinh seq2seq với bộ truy hồi dense non-parametric (DPR + FAISS), chứng minh sự vượt trội trên các tác vụ Open-domain QA.
2. **Karpukhin et al. (EMNLP 2020) — *Dense Passage Retrieval for Open-Domain Question Answering (DPR)***: Đặt nền móng cho dense retrieval hiện đại, sử dụng bi-encoder với hàm loss contrastive để mã hóa câu hỏi và passage vào không gian vector chung, vượt qua BM25 trong việc nắm bắt tương đồng ngữ nghĩa.
3. **Guu et al. (ICML 2020) — *REALM: Retrieval-Augmented Language Model Pre-training***: Đề xuất phương pháp pre-train mô hình ngôn ngữ có sự hỗ trợ của retrieval module từ đầu (unsupervised masked language modeling kết hợp neural retriever), minh chứng giá trị của external knowledge.
4. **Izacard & Grave (EACL 2021) — *Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering (Fusion-in-Decoder - FiD)***: Mở rộng khả năng xử lý nhiều passage (lên đến 100 passages) bằng cách mã hóa độc lập từng đoạn ở Encoder và dung hợp (fuse) toàn bộ thông tin tại Decoder, đạt SOTA trên Natural Questions và TriviaQA.
5. **Gao et al. (arXiv 2023 / 2024 Survey) — *Retrieval-Augmented Generation for Large Language Models: A Survey***: Tổng hợp và phân loại hệ sinh thái RAG thành 3 thế hệ (Naive RAG, Advanced RAG, Modular RAG); định hình các kỹ thuật cải tiến như Hybrid Search, Query Rewriting, Reranking và các bộ chỉ số đánh giá chuyên sâu (Faithfulness, Answer Relevance).

---

## 📐 Kiến trúc Pipeline research_v1

```
                        ┌───────────────────────────────────────────────────┐
                        │           RAG Pipeline (Đề tài 20)               │
                        └───────────────────────────────────────────────────┘

  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐    ┌───────────────┐
  │  TriviaQA    │───▶│  Chunking        │───▶│  Embedding         │───▶│  FAISS Index  │
  │  3.157 docs  │    │  200 words/chunk  │    │  all-MiniLM-L6-v2  │    │  IndexFlatIP  │
  │  (Bước 0)    │    │  40 word overlap  │    │  (Bước 2)          │    │  (Bước 2)     │
  └──────────────┘    └──────────────────┘    └────────────────────┘    └───────┬───────┘
                                                                               │
                              ┌─────────────────────────────────────────┐      │
                              │                                         ▼      ▼
  ┌──────────────┐    ┌───────┴──────────┐    ┌────────────────────┐  ┌─────────────────┐
  │  Câu hỏi     │───▶│  Query Encoder   │───▶│  Dense Retrieval   │──│  Top-K Passages │
  │  (User)      │    │  all-MiniLM-L6-v2│    │  cosine similarity │  │  (context)      │
  └──────────────┘    └──────────────────┘    └────────────────────┘  └────────┬────────┘
                              │                                                │
                              │    ┌────────────────────┐                      │
                              └───▶│  BM25 Sparse       │──── RRF Fusion ─────┘
                                   │  (Hybrid, Bước 6)  │     (cải tiến)
                                   └────────────────────┘

                                                                     ┌────────┴────────┐
                                                                     │                  │
                                                               ┌─────▼──────┐    ┌──────▼──────┐
                                                               │ T5-base    │    │ DistilBERT  │
                                                               │ Generator  │    │ Extractive  │
                                                               │ (fine-tuned│    │ QA Baseline │
                                                               │  Bước 3)   │    │ (Hệ 2)     │
                                                               └─────┬──────┘    └──────┬──────┘
                                                                     │                  │
                                                               ┌─────▼──────────────────▼──────┐
                                                               │     So sánh 3 hệ thống       │
                                                               │  1. Generator-only (no ctx)   │
                                                               │  2. Retrieval-only (extract)  │
                                                               │  3. RAG đầy đủ (ret + gen)    │
                                                               └───────────────────────────────┘
```

### 3 Hệ thống so sánh (theo yêu cầu đề bài)

| Hệ thống | Mô tả | Model |
|-----------|-------|-------|
| **Generator-only** | Fine-tune riêng với context rỗng, chỉ nhận câu hỏi lúc suy luận | T5-base closed-book |
| **Retrieval-only** | Trích xuất span từ passage (extractive baseline) | DistilBERT-SQuAD |
| **RAG đầy đủ** | Retrieval + Generation — pipeline ứng dụng ý tưởng RAG | FAISS + T5-base |

---

## 📊 Kết quả lịch sử — corpus 3.940 chunks

Các bảng sau là số liệu lịch sử trên corpus cũ. Kết quả research_v1 nằm ở đầu README; verified_v2 được giữ trong EVALUATION_RESULTS_V2. Không dùng các bảng cũ để mô tả bản mới. Web lịch sử dùng num_beams=4 và ngưỡng Cosine 0,25; script thí nghiệm dùng cấu hình riêng. Faithfulness ở đây chỉ là tỷ lệ token đáp án xuất hiện trong context. Các chỉ số citation và nhãn lỗi dưới đây là proxy theo tài liệu/chuỗi đáp án, chưa phải đánh giá ngữ nghĩa do người chấm.

### 1. So sánh 3 Hệ thống (End-to-End QA, trên toàn bộ 120 câu test)

| Hệ thống | Exact Match (EM) | F1 Score | Token-overlap proxy |
|---|:---:|:---:|:---:|
| **Generator-only** (không context) | 0.008 | 0.019 | N/A |
| **Retrieval-only** (DistilBERT extractive) | 0.250 | 0.326 | 0.704 |
| **RAG đầy đủ** (FAISS + T5 fine-tuned) | **0.325** | **0.376** | **0.858** |

> **Nhận xét cốt lõi:**
> - RAG đạt **F1=0.376** và **EM=0.325**, cao hơn hai baseline trong run lịch sử này; bảng không có ước lượng bất định cho chênh lệch.
> - **Token-overlap proxy đạt 0.858 (85.8%)** theo heuristic lịch sử; cần đánh giá ngữ nghĩa bổ sung để kết luận về faithfulness.

### 2. Phân rã lỗi & Độ chính xác trích dẫn (RAG đầy đủ)

| Chỉ số đánh giá | Kết quả | Ý nghĩa khoa học |
|---|:---:|---|
| **Doc-hit proxy lịch sử** | **0.908 (90.8%)** | Có nguồn được gán trong kết quả truy hồi; không chứng minh nguồn hỗ trợ đáp án |
| **Answer-string passage proxy** | **0.617 (61.7%)** | Tỷ lệ passage chứa chuỗi đáp án; không tương đương citation correctness ngữ nghĩa |
| **Không trúng tài liệu được gán** | **9.2%** (11/120) | Không có tài liệu liên quan theo qrels trong top-k |
| **EM sai dù trúng tài liệu** | **58.3%** (70/120) | Nhóm lỗi theo proxy; chưa xác lập lỗi do generation nếu chưa xem context thực dùng |
| **Hoàn toàn chính xác (Correct)** | **32.5%** (39/120) | Trả lời chính xác hoàn toàn (Exact Match) |

### 3. Đánh giá Mô hình Retrieval: Cổ điển (Tuần Baseline) vs Dense & Hybrid

Thực nghiệm đo lường trên toàn bộ 120 câu hỏi test (`data/test_qa.jsonl`) đối chiếu với 3,940 chunks trong corpus:

| Nhóm mô hình | Phương pháp Retrieval | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|:---:|:---:|:---:|:---:|
| **Classical Sparse** | **BM25 (Okapi)** | 0.8583 | 0.8917 | 0.9250 | **0.8112** |
| | **TF-IDF Vectorizer** | 0.8000 | 0.8583 | 0.8917 | 0.7753 |
| **Classical Embedding** | **Word2Vec (CBOW Avg Pooling)** | 0.0583 | 0.1000 | 0.1167 | 0.0499 |
| | **FastText (Subwords Avg Pooling)** | 0.0500 | 0.0583 | 0.1000 | 0.0519 |
| **Pre-trained Dense** | **Dense FAISS (all-MiniLM-L6-v2)** | 0.8167 | 0.8800 | 0.9417 | 0.6950 |
| **Ensemble Hybrid** | **Hybrid Search (BM25 + FAISS via RRF)** | **0.8583** | **0.9100** | **0.9583** | 0.7320 |

> **Quan sát trong run lịch sử:** BM25/TF-IDF đạt MRR 0.8112/0.7753; Word2Vec/FastText có Recall@5 khoảng 6–10%; Dense đạt Recall@10 0.9417 so với BM25 0.9250. Hybrid đạt Recall@5 0.9100 và Recall@10 0.9583. Các giải thích bằng tên riêng, corpus nhỏ hoặc biểu diễn ngữ nghĩa là giả thuyết; bảng này chưa có ablation để xác nhận nguyên nhân.

### 4. Ablation Study: chunk_size × top_k

| chunk_size | top_k | Recall (Dense) | Recall (Hybrid) |
|:----------:|:-----:|:--------------:|:---------------:|
| 100 | 3 | 0.880 | 0.870 |
| 100 | 5 | 0.880 | 0.900 |
| 100 | 10 | 0.910 | **0.940** |
| 200 | 3 | 0.860 | 0.840 |
| 200 | 5 | 0.880 | 0.910 |
| 200 | 10 | 0.910 | **0.940** |
| 300 | 3 | 0.880 | 0.840 |
| 300 | 5 | 0.890 | 0.910 |
| 300 | 10 | 0.910 | **0.950** |

> **Quan sát giới hạn ở bảng lịch sử:** Hybrid có Recall cao hơn Dense ở các cấu hình top_k ≥ 5 đã đo; chunk_size=300 + top_k=10 đạt 0.950. Đây không phải ablation research_v1 và chưa có kiểm định độ ổn định.

---

## 🛠️ Cài đặt & chạy pipeline lịch sử

Các lệnh trong mục này dành cho pipeline/corpus cũ. Vận hành research_v1 theo [HOC_VA_VAN_HANH.md](../HOC_VA_VAN_HANH.md) và [notebook nghiên cứu](../../notebooks/research_reproduce_colab.ipynb).

### Yêu cầu phần cứng
- GPU: RTX 3050 6GB trở lên (hoặc CPU, chạy chậm hơn)
- RAM: 16GB+
- Disk: ~5GB cho model + data

### 0. Cài đặt môi trường

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
python -m pip install -r requirements/ml.txt
```

Kiểm tra CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 1. Tải dataset

```bash
python -m experiments.download_dataset
```
Mặc định tải 800 câu từ TriviaQA (`rc.wikipedia`, split validation). Đổi `DATASET_CHOICE = "natural_questions"` nếu muốn dùng NQ. → Tạo `data/raw_documents.json`.

### 2. Chunking

```bash
python -m experiments.prepare_chunks
```
→ Tạo `data/corpus_chunks.jsonl` và `data/qa_pairs.jsonl`.

### 3. Tách train/test

```bash
python -m experiments.split_train_test
```
→ Tạo `data/test_qa.jsonl` (evaluate) và `data/training_pairs.jsonl` (fine-tune). Seed=42, test 15%.

### 4. Xây FAISS index

```bash
python -m experiments.build_index
```
→ Tạo `data/faiss.index` + `data/chunk_metadata.jsonl`. Chạy 1 lần.

### 5. Fine-tune generator

```bash
python -m experiments.finetune_generator
```
→ Model lưu tại `generator_finetuned/`. T5-base, batch_size=4 + fp16 + gradient_accumulation=4.

### 6. Đánh giá toàn diện (Evaluation)

```bash
# Đánh giá toàn bộ 120 câu test
python -m experiments.evaluate

# Hoặc đánh giá nhanh trên tập mẫu (ví dụ 10 câu)
python -m experiments.evaluate --samples 10
```
→ Xuất các chỉ số và proxy của evaluator lịch sử:
- **Recall@k**: Đo lường độc lập cho bộ truy hồi (retriever)
- **EM (Exact Match) & F1**: So sánh cả 3 hệ thống (`generator_only`, `retrieval_only`, `rag_full`)
- **Token-overlap proxy**: Tỷ lệ token đáp án xuất hiện trong context; chưa đánh giá quan hệ ngữ nghĩa hoặc ảo giác.
- **Citation proxies**: Trùng nguồn được gán và chứa chuỗi đáp án; chưa thay đánh giá citation correctness trên mẫu.
- **Phân nhóm lỗi theo proxy**: Phân nhóm bằng tài liệu/chuỗi đáp án, chưa xác lập nguyên nhân retrieval hoặc generation về ngữ nghĩa.

### 7. Cải tiến — Hybrid retrieval

```bash
python -m experiments.hybrid_retrieval
```
BM25 + Dense kết hợp bằng Reciprocal Rank Fusion (RRF, k=60).

### 8. Ablation study

```bash
python -m experiments.ablation
```
3 chunk_size × 3 top_k, so sánh Dense vs Hybrid. Kết quả → `results/ablation_results.csv`.

### 9. Thực nghiệm Retrieval Baselines (Tuần xây dựng baseline)

```bash
python -m experiments.baseline_retrieval
```
So sánh 4 phương pháp retrieval truyền thống (BM25, TF-IDF, Word2Vec CBOW, FastText subwords) trên tập test 120 câu hỏi, đo lường Recall@3, Recall@5, Recall@10, MRR → Lưu kết quả tại `results/baseline_results.json`.

---

## 🌐 Web demo lịch sử (`/legacy`)

Dự án bao gồm web demo **RAG Lab** với giao diện tiếng Việt, hỗ trợ **so sánh trực tiếp giữa Hệ thống RAG (FAISS Dense + fine-tuned T5-base) và Baseline BM25 (truy hồi từ khóa + trích câu)** trên tập TriviaQA.

```powershell
python app.py
```
Mở **http://127.0.0.1:8001/legacy** cho giao diện lịch sử này. Demo research_v1 dùng `/research` theo phần đầu README. Giao diện lịch sử hiển thị:
- **Bảng 3 khối so sánh kết quả trực diện**:
  1. **Hệ RAG đầy đủ (T5 + FAISS)**: Câu trả lời do T5 sinh ra dựa vào ngữ cảnh, kèm câu trả lời của cùng checkpoint T5 khi không có context (Generator-only); câu này có thể đúng hoặc sai.
  2. **Hệ Baseline BM25**: Câu trích nguyên văn từ đoạn văn theo tần suất từ khóa.
  3. **Đáp án tham chiếu (Ground Truth)**: Đối chiếu với đáp án chuẩn trong dataset TriviaQA.
- **So sánh tài liệu nguồn song song**: Tab xem Top-K nguồn cho RAG (Dense hoặc Hybrid) và BM25; Cosine không phải xác suất đúng.
- **Phân rã kỹ thuật**: Bảng điểm BM25 chi tiết (TF, IDF, contribution) và độ trễ từng giai đoạn.
- **Tự động tận dụng GPU (CUDA)**: Tự động chạy trên GPU (RTX 3050) nếu có hoặc chuyển sang CPU một cách an toàn.

Kiểm tra nhanh: cài `requirements/test.txt`, sau đó chạy `python -m unittest discover -s tests -v`. Lệnh kiểm thử engine cũ `python -m unittest test_engine -v` vẫn hoạt động.

Kiểm tra trước demo: `python -m scripts.check_demo`. Kiểm tra model local thực tế: `python -m scripts.check_demo --neural`. Xem [hướng dẫn demo](DEMO_READINESS.md) để chuẩn bị cache khi cài trên máy mới.

### Deploy lên Hugging Face Space

1. Upload model lên HF Hub:
   ```bash
   python -m scripts.upload_to_hf
   ```
2. Cấu hình Hugging Face Space:
   - Set `HF_MODEL_REPO_ID` environment variable hoặc sửa trực tiếp trong file
3. Tạo HF Space (Gradio), upload `deployment/huggingface/app.py` thành `app.py` và `deployment/huggingface/requirements.txt` thành `requirements.txt`. Đây là hai tệp độc lập để triển khai; không dùng launcher tương thích ở root.

---

## 📁 Cấu trúc dự án

```text
RAG/
├── configs/research_v1.json    # Cấu hình dữ liệu, retrieval, train và generation
├── experiments/
│   ├── research_data.py       # QA splits → documents/chunks → embedding/index
│   ├── research_pipeline.py   # Retrieval, context thực dùng, generation/extraction
│   ├── research_train.py      # Hai T5 riêng, checkpoint/loss/provenance
│   ├── research_evaluate.py   # Ba hệ + Hybrid/ablation, metrics và predictions
│   ├── metrics.py             # TriviaQA normalization và answer policies
│   └── legacy_metrics.py      # Bảo toàn evaluator lịch sử
├── ragqa/
│   ├── research_web.py        # API/demo research_v1 ba hệ
│   ├── web.py                 # FastAPI và route tương thích
│   ├── rag_engine.py          # Các khối neural dùng chung
│   └── static/                # Giao diện research và legacy
├── scripts/                   # Preflight, báo cáo, review, replay audit, đóng gói
├── data/research_v1/           # Corpus/index/splits; manifest và replay descriptor
├── checkpoints/research_v1/    # Hai run closed_book và rag; artifact cung cấp riêng
├── results/research_*/         # Predictions nén, metrics, manifests, source snapshots
├── tests/                     # Kiểm thử hợp đồng, hồi quy và provenance
├── notebooks/                 # Workflow Colab
├── requirements/              # demo, test, research và môi trường lịch sử
├── docs/                      # Protocol, kết quả, casebook, hướng dẫn học/demo
├── deployment/huggingface/      # Gradio app và requirements độc lập
├── app.py                     # Điểm chạy web
├── 00_*.py … 08_*.py           # Điểm gọi tương thích của pipeline lịch sử
└── outputs/                   # Gói release và log tạm, Git bỏ qua
```

Mã triển khai chính nằm trong các package; các tệp Python ở root chỉ giữ tương thích với lệnh cũ. Chỉnh sửa web trong `ragqa/`, thí nghiệm trong `experiments/`. Xem [bản đồ kiến trúc và bảng lệnh](../ARCHITECTURE.md).

`python -m experiments --list` liệt kê các bước mà không tải thư viện ML. Các lệnh `research_*` giữ riêng protocol mới; các module cũ được giữ để đọc lịch sử. Run nghiên cứu hoàn chỉnh được bảo vệ khỏi ghi đè. Không cần tạo lại corpus hoặc train lại để mở web khi đã có đúng artifact.

---

## 🔬 Chi tiết kỹ thuật

| Component | Specification |
|-----------|--------------|
| **Embedding Model** | `all-MiniLM-L6-v2` (22M params, 384-dim) |
| **Generator** | Hai `T5-base` fine-tune riêng; 1.865 QA mỗi hệ, 3 epochs, 351 steps/run |
| **Extractive QA** | `distilbert-base-cased-distilled-squad` |
| **Index** | FAISS `IndexFlatIP` (exact cosine search) |
| **Hybrid** | BM25Okapi + Dense, Reciprocal Rank Fusion (k=60) |
| **Training research_v1 local** | BF16, gradient checkpointing, effective batch=16; pilot FP16 thất bại được giữ trong provenance |
| **Seed** | 42 (torch, numpy, random); môi trường và artifact lưu trong manifest |
| **Dataset research_v1** | TriviaQA official train subset 2.000 ứng viên; validation subset chia 200 dev/300 test; 3.157 docs, 90.667 chunks |

---

## 📚 Tham khảo

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
- Guu, K. et al. (2020). *REALM: Retrieval-Augmented Language Model Pre-training.* ICML 2020. [arXiv:2002.08909](https://arxiv.org/abs/2002.08909)
- Izacard, G. & Grave, E. (2021). *Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering.* EACL 2021. [arXiv:2007.01282](https://arxiv.org/abs/2007.01282)
- Gao, Y. et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey.* [arXiv:2312.10997](https://arxiv.org/abs/2312.10997)
- Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
- Cormack, G. et al. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009.

---

## 📝 License

Dataset: TriviaQA (câu hỏi Apache 2.0; evidence từ Wikipedia CC BY-SA 3.0).
Code: MIT License.
