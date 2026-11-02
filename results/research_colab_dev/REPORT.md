# Kết quả research_v1 · dev · 200 QA

Số liệu được tính lại từ dự đoán đã lưu và đối chiếu với metrics.json (sai số tối đa 1e-12).

Dataset manifest SHA256: `b2abb8c2f7c17064d4c6656aad00c8f1e01f419b404323664c13d445e7563ffa`.
Closed-book model: `60ca4725f079cd65fb5607a23c46f9aac858a1f47a58b0c58e466e2281833f10`; RAG model: `4909c3007546c81e3523429b6e64cedca816ba3f0281348124a5a2444d5e693e`.

| Hệ | Official EM | Official F1 | Canonical EM | Canonical F1 |
| --- | ---: | ---: | ---: | ---: |
| T5 closed-book (fine-tune riêng) | 5.50% | 10.42% | 4.50% | 6.18% |
| Dense + DistilBERT extractive | 45.50% | 51.60% | 38.00% | 46.39% |
| Dense + T5 RAG | 51.50% | 57.80% | 47.50% | 54.42% |
| Hybrid RRF + cùng T5 RAG | 54.50% | 59.72% | 50.50% | 57.21% |
| Ablation: checkpoint RAG bỏ context | 7.00% | 16.27% | 4.50% | 7.45% |

## Retrieval

| Phương pháp | Doc hit@5 | Doc recall@5 | Official answer recall@5 | Canonical answer recall@5 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 93.00% | 76.55% | 82.50% | 76.50% | 87.78% |
| bm25 | 91.50% | 73.90% | 80.50% | 75.50% | 83.94% |
| hybrid | 95.00% | 80.57% | 86.50% | 80.50% | 89.01% |

Doc hit khác doc recall khi một câu có nhiều gold sources. Answer recall chỉ dò chuỗi; nguồn qrels là document-level. Đủ k=1/3/5/10 trong metrics.json.

## Hybrid so với Dense

- official EM: chênh +3.00 điểm phần trăm; paired bootstrap 95% CI [-2.00, +8.00].
- official F1: chênh +1.92 điểm phần trăm; paired bootstrap 95% CI [-3.02, +6.59].
- canonical EM: chênh +3.00 điểm phần trăm; paired bootstrap 95% CI [-2.00, +8.00].
- canonical F1: chênh +2.78 điểm phần trăm; paired bootstrap 95% CI [-2.23, +7.77].

CI theo câu hỏi; chưa thể hiện phương sai nhiều training seed. Không diễn giải Hybrid luôn tốt hơn.

## Phân rã lỗi lexical

| Nhánh | Nhóm | Số câu |
| --- | --- | ---: |
| rag_full | context_selection_or_truncation | 18 |
| rag_full | generation_wrong_with_answer_string_in_context | 43 |
| rag_full | retrieval_answer_string_missing | 40 |
| rag_full | correct | 95 |
| rag_full | corpus_answer_string_missing | 4 |
| rag_hybrid | context_selection_or_truncation | 22 |
| rag_hybrid | generation_wrong_with_answer_string_in_context | 41 |
| rag_hybrid | retrieval_answer_string_missing | 32 |
| rag_hybrid | correct | 101 |
| rag_hybrid | corpus_answer_string_missing | 4 |

## Ví dụ phục vụ báo cáo

Các nhóm dưới là proxy, cần đọc context để xác nhận ngữ nghĩa. Mỗi nhóm lấy tối đa hai câu đầu theo thứ tự split; không đại diện tỷ lệ toàn bộ dữ liệu.

| Nhóm | Question ID | Câu hỏi | Canonical | Dense RAG | Hybrid RAG |
| --- | --- | --- | --- | --- | --- |
| context_selection_or_truncation | qb_8485 | A sterlet is what type of creature? | Fish | Sturgeon | Sturgeon |
| generation_wrong_with_answer_string_in_context | qb_6193 | Tennis player Martina Hingis was born in which European country? | Switzerland | Czech Republic | Czech Republic |
| retrieval_answer_string_missing | qb_9451 | The Goya Gate is located at which European museum? | Prado Museum | Madrid | The De Young Museum |
| generation_wrong_with_answer_string_in_context | qb_7781 | In our solar system, the sun is composed of mostly which gas? | Hydrogen | Helium | Hydrogen |
| hybrid_rescue | qb_7781 | In our solar system, the sun is composed of mostly which gas? | Hydrogen | Helium | Hydrogen |
| retrieval_answer_string_missing | qb_7213 | Which land mammal has the longest tail? | Giraffe | Cowfish | Hippo |
| correct | qb_4336 | Gentoo, Adelie and Macaroni are all species of which bird? | Penguin | Penguin | Penguin |
| context_selection_or_truncation | qb_8367 | Which late American musician, who pioneered the styles of rock and roll and rockabilly, was born Vincent Eugene Craddock in 1935? | Gene Vincent | Johnny O'Keefe | Gene Vincent |
| hybrid_rescue | qb_8367 | Which late American musician, who pioneered the styles of rock and roll and rockabilly, was born Vincent Eugene Craddock in 1935? | Gene Vincent | Johnny O'Keefe | Gene Vincent |
| correct | qb_7077 | Neal Foulds, John Dunning and Mark Williams are associated with which sport? | Snooker | Snooker | Snooker |
| official_canonical_disagreement | qb_5139 | Who replaced Iraqi President Ahmed Hassan al-Bakr in July 1979? | Saddam Hussein | Saddam | Vice President |
| hybrid_regression | qb_7076 | Which religion uses a domed structure called a Stupa as a shrine? | Buddhism | Buddhism | Christianity |
| hybrid_regression | qb_5564 | Who is the host of the BBC television show QI? | Stephen Fry | Stephen Fry | John Lloyd |
| official_canonical_disagreement | qb_7680 | What is the title of the 18th Century novel by Lawrence Sterne, which was published in nine volumes? | (The Life and Opinions of) Tristram Shandy (Gentleman) | The Life and Opinions of Tristram Shandy | Tristram Shandy |
| corpus_answer_string_missing | qb_4157 | What is the art of hand-making cricket bats called? | Podshaving | Knocking | Knocking |
| corpus_answer_string_missing | qb_5669 | Barney the Owl is a mascot of which English football club? | Sheffield Wednesday FC | Sheffield United | Sheffield United |

Tra question_id trong predictions.jsonl.gz để xem passage, source IDs và context thực dùng. cases.csv chứa toàn bộ câu để lọc đúng/sai/Hybrid tốt hơn hoặc kém hơn.

## Giới hạn và phần cần bổ sung

- Corpus nhỏ, cùng miền; shared documents có thể xuất hiện giữa các split, pretraining contamination chưa kiểm soát.
- MiniLM cố định; T5 context nối chuỗi, không tái lập end-to-end RAG-Sequence/RAG-Token.
- Một seed, hai T5 cùng QA và budget, extractive dùng checkpoint SQuAD sẵn có.
- Faithfulness/citation người chấm: xem phiếu và summary riêng. Phiếu trống nghĩa NOT RUN; điểm EM/F1 không thay được nhãn ngữ nghĩa.
- Chạy Colab thực tế: không được suy ra từ việc notebook qua kiểm tra cú pháp.
- Split dev dùng phát triển và ví dụ demo; chỉ test đã khóa protocol mới là held-out evaluation.
