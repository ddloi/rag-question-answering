# Kết quả research_v1 · test · 300 QA

Số liệu được tính lại từ dự đoán đã lưu và đối chiếu với metrics.json (sai số tối đa 1e-12).

Dataset manifest SHA256: `b2abb8c2f7c17064d4c6656aad00c8f1e01f419b404323664c13d445e7563ffa`.
Closed-book model: `60ca4725f079cd65fb5607a23c46f9aac858a1f47a58b0c58e466e2281833f10`; RAG model: `4909c3007546c81e3523429b6e64cedca816ba3f0281348124a5a2444d5e693e`.

| Hệ | Official EM | Official F1 | Canonical EM | Canonical F1 |
| --- | ---: | ---: | ---: | ---: |
| T5 closed-book (fine-tune riêng) | 8.33% | 12.82% | 6.33% | 9.41% |
| Dense + DistilBERT extractive | 53.33% | 60.17% | 47.33% | 55.47% |
| Dense + T5 RAG | 57.67% | 62.68% | 53.67% | 60.39% |
| Hybrid RRF + cùng T5 RAG | 56.33% | 61.36% | 52.67% | 59.20% |
| Ablation: checkpoint RAG bỏ context | 6.67% | 12.26% | 3.33% | 5.63% |

## Retrieval

| Phương pháp | Doc hit@5 | Doc recall@5 | Official answer recall@5 | Canonical answer recall@5 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 94.00% | 79.33% | 88.33% | 84.00% | 89.18% |
| bm25 | 93.67% | 78.86% | 84.67% | 80.67% | 85.15% |
| hybrid | 95.00% | 80.97% | 88.67% | 84.67% | 90.33% |

Doc hit khác doc recall khi một câu có nhiều gold sources. Answer recall chỉ dò chuỗi; nguồn qrels là document-level. Đủ k=1/3/5/10 trong metrics.json.

## Hybrid so với Dense

- official EM: chênh -1.33 điểm phần trăm; paired bootstrap 95% CI [-6.00, +3.33].
- official F1: chênh -1.32 điểm phần trăm; paired bootstrap 95% CI [-5.65, +3.29].
- canonical EM: chênh -1.00 điểm phần trăm; paired bootstrap 95% CI [-5.67, +3.67].
- canonical F1: chênh -1.19 điểm phần trăm; paired bootstrap 95% CI [-5.37, +3.32].

CI theo câu hỏi; chưa thể hiện phương sai nhiều training seed. Không diễn giải Hybrid luôn tốt hơn.

## Phân rã lỗi lexical

| Nhánh | Nhóm | Số câu |
| --- | --- | ---: |
| rag_full | context_selection_or_truncation | 34 |
| rag_full | retrieval_answer_string_missing | 42 |
| rag_full | correct | 161 |
| rag_full | generation_wrong_with_answer_string_in_context | 60 |
| rag_full | corpus_answer_string_missing | 3 |
| rag_hybrid | generation_wrong_with_answer_string_in_context | 64 |
| rag_hybrid | context_selection_or_truncation | 34 |
| rag_hybrid | retrieval_answer_string_missing | 41 |
| rag_hybrid | correct | 158 |
| rag_hybrid | corpus_answer_string_missing | 3 |

## Ví dụ phục vụ báo cáo

Các nhóm dưới là proxy, cần đọc context để xác nhận ngữ nghĩa. Mỗi nhóm lấy tối đa hai câu đầu theo thứ tự split; không đại diện tỷ lệ toàn bộ dữ liệu.

| Nhóm | Question ID | Câu hỏi | Canonical | Dense RAG | Hybrid RAG |
| --- | --- | --- | --- | --- | --- |
| context_selection_or_truncation | qb_9571 | Which English city is known as ‘The Lace City’? | Nottingham | Bath | Oxford |
| context_selection_or_truncation | qb_9289 | What is the title of the first novel to be published in the ‘Chronicles of Narnia’ series by C S Lewis? | The Lion, The Witch and The Wardrobe | The Horse and His Boy | The Horse and His Boy |
| retrieval_answer_string_missing | qb_3790 | Ommatophobia is the irrational fear of which part of the body? | Eyes | Arachnophobia | bowel |
| hybrid_rescue | qb_7586 | Who is the narrator on the UK television show The Only Way Is Essex? | Denise van Outen | Amy Childs | Denise van Outen |
| correct | qb_7790 | Late English criminal Bruce Reynolds masterminded which infamous robbery, which he later referred to as his ‘Sistine Chapel ceiling’ | Great Train Robbery | Great Train Robbery | Great Train Robbery |
| correct | qb_4064 | In which US state is the world’s largest volcano? | Hawaii | Hawaii | Ohio |
| hybrid_regression | qb_4064 | In which US state is the world’s largest volcano? | Hawaii | Hawaii | Ohio |
| generation_wrong_with_answer_string_in_context | qb_9094 | In 1968, ‘The Troubles’ began in which European country between elements of the nationalist community and the unionist community? | Northern Ireland | Ireland | Ireland |
| generation_wrong_with_answer_string_in_context | qb_8947 | The French town of Bordeaux lies on which river? | The Garonne River | Garonne | Garonne |
| official_canonical_disagreement | qb_8947 | The French town of Bordeaux lies on which river? | The Garonne River | Garonne | Garonne |
| retrieval_answer_string_missing | qb_3644 | Which British-American actress died on 23rd March 2011, aged 79? | Elizabeth Taylor | Naomie Harris | Naomie Harris |
| official_canonical_disagreement | qb_7768 | Philippa of Hainault was the wife of which English monarch? | King Edward III | Edward III | Edward III |
| hybrid_regression | qb_9264 | The Kingdom of Bahrain lies in which body of water? | Persian Gulf | Persian Gulf | Aquifer |
| hybrid_rescue | qb_3880 | Which US actor appeared in the films ‘Ironweed’, ‘The Pledge’ and ‘Easy Rider’? | Jack Nicholson | Peter Fonda | Jack Nicholson |
| corpus_answer_string_missing | qb_3803 | Maria Dickin was the founder of which British animal charity in 1917? | Peoples Dispensary for Sick Animals | People's Dispensary for Sick Animals | People's Dispensary for Sick Animals |
| corpus_answer_string_missing | qb_6828 | What is the young of a hawk called? | Eyas | Buteo | Buteo |

Tra question_id trong predictions.jsonl.gz để xem passage, source IDs và context thực dùng. cases.csv chứa toàn bộ câu để lọc đúng/sai/Hybrid tốt hơn hoặc kém hơn.

## Giới hạn và phần cần bổ sung

- Corpus nhỏ, cùng miền; shared documents có thể xuất hiện giữa các split, pretraining contamination chưa kiểm soát.
- MiniLM cố định; T5 context nối chuỗi, không tái lập end-to-end RAG-Sequence/RAG-Token.
- Một seed, hai T5 cùng QA và budget, extractive dùng checkpoint SQuAD sẵn có.
- Faithfulness/citation người chấm: xem phiếu và summary riêng. Phiếu trống nghĩa NOT RUN; điểm EM/F1 không thay được nhãn ngữ nghĩa.
- Chạy Colab thực tế: không được suy ra từ việc notebook qua kiểm tra cú pháp.
- Split dev dùng phát triển và ví dụ demo; chỉ test đã khóa protocol mới là held-out evaluation.
