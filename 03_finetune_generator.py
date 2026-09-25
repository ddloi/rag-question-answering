"""
BƯỚC 4: Fine-tune generator (T5-base hoặc BART-base)
-------------------------------------------------------
Fine-tune trên định dạng: input = "question: {câu hỏi} context: {passage}"
                           target = câu trả lời

Đây chính là phần "generation" của RAG — mô hình học cách sinh câu trả lời
DỰA VÀO context được cung cấp, thay vì chỉ dựa vào parametric knowledge.

Cấu hình dưới đây đã tối ưu cho RTX 3050 6GB:
- fp16 (mixed precision) bật sẵn
- batch_size nhỏ + gradient_accumulation để có effective batch lớn hơn
- gradient_checkpointing để tiết kiệm thêm VRAM nếu cần
"""

import json
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# ---------------------------------------------------------------
# CẤU HÌNH — đổi MODEL_NAME sang "facebook/bart-base" nếu muốn dùng BART
# ---------------------------------------------------------------
MODEL_NAME = "t5-base"
DATA_DIR = Path("data")
OUTPUT_DIR = "generator_finetuned"
MAX_INPUT_LEN = 384
MAX_TARGET_LEN = 64

# NOTE: bước này cần "training_pairs.jsonl" chứa các mẫu đã ghép sẵn
# (câu hỏi + passage đúng + câu trả lời) — thường lấy từ gold passage
# trong qa_pairs.jsonl kết hợp corpus_chunks.jsonl. Nếu bạn có dataset
# gốc (vd Natural Questions) đã có sẵn positive passage, dùng luôn.


def load_training_data():
    records = []
    with open(DATA_DIR / "training_pairs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def preprocess_function(examples, tokenizer):
    inputs = [
        f"question: {q} context: {c}"
        for q, c in zip(examples["question"], examples["context"])
    ]
    targets = examples["answer"]

    model_inputs = tokenizer(
        inputs, max_length=MAX_INPUT_LEN, truncation=True, padding=False
    )
    labels = tokenizer(
        text_target=targets, max_length=MAX_TARGET_LEN, truncation=True, padding=False
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    dataset = load_training_data()
    dataset = dataset.train_test_split(test_size=0.1, seed=42)  # ghi rõ seed để tái lập

    tokenized = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,       # 6GB VRAM: 4 an toàn cho t5-base
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,        # effective batch = 4*4 = 16
        num_train_epochs=3,
        learning_rate=3e-4,
        weight_decay=0.01,
        fp16=True,                            # BẮT BUỘC cho 6GB VRAM
        gradient_checkpointing=True,          # tiết kiệm thêm VRAM, đổi lấy tốc độ
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=42,                              # ghi rõ seed để đảm bảo khả năng tái lập
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Đã fine-tune xong, model lưu tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
