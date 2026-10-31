"""Recover official answer aliases/evidence for the existing subset, without replacing it."""
import json
import unicodedata
from pathlib import Path

from ragqa.paths import DATA_DIR, RAW_DOCUMENTS_PATH

DATASET = "mandarjoshi/trivia_qa"
REVISION = "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"


def key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def main():
    from datasets import load_dataset
    raw = json.loads(RAW_DOCUMENTS_PATH.read_text(encoding="utf-8"))
    targets = {key(qa["question"]): doc["id"] for doc in raw for qa in doc["qa"]}
    output = DATA_DIR / "provenance"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "triviaqa_official.jsonl"
    if destination.exists():
        raise FileExistsError(f"Preserving existing recovery: {destination}")
    found = set()
    dataset = load_dataset(DATASET, "rc.wikipedia", split="validation", streaming=True, revision=REVISION)
    with destination.open("w", encoding="utf-8") as stream:
        for offset, ex in enumerate(dataset):
            question_key = key(ex["question"])
            if question_key in targets and question_key not in found:
                record = {"doc_id": targets[question_key], "source_id": ex["question_id"],
                          "question": ex["question"], "answer": ex["answer"],
                          "entity_pages": ex["entity_pages"], "dataset": DATASET,
                          "revision": REVISION, "source_offset": offset}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                found.add(question_key)
                if len(found) % 100 == 0:
                    print(f"Recovered {len(found)}/{len(targets)}", flush=True)
            if len(found) == len(targets):
                break
            if offset > 20000:
                break
    print(f"Recovered {len(found)}/{len(targets)} -> {destination}", flush=True)
    if len(found) != len(targets):
        raise RuntimeError("Incomplete source recovery; do not silently treat missing records as verified")


if __name__ == "__main__":
    main()
