"""Discover and run pipeline stages without importing ML libraries to list them."""

import argparse
import runpy
import sys

STAGES = {
    "download": "download_dataset",
    "chunk": "prepare_chunks",
    "split": "split_train_test",
    "index": "build_index",
    "train": "finetune_generator",
    "rag": "rag_pipeline",
    "evaluate": "evaluate",
    "hybrid": "hybrid_retrieval",
    "ablation": "ablation",
    "baseline": "baseline_retrieval",
    "verify": "verify_dataset",
    "research-data": "research_data",
    "research-train": "research_train",
    "research-evaluate": "research_evaluate",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List or run a RAG experiment stage. No stage runs implicitly."
    )
    parser.add_argument("--list", action="store_true", help="List stages without loading models")
    parser.add_argument("stage", nargs="?", choices=STAGES)
    parser.add_argument("stage_args", nargs=argparse.REMAINDER, help="Arguments passed to the chosen stage")
    args = parser.parse_args()
    if args.list:
        for name, module in STAGES.items():
            print(f"{name:10s} python -m experiments.{module}")
        return
    if not args.stage:
        parser.print_help()
        return
    module = f"experiments.{STAGES[args.stage]}"
    sys.argv = [module, *args.stage_args]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
