"""Shared repository paths, independent of the current working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "generator_finetuned"
RAW_DOCUMENTS_PATH = DATA_DIR / "raw_documents.json"
RESULTS_DIR = PROJECT_ROOT / "results"
PIPELINE_DIR = PROJECT_ROOT / ".pipeline"
BASELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline_retrieval.json"


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a config path from the repository, preserving absolute paths."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
