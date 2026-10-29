"""Compatibility entry point; implementation: experiments/hybrid_retrieval.py.

Preferred command: python -m experiments.hybrid_retrieval
"""

from experiments.hybrid_retrieval import *  # noqa: F403 - preserve the original public API


if __name__ == "__main__":
    compare_retrievers_demo()
