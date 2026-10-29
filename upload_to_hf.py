"""Compatibility entry point. Prefer: python -m scripts.upload_to_hf."""

from scripts.upload_to_hf import *  # noqa: F403

if __name__ == "__main__":
    main()
