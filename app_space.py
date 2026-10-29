"""Compatibility launcher for deployment.huggingface.app.

For a standalone Hugging Face Space, upload deployment/huggingface/app.py
as app.py alongside deployment/huggingface/requirements.txt.
"""

from deployment.huggingface.app import *  # noqa: F403

if __name__ == "__main__":
    demo.launch()
