"""Compatibility entry point for the local web demo.

Implementation: ragqa.web. Both python app.py and uvicorn app:app remain valid.
"""

from ragqa.web import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
