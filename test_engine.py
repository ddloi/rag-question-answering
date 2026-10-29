"""Compatibility test entry point; canonical tests live in tests/test_engine.py.

Preferred command: python -m unittest discover -s tests -v
"""

import unittest


def load_tests(loader, tests, pattern):
    return loader.loadTestsFromName("tests.test_engine")


if __name__ == "__main__":
    unittest.main()
