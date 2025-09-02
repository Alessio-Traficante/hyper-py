"""
Hyper-Py package initializer.

This file defines the public API and version metadata.
It does not run the application automatically; use
either `python -m hyper_py` or the console script entry point.
"""

from importlib.metadata import version, PackageNotFoundError

from .hyper import start_hyper

try:
    __version__ = version("hyper-py-photometry")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = [
    "start_hyper",
    "__version__",
]
