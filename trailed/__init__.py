"""
TRAILED - Topological Regularization and Integrity Learning for EHR Data.

This package re-exports the current DECT Python implementation under the
`trailed` package name while functionality remains unchanged.
"""

from dect import *  # noqa: F403
from dect import __all__ as _dect_all
from dect import __path__ as _dect_path

__all__ = list(_dect_all)
__path__ = _dect_path
