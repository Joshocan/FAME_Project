"""Evaluation utilities for Feature Models.

Current modules:
 - coverage: semantic recall-style coverage of generated vs ground-truth FMs.

Designed to be extended (structure, constraints, traceability, etc.).
"""

from .coverage import CoverageEvaluator, CoverageConfig, coverage_score

__all__ = ["CoverageEvaluator", "CoverageConfig", "coverage_score"]
