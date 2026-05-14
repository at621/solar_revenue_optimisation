"""Evaluation harness (Module 8): paired-seed N-episode runner + significance tests."""

from solarbess.eval.harness import EvaluationReport, run_paired_evaluation
from solarbess.eval.stats import (
    ComparisonResult,
    empirical_cvar,
    paired_t_test,
    wilcoxon_signed_rank,
)

__all__ = [
    "EvaluationReport",
    "run_paired_evaluation",
    "ComparisonResult",
    "empirical_cvar",
    "paired_t_test",
    "wilcoxon_signed_rank",
]
