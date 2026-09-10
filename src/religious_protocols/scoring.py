"""Scoring utilities for receiver-centered decoding experiments."""

from __future__ import annotations

import math

_EPS = 1e-12


def _clip_probability(p: float) -> float:
    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return min(max(float(p), _EPS), 1.0 - _EPS)


def log_score_gain(prior_correct: float, posterior_correct: float) -> float:
    """Return log-score improvement for the correct target.

    Positive values mean the cue moved probability mass toward the correct
    target. Negative values mean the receiver became more confidently wrong.
    Natural logarithms are used, so values are in nats.
    """

    p0 = _clip_probability(prior_correct)
    p1 = _clip_probability(posterior_correct)
    return math.log(p1) - math.log(p0)


def update_magnitude_bernoulli(prior_correct: float, posterior_correct: float) -> float:
    """KL divergence D_KL(posterior || prior) for a Bernoulli target.

    This measures how much the receiver's belief changed, not whether the
    change was correct. It is therefore intentionally separate from
    ``log_score_gain``.
    """

    p0 = _clip_probability(prior_correct)
    p1 = _clip_probability(posterior_correct)
    return (
        p1 * math.log(p1 / p0)
        + (1.0 - p1) * math.log((1.0 - p1) / (1.0 - p0))
    )
