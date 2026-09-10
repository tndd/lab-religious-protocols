"""Diagnostics for testing one-codebook assumptions.

These helpers are screening tools, not a replacement for formal Cultural
Consensus Theory model comparison. The spectral ratio mirrors the familiar
first/second eigenvalue heuristic used with informal consensus analyses.
"""

from __future__ import annotations

import numpy as np


def informal_consensus_eigen_ratio(responses: np.ndarray) -> float:
    """Return first/second eigenvalue ratio of respondent correlations.

    Rows are respondents and columns are binary/categorical-coded items. For
    binary simulations this is a computationally cheap spectral screen for a
    dominant one-factor response pattern. It should be interpreted as a gate,
    not as proof that only one cultural model exists.
    """

    x = np.asarray(responses, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 3:
        raise ValueError("responses must be a 2D matrix with >=3 rows/items")

    centered = x - x.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, ddof=1, keepdims=True)
    if np.any(scale <= 0):
        raise ValueError("each respondent must vary across items")
    z = centered / scale

    # Non-zero eigenvalues of corr(respondents) are proportional to squared
    # singular values of the row-standardized response matrix. Working in the
    # smaller N x J matrix avoids constructing a potentially large N x N
    # correlation matrix.
    singular = np.linalg.svd(z, compute_uv=False, full_matrices=False)
    if singular.size < 2 or singular[1] <= 0:
        return float("inf")
    return float((singular[0] / singular[1]) ** 2)
