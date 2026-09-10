"""Binary cultural-consensus style latent answer-key estimation.

This module is a deliberately small baseline inspired by Cultural Consensus
Theory (CCT). It is not presented as a novel estimator; its purpose is to
stress-test the receiver-side measurement architecture used in this project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConsensusFit:
    truth_probability: np.ndarray
    competence: np.ndarray
    iterations: int
    converged: bool


def fit_binary_consensus(
    responses: np.ndarray,
    *,
    max_iter: int = 500,
    tol: float = 1e-8,
    min_competence: float = 0.5001,
    max_competence: float = 0.999,
) -> ConsensusFit:
    """Jointly estimate binary item truth and respondent competence.

    The model assumes one shared answer key, no item-specific difficulty, and
    no response bias. Those assumptions are intentionally restrictive: later
    experiments use their failure modes to motivate richer codebook models.
    """

    x = np.asarray(responses, dtype=float)
    if x.ndim != 2 or x.size == 0:
        raise ValueError("responses must be a non-empty 2D array")
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("responses must contain only 0/1 values")
    n_people, _ = x.shape
    if n_people < 3:
        raise ValueError("at least three respondents are required")

    # Majority-vote initialization. The estimator is symmetric under a global
    # label flip, so the soft majority fixes the orientation for simulations.
    truth = np.clip(x.mean(axis=0), 1e-4, 1.0 - 1e-4)
    competence = np.full(n_people, 0.75, dtype=float)

    converged = False
    for iteration in range(1, max_iter + 1):
        old_truth = truth.copy()
        old_competence = competence.copy()

        # E-step: posterior P(T_j=1 | responses, competence), uniform prior.
        logit_c = np.log(competence / (1.0 - competence))
        log_odds = ((2.0 * x - 1.0) * logit_c[:, None]).sum(axis=0)

        truth = np.empty_like(log_odds)
        positive = log_odds >= 0
        truth[positive] = 1.0 / (1.0 + np.exp(-log_odds[positive]))
        exp_lo = np.exp(log_odds[~positive])
        truth[~positive] = exp_lo / (1.0 + exp_lo)
        truth = np.clip(truth, 1e-9, 1.0 - 1e-9)

        # M-step: expected agreement with the latent answer key.
        expected_agreement = (
            x * truth[None, :]
            + (1.0 - x) * (1.0 - truth[None, :])
        )
        competence = np.clip(
            expected_agreement.mean(axis=1),
            min_competence,
            max_competence,
        )

        change = max(
            float(np.max(np.abs(truth - old_truth))),
            float(np.max(np.abs(competence - old_competence))),
        )
        if change < tol:
            converged = True
            break

    return ConsensusFit(
        truth_probability=truth,
        competence=competence,
        iterations=iteration,
        converged=converged,
    )
