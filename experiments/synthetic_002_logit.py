"""Synthetic benchmark using log-odds updates instead of additive probability updates.

Why this exists
---------------
``synthetic_001.py`` deliberately exposed a boundary artifact: additive updates
in probability space can drive some posteriors to (or below) zero, after which
clipping produces enormous negative log-score penalties. That is useful as a
smoke test, but it is not a good default generative model for receiver belief
updates.

This version applies the cue in log-odds space. That keeps probabilities inside
(0, 1) naturally and lets us test the intended qualitative hypothesis:
receivers with deeper shared codebooks should, on average, decode the cue more
accurately, while some receivers still update in the wrong direction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from religious_protocols.scoring import log_score_gain, update_magnitude_bernoulli

RNG = np.random.default_rng(20260911)
N = 10_000


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    prior = RNG.uniform(0.05, 0.45, size=N)
    codebook_depth = RNG.uniform(0.0, 1.0, size=N)

    # Cue evidence is expressed in log-odds units. Deeper codebooks tend to
    # yield more evidence for the correct target, but noise preserves genuine
    # misdecoding for some receivers.
    evidence = 1.8 * (codebook_depth - 0.25) + RNG.normal(0.0, 0.75, size=N)
    posterior = expit(logit(prior) + evidence)

    rows = pd.DataFrame(
        {
            "receiver_id": np.arange(N),
            "codebook_depth": codebook_depth,
            "prior_correct": prior,
            "evidence_log_odds": evidence,
            "posterior_correct": posterior,
        }
    )

    rows["predictive_gain_nats"] = [
        log_score_gain(a, b)
        for a, b in zip(rows.prior_correct, rows.posterior_correct, strict=True)
    ]
    rows["update_magnitude_nats"] = [
        update_magnitude_bernoulli(a, b)
        for a, b in zip(rows.prior_correct, rows.posterior_correct, strict=True)
    ]

    out = Path("results")
    out.mkdir(exist_ok=True)
    rows.to_csv(out / "synthetic_002_receivers.csv", index=False)

    summary = pd.DataFrame(
        {
            "metric": [
                "n_receivers",
                "mean_predictive_gain_nats",
                "median_predictive_gain_nats",
                "mean_update_magnitude_nats",
                "share_negative_predictive_gain",
                "corr_codebook_predictive_gain",
                "corr_codebook_update_magnitude",
            ],
            "value": [
                len(rows),
                rows.predictive_gain_nats.mean(),
                rows.predictive_gain_nats.median(),
                rows.update_magnitude_nats.mean(),
                (rows.predictive_gain_nats < 0).mean(),
                rows[["codebook_depth", "predictive_gain_nats"]].corr().iloc[0, 1],
                rows[["codebook_depth", "update_magnitude_nats"]].corr().iloc[0, 1],
            ],
        }
    )
    summary.to_csv(out / "synthetic_002_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote results to: {out.resolve()}")


if __name__ == "__main__":
    main()
