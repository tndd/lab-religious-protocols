"""Synthetic smoke test for receiver update metrics.

This is deliberately simple: it verifies that the research code runs end to
end on Kaggle and demonstrates why update magnitude and predictive gain must
remain separate quantities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from religious_protocols.scoring import log_score_gain, update_magnitude_bernoulli


RNG = np.random.default_rng(20260911)
N = 10_000


def main() -> None:
    prior = RNG.uniform(0.05, 0.45, size=N)
    codebook_depth = RNG.uniform(0.0, 1.0, size=N)

    # Higher codebook depth tends to move receivers toward the correct target,
    # but noise allows some receivers to update in the wrong direction.
    delta = 0.55 * (codebook_depth - 0.35) + RNG.normal(0.0, 0.18, size=N)
    posterior = np.clip(prior + delta, 1e-6, 1 - 1e-6)

    rows = pd.DataFrame(
        {
            "receiver_id": np.arange(N),
            "codebook_depth": codebook_depth,
            "prior_correct": prior,
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
    rows.to_csv(out / "synthetic_001_receivers.csv", index=False)

    summary = pd.DataFrame(
        {
            "metric": [
                "n_receivers",
                "mean_predictive_gain_nats",
                "mean_update_magnitude_nats",
                "share_negative_predictive_gain",
                "corr_codebook_predictive_gain",
            ],
            "value": [
                len(rows),
                rows.predictive_gain_nats.mean(),
                rows.update_magnitude_nats.mean(),
                (rows.predictive_gain_nats < 0).mean(),
                rows[["codebook_depth", "predictive_gain_nats"]].corr().iloc[0, 1],
            ],
        }
    )
    summary.to_csv(out / "synthetic_001_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote results to: {out.resolve()}")


if __name__ == "__main__":
    main()
