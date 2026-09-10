"""Map when a one-codebook competence score misreads plural interpretation.

The synthetic population contains two communities with equal latent competence
distributions. Community B differs systematically on a fraction of cues. A
single-consensus fit can then manufacture a competence gap that does not exist.
We sweep minority share and divergent-cue share, and compare the pooled fit
with group-specific fits as an oracle rescue benchmark.

This is explicitly a stress test of our measurement design. Multi-culture CCT
already models multiple latent consensus truths; the experiment quantifies why
we need that richer class rather than claiming a new consensus method.
"""

from __future__ import annotations

from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from religious_protocols.consensus import fit_binary_consensus

BASE_SEED = 20260911
N = 800
J = 120
REPLICATES = 40
GROUP_SHARES = np.arange(0.05, 0.50, 0.05)
DIVERGENCE_SHARES = np.arange(0.0, 0.61, 0.10)


def make_responses(
    rng: np.random.Generator,
    key: np.ndarray,
    competence: np.ndarray,
) -> np.ndarray:
    correct = rng.random((competence.size, key.size)) < competence[:, None]
    return np.where(correct, key[None, :], 1 - key[None, :]).astype(np.int8)


def one_run(
    rng: np.random.Generator,
    group_share: float,
    divergence_share: float,
) -> dict[str, float]:
    depth = rng.uniform(0.0, 1.0, N)
    true_competence = 0.52 + 0.44 * depth
    group_b = rng.random(N) < group_share

    if not group_b.any():
        group_b[rng.integers(0, N)] = True
    if group_b.all():
        group_b[rng.integers(0, N)] = False
    group_a = ~group_b

    key_a = rng.integers(0, 2, J, dtype=np.int8)
    flip = rng.random(J) < divergence_share
    key_b = key_a.copy()
    key_b[flip] = 1 - key_b[flip]

    responses = np.empty((N, J), dtype=np.int8)
    responses[group_a] = make_responses(
        rng, key_a, true_competence[group_a]
    )
    responses[group_b] = make_responses(
        rng, key_b, true_competence[group_b]
    )

    pooled = fit_binary_consensus(responses)
    fit_a = fit_binary_consensus(responses[group_a])
    fit_b = fit_binary_consensus(responses[group_b])

    pooled_bias_a = float(
        (pooled.competence[group_a] - true_competence[group_a]).mean()
    )
    pooled_bias_b = float(
        (pooled.competence[group_b] - true_competence[group_b]).mean()
    )
    oracle_bias_a = float(
        (fit_a.competence - true_competence[group_a]).mean()
    )
    oracle_bias_b = float(
        (fit_b.competence - true_competence[group_b]).mean()
    )

    inferred_key = (pooled.truth_probability >= 0.5).astype(np.int8)

    return {
        "realized_group_b_share": float(group_b.mean()),
        "realized_divergence_share": float(flip.mean()),
        "pooled_bias_group_a": pooled_bias_a,
        "pooled_bias_group_b": pooled_bias_b,
        "spurious_gap_b_minus_a": float(
            pooled.competence[group_b].mean()
            - pooled.competence[group_a].mean()
        ),
        "truth_key_accuracy_vs_group_a": float(
            (inferred_key == key_a).mean()
        ),
        "oracle_group_specific_abs_bias": float(
            (abs(oracle_bias_a) + abs(oracle_bias_b)) / 2.0
        ),
        "pooled_group_b_abs_bias": abs(pooled_bias_b),
    }


def main() -> None:
    rows: list[dict[str, float | int]] = []

    for group_share, divergence_share in itertools.product(
        GROUP_SHARES, DIVERGENCE_SHARES
    ):
        for replicate in range(REPLICATES):
            seed = np.random.SeedSequence(
                [
                    BASE_SEED,
                    int(round(group_share * 100)),
                    int(round(divergence_share * 100)),
                    replicate,
                ]
            )
            rng = np.random.default_rng(seed)
            result = one_run(
                rng,
                float(group_share),
                float(divergence_share),
            )
            result.update(
                {
                    "target_group_b_share": float(group_share),
                    "target_divergence_share": float(divergence_share),
                    "replicate": replicate,
                }
            )
            rows.append(result)

    raw = pd.DataFrame(rows)
    metrics = [
        "pooled_bias_group_a",
        "pooled_bias_group_b",
        "spurious_gap_b_minus_a",
        "truth_key_accuracy_vs_group_a",
        "oracle_group_specific_abs_bias",
        "pooled_group_b_abs_bias",
    ]

    summary = (
        raw.groupby(
            ["target_group_b_share", "target_divergence_share"]
        )[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(x) for x in column if x != "")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]

    out = Path("results")
    out.mkdir(exist_ok=True)
    raw.to_csv(out / "synthetic_004_failure_surface_raw.csv", index=False)
    summary.to_csv(
        out / "synthetic_004_failure_surface_summary.csv", index=False
    )

    worst = summary.loc[summary["pooled_group_b_abs_bias_mean"].idxmax()]
    print(f"scenarios={len(summary)}, runs={len(raw)}")
    print(
        "worst minority abs bias="
        f"{worst['pooled_group_b_abs_bias_mean']:.4f} "
        f"at group_share={worst['target_group_b_share']:.2f}, "
        f"divergence={worst['target_divergence_share']:.2f}"
    )
    print(
        "oracle group-specific abs bias there="
        f"{worst['oracle_group_specific_abs_bias_mean']:.4f}"
    )
    print(
        "spurious B-A competence gap there="
        f"{worst['spurious_gap_b_minus_a_mean']:.4f}"
    )
    print(
        "pooled key accuracy vs group A there="
        f"{worst['truth_key_accuracy_vs_group_a_mean']:.4f}"
    )
    print(f"\nWrote results to: {out.resolve()}")


if __name__ == "__main__":
    main()
