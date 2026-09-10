"""When does a 3:1 one-factor consensus screen miss harmful subgroup bias?

We generate populations with equal latent competence distributions but two
coherent answer keys. Community B differs from the dominant key on a fraction
of items. We then ask whether a correlation-matrix first/second eigenvalue
ratio >= 3 (a common informal CCT heuristic) can coexist with a substantively
large competence bias for community B.

This is a design stress test, not a claim that the heuristic or CCT is flawed
in general. Formal residual-agreement and multi-culture models are the rescue
path when the one-consensus assumption is doubtful.
"""

from __future__ import annotations

from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from religious_protocols.consensus import fit_binary_consensus
from religious_protocols.diagnostics import informal_consensus_eigen_ratio

BASE_SEED = 20260911
N = 300
REPLICATES = 30
ITEM_COUNTS = [48, 72, 120]
GROUP_SHARES = [0.05, 0.10, 0.20, 0.30, 0.40]
DIVERGENCE_SHARES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
BIAS_THRESHOLD = 0.10
RATIO_THRESHOLD = 3.0


def make_responses(
    rng: np.random.Generator,
    key: np.ndarray,
    competence: np.ndarray,
) -> np.ndarray:
    correct = rng.random((competence.size, key.size)) < competence[:, None]
    return np.where(correct, key[None, :], 1 - key[None, :]).astype(np.int8)


def one_run(
    rng: np.random.Generator,
    j: int,
    group_share: float,
    divergence_share: float,
) -> dict[str, float]:
    depth = rng.uniform(0.0, 1.0, N)
    competence = 0.52 + 0.44 * depth
    group_b = rng.random(N) < group_share
    if group_b.sum() < 4:
        group_b[rng.choice(N, size=4, replace=False)] = True
    group_a = ~group_b

    key_a = rng.integers(0, 2, j, dtype=np.int8)
    flip = rng.random(j) < divergence_share
    key_b = key_a.copy()
    key_b[flip] = 1 - key_b[flip]

    responses = np.empty((N, j), dtype=np.int8)
    responses[group_a] = make_responses(rng, key_a, competence[group_a])
    responses[group_b] = make_responses(rng, key_b, competence[group_b])

    fit = fit_binary_consensus(responses)
    ratio = informal_consensus_eigen_ratio(responses)
    bias_b = float((fit.competence[group_b] - competence[group_b]).mean())
    inferred_key = (fit.truth_probability >= 0.5).astype(np.int8)

    realized_d = float(flip.mean())
    mean_c_b = float(competence[group_b].mean())
    analytic_unclipped_bias = -realized_d * (2.0 * mean_c_b - 1.0)

    return {
        "realized_group_b_share": float(group_b.mean()),
        "realized_divergence_share": realized_d,
        "mean_true_competence_b": mean_c_b,
        "estimated_bias_b": bias_b,
        "analytic_unclipped_bias": analytic_unclipped_bias,
        "abs_formula_error": abs(bias_b - analytic_unclipped_bias),
        "eigen_ratio_1_to_2": ratio,
        "screen_pass_3_to_1": float(ratio >= RATIO_THRESHOLD),
        "harmful_bias_ge_0_10": float(abs(bias_b) >= BIAS_THRESHOLD),
        "dangerous_false_pass": float(
            ratio >= RATIO_THRESHOLD and abs(bias_b) >= BIAS_THRESHOLD
        ),
        "dominant_key_accuracy": float((inferred_key == key_a).mean()),
    }


def main() -> None:
    rows: list[dict[str, float | int]] = []
    for j, group_share, divergence_share in itertools.product(
        ITEM_COUNTS, GROUP_SHARES, DIVERGENCE_SHARES
    ):
        for replicate in range(REPLICATES):
            seed = np.random.SeedSequence(
                [
                    BASE_SEED,
                    j,
                    int(group_share * 100),
                    int(divergence_share * 100),
                    replicate,
                ]
            )
            rng = np.random.default_rng(seed)
            r = one_run(rng, j, group_share, divergence_share)
            r.update(
                {
                    "items": j,
                    "target_group_b_share": group_share,
                    "target_divergence_share": divergence_share,
                    "replicate": replicate,
                }
            )
            rows.append(r)

    raw = pd.DataFrame(rows)
    group_cols = ["items", "target_group_b_share", "target_divergence_share"]
    metrics = [
        "estimated_bias_b",
        "analytic_unclipped_bias",
        "abs_formula_error",
        "eigen_ratio_1_to_2",
        "screen_pass_3_to_1",
        "harmful_bias_ge_0_10",
        "dangerous_false_pass",
        "dominant_key_accuracy",
    ]
    summary = raw.groupby(group_cols)[metrics].mean().reset_index()

    out = Path("results")
    out.mkdir(exist_ok=True)
    raw.to_csv(out / "synthetic_005_consensus_gate_raw.csv", index=False)
    summary.to_csv(out / "synthetic_005_consensus_gate_summary.csv", index=False)

    harmful = raw[raw["harmful_bias_ge_0_10"] == 1.0]
    false_pass_rate_among_harmful = (
        float(harmful["screen_pass_3_to_1"].mean()) if len(harmful) else float("nan")
    )
    formula_region = raw[raw["realized_divergence_share"] <= 0.40]

    print(f"runs={len(raw)}, grid_cells={len(summary)}")
    print(f"harmful runs (|bias|>=0.10)={len(harmful)}")
    print(
        "3:1 screen pass rate among harmful runs="
        f"{false_pass_rate_among_harmful:.3f}"
    )
    print(
        "mean |simulation bias - analytic formula| for realized d<=0.40="
        f"{formula_region['abs_formula_error'].mean():.4f}"
    )
    print(
        "dominant-key mean accuracy="
        f"{raw['dominant_key_accuracy'].mean():.4f}"
    )

    danger = summary[summary["dangerous_false_pass"] > 0]
    if not danger.empty:
        top = danger.sort_values("dangerous_false_pass", ascending=False).iloc[0]
        print(
            "worst false-pass cell: "
            f"J={int(top['items'])}, "
            f"group={top['target_group_b_share']:.2f}, "
            f"divergence={top['target_divergence_share']:.2f}, "
            f"false-pass rate={top['dangerous_false_pass']:.3f}, "
            f"mean ratio={top['eigen_ratio_1_to_2']:.2f}, "
            f"mean bias={top['estimated_bias_b']:.3f}"
        )

    print(f"\nWrote results to: {out.resolve()}")


if __name__ == "__main__":
    main()
