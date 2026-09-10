"""Can a calibration battery recover receiver codebook competence?

Scenario A matches a one-consensus Cultural Consensus Theory baseline.
Scenario B introduces a minority codebook that systematically interprets a
subset of cues differently. The point is not to rediscover multi-culture CCT;
it is to show what a single-codebook receiver score would misread in our
proposed measurement architecture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from religious_protocols.consensus import fit_binary_consensus

RNG = np.random.default_rng(20260911)
N = 2_000
J_CAL = 120
J_TEST = 120


def make_responses(key: np.ndarray, competence: np.ndarray) -> np.ndarray:
    correct = RNG.random((competence.size, key.size)) < competence[:, None]
    return np.where(correct, key[None, :], 1 - key[None, :]).astype(np.int8)


def run_scenario(name: str, *, mixed: bool) -> dict[str, float | str]:
    codebook_depth = RNG.uniform(0.0, 1.0, N)
    true_competence = 0.52 + 0.44 * codebook_depth
    group_b = RNG.random(N) < 0.30 if mixed else np.zeros(N, dtype=bool)

    key_cal = RNG.integers(0, 2, J_CAL, dtype=np.int8)
    key_test = RNG.integers(0, 2, J_TEST, dtype=np.int8)
    flip_cal = RNG.random(J_CAL) < 0.30 if mixed else np.zeros(J_CAL, dtype=bool)
    flip_test = RNG.random(J_TEST) < 0.30 if mixed else np.zeros(J_TEST, dtype=bool)

    cal = np.empty((N, J_CAL), dtype=np.int8)
    test = np.empty((N, J_TEST), dtype=np.int8)

    group_a = ~group_b
    cal[group_a] = make_responses(key_cal, true_competence[group_a])
    test[group_a] = make_responses(key_test, true_competence[group_a])

    if group_b.any():
        key_cal_b = key_cal.copy()
        key_test_b = key_test.copy()
        key_cal_b[flip_cal] = 1 - key_cal_b[flip_cal]
        key_test_b[flip_test] = 1 - key_test_b[flip_test]
        cal[group_b] = make_responses(key_cal_b, true_competence[group_b])
        test[group_b] = make_responses(key_test_b, true_competence[group_b])

    fit = fit_binary_consensus(cal)
    inferred_key = (fit.truth_probability >= 0.5).astype(np.int8)

    consensus_test_accuracy = (test == key_test[None, :]).mean(axis=1)

    own_key_test = np.tile(key_test, (N, 1))
    if group_b.any():
        own_key_test[np.ix_(group_b, flip_test)] ^= 1
    own_test_accuracy = (test == own_key_test).mean(axis=1)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(a, b)[0, 1])

    result: dict[str, float | str] = {
        "scenario": name,
        "n_receivers": N,
        "calibration_items": J_CAL,
        "test_items": J_TEST,
        "group_b_share": float(group_b.mean()),
        "truth_key_accuracy": float((inferred_key == key_cal).mean()),
        "corr_true_competence_estimated": corr(true_competence, fit.competence),
        "corr_estimated_competence_consensus_test_accuracy": corr(
            fit.competence, consensus_test_accuracy
        ),
        "corr_estimated_competence_own_codebook_test_accuracy": corr(
            fit.competence, own_test_accuracy
        ),
        "mean_true_competence_group_a": float(true_competence[group_a].mean()),
        "mean_estimated_competence_group_a": float(fit.competence[group_a].mean()),
        "mean_bias_group_a": float(
            (fit.competence[group_a] - true_competence[group_a]).mean()
        ),
        "iterations": fit.iterations,
        "converged": float(fit.converged),
    }

    if group_b.any():
        result.update(
            {
                "corr_true_estimated_group_a": corr(
                    true_competence[group_a], fit.competence[group_a]
                ),
                "corr_true_estimated_group_b": corr(
                    true_competence[group_b], fit.competence[group_b]
                ),
                "mean_true_competence_group_b": float(true_competence[group_b].mean()),
                "mean_estimated_competence_group_b": float(fit.competence[group_b].mean()),
                "mean_bias_group_b": float(
                    (fit.competence[group_b] - true_competence[group_b]).mean()
                ),
            }
        )

    return result


def main() -> None:
    rows = pd.DataFrame(
        [
            run_scenario("single_codebook", mixed=False),
            run_scenario("mixed_codebook", mixed=True),
        ]
    )

    out = Path("results")
    out.mkdir(exist_ok=True)
    rows.to_csv(out / "synthetic_003_consensus_summary.csv", index=False)

    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(rows.to_string(index=False))
    print(f"\nWrote results to: {out.resolve()}")


if __name__ == "__main__":
    main()
