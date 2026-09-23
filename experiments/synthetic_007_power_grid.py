"""Monte Carlo power surface for the Study 1 receiver × exposure interaction.

The design is deliberately crossed: every synthetic participant sees every cue.
Both participant and cue have random intercepts *and* random slopes relevant to
the cross-level interaction. Inference uses a two-way cluster-robust sandwich
(participant and cue), so the simulation does not pretend P*C observations are
independent.

This is a planning stress test, not a preregistered human-study power analysis.
Variance components must later be replaced by pilot estimates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results")
OUT.mkdir(exist_ok=True)

PARTICIPANTS = [80, 120, 180, 240]
CUES = [36, 48, 60, 72]
BETAS = [0.00, 0.05, 0.10, 0.15, 0.20]
REPS = 120
ALPHA_Z = 1.959963984540054

# Synthetic variance assumptions, all on standardized outcome scale.
SIGMA_PARTICIPANT_INTERCEPT = 0.45
SIGMA_CUE_INTERCEPT = 0.45
SIGMA_PARTICIPANT_THETA_SLOPE = 0.20
SIGMA_CUE_K_SLOPE = 0.20
SIGMA_RESIDUAL = 1.00

SEED = 20260923


def ols_two_way_cluster(
    y: np.ndarray,
    X: np.ndarray,
    participant_id: np.ndarray,
    cue_id: np.ndarray,
    n_participants: int,
    n_cues: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS coefficient + naive and two-way cluster-robust standard errors."""
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta

    sigma2 = float(resid @ resid) / max(n - k, 1)
    naive_var = xtx_inv * sigma2
    naive_se = np.sqrt(np.maximum(np.diag(naive_var), 0.0))

    scores = X * resid[:, None]

    p_score = np.zeros((n_participants, k), dtype=float)
    np.add.at(p_score, participant_id, scores)
    meat_p = p_score.T @ p_score

    c_score = np.zeros((n_cues, k), dtype=float)
    np.add.at(c_score, cue_id, scores)
    meat_c = c_score.T @ c_score

    # Intersection of participant and cue clusters is each observation.
    meat_i = scores.T @ scores

    # Cameron-Gelbach-Miller style inclusion-exclusion with small-sample factors.
    fp = (n_participants / max(n_participants - 1, 1)) * ((n - 1) / max(n - k, 1))
    fc = (n_cues / max(n_cues - 1, 1)) * ((n - 1) / max(n - k, 1))
    fi = n / max(n - k, 1)

    meat = fp * meat_p + fc * meat_c - fi * meat_i
    cov = xtx_inv @ meat @ xtx_inv
    robust_se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return beta, naive_se, robust_se


def simulate_once(
    rng: np.random.Generator,
    n_participants: int,
    n_cues: int,
    beta_interaction: float,
) -> dict[str, float]:
    K = rng.normal(size=n_participants)
    theta = rng.normal(size=n_cues)

    p_int = rng.normal(scale=SIGMA_PARTICIPANT_INTERCEPT, size=n_participants)
    c_int = rng.normal(scale=SIGMA_CUE_INTERCEPT, size=n_cues)
    p_theta = rng.normal(scale=SIGMA_PARTICIPANT_THETA_SLOPE, size=n_participants)
    c_K = rng.normal(scale=SIGMA_CUE_K_SLOPE, size=n_cues)

    pid = np.repeat(np.arange(n_participants), n_cues)
    cid = np.tile(np.arange(n_cues), n_participants)

    K_obs = K[pid]
    theta_obs = theta[cid]
    interaction = K_obs * theta_obs

    # Nonzero main effects prevent the interaction from living in an unrealistically
    # empty model; random slopes are the key dependence terms for power.
    y = (
        0.25 * K_obs
        + 0.20 * theta_obs
        + beta_interaction * interaction
        + p_int[pid]
        + c_int[cid]
        + p_theta[pid] * theta_obs
        + c_K[cid] * K_obs
        + rng.normal(scale=SIGMA_RESIDUAL, size=len(pid))
    )

    X = np.column_stack(
        [np.ones(len(pid)), K_obs, theta_obs, interaction]
    )
    b, naive_se, robust_se = ols_two_way_cluster(
        y, X, pid, cid, n_participants, n_cues
    )

    idx = 3
    z_robust = b[idx] / robust_se[idx] if robust_se[idx] > 0 else np.nan
    z_naive = b[idx] / naive_se[idx] if naive_se[idx] > 0 else np.nan
    return {
        "beta_hat": float(b[idx]),
        "se_two_way": float(robust_se[idx]),
        "se_naive": float(naive_se[idx]),
        "reject_two_way": float(abs(z_robust) > ALPHA_Z),
        "reject_naive": float(abs(z_naive) > ALPHA_Z),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | int]] = []

    total = len(PARTICIPANTS) * len(CUES) * len(BETAS) * REPS
    done = 0
    for n_participants in PARTICIPANTS:
        for n_cues in CUES:
            for beta_interaction in BETAS:
                for rep in range(REPS):
                    result = simulate_once(
                        rng, n_participants, n_cues, beta_interaction
                    )
                    rows.append(
                        {
                            "n_participants": n_participants,
                            "n_cues": n_cues,
                            "beta_interaction": beta_interaction,
                            "rep": rep,
                            **result,
                        }
                    )
                    done += 1
                print(
                    f"{done}/{total}: P={n_participants}, C={n_cues}, "
                    f"beta={beta_interaction:.2f}"
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "synthetic_007_power_raw.csv", index=False)

    grid = (
        raw.groupby(["n_participants", "n_cues", "beta_interaction"], as_index=False)
        .agg(
            power_two_way=("reject_two_way", "mean"),
            rejection_naive=("reject_naive", "mean"),
            mean_beta_hat=("beta_hat", "mean"),
            sd_beta_hat=("beta_hat", "std"),
            mean_se_two_way=("se_two_way", "mean"),
            mean_se_naive=("se_naive", "mean"),
        )
    )
    grid["se_inflation_two_way_over_naive"] = (
        grid["mean_se_two_way"] / grid["mean_se_naive"]
    )
    grid.to_csv(OUT / "synthetic_007_power_grid.csv", index=False)

    # Compact planning views.
    power80 = grid[grid["power_two_way"] >= 0.80].copy()
    if len(power80):
        power80 = power80.sort_values(
            ["beta_interaction", "n_participants", "n_cues"]
        )
    power80.to_csv(OUT / "synthetic_007_power_ge80.csv", index=False)

    null = grid[grid["beta_interaction"] == 0.0]
    summary = {
        "seed": SEED,
        "reps_per_cell": REPS,
        "n_simulations": int(len(raw)),
        "participant_grid": PARTICIPANTS,
        "cue_grid": CUES,
        "interaction_grid": BETAS,
        "variance_assumptions": {
            "participant_intercept_sd": SIGMA_PARTICIPANT_INTERCEPT,
            "cue_intercept_sd": SIGMA_CUE_INTERCEPT,
            "participant_theta_slope_sd": SIGMA_PARTICIPANT_THETA_SLOPE,
            "cue_K_slope_sd": SIGMA_CUE_K_SLOPE,
            "residual_sd": SIGMA_RESIDUAL,
        },
        "mean_null_rejection_two_way": float(null["power_two_way"].mean()),
        "mean_null_rejection_naive": float(null["rejection_naive"].mean()),
        "max_null_rejection_two_way": float(null["power_two_way"].max()),
        "max_null_rejection_naive": float(null["rejection_naive"].max()),
        "median_se_inflation": float(
            grid["se_inflation_two_way_over_naive"].median()
        ),
        "warning": (
            "Synthetic planning stress test only. Replace variance components and "
            "effect-size priors with pilot estimates before preregistration."
        ),
    }
    (OUT / "synthetic_007_power_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
