"""Can held-out prediction rescue minority codebooks missed by a simple gate?

Reference respondents are split from evaluation respondents. Pooled and
stratum-specific cultural answer keys are estimated only from the reference
sample. Each evaluation respondent's competence is calibrated on one half of
items and scored on the other half. The key outcome is out-of-sample log-score
improvement for the minority stratum.

This follows the project's methodological principle: consensus diagnostics are
screens; predictive generalization decides whether added reference structure is
useful. It is not presented as a new Cultural Consensus Theory estimator.
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
J = 72
REPLICATES = 50
GROUP_SHARES = [0.05, 0.10, 0.20, 0.30, 0.40]
DIVERGENCE_SHARES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
REFERENCE_FRACTION = 0.60


def make_responses(rng, key, competence):
    correct = rng.random((competence.size, key.size)) < competence[:, None]
    return np.where(correct, key[None, :], 1 - key[None, :]).astype(np.int8)


def calibrate_and_score(responses, key, cal_idx, test_idx):
    agreement = (responses[:, cal_idx] == key[cal_idx][None, :]).mean(axis=1)
    c = np.clip(agreement, 0.5001, 0.999)
    matches = responses[:, test_idx] == key[test_idx][None, :]
    probs = np.where(matches, c[:, None], 1.0 - c[:, None])
    return np.log(np.clip(probs, 1e-12, 1.0)).mean(axis=1)


def one_run(rng, group_share, divergence_share):
    depth = rng.uniform(0.0, 1.0, N)
    competence = 0.52 + 0.44 * depth
    group_b = rng.random(N) < group_share
    if group_b.sum() < 8:
        group_b[rng.choice(np.where(~group_b)[0], size=8-group_b.sum(), replace=False)] = True
    group_a = ~group_b

    key_a = rng.integers(0, 2, J, dtype=np.int8)
    flip = rng.random(J) < divergence_share
    key_b = key_a.copy()
    key_b[flip] = 1 - key_b[flip]

    x = np.empty((N, J), dtype=np.int8)
    x[group_a] = make_responses(rng, key_a, competence[group_a])
    x[group_b] = make_responses(rng, key_b, competence[group_b])

    # Stratified reference/evaluation respondent split.
    ref = np.zeros(N, dtype=bool)
    for mask in [group_a, group_b]:
        ids = np.where(mask)[0]
        rng.shuffle(ids)
        n_ref = max(4, int(round(len(ids) * REFERENCE_FRACTION)))
        n_ref = min(n_ref, len(ids)-2)
        ref[ids[:n_ref]] = True
    ev = ~ref

    fit_pool = fit_binary_consensus(x[ref])
    fit_a = fit_binary_consensus(x[ref & group_a])
    fit_b = fit_binary_consensus(x[ref & group_b])
    k_pool = (fit_pool.truth_probability >= 0.5).astype(np.int8)
    k_a = (fit_a.truth_probability >= 0.5).astype(np.int8)
    k_b = (fit_b.truth_probability >= 0.5).astype(np.int8)

    perm = rng.permutation(J)
    cal_idx = perm[: J // 2]
    test_idx = perm[J // 2 :]

    score_pool = calibrate_and_score(x[ev], k_pool, cal_idx, test_idx)
    ids_ev = np.where(ev)[0]
    score_strat = np.empty(len(ids_ev), dtype=float)
    ev_b_local = group_b[ids_ev]
    if (~ev_b_local).any():
        score_strat[~ev_b_local] = calibrate_and_score(
            x[ids_ev[~ev_b_local]], k_a, cal_idx, test_idx
        )
    if ev_b_local.any():
        score_strat[ev_b_local] = calibrate_and_score(
            x[ids_ev[ev_b_local]], k_b, cal_idx, test_idx
        )

    delta = score_strat - score_pool
    ratio = informal_consensus_eigen_ratio(x[ref])

    return {
        "realized_group_b_share": float(group_b.mean()),
        "realized_divergence_share": float(flip.mean()),
        "reference_b_n": int((ref & group_b).sum()),
        "evaluation_b_n": int((ev & group_b).sum()),
        "eigen_ratio": ratio,
        "screen_pass_3_to_1": float(ratio >= 3.0),
        "pooled_key_accuracy_a": float((k_pool == key_a).mean()),
        "stratum_key_accuracy_b": float((k_b == key_b).mean()),
        "minority_logscore_gain_strat_vs_pool": float(delta[ev_b_local].mean()),
        "majority_logscore_gain_strat_vs_pool": float(delta[~ev_b_local].mean()),
        "minority_rescue_gt_0_05": float(delta[ev_b_local].mean() > 0.05),
        "hidden_predictive_rescue": float(
            ratio >= 3.0 and delta[ev_b_local].mean() > 0.05
        ),
    }


def main():
    rows = []
    for g, d in itertools.product(GROUP_SHARES, DIVERGENCE_SHARES):
        for r in range(REPLICATES):
            seed = np.random.SeedSequence(
                [BASE_SEED, int(g*100), int(d*100), r]
            )
            out = one_run(np.random.default_rng(seed), g, d)
            out.update({
                "target_group_b_share": g,
                "target_divergence_share": d,
                "replicate": r,
            })
            rows.append(out)

    raw = pd.DataFrame(rows)
    metrics = [
        "reference_b_n", "evaluation_b_n", "eigen_ratio",
        "screen_pass_3_to_1", "pooled_key_accuracy_a",
        "stratum_key_accuracy_b", "minority_logscore_gain_strat_vs_pool",
        "majority_logscore_gain_strat_vs_pool", "minority_rescue_gt_0_05",
        "hidden_predictive_rescue",
    ]
    summary = raw.groupby(
        ["target_group_b_share", "target_divergence_share"]
    )[metrics].mean().reset_index()

    outdir = Path("results")
    outdir.mkdir(exist_ok=True)
    raw.to_csv(outdir / "synthetic_006_predictive_rescue_raw.csv", index=False)
    summary.to_csv(outdir / "synthetic_006_predictive_rescue_summary.csv", index=False)

    passed = raw[raw.screen_pass_3_to_1 == 1]
    print(f"runs={len(raw)}, grid_cells={len(summary)}")
    print(f"3:1 screen pass rate={raw.screen_pass_3_to_1.mean():.3f}")
    print(
        "among screen-pass runs, share where subgroup model improves minority "
        f">0.05 nats/item={passed.minority_rescue_gt_0_05.mean():.3f}"
    )
    print(
        "mean minority held-out gain, stratified minus pooled="
        f"{raw.minority_logscore_gain_strat_vs_pool.mean():.4f} nats/item"
    )
    print(
        "mean majority held-out gain, stratified minus pooled="
        f"{raw.majority_logscore_gain_strat_vs_pool.mean():.4f} nats/item"
    )
    top = summary.sort_values(
        "minority_logscore_gain_strat_vs_pool", ascending=False
    ).iloc[0]
    print(
        "largest mean minority rescue: "
        f"group={top.target_group_b_share:.2f}, "
        f"divergence={top.target_divergence_share:.2f}, "
        f"gain={top.minority_logscore_gain_strat_vs_pool:.3f}, "
        f"screen pass={top.screen_pass_3_to_1:.2f}, "
        f"B reference n={top.reference_b_n:.1f}"
    )
    print(f"\nWrote results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
