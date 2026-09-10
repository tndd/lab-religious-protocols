import numpy as np

from religious_protocols.consensus import fit_binary_consensus


def test_consensus_recovers_easy_key_and_competence_order():
    rng = np.random.default_rng(42)
    n_receivers, n_items = 80, 100
    key = rng.integers(0, 2, n_items)
    competence = np.linspace(0.55, 0.95, n_receivers)

    correct = rng.random((n_receivers, n_items)) < competence[:, None]
    responses = np.where(correct, key[None, :], 1 - key[None, :])

    fit = fit_binary_consensus(responses)
    inferred_key = fit.truth_probability >= 0.5

    assert (inferred_key == key).mean() > 0.95
    assert np.corrcoef(competence, fit.competence)[0, 1] > 0.8
    assert fit.converged
