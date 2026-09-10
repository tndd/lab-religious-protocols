import numpy as np

from religious_protocols.diagnostics import informal_consensus_eigen_ratio


def _responses(rng, n, j, competence, key):
    correct = rng.random((n, j)) < competence
    return np.where(correct, key[None, :], 1 - key[None, :])


def test_spectral_ratio_higher_for_clean_single_codebook():
    rng = np.random.default_rng(7)
    n, j = 160, 100
    key = rng.integers(0, 2, j)
    single = _responses(rng, n, j, 0.80, key)

    split = single.copy()
    flip = np.arange(j) < j // 2
    split[n // 2 :, flip] = 1 - split[n // 2 :, flip]

    assert informal_consensus_eigen_ratio(single) > informal_consensus_eigen_ratio(split)
