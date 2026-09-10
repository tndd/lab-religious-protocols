import math

import pytest

from religious_protocols.scoring import log_score_gain, update_magnitude_bernoulli


def test_correct_update_has_positive_predictive_gain():
    assert log_score_gain(0.2, 0.7) > 0


def test_wrong_update_has_negative_predictive_gain():
    assert log_score_gain(0.2, 0.05) < 0


def test_both_correct_and_wrong_large_updates_have_positive_magnitude():
    correct_move = update_magnitude_bernoulli(0.2, 0.7)
    wrong_move = update_magnitude_bernoulli(0.2, 0.05)
    assert correct_move > 0
    assert wrong_move > 0


def test_no_update_has_zero_magnitude_and_gain():
    assert update_magnitude_bernoulli(0.4, 0.4) == pytest.approx(0.0, abs=1e-12)
    assert log_score_gain(0.4, 0.4) == pytest.approx(0.0, abs=1e-12)


def test_gain_matches_log_ratio():
    assert log_score_gain(0.25, 0.5) == pytest.approx(math.log(2.0))


def test_invalid_probability_rejected():
    with pytest.raises(ValueError):
        log_score_gain(-0.1, 0.5)
