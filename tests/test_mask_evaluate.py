import numpy as np

from earthfire.mask_evaluate import _score


def test_probability_metrics_handle_ties_without_optimism():
    result = _score(np.array([1, 0, 0, 0]), np.array([.2, .2, .2, .2]))
    assert np.isclose(result["average_precision"], .25)
    assert np.isclose(result["roc_auc"], .5)
    assert np.isclose(result["brier"], .19)
