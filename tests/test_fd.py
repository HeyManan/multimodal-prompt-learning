"""Checks of the Frechet distance. Run from the repository root:

    PYTHONPATH="${PWD}" pytest tests/test_fd.py
"""
import numpy as np

from utils.distance_measures import frechet_distance, mean_and_covariance

DIMENSIONS = 16
SAMPLES = 2000


def random_features(seed, shift=0.0):
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(DIMENSIONS, DIMENSIONS))
    return rng.normal(size=(SAMPLES, DIMENSIONS)) @ mixing + shift


def fd_of_samples(a, b):
    return frechet_distance(*mean_and_covariance(a), *mean_and_covariance(b))


def test_fd_of_identical_sets_is_zero():
    features = random_features(seed=1)
    assert abs(fd_of_samples(features, features)) < 1e-6


def test_fd_is_symmetric():
    a = random_features(seed=1)
    b = random_features(seed=2, shift=0.5)
    assert np.isclose(fd_of_samples(a, b), fd_of_samples(b, a), rtol=1e-9, atol=1e-9)


def test_fd_matches_closed_form_for_diagonal_gaussians():
    rng = np.random.default_rng(3)
    mu1, mu2 = rng.normal(size=DIMENSIONS), rng.normal(size=DIMENSIONS)
    var1, var2 = rng.uniform(0.1, 2.0, DIMENSIONS), rng.uniform(0.1, 2.0, DIMENSIONS)

    expected = np.sum((mu1 - mu2) ** 2) + np.sum((np.sqrt(var1) - np.sqrt(var2)) ** 2)
    found = frechet_distance(mu1, np.diag(var1), mu2, np.diag(var2))
    assert np.isclose(found, expected, rtol=1e-9, atol=1e-9)


def test_fd_is_not_negative():
    for seed in range(5):
        a = random_features(seed=10 + seed)
        b = random_features(seed=20 + seed, shift=0.1 * seed)
        assert fd_of_samples(a, b) >= 0
