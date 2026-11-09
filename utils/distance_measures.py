"""Distances between two sets of image features: Frechet distance and nearest-neighbor distance."""
import logging

import numpy as np
from scipy import linalg

# sqrtm of a product of covariances can return a tiny imaginary part from rounding. Anything
# larger means the result is not a usable square root.
MAX_IMAGINARY = 1e-3
# Added to both covariance diagonals when sqrtm fails, the same fallback as pytorch-fid.
SQRTM_RETRY_OFFSET = 1e-6


def normalize_rows(features):
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Expected nonzero feature vectors, found {} zero rows.".format(int(np.sum(norms == 0))))
    return features / norms


def mean_and_covariance(features):
    features = np.asarray(features, dtype=np.float64)
    return features.mean(axis=0), np.cov(features, rowvar=False)


def covariance_product_sqrt(cov1, cov2):
    covmean = linalg.sqrtm(cov1 @ cov2, disp=False)[0]
    if not np.isfinite(covmean).all():
        logging.warning(
            "sqrtm returned inf or nan; retrying with %g added to both covariance diagonals.",
            SQRTM_RETRY_OFFSET,
        )
        offset = np.eye(cov1.shape[0]) * SQRTM_RETRY_OFFSET
        covmean = linalg.sqrtm((cov1 + offset) @ (cov2 + offset), disp=False)[0]
        if not np.isfinite(covmean).all():
            raise RuntimeError("Expected a finite matrix square root after the retry, found inf or nan.")

    if np.iscomplexobj(covmean):
        max_imaginary = np.max(np.abs(covmean.imag))
        if max_imaginary >= MAX_IMAGINARY:
            raise RuntimeError(
                "Expected max |imag| of the matrix square root below {}, found {}.".format(
                    MAX_IMAGINARY, max_imaginary
                )
            )
        covmean = covmean.real
    return covmean


def frechet_distance(mu1, cov1, mu2, cov2):
    """||mu1 - mu2||^2 + Tr(cov1 + cov2 - 2 (cov1 cov2)^(1/2)), in float64."""
    mu1, mu2 = np.asarray(mu1, dtype=np.float64), np.asarray(mu2, dtype=np.float64)
    cov1, cov2 = np.asarray(cov1, dtype=np.float64), np.asarray(cov2, dtype=np.float64)
    diff = mu1 - mu2
    covmean = covariance_product_sqrt(cov1, cov2)
    # Trace terms use the covariances without the retry offset, as pytorch-fid does.
    return float(diff @ diff + np.trace(cov1) + np.trace(cov2) - 2 * np.trace(covmean))


def nearest_neighbor_distance(features, reference):
    """Mean over the rows of features of (1 - highest cosine similarity to any reference row)."""
    similarity = normalize_rows(features) @ normalize_rows(reference).T
    return float(np.mean(1 - similarity.max(axis=1)))
