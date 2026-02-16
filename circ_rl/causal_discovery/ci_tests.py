"""Conditional independence tests for causal discovery.

Implements statistical tests for conditional independence:
X _||_ Y | Z (X is independent of Y given Z).

Used by the PC algorithm (``pc_algorithm.py``) to learn the causal skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats


class CITestMethod(Enum):
    """Available conditional independence test methods."""

    FISHER_Z = "fisher_z"
    KERNEL_CI = "kernel_ci"
    CHI_SQUARED = "chi_squared"


@dataclass(frozen=True)
class CITestResult:
    """Result of a conditional independence test.

    :param independent: Whether the null hypothesis (independence) is accepted.
    :param p_value: p-value of the test statistic.
    :param statistic: The raw test statistic value.
    :param conditioning_set: The set of variable indices conditioned on.
    """

    independent: bool
    p_value: float
    statistic: float
    conditioning_set: frozenset[int]


def causal_ci_test_fisher_z(
    data: np.ndarray,
    x_idx: int,
    y_idx: int,
    conditioning_idxs: list[int],
    alpha: float = 0.05,
) -> CITestResult:
    r"""Fisher's z-transform test for conditional independence.

    Tests :math:`H_0: X \perp\!\!\!\perp Y \mid Z` using partial correlations
    and Fisher's z-transform. Assumes the data follows a multivariate
    Gaussian distribution.

    The test statistic is:

    .. math::

        z = \frac{1}{2} \sqrt{n - |Z| - 3} \cdot
            \ln\left(\frac{1 + r_{XY \cdot Z}}{1 - r_{XY \cdot Z}}\right)

    where :math:`r_{XY \cdot Z}` is the partial correlation of X and Y
    given Z, and n is the sample size.

    :param data: Data matrix of shape ``(n_samples, n_variables)``.
    :param x_idx: Column index of variable X.
    :param y_idx: Column index of variable Y.
    :param conditioning_idxs: Column indices of conditioning variables Z.
    :param alpha: Significance level for the test.
    :returns: CITestResult with independence decision.
    :raises ValueError: If there are insufficient samples for the test.
    """
    n_samples = data.shape[0]
    cond_set = frozenset(conditioning_idxs)
    n_cond = len(conditioning_idxs)

    min_samples = n_cond + 3 + 1
    if n_samples < min_samples:
        raise ValueError(
            f"Insufficient samples for Fisher-z test: got {n_samples}, "
            f"need at least {min_samples} (conditioning set size={n_cond})"
        )

    partial_corr = _partial_correlation(data, x_idx, y_idx, conditioning_idxs)

    partial_corr = np.clip(partial_corr, -1 + 1e-10, 1 - 1e-10)

    # Fisher z-transform
    z_val = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr))
    # Standard error under H0
    se = 1.0 / np.sqrt(n_samples - n_cond - 3)
    # Test statistic (standard normal under H0)
    statistic = float(abs(z_val / se))

    p_value = float(2.0 * (1.0 - stats.norm.cdf(statistic)))

    return CITestResult(
        independent=p_value >= alpha,
        p_value=p_value,
        statistic=statistic,
        conditioning_set=cond_set,
    )


def causal_ci_test_kernel(
    data: np.ndarray,
    x_idx: int,
    y_idx: int,
    conditioning_idxs: list[int],
    alpha: float = 0.05,
    n_permutations: int = 200,
    max_samples: int = 1000,
) -> CITestResult:
    r"""Kernel-based conditional independence test using HSIC.

    Uses the Hilbert-Schmidt Independence Criterion (HSIC) with RBF
    kernels and a permutation test for significance.  HSIC detects
    arbitrary non-linear dependencies, unlike the Fisher-z test which
    only detects linear associations.

    For the conditional case :math:`X \perp\!\!\!\perp Y \mid Z`:

    1. Regress X on Z using kernel smoothing to get residuals
       :math:`\epsilon_X`.
    2. Regress Y on Z using kernel smoothing to get residuals
       :math:`\epsilon_Y`.
    3. Test independence of residuals via HSIC permutation test.

    :param data: Data matrix of shape ``(n_samples, n_variables)``.
    :param x_idx: Column index of variable X.
    :param y_idx: Column index of variable Y.
    :param conditioning_idxs: Column indices of conditioning variables Z.
    :param alpha: Significance level.
    :param n_permutations: Number of permutations for the permutation test.
    :param max_samples: Maximum samples to use (subsamples if exceeded).
    :returns: CITestResult with independence decision.
    """
    n_samples = data.shape[0]
    cond_set = frozenset(conditioning_idxs)

    # Subsample for computational efficiency (HSIC is O(n^2))
    if n_samples > max_samples:
        rng_sub = np.random.RandomState(42)
        idx = rng_sub.choice(n_samples, max_samples, replace=False)
        data = data[idx]
        n_samples = max_samples

    x = data[:, x_idx]  # (n_samples,)
    y = data[:, y_idx]  # (n_samples,)

    if len(conditioning_idxs) == 0:
        K_x_c = _centered_rbf_kernel(x)  # (n_samples, n_samples)
        K_y_c = _centered_rbf_kernel(y)  # (n_samples, n_samples)
    else:
        z = data[:, conditioning_idxs]  # (n_samples, n_cond)
        res_x = _kernel_residuals(x, z)
        res_y = _kernel_residuals(y, z)
        K_x_c = _centered_rbf_kernel(res_x)  # (n_samples, n_samples)
        K_y_c = _centered_rbf_kernel(res_y)  # (n_samples, n_samples)

    observed_stat = float(np.sum(K_x_c * K_y_c)) / (n_samples * n_samples)

    rng = np.random.RandomState(42)
    count_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n_samples)
        # Permuting rows+cols of centered K is equivalent to
        # centering the permuted kernel (proven via centering algebra)
        K_x_perm = K_x_c[np.ix_(perm, perm)]
        perm_stat = float(np.sum(K_x_perm * K_y_c)) / (n_samples * n_samples)
        if perm_stat >= observed_stat:
            count_ge += 1

    p_value = float((count_ge + 1) / (n_permutations + 1))

    return CITestResult(
        independent=p_value >= alpha,
        p_value=p_value,
        statistic=float(observed_stat),
        conditioning_set=cond_set,
    )


def _hsic_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the biased HSIC statistic between 1D arrays.

    Uses RBF kernels with the median heuristic for bandwidth selection.
    HSIC measures non-linear statistical dependence; it equals zero iff
    X and Y are independent (in the population, for characteristic kernels).

    :param x: First variable, shape ``(n,)``.
    :param y: Second variable, shape ``(n,)``.
    :returns: HSIC statistic (non-negative scalar).
    """
    n = len(x)
    K_x_c = _centered_rbf_kernel(x)  # (n, n)
    K_y_c = _centered_rbf_kernel(y)  # (n, n)
    return float(np.sum(K_x_c * K_y_c)) / (n * n)


def _centered_rbf_kernel(x: np.ndarray) -> np.ndarray:
    """Compute a double-centered RBF kernel matrix for a 1D array.

    Bandwidth is set via the median heuristic (median of pairwise
    distances).

    :param x: Input array, shape ``(n,)``.
    :returns: Double-centered kernel matrix, shape ``(n, n)``.
    """
    dists_sq = (x[:, None] - x[None, :]) ** 2  # (n, n)
    # Median heuristic: bandwidth = median of pairwise distances
    n = len(x)
    upper_tri = dists_sq[np.triu_indices(n, k=1)]
    bandwidth = float(np.sqrt(np.median(upper_tri))) + 1e-10
    K = np.exp(-dists_sq / (2.0 * bandwidth * bandwidth))  # (n, n)
    # Double center: K_c = K - row_mean - col_mean + grand_mean
    row_mean = K.mean(axis=1, keepdims=True)  # (n, 1)
    col_mean = K.mean(axis=0, keepdims=True)  # (1, n)
    grand_mean = K.mean()
    return K - row_mean - col_mean + grand_mean  # (n, n)


def _partial_correlation(
    data: np.ndarray,
    x_idx: int,
    y_idx: int,
    cond_idxs: list[int],
) -> float:
    """Compute the partial correlation of X and Y given Z.

    Uses the recursive formula via the correlation matrix when the
    conditioning set is small, or linear regression residuals otherwise.

    :param data: Data matrix of shape ``(n_samples, n_variables)``.
    :param x_idx: Column index of X.
    :param y_idx: Column index of Y.
    :param cond_idxs: Column indices of Z.
    :returns: Partial correlation coefficient in [-1, 1].
    """
    if len(cond_idxs) == 0:
        return float(np.corrcoef(data[:, x_idx], data[:, y_idx])[0, 1])

    # Use regression-based approach: regress X and Y on Z, correlate residuals
    z = data[:, cond_idxs]  # (n_samples, n_cond)
    x = data[:, x_idx]  # (n_samples,)
    y = data[:, y_idx]  # (n_samples,)

    # Add intercept
    z_aug = np.column_stack([np.ones(z.shape[0]), z])  # (n_samples, n_cond+1)

    # Least squares regression
    coeffs_x, _, _, _ = np.linalg.lstsq(z_aug, x, rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(z_aug, y, rcond=None)

    res_x = x - z_aug @ coeffs_x  # (n_samples,)
    res_y = y - z_aug @ coeffs_y  # (n_samples,)

    corr_matrix = np.corrcoef(res_x, res_y)
    return float(corr_matrix[0, 1])


def _kernel_residuals(
    target: np.ndarray,
    predictors: np.ndarray,
    bandwidth: float | None = None,
) -> np.ndarray:
    """Compute residuals from Nadaraya-Watson kernel regression.

    :param target: Target variable of shape ``(n_samples,)``.
    :param predictors: Predictor matrix of shape ``(n_samples, n_predictors)``.
    :param bandwidth: Kernel bandwidth. If None, uses Scott's rule.
    :returns: Residuals of shape ``(n_samples,)``.
    """
    n_samples, n_pred = predictors.shape

    if bandwidth is None:
        bandwidth = float(n_samples ** (-1.0 / (n_pred + 4)))

    # Pairwise squared distances
    # (n_samples, n_samples)
    diffs = predictors[:, np.newaxis, :] - predictors[np.newaxis, :, :]
    sq_dists = np.sum(diffs**2, axis=2)

    # RBF kernel weights
    weights = np.exp(-sq_dists / (2.0 * bandwidth**2))
    weight_sums = weights.sum(axis=1, keepdims=True)
    weights = weights / np.maximum(weight_sums, 1e-10)

    # Kernel regression prediction
    predictions = weights @ target  # (n_samples,)

    return target - predictions
