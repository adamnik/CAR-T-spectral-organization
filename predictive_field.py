# Shared predictive-field calculations

import numpy as np


METRICS = [
    "prognostic_channeling",
    "signed_graph_variation",
    "signed_neighbor_cosine",
]
DESIRED_SIGN = np.asarray([1.0, -1.0, 1.0])


def weighted_correlation(values, weights):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    centered = values - np.sum(weights[:, None] * values, axis=0)
    covariance = (centered * weights[:, None]).T @ centered
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    correlation = covariance / scale[:, None] / scale[None, :]
    correlation = np.clip((correlation + correlation.T) / 2, -1, 1)
    np.fill_diagonal(correlation, 1)
    return correlation


def dataset_residuals(values, weights, datasets):
    residuals = np.asarray(values, dtype=float).copy()
    for dataset in np.unique(datasets):
        keep = datasets == dataset
        group_weights = weights[keep]
        group_weights = group_weights / group_weights.sum()
        residuals[keep] -= np.sum(
            values[keep] * group_weights[:, None], axis=0
        )
    return residuals


def field_metrics(correlation, beta):
    beta = np.asarray(beta, dtype=float)
    first, second = np.triu_indices(len(beta), 1)
    edge = correlation[first, second]
    weight = edge**2
    sign = np.where(edge < 0, -1.0, 1.0)

    degree = np.zeros(len(beta))
    np.add.at(degree, first, weight)
    np.add.at(degree, second, weight)
    adjacency = np.zeros_like(correlation)
    adjacency[first, second] = weight * sign
    adjacency[second, first] = weight * sign
    neighbor = adjacency @ beta / np.maximum(degree, 1e-12)

    beta_norm = np.maximum(beta @ beta, 1e-12)
    variation_norm = np.maximum(np.sum(degree * beta**2), 1e-12)
    cosine_norm = np.sqrt(
        np.maximum(np.sum(degree * beta**2), 1e-12)
        * np.maximum(np.sum(degree * neighbor**2), 1e-12)
    )
    return np.asarray([
        beta @ correlation @ beta / beta_norm,
        np.sum(weight * (beta[first] - sign * beta[second])**2) / variation_norm,
        np.sum(degree * beta * neighbor) / cosine_norm,
    ])


def field_contrast(high_correlation, low_correlation, beta):
    return field_metrics(high_correlation, beta) - field_metrics(low_correlation, beta)


def permute_within_dataset(values, dataset_indices, rng):
    permuted = np.asarray(values, dtype=float).copy()
    for index in dataset_indices:
        permuted[index] = permuted[rng.permutation(index)]
    return permuted


def permutation_summary(observed, null):
    null_mean = null.mean(axis=0)
    null_sd = np.maximum(null.std(axis=0, ddof=1), 1e-12)
    z = (observed - null_mean) / null_sd
    p = (
        1 + np.sum(null * DESIRED_SIGN >= observed * DESIRED_SIGN, axis=0)
    ) / (len(null) + 1)

    oriented_z = z * DESIRED_SIGN
    joint_z = oriented_z.mean()
    joint_null = ((null - null_mean) / null_sd * DESIRED_SIGN).mean(axis=1)
    joint_p = (1 + np.sum(joint_null >= joint_z)) / (len(null) + 1)
    return z, p, joint_z, joint_p, joint_null


def benjamini_hochberg(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    answer = np.empty(len(values), dtype=float)
    answer[order] = np.minimum(adjusted, 1)
    return answer
