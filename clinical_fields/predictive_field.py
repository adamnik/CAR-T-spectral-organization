# Shared predictive-field calculations

import json

import numpy as np
import pandas as pd


METRICS = ['prognostic_channeling', 'signed_graph_variation', 'signed_neighbor_cosine']
DESIRED_SIGN = np.asarray([1.0, -1.0, 1.0])


def weighted_correlation_and_scale(values, weights):
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Correlation weights have no positive finite mass.')
    weights = weights / total
    centered = values - np.sum(weights[:, None] * values, axis=0)
    covariance = (centered * weights[:, None]).T @ centered
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    correlation = covariance / scale[:, None] / scale[None, :]
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1)
    return correlation, scale


def weighted_correlation(values, weights):
    return weighted_correlation_and_scale(values, weights)[0]


def dataset_residuals(values, weights, datasets):
    residuals = np.asarray(values, dtype=float).copy()
    for dataset in np.unique(datasets):
        keep = datasets == dataset
        group_weights = weights[keep]
        group_weights = group_weights / group_weights.sum()
        residuals[keep] -= np.sum(values[keep] * group_weights[:, None], axis=0)
    return residuals


def posterior_states(values, particle_weights, datasets, draws, random_state):
    rng = np.random.default_rng(random_state)
    patients, particles, features = values.shape
    adjusted = np.empty((draws, patients, features), dtype=np.float32)
    bootstrap_weights = np.empty((draws, patients), dtype=np.float32)
    for draw in range(draws):
        particle_index = np.asarray([
            rng.choice(particles, p=particle_weights[patient])
            for patient in range(patients)
        ])
        weights = rng.dirichlet(np.ones(patients))
        state = values[np.arange(patients), particle_index]
        adjusted[draw] = dataset_residuals(state, weights, datasets)
        bootstrap_weights[draw] = weights
    return adjusted, bootstrap_weights


def mean_correlations(states, bootstrap_weights, membership, indices, feature_indices=None,
                      return_scales=False):
    if feature_indices is None:
        feature_indices = np.arange(states.shape[2])
    adverse_sum = np.zeros((len(feature_indices), len(feature_indices)), dtype=float)
    favorable_sum = np.zeros_like(adverse_sum)
    if return_scales:
        adverse_scales = np.empty((states.shape[0], len(feature_indices)), dtype=np.float32)
        favorable_scales = np.empty_like(adverse_scales)
    membership = np.asarray(membership, dtype=float)
    for draw in range(states.shape[0]):
        values = states[draw, indices][:, feature_indices]
        weights = bootstrap_weights[draw, indices]
        adverse_correlation, adverse_scale = weighted_correlation_and_scale(
            values, weights * membership
        )
        favorable_correlation, favorable_scale = weighted_correlation_and_scale(
            values, weights * (1 - membership)
        )
        adverse_sum += adverse_correlation
        favorable_sum += favorable_correlation
        if return_scales:
            adverse_scales[draw] = adverse_scale
            favorable_scales[draw] = favorable_scale
    adverse = adverse_sum / states.shape[0]
    favorable = favorable_sum / states.shape[0]
    if return_scales:
        return adverse, favorable, adverse_scales, favorable_scales
    return adverse, favorable


def field_metrics(correlation, beta):
    beta = np.asarray(beta, dtype=float)
    first, second = np.triu_indices(len(beta), 1)
    edge = correlation[first, second]
    # Edge strength is squared correlation; retain the sign separately
    weight = edge**2
    sign = np.where(edge < 0, -1.0, 1.0)

    degree = np.zeros(len(beta))
    np.add.at(degree, first, weight)
    np.add.at(degree, second, weight)
    A = np.zeros_like(correlation)
    A[first, second] = weight * sign
    A[second, first] = weight * sign
    neighbor_beta = A @ beta / np.maximum(degree, 1e-12)

    beta_norm = np.maximum(beta @ beta, 1e-12)
    variation_norm = np.maximum(np.sum(degree * beta**2), 1e-12)
    cosine_norm = np.sqrt(
        np.maximum(np.sum(degree * beta**2), 1e-12)
        * np.maximum(np.sum(degree * neighbor_beta**2), 1e-12)
    )
    channeling = beta @ correlation @ beta / beta_norm
    edge_difference = beta[first] - sign * beta[second]
    variation = np.sum(weight * edge_difference**2) / variation_norm
    cosine = np.sum(degree * beta * neighbor_beta) / cosine_norm
    return np.asarray([channeling, variation, cosine])


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
    more_extreme = null * DESIRED_SIGN >= observed * DESIRED_SIGN
    p = (1 + np.sum(more_extreme, axis=0)) / (len(null) + 1)

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


def particle_profiles(path, membership, prefix):
    with np.load(path, allow_pickle=False) as archive:
        names = archive['feature_names'].astype(str)
        keep = np.char.startswith(names, prefix)
        if not keep.any():
            raise ValueError(f'No particle features start with {prefix}')
        patients = pd.DataFrame({
            'Dataset': archive['datasets'].astype(str),
            'sample_id': archive['patient_ids'].astype(str),
            'particle_row': np.arange(len(archive['patient_ids'])),
        })
        patients = membership.merge(patients, on=['Dataset', 'sample_id'],
                                    how='left', validate='one_to_one')
        if patients.particle_row.isna().any():
            raise ValueError('Missing particle profiles for patients with regime memberships')
        patients = patients.sort_values(['Dataset', 'sample_id']).reset_index(drop=True)
        rows = patients.particle_row.to_numpy(int)
        values = archive['values'][rows][:, :, keep].astype(float)
        weights = archive['particle_weights'][rows].astype(float)
    weights /= weights.sum(axis=1, keepdims=True)
    return patients, names[keep], values, weights


def field_test(states, weights, membership, indices, groups, effects, removed,
               bundle_path, metadata, permutations, seed):
    # Observed mean matrices and the scaling needed for the CLR sensitivity.
    features = effects.index.to_numpy(str)
    keep = np.flatnonzero(~np.isin(features, removed))
    effects = effects.iloc[keep]
    beta = np.log(effects.HR_median.to_numpy(float))
    adverse, favorable, sd_high, sd_low = mean_correlations(
        states, weights, membership, indices, keep, return_scales=True
    )
    observed = field_contrast(adverse, favorable, beta)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        bundle_path, metadata_json=json.dumps(metadata), clinical_beta=beta,
        feature_names=features[keep], removed_coordinates=np.asarray(removed, dtype=str),
        coefficient_feature_scales=effects.scale.to_numpy(float),
        adverse_mean_correlation=adverse, favorable_mean_correlation=favorable,
        adverse_feature_sd_by_draw=sd_high, favorable_feature_sd_by_draw=sd_low,
    )

    rng = np.random.default_rng(seed)
    null = np.empty((permutations, len(METRICS)))
    for draw in range(permutations):
        permuted = permute_within_dataset(membership, groups, rng)
        high, low = mean_correlations(states, weights, permuted, indices, keep)
        null[draw] = field_contrast(high, low, beta)

    z, p, joint_z, joint_p, _ = permutation_summary(observed, null)
    result = {'joint_oriented_z': joint_z, 'joint_permutation_p': joint_p}
    for i, metric in enumerate(METRICS):
        result[f'contrast__{metric}'] = observed[i]
        result[f'z__{metric}'] = z[i]
        result[f'p__{metric}'] = p[i]
    return result, null


def combine_folds(folds, null_lookup, probability_column='permutation_p'):
    rows = []
    standardized = []
    for fold in folds.itertuples(index=False):
        null = null_lookup[fold.heldout_dataset]
        null_mean = null.mean(axis=0)
        null_sd = np.maximum(null.std(axis=0, ddof=1), 1e-12)
        standardized.append((null - null_mean) / null_sd * DESIRED_SIGN)
    standardized = np.stack(standardized)
    for metric_index, metric in enumerate(METRICS):
        observed = np.mean(folds[f'z__{metric}'].to_numpy(float) * DESIRED_SIGN[metric_index])
        null = standardized[:, :, metric_index].mean(axis=0)
        rows.append({
            'metric': metric, 'datasets': len(folds), 'mean_oriented_z': observed,
            probability_column: (1 + np.sum(null >= observed)) / (len(null) + 1),
            'datasets_in_expected_direction': int(np.sum(
                folds[f'contrast__{metric}'].to_numpy(float) * DESIRED_SIGN[metric_index] > 0
            )),
        })
    observed = folds.joint_oriented_z.mean()
    null = standardized.mean(axis=(0, 2))
    rows.append({
        'metric': 'three_metric_joint', 'datasets': len(folds),
        'mean_oriented_z': observed,
        probability_column: (1 + np.sum(null >= observed)) / (len(null) + 1),
        'datasets_in_expected_direction': int(np.sum(folds.joint_oriented_z > 0)),
    })
    return pd.DataFrame(rows)
