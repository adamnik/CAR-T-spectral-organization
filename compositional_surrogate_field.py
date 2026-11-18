# Constraint-aware clinical fields for CLR predictive geometry

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_io import safe_name
from analyze_cross_system_predictive_field import ASA_GEP_FEATURES
from predictive_field import METRICS, benjamini_hochberg, combine_folds, permutation_summary


SURROGATE_FAMILIES = [
    "permuted_field",
    "sign_randomized_field",
    "norm_matched_isotropic_field",
]


def raw_clr_field(beta, coefficient_scales, constrained=True):
    beta = np.asarray(beta, dtype=float)
    coefficient_scales = np.asarray(coefficient_scales, dtype=float)
    if not np.isfinite(coefficient_scales).all() or np.any(coefficient_scales <= 0):
        raise ValueError("Coefficient feature scales must be positive and finite.")
    field = beta / coefficient_scales
    if constrained:
        field -= field.mean()
    if np.linalg.norm(field) <= 1e-12:
        raise ValueError("The clinical field has zero norm.")
    return field


def mean_tangent_map(feature_sd_by_draw):
    feature_sd_by_draw = np.asarray(feature_sd_by_draw, dtype=float)
    if not np.isfinite(feature_sd_by_draw).all() or np.any(feature_sd_by_draw <= 0):
        raise ValueError("Feature-SD vectors must be positive and finite.")
    dimensions = feature_sd_by_draw.shape[1]
    identity = np.eye(dimensions)
    answer = np.zeros((dimensions, dimensions), dtype=float)
    for scale in feature_sd_by_draw:
        direction = scale / np.linalg.norm(scale)
        projection = identity - np.outer(direction, direction)
        answer += projection @ np.diag(scale)
    return answer / len(feature_sd_by_draw)


def mean_standardization_map(feature_sd_by_draw):
    feature_sd_by_draw = np.asarray(feature_sd_by_draw, dtype=float)
    if not np.isfinite(feature_sd_by_draw).all() or np.any(feature_sd_by_draw <= 0):
        raise ValueError("Feature-SD vectors must be positive and finite.")
    return np.diag(feature_sd_by_draw.mean(axis=0))


def regime_field_map(feature_sd_by_draw, constrained=True):
    if constrained:
        return mean_tangent_map(feature_sd_by_draw)
    return mean_standardization_map(feature_sd_by_draw)


def normalize_rows(values, target_norm):
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("A surrogate field has zero norm.")
    return values * (target_norm / norms)[:, None]


def surrogate_fields(field, family, count, random_state=0, constrained=True):
    field = np.asarray(field, dtype=float)
    rng = np.random.default_rng(random_state)
    dimensions = len(field)
    target_norm = np.linalg.norm(field)

    if family == "permuted_field":
        order = np.argsort(rng.random((count, dimensions)), axis=1)
        values = field[order]
    elif family == "sign_randomized_field":
        values = np.abs(field)[None, :] * rng.choice(
            [-1.0, 1.0], size=(count, dimensions)
        )
    elif family == "norm_matched_isotropic_field":
        values = rng.normal(size=(count, dimensions))
    else:
        raise ValueError(f"Unknown surrogate family: {family}")

    if constrained:
        values -= values.mean(axis=1, keepdims=True)
    return normalize_rows(values, target_norm)


def transformed_fields(raw_fields, feature_sd_by_draw, constrained=True):
    raw_fields = np.atleast_2d(np.asarray(raw_fields, dtype=float))
    transformation = regime_field_map(feature_sd_by_draw, constrained)
    return raw_fields @ transformation.T


def projection_diagnostics(feature_sd_by_draw, raw_field):
    residuals = []
    fractions_removed = []
    directions = []
    for scale in np.asarray(feature_sd_by_draw, dtype=float):
        direction = scale / np.linalg.norm(scale)
        unprojected = scale * raw_field
        projected = unprojected - direction * (direction @ unprojected)
        residuals.append(
            abs(direction @ projected) / max(np.linalg.norm(projected), 1e-12)
        )
        fractions_removed.append(
            np.linalg.norm(unprojected - projected)
            / max(np.linalg.norm(unprojected), 1e-12)
        )
        directions.append(direction)
    directions = np.asarray(directions)
    representative = directions.mean(axis=0)
    representative /= np.linalg.norm(representative)
    cosines = directions @ representative
    return {
        "maximum_constraint_residual": float(np.max(residuals)),
        "median_fraction_field_norm_removed": float(np.median(fractions_removed)),
        "maximum_fraction_field_norm_removed": float(np.max(fractions_removed)),
        "minimum_constraint_direction_cosine": float(np.min(cosines)),
    }


def field_metric_matrices(correlation):
    correlation = np.asarray(correlation, dtype=float)
    first, second = np.triu_indices(len(correlation), 1)
    edge = correlation[first, second]
    weight = edge ** 2
    sign = np.where(edge < 0, -1.0, 1.0)

    degree = np.zeros(len(correlation), dtype=float)
    np.add.at(degree, first, weight)
    np.add.at(degree, second, weight)

    adjacency = np.zeros_like(correlation)
    adjacency[first, second] = weight * sign
    adjacency[second, first] = weight * sign
    laplacian = np.diag(degree) - adjacency
    return correlation, adjacency, laplacian, degree


def field_metrics_batch(correlation, fields):
    fields = np.atleast_2d(np.asarray(fields, dtype=float))
    correlation, adjacency, laplacian, degree = field_metric_matrices(correlation)

    field_norm = np.maximum(np.sum(fields ** 2, axis=1), 1e-12)
    channeling = np.sum((fields @ correlation) * fields, axis=1) / field_norm

    degree_energy = np.maximum(np.sum(fields ** 2 * degree[None, :], axis=1), 1e-12)
    variation = np.sum((fields @ laplacian) * fields, axis=1) / degree_energy

    neighbor = (fields @ adjacency) / np.maximum(degree[None, :], 1e-12)
    neighbor_energy = np.maximum(
        np.sum(neighbor ** 2 * degree[None, :], axis=1), 1e-12
    )
    cosine = (
        np.sum(fields * neighbor * degree[None, :], axis=1)
        / np.sqrt(degree_energy * neighbor_energy)
    )
    return np.column_stack([channeling, variation, cosine])



def evaluate_fields(path, count, seed, unprojected=False):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata_json']))
        beta = archive['clinical_beta']
        adverse = archive['adverse_mean_correlation']
        favorable = archive['favorable_mean_correlation']
        removed = set(archive['removed_coordinates'].astype(str))
        if meta['representation'] == 'TCAT_GEP' and meta['context'] == 'TCAT_SCORE_ASA_CLR':
            if removed != set(ASA_GEP_FEATURES) or removed.intersection(archive['feature_names']):
                raise ValueError('Recompute the ASA geometry with its four component GEPs excluded')
        constrained = not removed
        if unprojected:
            raw = beta
            high_map = low_map = np.eye(len(beta))
        else:
            raw = raw_clr_field(beta, archive['coefficient_feature_scales'], constrained)
            high_map = regime_field_map(archive['adverse_feature_sd_by_draw'], constrained)
            low_map = regime_field_map(archive['favorable_feature_sd_by_draw'], constrained)

    adverse_metrics = field_metrics_batch(adverse, high_map @ raw)
    favorable_metrics = field_metrics_batch(favorable, low_map @ raw)
    observed = (adverse_metrics - favorable_metrics)[0]
    for i, family in enumerate(SURROGATE_FAMILIES):
        family_seed = seed + 100 * i
        if unprojected and family == 'permuted_field':
            # Notebook 78 permuted each vector separately.
            rng = np.random.default_rng(family_seed)
            fields = np.stack([rng.permutation(raw) for _ in range(count)])
        else:
            fields = surrogate_fields(raw, family, count, family_seed, constrained and not unprojected)
        adverse_metrics = field_metrics_batch(adverse, fields @ high_map.T)
        favorable_metrics = field_metrics_batch(favorable, fields @ low_map.T)
        null = adverse_metrics - favorable_metrics
        z, p, joint_z, joint_p, _ = permutation_summary(observed, null)
        row = {'representation': meta['representation'], 'context': meta['context'],
               'heldout_dataset': meta.get('heldout_dataset', ''), 'surrogate_family': family,
               'joint_oriented_z': joint_z, 'joint_surrogate_p': joint_p}
        for j, metric in enumerate(METRICS):
            row[f'contrast__{metric}'] = observed[j]
            row[f'z__{metric}'] = z[j]
            row[f'p__{metric}'] = p[j]
        yield row, null


def main():
    parser = argparse.ArgumentParser(description='Clinical-field specificity and Table S5')
    parser.add_argument('pooled_dir', type=Path)
    parser.add_argument('heldout_dir', type=Path, help='Parent of the TCAT_GEP and atlas_cell_type fold outputs')
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--surrogates', type=int, default=10000)
    parser.add_argument('--random-state', type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pooled = []
    contexts = pd.read_csv(args.pooled_dir / 'selected_contexts.csv').context
    paths = sorted(args.pooled_dir / 'compositional_sensitivity_bundles' / field / (safe_name(context) + '.npz')
                   for field in ['TCAT_GEP', 'atlas_cell_type'] for context in contexts)
    if not paths:
        raise ValueError('No pooled field bundles')
    for number, path in enumerate(paths, start=1):
        seed = args.random_state + 100000 + 1000 * number
        for row, _ in evaluate_fields(path, args.surrogates, seed):
            pooled.append(row)
    pooled = pd.DataFrame(pooled)
    groups = pooled.groupby(['representation', 'surrogate_family'])
    for column, output in [('joint_surrogate_p', 'joint_FDR')] + [(f'p__{m}', f'FDR__{m}') for m in METRICS]:
        pooled[output] = groups[column].transform(benjamini_hochberg)
    pooled.to_csv(args.output_dir / 'pooled_compositional_surrogate_results.csv', index=False)

    combined_rows = []
    for representation in ['TCAT_GEP', 'atlas_cell_type']:
        fold_dir = args.heldout_dir / representation
        support = pd.read_csv(fold_dir / 'table_s4.csv').sort_values('Dataset')
        eligible = support.loc[support.eligible, 'Dataset'].tolist()
        if not eligible:
            raise ValueError(f'No eligible held-out cohorts for {representation}')
        for unprojected in [False, True]:
            fold_rows = []
            nulls = {family: {} for family in SURROGATE_FAMILIES}
            for number, dataset in enumerate(eligible, start=1):
                # Notebook 78 indexed all folds. Notebook 80 indexed eligible folds.
                fold_number = support.Dataset.tolist().index(dataset) + 1 if unprojected else number
                seed = args.random_state + 1000 * fold_number
                path = fold_dir / 'compositional_sensitivity_bundles' / (safe_name(dataset) + '__O_lambda.npz')
                for row, null in evaluate_fields(path, args.surrogates, seed, unprojected):
                    fold_rows.append(row)
                    nulls[row['surrogate_family']][dataset] = null
            folds = pd.DataFrame(fold_rows)
            for family, frame in folds.groupby('surrogate_family'):
                combined = combine_folds(frame, nulls[family], 'surrogate_p')
                combined['representation'] = representation
                combined['surrogate_family'] = family
                combined['unprojected'] = unprojected
                combined_rows.append(combined)
    combined = pd.concat(combined_rows, ignore_index=True)
    combined.to_csv(args.output_dir / 'crossfit_surrogate_combined.csv', index=False)

    keys = ['representation', 'surrogate_family']
    spectral = pooled.loc[pooled.context.eq('O_lambda'), keys + ['joint_surrogate_p']]
    joint = combined.loc[~combined.unprojected & combined.metric.eq('three_metric_joint')]
    table = spectral.merge(joint[keys + ['surrogate_p']], on=keys, validate='one_to_one')
    if len(table) != 2 * len(SURROGATE_FAMILIES):
        raise ValueError('Incomplete pooled or held-out O-lambda results')
    table.rename(columns={'joint_surrogate_p': 'pooled', 'surrogate_p': 'heldout'}).to_csv(
        args.output_dir / 'table_s5.csv', index=False)


if __name__ == '__main__':
    main()
