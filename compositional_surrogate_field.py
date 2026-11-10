# Constraint-aware clinical fields for CLR predictive geometry

import numpy as np


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
