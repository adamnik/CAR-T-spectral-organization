# Shared calculations for the Bayesian survival models

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import t as student_t


TIME_EDGES = np.asarray([0, 1, 3, 6, 9, 12, 18, 24, 36, 60, 120], dtype=float)
FIRST_STAGE_DF = 4.0

GLOBAL_EFFECT_SD = 0.45
DATASET_EFFECT_SD = 0.40
BASELINE_RATE_MEDIAN = 0.05
BASELINE_LOG_RATE_SD = 1.00
BASELINE_SMOOTHNESS_SD = 0.30


def load_olambda_analysis(
    asymptote_path, outcome_path, minimum_cells, n_particles, particles_path=None
):
    asymptotes = pd.read_csv(asymptote_path)
    outcomes = pd.read_csv(outcome_path)
    if "event_observed" not in outcomes.columns and "event" in outcomes.columns:
        outcomes = outcomes.rename(columns={"event": "event_observed"})

    # Match patients across the two tables
    outcome_columns = ['Dataset', 'sample_id', 'event_observed', 't_start', 't_end']
    data = asymptotes.merge(
        outcomes[outcome_columns],
        on=["Dataset", "sample_id"],
        how="inner",
        validate="one_to_one",
    )
    numeric_columns = [
        "n_cells_available",
        "O_lambda_inf_central",
        "event_observed",
        "t_start",
        "t_end",
    ]
    if particles_path is None:
        candidate = Path(asymptote_path).parent / "patient_olambda_particles.csv"
        if candidate.exists():
            particles_path = candidate
    if particles_path is None:
        numeric_columns.extend([
            "log_O_minus_1_inf_location",
            "log_O_minus_1_inf_scale",
        ])
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    # Keep the analysis cohort
    data = data.dropna(subset=numeric_columns).copy()
    keep = (
        (data.n_cells_available >= minimum_cells)
        & data.event_observed.isin([0, 1])
        & (data.t_end > 0)
        & (data.t_end >= data.t_start)
    )
    if particles_path is None:
        keep &= data.log_O_minus_1_inf_scale > 0
    data = data[keep].copy()
    data.event_observed = data.event_observed.astype("int32")
    data = data.sort_values(["Dataset", "sample_id"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("No patients satisfy the analysis requirements.")
    if data.t_end.max() > TIME_EDGES[-1]:
        raise ValueError(f'Follow-up extends beyond the last time edge ({TIME_EDGES[-1]} months).')

    if particles_path is not None:
        particle_table = pd.read_csv(particles_path)
        particle_columns = sorted(
            column for column in particle_table if column.startswith("particle_")
        )
        if len(particle_columns) != n_particles:
            raise ValueError(
                f"Expected {n_particles} O_lambda particles, found {len(particle_columns)}."
            )
        particle_table = particle_table[
            ["Dataset", "sample_id", *particle_columns]
        ]
        data = data.merge(
            particle_table,
            on=["Dataset", "sample_id"],
            how="left",
            validate="one_to_one",
        ).sort_values(["Dataset", "sample_id"]).reset_index(drop=True)
        raw_particles = data[particle_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        data = data.drop(columns=particle_columns)
        if np.any(~np.isfinite(raw_particles)) or np.any(raw_particles <= 1):
            raise ValueError("O_lambda particles must be finite and greater than one.")
        particle_source = str(particles_path)
    else:
        # Compatibility path for older asymptote tables without saved particles
        probabilities = (np.arange(n_particles) + 0.5) / n_particles
        offsets = student_t.ppf(probabilities, df=FIRST_STAGE_DF)
        z_particles = (
            data.log_O_minus_1_inf_location.to_numpy(float)[:, None]
            + data.log_O_minus_1_inf_scale.to_numpy(float)[:, None]
            * offsets[None, :]
        )
        raw_particles = 1.0 + np.exp(z_particles)
        particle_source = "Student-t4 approximation from asymptote summary"

    central = data.O_lambda_inf_central.to_numpy(float)
    center = float(central.mean())
    scale = float(central.std(ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The patient-level O_lambda scale is not positive.")
    standardized_particles = (raw_particles - center) / scale
    particle_weights = np.full((len(data), n_particles), 1.0 / n_particles)

    scaling = {
        'center': center,
        'scale': scale,
        'particle_source': particle_source,
    }
    return data, raw_particles, standardized_particles, particle_weights, scaling


def load_tcat_analysis(estimates_path, particles_path, outcome_path, minimum_cells):
    """Match patient TCAT particles, central estimates, and survival outcomes."""
    estimates = pd.read_csv(estimates_path)
    outcomes = pd.read_csv(outcome_path)
    if "event_observed" not in outcomes.columns and "event" in outcomes.columns:
        outcomes = outcomes.rename(columns={"event": "event_observed"})

    archive = np.load(particles_path, allow_pickle=False)
    values = np.asarray(archive["values"], dtype=float)
    particle_weights = np.asarray(archive["particle_weights"], dtype=float)
    patient_ids = archive["patient_ids"].astype(str)
    datasets = archive["datasets"].astype(str)
    feature_names = archive["feature_names"].astype(str).tolist()
    n_cells = np.asarray(archive["n_cells"], dtype=int)

    particle_index = pd.DataFrame(
        {
            "Dataset": datasets,
            "sample_id": patient_ids,
            "n_cells": n_cells,
            "particle_index": np.arange(len(patient_ids)),
        }
    )
    data = particle_index.merge(
        estimates[["Dataset", "sample_id", *feature_names]],
        on=["Dataset", "sample_id"],
        how="inner",
        validate="one_to_one",
    )
    outcome_columns = ["Dataset", "sample_id", "event_observed", "t_start", "t_end"]
    data = data.merge(
        outcomes[outcome_columns],
        on=["Dataset", "sample_id"],
        how="inner",
        validate="one_to_one",
    )

    numeric_columns = ["n_cells", "event_observed", "t_start", "t_end", *feature_names]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=numeric_columns).copy()
    keep = (
        (data.n_cells >= minimum_cells)
        & data.event_observed.isin([0, 1])
        & (data.t_end > 0)
        & (data.t_end >= data.t_start)
    )
    data = data.loc[keep].sort_values(["Dataset", "sample_id"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("No patients satisfy the TCAT analysis requirements.")
    if data.t_end.max() > TIME_EDGES[-1]:
        raise ValueError(f"Follow-up extends beyond the last time edge ({TIME_EDGES[-1]} months).")

    order = data.particle_index.to_numpy(int)
    values = values[order]
    particle_weights = particle_weights[order]
    if values.shape[:2] != particle_weights.shape or values.shape[2] != len(feature_names):
        raise ValueError("TCAT particle dimensions do not match the saved metadata.")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(particle_weights)):
        raise ValueError("TCAT particles and weights must be finite.")
    particle_weights = particle_weights / particle_weights.sum(axis=1, keepdims=True)
    central = data[feature_names].to_numpy(float)
    data = data.drop(columns="particle_index")
    data.event_observed = data.event_observed.astype("int32")
    return data, central, values, particle_weights, feature_names


def exposure(start, end):
    """Months spent in each hazard bin between start and end."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    return np.maximum(
        0.0,
        np.minimum(end[:, None], TIME_EDGES[1:][None, :])
        - np.maximum(start[:, None], TIME_EDGES[:-1][None, :]),
    )


def survival_arrays(data):
    datasets = sorted(data.Dataset.astype(str).unique())
    dataset_index = {dataset: i for i, dataset in enumerate(datasets)}
    dataset_index = data.Dataset.astype(str).map(dataset_index).to_numpy("int32")
    events = data.event_observed.to_numpy("int32")
    t_start = data.t_start.to_numpy(float)
    t_end = data.t_end.to_numpy(float)
    exact = (events == 1) & np.isclose(t_start, t_end)
    event_bin = np.searchsorted(TIME_EDGES[1:], t_end, side='left')
    event_bin = np.clip(event_bin, 0, len(TIME_EDGES) - 2).astype('int32')

    return {
        "datasets": datasets,
        "dataset_index": dataset_index,
        "events": events,
        "exact": exact,
        "event_selector": np.eye(len(TIME_EDGES) - 1)[event_bin],
        "exposure_before_interval": exposure(np.zeros(len(data)), t_start),
        "exposure_during_interval": exposure(t_start, t_end),
    }


def model_coordinates(data, arrays):
    return {
        "observation": np.arange(len(data)),
        "dataset": arrays["datasets"],
        "time_bin": np.arange(len(TIME_EDGES) - 1),
        "time_transition": np.arange(len(TIME_EDGES) - 2),
    }


def baseline_parameters(arrays):
    baseline_initial = pm.Normal(
        "baseline_log_rate_initial",
        np.log(BASELINE_RATE_MEDIAN),
        BASELINE_LOG_RATE_SD,
        dims="dataset",
    )
    baseline_smoothness = pm.HalfNormal('baseline_smoothness', BASELINE_SMOOTHNESS_SD)
    baseline_innovation = pm.Normal('baseline_innovation', 0.0, 1.0, dims=('dataset', 'time_transition'))
    # Random walk across time bins
    midpoints = 0.5 * (TIME_EDGES[:-1] + TIME_EDGES[1:])
    increments = (
        baseline_smoothness * pt.sqrt(pt.as_tensor_variable(np.diff(midpoints)))[None, :]
        * baseline_innovation
    )
    baseline_log_rate = pm.Deterministic(
        "baseline_log_rate",
        pt.concatenate([
            baseline_initial[:, None],
            baseline_initial[:, None] + pt.cumsum(increments, axis=1),
        ], axis=1),
        dims=("dataset", "time_bin"),
    )

    dataset_index = pt.as_tensor_variable(arrays["dataset_index"])
    return baseline_log_rate[dataset_index]


def hierarchical_parameters(arrays):
    global_effect = pm.Normal("global_effect", 0.0, GLOBAL_EFFECT_SD)
    dataset_effect_sd = pm.HalfNormal("dataset_effect_sd", DATASET_EFFECT_SD)
    dataset_effect_raw = pm.Normal('dataset_effect_raw', 0.0, 1.0, dims='dataset')
    dataset_effect = pm.Deterministic(
        "dataset_effect",
        global_effect + dataset_effect_sd * dataset_effect_raw,
        dims="dataset",
    )
    dataset_index = pt.as_tensor_variable(arrays["dataset_index"])
    return baseline_parameters(arrays), dataset_effect[dataset_index]


def interval_loglikelihood(hazard, arrays):
    """Log likelihood for patient x time-bin hazards, optionally with particles."""
    exposure_before = pt.as_tensor_variable(arrays["exposure_before_interval"])
    exposure_during = pt.as_tensor_variable(arrays["exposure_during_interval"])
    event_selector = pt.as_tensor_variable(arrays["event_selector"])
    if hazard.ndim == 3:
        exposure_before = exposure_before[:, None, :]
        exposure_during = exposure_during[:, None, :]
        event_selector = event_selector[:, None, :]

    H_start = pt.sum(hazard * exposure_before, axis=-1)
    H_interval = pt.sum(hazard * exposure_during, axis=-1)
    exact_hazard = pt.sum(hazard * event_selector, axis=-1)

    # Exact, interval-censored, and right-censored observations
    # Small floors avoid log(0) in inactive branches
    log_exact = pt.log(pt.maximum(exact_hazard, 1e-30)) - H_start
    log_interval = -H_start + pt.log(-pt.expm1(-pt.maximum(H_interval, 1e-30)))
    log_censored = -(H_start + H_interval)

    exact = pt.as_tensor_variable(arrays["exact"])
    interval_event = pt.as_tensor_variable((arrays['events'] == 1) & ~arrays['exact'])
    if hazard.ndim == 3:
        exact = exact[:, None]
        interval_event = interval_event[:, None]

    return pt.switch(exact, log_exact, pt.switch(interval_event, log_interval, log_censored))


def marginalize_particles(conditional_loglikelihood, particle_weights):
    log_weights = pt.log(pt.as_tensor_variable(particle_weights))
    return pm.math.logsumexp(conditional_loglikelihood + log_weights, axis=1)


def effect_summary(effect):
    """Posterior hazard-ratio interval and probability of a positive effect."""
    hazard_ratio = np.exp(effect)
    return {
        "HR_median": float(np.median(hazard_ratio)),
        "HR_q025": float(np.quantile(hazard_ratio, 0.025)),
        "HR_q975": float(np.quantile(hazard_ratio, 0.975)),
        "P_effect_gt_0": float(np.mean(effect > 0)),
    }


def alignment_summary(dataset_effect):
    all_positive = np.all(dataset_effect > 0, axis=1)
    all_negative = np.all(dataset_effect < 0, axis=1)
    return {
        "P_all_datasets_gt_0": float(np.mean(all_positive)),
        "P_all_datasets_lt_0": float(np.mean(all_negative)),
        "P_all_datasets_same_direction": float(np.mean(all_positive | all_negative)),
    }


def summarize_effects(idata, datasets):
    global_effect = np.asarray(idata.posterior.global_effect).reshape(-1)
    dataset_effect = idata.posterior.dataset_effect.stack(sample=('chain', 'draw'))
    dataset_effect = dataset_effect.transpose('sample', 'dataset').values
    heterogeneity = np.asarray(idata.posterior.dataset_effect_sd).reshape(-1)

    result = effect_summary(global_effect)
    result["dataset_effect_sd_median"] = float(np.median(heterogeneity))
    result.update(alignment_summary(dataset_effect))
    dataset_rows = []
    for index, dataset in enumerate(datasets):
        dataset_rows.append({"Dataset": dataset, **effect_summary(dataset_effect[:, index])})
    return result, pd.DataFrame(dataset_rows)


def diagnostics(idata, variable_names):
    summary = az.summary(idata, var_names=variable_names, round_to=None)
    values = {
        "divergences": int(np.asarray(idata.sample_stats.diverging).sum()),
        "max_rhat": float(summary.r_hat.max()),
        "min_ess_bulk": float(summary.ess_bulk.min()),
        "min_ess_tail": float(summary.ess_tail.min()),
    }
    return summary, values
