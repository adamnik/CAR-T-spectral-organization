"""Shared interval-censored Bayesian survival utilities."""

import json
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


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def load_olambda_analysis(asymptote_path, outcome_path, minimum_cells, n_particles):
    asymptotes = pd.read_csv(asymptote_path)
    outcomes = pd.read_csv(outcome_path)
    if "event_observed" not in outcomes.columns and "event" in outcomes.columns:
        outcomes = outcomes.rename(columns={"event": "event_observed"})

    estimate_columns = {
        "Dataset",
        "sample_id",
        "n_cells_available",
        "O_lambda_inf_central",
        "log_O_minus_1_inf_location",
        "log_O_minus_1_inf_scale",
    }
    outcome_columns = {
        "Dataset",
        "sample_id",
        "event_observed",
        "t_start",
        "t_end",
    }
    missing_estimates = estimate_columns.difference(asymptotes.columns)
    missing_outcomes = outcome_columns.difference(outcomes.columns)
    if missing_estimates:
        raise ValueError(f"Asymptote file is missing columns: {sorted(missing_estimates)}")
    if missing_outcomes:
        raise ValueError(f"Outcome file is missing columns: {sorted(missing_outcomes)}")
    if asymptotes.duplicated(["Dataset", "sample_id"]).any():
        raise ValueError("Asymptote file contains duplicated dataset/patient rows.")
    if outcomes.duplicated(["Dataset", "sample_id"]).any():
        raise ValueError("Outcome file contains duplicated dataset/patient rows.")

    outcome_columns_ordered = [
        "Dataset", "sample_id", "event_observed", "t_start", "t_end"
    ]
    data = asymptotes.merge(
        outcomes[outcome_columns_ordered],
        on=["Dataset", "sample_id"],
        how="inner",
        validate="one_to_one",
    )
    numeric_columns = [
        "n_cells_available",
        "O_lambda_inf_central",
        "log_O_minus_1_inf_location",
        "log_O_minus_1_inf_scale",
        "event_observed",
        "t_start",
        "t_end",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=numeric_columns).copy()
    data = data.loc[
        data.n_cells_available.ge(minimum_cells)
        & data.log_O_minus_1_inf_scale.gt(0)
        & data.event_observed.isin([0, 1])
        & data.t_end.gt(0)
        & data.t_end.ge(data.t_start)
    ].copy()
    data.event_observed = data.event_observed.astype("int32")
    data = data.sort_values(["Dataset", "sample_id"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("No patients satisfy the analysis requirements.")
    if data.t_end.max() > TIME_EDGES[-1]:
        raise ValueError(
            f"Follow-up extends beyond the last time edge ({TIME_EDGES[-1]} months)."
        )

    central = data.O_lambda_inf_central.to_numpy(float)
    center = float(central.mean())
    scale = float(central.std(ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The patient-level O_lambda scale is not positive.")

    probabilities = (np.arange(n_particles) + 0.5) / n_particles
    offsets = student_t.ppf(probabilities, df=FIRST_STAGE_DF)
    z_particles = (
        data.log_O_minus_1_inf_location.to_numpy(float)[:, None]
        + data.log_O_minus_1_inf_scale.to_numpy(float)[:, None]
        * offsets[None, :]
    )
    raw_particles = 1.0 + np.exp(z_particles)
    standardized_particles = (raw_particles - center) / scale
    particle_weights = np.full((len(data), n_particles), 1.0 / n_particles)

    return data, raw_particles, standardized_particles, particle_weights, {
        "center": center,
        "scale": scale,
    }


def exposure(start, end):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    return np.maximum(
        0.0,
        np.minimum(end[:, None], TIME_EDGES[1:][None, :])
        - np.maximum(start[:, None], TIME_EDGES[:-1][None, :]),
    )


def survival_arrays(data):
    datasets = sorted(data.Dataset.astype(str).unique())
    dataset_index = {
        dataset: index for index, dataset in enumerate(datasets)
    }
    dataset_index = data.Dataset.astype(str).map(dataset_index).to_numpy("int32")
    events = data.event_observed.to_numpy("int32")
    t_start = data.t_start.to_numpy(float)
    t_end = data.t_end.to_numpy(float)
    exact = (events == 1) & np.isclose(t_start, t_end)
    event_bin = np.clip(
        np.searchsorted(TIME_EDGES[1:], t_end, side="left"),
        0,
        len(TIME_EDGES) - 2,
    ).astype("int32")

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


def hierarchical_parameters(arrays):
    global_effect = pm.Normal("global_effect", 0.0, GLOBAL_EFFECT_SD)
    dataset_effect_sd = pm.HalfNormal("dataset_effect_sd", DATASET_EFFECT_SD)
    dataset_effect_raw = pm.Normal(
        "dataset_effect_raw", 0.0, 1.0, dims="dataset"
    )
    dataset_effect = pm.Deterministic(
        "dataset_effect",
        global_effect + dataset_effect_sd * dataset_effect_raw,
        dims="dataset",
    )

    baseline_initial = pm.Normal(
        "baseline_log_rate_initial",
        np.log(BASELINE_RATE_MEDIAN),
        BASELINE_LOG_RATE_SD,
        dims="dataset",
    )
    baseline_smoothness = pm.HalfNormal(
        "baseline_smoothness", BASELINE_SMOOTHNESS_SD
    )
    baseline_innovation = pm.Normal(
        "baseline_innovation",
        0.0,
        1.0,
        dims=("dataset", "time_transition"),
    )
    midpoints = 0.5 * (TIME_EDGES[:-1] + TIME_EDGES[1:])
    increments = (
        baseline_smoothness
        * pt.sqrt(pt.as_tensor_variable(np.diff(midpoints)))[None, :]
        * baseline_innovation
    )
    baseline_log_rate = pm.Deterministic(
        "baseline_log_rate",
        pt.concatenate(
            [
                baseline_initial[:, None],
                baseline_initial[:, None] + pt.cumsum(increments, axis=1),
            ],
            axis=1,
        ),
        dims=("dataset", "time_bin"),
    )

    dataset_index = pt.as_tensor_variable(arrays["dataset_index"])
    return baseline_log_rate[dataset_index], dataset_effect[dataset_index]


def interval_loglikelihood(hazard, arrays):
    if hazard.ndim == 3:
        exposure_before = pt.as_tensor_variable(
            arrays["exposure_before_interval"]
        )[:, None, :]
        exposure_during = pt.as_tensor_variable(
            arrays["exposure_during_interval"]
        )[:, None, :]
        event_selector = pt.as_tensor_variable(arrays["event_selector"])[:, None, :]
    else:
        exposure_before = pt.as_tensor_variable(arrays["exposure_before_interval"])
        exposure_during = pt.as_tensor_variable(arrays["exposure_during_interval"])
        event_selector = pt.as_tensor_variable(arrays["event_selector"])

    cumulative_before = pt.sum(hazard * exposure_before, axis=-1)
    cumulative_during = pt.sum(hazard * exposure_during, axis=-1)
    exact_hazard = pt.sum(hazard * event_selector, axis=-1)

    exact_loglikelihood = pt.log(pt.maximum(exact_hazard, 1e-30)) - cumulative_before
    interval_loglikelihood = -cumulative_before + pt.log(
        -pt.expm1(-pt.maximum(cumulative_during, 1e-30))
    )
    censored_loglikelihood = -(cumulative_before + cumulative_during)

    exact = pt.as_tensor_variable(arrays["exact"])
    interval_event = pt.as_tensor_variable(
        (arrays["events"] == 1) & (~arrays["exact"])
    )
    if hazard.ndim == 3:
        exact = exact[:, None]
        interval_event = interval_event[:, None]

    return pt.switch(
        exact,
        exact_loglikelihood,
        pt.switch(
            interval_event,
            interval_loglikelihood,
            censored_loglikelihood,
        ),
    )


def marginalize_particles(conditional_loglikelihood, particle_weights):
    log_weights = pt.log(pt.as_tensor_variable(particle_weights))
    return pm.math.logsumexp(
        conditional_loglikelihood + log_weights,
        axis=1,
    )


def sample_model(draws, tune, chains, cores, target_accept, seed):
    return pm.sample(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        init="adapt_diag",
        random_seed=seed,
        progressbar=True,
        return_inferencedata=True,
    )


def summarize_effects(idata, datasets):
    global_effect = np.asarray(idata.posterior.global_effect).reshape(-1)
    dataset_effect = (
        idata.posterior.dataset_effect.stack(sample=("chain", "draw"))
        .transpose("sample", "dataset")
        .values
    )
    heterogeneity = np.asarray(idata.posterior.dataset_effect_sd).reshape(-1)

    result = {
        "HR_median": float(np.median(np.exp(global_effect))),
        "HR_q025": float(np.quantile(np.exp(global_effect), 0.025)),
        "HR_q975": float(np.quantile(np.exp(global_effect), 0.975)),
        "P_effect_gt_0": float(np.mean(global_effect > 0)),
        "dataset_effect_sd_median": float(np.median(heterogeneity)),
        "P_all_datasets_gt_0": float(np.mean(np.all(dataset_effect > 0, axis=1))),
    }
    dataset_rows = []
    for index, dataset in enumerate(datasets):
        values = dataset_effect[:, index]
        dataset_rows.append(
            {
                "Dataset": dataset,
                "HR_median": float(np.median(np.exp(values))),
                "HR_q025": float(np.quantile(np.exp(values), 0.025)),
                "HR_q975": float(np.quantile(np.exp(values), 0.975)),
                "P_effect_gt_0": float(np.mean(values > 0)),
            }
        )
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
