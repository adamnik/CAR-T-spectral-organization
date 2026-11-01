# Bayesian regressions of latent O-lambda on technical covariates

import argparse
import json
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


GLOBAL_EFFECT_SD = 0.50
DATASET_EFFECT_SD = 0.40
DATASET_INTERCEPT_SD = 0.50


def particle_columns(frame):
    columns = [column for column in frame if column.startswith("particle_")]
    return sorted(columns, key=lambda column: int(column.split("_")[-1]))


def load_analysis(estimates_path, technical_particles_path, olambda_particles_path,
                  minimum_cells):
    estimates = pd.read_csv(estimates_path)
    archive = np.load(technical_particles_path, allow_pickle=False)
    technical_values = np.asarray(archive["values"], dtype=float)
    technical_weights = np.asarray(archive["particle_weights"], dtype=float)
    technical_patients = archive["patient_ids"].astype(str)
    technical_datasets = archive["datasets"].astype(str)
    feature_names = archive["feature_names"].astype(str).tolist()
    technical_cells = np.asarray(archive["n_cells"], dtype=int)

    technical_index = pd.DataFrame({
        "Dataset": technical_datasets,
        "sample_id": technical_patients,
        "n_cells": technical_cells,
        "technical_index": np.arange(len(technical_patients)),
    })
    data = technical_index.merge(
        estimates[["Dataset", "sample_id", *feature_names]],
        on=["Dataset", "sample_id"],
        how="inner",
        validate="one_to_one",
    )

    olambda = pd.read_csv(olambda_particles_path)
    olambda_columns = particle_columns(olambda)
    if not olambda_columns:
        raise ValueError("No particle_* columns were found in the O-lambda table.")
    olambda = olambda[["Dataset", "sample_id", *olambda_columns]]
    data = data.merge(olambda, on=['Dataset', 'sample_id'], how='inner', validate='one_to_one')
    data = data.loc[data.n_cells >= minimum_cells].copy()
    data = data.sort_values(["Dataset", "sample_id"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("No patients satisfy the analysis requirements.")

    order = data.technical_index.to_numpy(int)
    technical_values = technical_values[order]
    technical_weights = technical_weights[order]
    olambda_values = data[olambda_columns].apply(pd.to_numeric, errors='coerce')
    olambda_values = olambda_values.to_numpy(float)
    data = data.drop(columns=["technical_index", *olambda_columns])
    central_x = data[feature_names].apply(pd.to_numeric, errors='coerce')
    central_x = central_x.to_numpy(float)
    central_y = olambda_values.mean(axis=1)

    if technical_values.shape[:2] != technical_weights.shape:
        raise ValueError("Technical particle values and weights do not align.")
    if technical_values.shape[2] != len(feature_names):
        raise ValueError("Technical feature names do not align with particle values.")
    arrays = [technical_values, technical_weights, olambda_values, central_x, central_y]
    if any(np.any(~np.isfinite(values)) for values in arrays):
        raise ValueError("Analysis values must be finite.")
    if np.any(technical_weights < 0) or np.any(technical_weights.sum(axis=1) <= 0):
        raise ValueError("Technical particle weights must be nonnegative and have positive mass.")
    technical_weights /= technical_weights.sum(axis=1, keepdims=True)
    olambda_weights = np.full(olambda_values.shape, 1.0 / olambda_values.shape[1], dtype=float)
    return (
        data, feature_names,
        central_x, technical_values, technical_weights,
        central_y, olambda_values, olambda_weights,
    )


def standardize(central_x, particle_x, central_y, particle_y, training_rows=None):
    if training_rows is None:
        training_rows = np.arange(len(central_y))
    # Use training patients only when holding out a dataset
    x_center = central_x[training_rows].mean(axis=0)
    x_scale = central_x[training_rows].std(axis=0, ddof=1)
    y_center = central_y[training_rows].mean()
    y_scale = central_y[training_rows].std(ddof=1)
    if np.any(~np.isfinite(x_scale)) or np.any(x_scale <= 0) or y_scale <= 0:
        raise ValueError("Every technical covariate and O-lambda must have positive variation.")
    values = {
        "central_x": (central_x - x_center) / x_scale,
        "particle_x": (particle_x - x_center[None, None, :]) / x_scale[None, None, :],
        "central_y": (central_y - y_center) / y_scale,
        "particle_y": (particle_y - y_center) / y_scale,
    }
    scaling = {
        "x_center": x_center,
        "x_scale": x_scale,
        "y_center": float(y_center),
        "y_scale": float(y_scale),
    }
    return values, scaling


def dataset_arrays(data, rows):
    observed = data.iloc[rows]
    datasets = sorted(observed.Dataset.astype(str).unique())
    lookup = {dataset: index for index, dataset in enumerate(datasets)}
    index = observed.Dataset.astype(str).map(lookup).to_numpy("int32")
    return datasets, index


def within_dataset(values, dataset_index):
    centered = np.asarray(values, dtype=float).copy()
    for dataset in np.unique(dataset_index):
        keep = dataset_index == dataset
        centered[keep] -= centered[keep].mean(axis=0)
    return centered


def patient_loglikelihood(observed_y, predictor_mean, residual_sd,
                          y_weights, x_weights):
    # Integrate over every O-lambda / technical-particle pair
    y = pt.as_tensor_variable(observed_y)[:, :, None]
    mean = predictor_mean[:, None, :]
    logp = pm.logp(pm.Normal.dist(mu=mean, sigma=residual_sd), y)
    logp += pt.log(pt.as_tensor_variable(y_weights))[:, :, None]
    logp += pt.log(pt.as_tensor_variable(x_weights))[:, None, :]
    return pt.logsumexp(logp, axis=(1, 2))


def intercepts(dataset_names):
    global_intercept = pm.Normal("global_intercept", 0, 1)
    dataset_intercept_sd = pm.HalfNormal("dataset_intercept_sd", DATASET_INTERCEPT_SD)
    dataset_intercept_raw = pm.Normal('dataset_intercept_raw', 0, 1, dims='dataset')
    dataset_intercept = pm.Deterministic(
        "dataset_intercept",
        global_intercept + dataset_intercept_sd * dataset_intercept_raw,
        dims="dataset",
    )
    return global_intercept, dataset_intercept


def sampling_kwargs(args, random_state):
    return {
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "target_accept": args.target_accept,
        "init": "adapt_diag",
        "random_seed": random_state,
        "progressbar": True,
        "return_inferencedata": True,
    }


def diagnostic_summary(idata, variables):
    summary = az.summary(idata, var_names=variables, round_to=None)
    return summary, {
        "divergences": int(idata.sample_stats.diverging.sum()),
        "max_rhat": float(summary.r_hat.max()),
        "min_ess_bulk": float(summary.ess_bulk.min()),
        "min_ess_tail": float(summary.ess_tail.min()),
    }


def individual_model(feature, feature_index, data, arrays, weights, args, output_dir):
    central_x, particle_x, central_y, particle_y = arrays
    x_weights, y_weights = weights
    datasets, dataset_index = dataset_arrays(data, np.arange(len(data)))
    coordinates = {"dataset": datasets, "observation": np.arange(len(data))}

    with pm.Model(coords=coordinates):
        _, dataset_intercept = intercepts(datasets)
        global_effect = pm.Normal("global_effect", 0, GLOBAL_EFFECT_SD)
        dataset_effect_sd = pm.HalfNormal("dataset_effect_sd", DATASET_EFFECT_SD)
        dataset_effect_raw = pm.Normal("dataset_effect_raw", 0, 1, dims="dataset")
        dataset_effect = pm.Deterministic(
            "dataset_effect",
            global_effect + dataset_effect_sd * dataset_effect_raw,
            dims="dataset",
        )
        residual_sd = pm.HalfNormal("residual_sd", 1)
        predictor_mean = (
            dataset_intercept[dataset_index, None]
            + dataset_effect[dataset_index, None]
            * pt.as_tensor_variable(particle_x[:, :, feature_index])
        )
        likelihood = patient_loglikelihood(
            particle_y, predictor_mean, residual_sd, y_weights, x_weights
        )
        pm.Potential("regression_loglikelihood", pt.sum(likelihood))
        centered_x = within_dataset(central_x[:, [feature_index]], dataset_index).ravel()
        technical_prediction = (
            dataset_effect[dataset_index]
            * pt.as_tensor_variable(centered_x)
        )
        predicted_variance = pt.var(technical_prediction)
        pm.Deterministic(
            "technical_R2",
            predicted_variance / (predicted_variance + residual_sd**2),
        )

        started = time.time()
        idata = pm.sample(**sampling_kwargs(args, args.random_state + feature_index))
        elapsed = time.time() - started

    global_draws = np.asarray(idata.posterior.global_effect).reshape(-1)
    dataset_draws = np.asarray(idata.posterior.dataset_effect).reshape(-1, len(datasets))
    r2_draws = np.asarray(idata.posterior.technical_R2).reshape(-1)
    summary, diagnostics = diagnostic_summary(
        idata,
        [
            "global_effect", "dataset_effect_sd", "dataset_effect",
            "global_intercept", "dataset_intercept_sd", "dataset_intercept",
            "residual_sd", "technical_R2",
        ],
    )
    result = {
        "feature": feature,
        "effect_median": np.median(global_draws),
        "effect_q025": np.quantile(global_draws, 0.025),
        "effect_q975": np.quantile(global_draws, 0.975),
        "P_effect_gt_0": np.mean(global_draws > 0),
        "P_all_datasets_gt_0": np.mean(np.all(dataset_draws > 0, axis=1)),
        "P_all_datasets_lt_0": np.mean(np.all(dataset_draws < 0, axis=1)),
        "P_all_datasets_same_direction": np.mean(
            np.all(dataset_draws > 0, axis=1) | np.all(dataset_draws < 0, axis=1)
        ),
        "dataset_effect_sd_median": np.median(np.asarray(idata.posterior.dataset_effect_sd)),
        "technical_R2_median": np.median(r2_draws),
        "technical_R2_q025": np.quantile(r2_draws, 0.025),
        "technical_R2_q975": np.quantile(r2_draws, 0.975),
        "patients": len(data),
        "datasets": len(datasets),
        "wall_time_seconds": elapsed,
        **diagnostics,
    }
    dataset_results = []
    for index, dataset in enumerate(datasets):
        values = dataset_draws[:, index]
        dataset_results.append({
            "feature": feature,
            "Dataset": dataset,
            "effect_median": np.median(values),
            "effect_q025": np.quantile(values, 0.025),
            "effect_q975": np.quantile(values, 0.975),
            "P_effect_gt_0": np.mean(values > 0),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(output_dir / "model_result.csv", index=False)
    pd.DataFrame(dataset_results).to_csv(output_dir / "dataset_effects.csv", index=False)
    summary.to_csv(output_dir / "posterior_summary.csv")
    idata.to_netcdf(output_dir / "posterior.nc")
    return result


def joint_model(data, feature_names, arrays, weights, args, output_dir,
                training_rows=None, save_posterior=True, random_state=None):
    central_x, particle_x, central_y, particle_y = arrays
    x_weights, y_weights = weights
    if training_rows is None:
        training_rows = np.arange(len(data))
    training_rows = np.asarray(training_rows, dtype=int)
    datasets, dataset_index = dataset_arrays(data, training_rows)
    coordinates = {
        "dataset": datasets,
        "feature": feature_names,
        "observation": np.arange(len(training_rows)),
    }
    x = particle_x[training_rows]
    y = particle_y[training_rows]

    with pm.Model(coords=coordinates):
        global_intercept, dataset_intercept = intercepts(datasets)
        coefficient_scale = pm.HalfNormal("coefficient_scale", GLOBAL_EFFECT_SD)
        coefficient = pm.Normal('coefficient', 0, coefficient_scale, dims='feature')
        residual_sd = pm.HalfNormal("residual_sd", 1)
        predictor_mean = dataset_intercept[dataset_index, None] + pt.sum(
            pt.as_tensor_variable(x) * coefficient[None, None, :],
            axis=2,
        )
        likelihood = patient_loglikelihood(
            y,
            predictor_mean,
            residual_sd,
            y_weights[training_rows],
            x_weights[training_rows],
        )
        pm.Potential("regression_loglikelihood", pt.sum(likelihood))
        centered_x = within_dataset(central_x[training_rows], dataset_index)
        technical_prediction = pt.sum(
            pt.as_tensor_variable(centered_x) * coefficient[None, :],
            axis=1,
        )
        predicted_variance = pt.var(technical_prediction)
        pm.Deterministic(
            "technical_R2",
            predicted_variance / (predicted_variance + residual_sd**2),
        )

        started = time.time()
        idata = pm.sample(
            **sampling_kwargs(args, args.random_state if random_state is None else random_state)
        )
        elapsed = time.time() - started

    coefficient_draws = np.asarray(idata.posterior.coefficient).reshape(-1, len(feature_names))
    r2_draws = np.asarray(idata.posterior.technical_R2).reshape(-1)
    summary, diagnostics = diagnostic_summary(
        idata,
        [
            "coefficient", "coefficient_scale", "global_intercept",
            "dataset_intercept_sd", "dataset_intercept", "residual_sd", "technical_R2",
        ],
    )
    coefficient_results = []
    for index, feature in enumerate(feature_names):
        values = coefficient_draws[:, index]
        coefficient_results.append({
            "feature": feature,
            "effect_median": np.median(values),
            "effect_q025": np.quantile(values, 0.025),
            "effect_q975": np.quantile(values, 0.975),
            "P_effect_gt_0": np.mean(values > 0),
        })
    result = {
        "technical_R2_median": np.median(r2_draws),
        "technical_R2_q025": np.quantile(r2_draws, 0.025),
        "technical_R2_q975": np.quantile(r2_draws, 0.975),
        "patients": len(training_rows),
        "datasets": len(datasets),
        "wall_time_seconds": elapsed,
        **diagnostics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(output_dir / "model_result.csv", index=False)
    pd.DataFrame(coefficient_results).to_csv(output_dir / 'coefficient_results.csv', index=False)
    summary.to_csv(output_dir / "posterior_summary.csv")
    if save_posterior:
        idata.to_netcdf(output_dir / "posterior.nc")
    return result, coefficient_results, idata


def run_individual(data, feature_names, arrays, weights, args):
    root = args.output_dir / "individual"
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for index, feature in enumerate(feature_names):
        output = root / feature
        result_path = output / "model_result.csv"
        print("=" * 100, flush=True)
        print(f"[{index + 1}/{len(feature_names)}] {feature}", flush=True)
        if result_path.exists() and not args.overwrite:
            print("  completed; loading saved result", flush=True)
            results.append(pd.read_csv(result_path).iloc[0].to_dict())
            continue
        result = individual_model(feature, index, data, arrays, weights, args, output)
        results.append(result)
        print(pd.DataFrame([result]).to_string(index=False), flush=True)
    pd.DataFrame(results).to_csv(root / "individual_results.csv", index=False)


def run_joint(data, feature_names, arrays, weights, args):
    output = args.output_dir / "joint"
    result_path = output / "model_result.csv"
    if result_path.exists() and not args.overwrite:
        print("Joint model already completed.", flush=True)
        print(pd.read_csv(result_path).to_string(index=False), flush=True)
        return
    result, coefficients, _ = joint_model(data, feature_names, arrays, weights, args, output)
    print(pd.DataFrame([result]).to_string(index=False), flush=True)
    print(pd.DataFrame(coefficients).to_string(index=False), flush=True)


def run_lodo(data, feature_names, original, technical_weights, olambda_weights, args):
    central_x, particle_x, central_y, particle_y = original
    root = args.output_dir / "leave_one_dataset_out"
    root.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    datasets = sorted(data.Dataset.astype(str).unique())

    for fold, heldout in enumerate(datasets):
        output = root / heldout
        prediction_path = output / "predictions.csv"
        print("=" * 100, flush=True)
        print(f"[{fold + 1}/{len(datasets)}] held-out dataset: {heldout}", flush=True)
        if prediction_path.exists() and not args.overwrite:
            print("  completed; loading saved predictions", flush=True)
            prediction_rows.append(pd.read_csv(prediction_path))
            continue

        training = np.flatnonzero(data.Dataset.astype(str).to_numpy() != heldout)
        testing = np.flatnonzero(data.Dataset.astype(str).to_numpy() == heldout)
        standardized, scaling = standardize(
            central_x, particle_x, central_y, particle_y,
            training_rows=training,
        )
        arrays = (
            standardized["central_x"], standardized["particle_x"],
            standardized["central_y"], standardized["particle_y"],
        )
        weights = (technical_weights, olambda_weights)
        _, _, idata = joint_model(
            data, feature_names, arrays, weights, args, output,
            training_rows=training,
            save_posterior=args.save_lodo_posteriors,
            random_state=args.random_state + 100 + fold,
        )
        intercept = np.asarray(idata.posterior.global_intercept).reshape(-1)
        coefficients = np.asarray(idata.posterior.coefficient).reshape(-1, len(feature_names))
        predicted_standardized = (
            intercept[:, None]
            + coefficients @ standardized["central_x"][testing].T
        )
        predicted = (
            scaling["y_center"]
            + scaling["y_scale"] * predicted_standardized
        )
        residual_sd = np.asarray(idata.posterior.residual_sd).reshape(-1)
        rng = np.random.default_rng(args.random_state + 1000 + fold)
        predictive_standardized = predicted_standardized + rng.normal(
            size=predicted_standardized.shape
        ) * residual_sd[:, None]
        predictive = (
            scaling["y_center"]
            + scaling["y_scale"] * predictive_standardized
        )
        frame = data.iloc[testing][["Dataset", "sample_id", "n_cells"]].copy()
        frame["observed_O_lambda_inf"] = central_y[testing]
        frame["predicted_median"] = np.median(predicted, axis=0)
        frame["predictive_q025"] = np.quantile(predictive, 0.025, axis=0)
        frame["predictive_q975"] = np.quantile(predictive, 0.975, axis=0)
        frame["covered_by_95_interval"] = (
            (frame.observed_O_lambda_inf >= frame.predictive_q025)
            & (frame.observed_O_lambda_inf <= frame.predictive_q975)
        )
        frame.to_csv(prediction_path, index=False)
        prediction_rows.append(frame)

    predictions = pd.concat(prediction_rows, ignore_index=True, sort=False)
    predictions["error"] = predictions.predicted_median - predictions.observed_O_lambda_inf
    predictions["absolute_error"] = predictions.error.abs()
    predictions["squared_error"] = predictions.error**2
    correlation = predictions[["predicted_median", "observed_O_lambda_inf"]].corr().iloc[0, 1]
    summary = pd.DataFrame([{
        "patients": len(predictions),
        "datasets": predictions.Dataset.nunique(),
        "MAE": predictions.absolute_error.mean(),
        "RMSE": np.sqrt(predictions.squared_error.mean()),
        "pearson_r": correlation,
        "coverage_95": predictions.covered_by_95_interval.mean(),
    }])
    predictions.to_csv(root / "lodo_predictions.csv", index=False)
    summary.to_csv(root / "lodo_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit uncertainty-aware Bayesian regressions of latent O-lambda "
            "on five technical covariates."
        )
    )
    parser.add_argument("technical_estimates_csv", type=Path)
    parser.add_argument("technical_particles_npz", type=Path)
    parser.add_argument("olambda_particles_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument('--mode', choices=['individual', 'joint', 'lodo', 'all'], default='all')
    parser.add_argument("--minimum-cells", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--save-lodo-posteriors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (data, feature_names,
     central_x, technical_values, technical_weights,
     central_y, olambda_values, olambda_weights) = load_analysis(
        args.technical_estimates_csv, args.technical_particles_npz,
        args.olambda_particles_csv, args.minimum_cells,
    )
    standardized, scaling = standardize(central_x, technical_values, central_y, olambda_values)
    arrays = (
        standardized["central_x"], standardized["particle_x"],
        standardized["central_y"], standardized["particle_y"],
    )
    weights = (technical_weights, olambda_weights)
    data.to_csv(args.output_dir / "analysis_cohort.csv", index=False)
    manifest = {
        "technical_estimates_csv": str(args.technical_estimates_csv),
        "technical_particles_npz": str(args.technical_particles_npz),
        "olambda_particles_csv": str(args.olambda_particles_csv),
        "mode": args.mode,
        "minimum_cells": args.minimum_cells,
        "features": feature_names,
        "patients": len(data),
        "datasets": sorted(data.Dataset.astype(str).unique()),
        "x_center": dict(zip(feature_names, scaling["x_center"])),
        "x_scale": dict(zip(feature_names, scaling["x_scale"])),
        "O_lambda_center": scaling["y_center"],
        "O_lambda_scale": scaling["y_scale"],
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "target_accept": args.target_accept,
        "random_state": args.random_state,
        "uncertainty_integration": (
            "Cartesian marginalization over independent technical and O-lambda particles"
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"Analysis cohort: {len(data):,} patients | "
        f"{data.Dataset.nunique():,} datasets | "
        f"{len(feature_names):,} technical covariates | "
        f"{technical_values.shape[1]:,} technical particles | "
        f"{olambda_values.shape[1]:,} O-lambda particles",
        flush=True,
    )

    if args.mode in ["individual", "all"]:
        run_individual(data, feature_names, arrays, weights, args)
    if args.mode in ["joint", "all"]:
        run_joint(data, feature_names, arrays, weights, args)
    if args.mode in ["lodo", "all"]:
        original = (central_x, technical_values, central_y, olambda_values)
        run_lodo(data, feature_names, original, technical_weights, olambda_weights, args)


if __name__ == "__main__":
    main()
