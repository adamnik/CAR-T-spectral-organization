# Joint smooth survival regimes for TCAT features and O-lambda

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from analysis_io import feature_kind
from fit_tcat_survival import add_olambda_particles, sample_model, summarize_joint
from survival_model import add_adjustment_arguments, adjustment_design, adjustment_parameters
from survival_model import baseline_parameters, diagnostics, interval_loglikelihood
from survival_model import load_tcat_analysis, marginalize_particles, model_coordinates
from survival_model import parse_adjustments, summarize_adjustments, survival_arrays


JOINT_GLOBAL_SCALE = 0.30
JOINT_DATASET_SCALE = 0.20


def fixed_crossovers(feature_names, tcat_results_path, olambda_result_path):
    tcat = pd.read_csv(tcat_results_path).set_index("feature")
    olambda = pd.read_csv(olambda_result_path).iloc[0]
    crossovers = {}
    for feature in feature_names:
        if feature == "O_lambda_inf":
            crossovers[feature] = float(olambda.crossover_median)
        elif feature in tcat.index:
            crossovers[feature] = float(tcat.loc[feature, "crossover_median"])
        else:
            raise ValueError(f"No individual change-point estimate was found for {feature!r}.")
    return crossovers


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit one joint survival model for all TCAT and O-lambda regimes, "
            "holding each crossover at its individual posterior median."
        )
    )
    parser.add_argument("estimates_csv", type=Path)
    parser.add_argument("particles_npz", type=Path)
    parser.add_argument("outcomes_csv", type=Path)
    parser.add_argument("olambda_estimates_csv", type=Path)
    parser.add_argument("olambda_particles_csv", type=Path)
    parser.add_argument("tcat_regime_results_csv", type=Path)
    parser.add_argument("olambda_regime_result_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-cells", type=int, default=1000)
    parser.add_argument(
        "--feature-kind", choices=["all", "gep", "cell_type"], default="all"
    )
    parser.add_argument("--transition-scale-sd", type=float, default=0.10)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    add_adjustment_arguments(parser)
    args = parser.parse_args()
    if args.transition_scale_sd <= 0:
        raise ValueError("--transition-scale-sd must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "joint_feature_results.csv"
    if result_path.exists() and not args.overwrite:
        print(f"RESUME: existing completed joint model at {result_path}", flush=True)
        print(pd.read_csv(result_path).to_string(index=False), flush=True)
        return
    adjustments = parse_adjustments(args.adjustment)

    data, central, particles, particle_weights, feature_names = load_tcat_analysis(
        args.estimates_csv, args.particles_npz, args.outcomes_csv, args.minimum_cells,
        adjustment_columns=adjustments,
    )
    if args.feature_kind != "all":
        selected = [
            index for index, feature in enumerate(feature_names)
            if feature_kind(feature).lower() == args.feature_kind
        ]
        if not selected:
            raise ValueError(f"No {args.feature_kind} features were found.")
        central = central[:, selected]
        particles = particles[:, :, selected]
        feature_names = [feature_names[index] for index in selected]
    data, central, particles, particle_weights, feature_names = add_olambda_particles(
        data, central, particles, particle_weights, feature_names,
        args.olambda_estimates_csv, args.olambda_particles_csv, args.random_state,
    )
    crossovers = fixed_crossovers(
        feature_names, args.tcat_regime_results_csv, args.olambda_regime_result_csv
    )
    center = central.mean(axis=0)
    scale = central.std(axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Every joint feature must have patient-level variation.")
    standardized = (particles - center[None, None, :]) / scale[None, None, :]
    crossover = np.asarray([crossovers[feature] for feature in feature_names])
    standardized_crossover = (crossover - center) / scale
    distance = (
        standardized - standardized_crossover[None, None, :]
    ) / args.transition_scale_sd
    activation = 1.0 / (1.0 + np.exp(-np.clip(distance, -40, 40)))

    arrays = survival_arrays(data)
    adjustment_matrix, adjustment_table = adjustment_design(data, adjustments)
    coords = model_coordinates(data, arrays)
    coords["feature"] = feature_names
    if not adjustment_table.empty:
        coords["adjustment"] = adjustment_table.adjustment.tolist()

    with pm.Model(coords=coords):
        baseline_log_rate = baseline_parameters(arrays)
        clinical_adjustment = adjustment_parameters(
            adjustment_matrix, args.adjustment_effect_sd
        )
        global_scale = pm.HalfNormal("joint_global_scale", JOINT_GLOBAL_SCALE)
        global_raw = pm.Normal("joint_global_raw", 0.0, 1.0, dims="feature")
        global_effect = pm.Deterministic(
            "joint_global_effect", global_scale * global_raw, dims="feature"
        )
        dataset_scale = pm.HalfNormal("joint_dataset_scale", JOINT_DATASET_SCALE)
        dataset_raw = pm.Normal(
            "joint_dataset_raw", 0.0, 1.0, dims=("dataset", "feature")
        )
        dataset_effect = pm.Deterministic(
            "joint_dataset_effect",
            global_effect[None, :] + dataset_scale * dataset_raw,
            dims=("dataset", "feature"),
        )
        patient_effect = dataset_effect[pt.as_tensor_variable(arrays["dataset_index"])]
        linear_predictor = pt.sum(
            pt.as_tensor_variable(activation) * patient_effect[:, None, :], axis=2
        )
        hazard = pt.exp(
            baseline_log_rate[:, None, :]
            + clinical_adjustment[:, None, None]
            + linear_predictor[:, :, None]
        )
        conditional = interval_loglikelihood(hazard, arrays)
        patient_loglikelihood = marginalize_particles(conditional, particle_weights)
        pm.Deterministic("patient_loglikelihood", patient_loglikelihood, dims="observation")
        pm.Potential("survival_loglikelihood", pt.sum(patient_loglikelihood))
        started = time.time()
        idata = sample_model(args)
        elapsed = time.time() - started

    variables = [
        "joint_global_scale", "joint_dataset_scale", "joint_global_effect",
        "joint_dataset_effect", "baseline_log_rate_initial", "baseline_smoothness",
        "baseline_innovation",
    ]
    if not adjustment_table.empty:
        variables.append("adjustment_effect")
    posterior_summary, diagnostic_values = diagnostics(idata, variables)
    feature_results, dataset_results = summarize_joint(
        idata, feature_names, arrays["datasets"], "joint-smooth-regimes"
    )
    for key, value in diagnostic_values.items():
        feature_results[key] = value
    feature_results["crossover_median_fixed"] = [crossovers[x] for x in feature_names]
    feature_results["transition_scale_sd"] = args.transition_scale_sd
    feature_results["patients"] = len(data)
    feature_results["events"] = int(data.event_observed.sum())
    feature_results["datasets"] = len(arrays["datasets"])
    feature_results["particles"] = particles.shape[1]
    feature_results["wall_time_seconds"] = elapsed

    data.to_csv(args.output_dir / "analysis_cohort.csv", index=False)
    feature_results.to_csv(args.output_dir / "joint_feature_results.csv", index=False)
    dataset_results.to_csv(args.output_dir / "joint_dataset_effects.csv", index=False)
    summarize_adjustments(idata, adjustment_table).to_csv(
        args.output_dir / "adjustment_effects.csv", index=False
    )
    pd.DataFrame({
        "feature": feature_names,
        "center": center,
        "scale": scale,
        "crossover_median_fixed": crossover,
    }).to_csv(args.output_dir / "feature_scaling_and_crossovers.csv", index=False)
    posterior_summary.to_csv(args.output_dir / "posterior_summary.csv")
    idata.to_netcdf(args.output_dir / "posterior.nc")
    manifest = {
        "model": "joint smooth TCAT and O_lambda survival regimes",
        "feature_kind": args.feature_kind,
        "crossovers": "fixed at individual Bayesian posterior medians",
        "features": feature_names,
        "minimum_cells": args.minimum_cells,
        "transition_scale_sd": args.transition_scale_sd,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "target_accept": args.target_accept,
        "random_state": args.random_state,
        "adjustments": adjustments,
        "adjustment_effect_prior": f"Normal(0, {args.adjustment_effect_sd})",
    }
    (args.output_dir / "joint_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(feature_results.sort_values("P_effect_gt_0", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
