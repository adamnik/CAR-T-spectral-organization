# Smooth survival regimes for individual patient-level TCAT features

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from analysis_io import safe_name
from fit_tcat_survival import feature_kind, individual_features, sample_model
from survival_model import diagnostics, hierarchical_parameters, interval_loglikelihood
from survival_model import load_tcat_analysis, marginalize_particles, model_coordinates
from survival_model import summarize_effects, survival_arrays


def regime_model(feature, feature_index, data, central, particles, particle_weights,
                 arrays, args, output_dir):
    values = central[:, feature_index]
    center = float(values.mean())
    scale = float(values.std(ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Feature {feature!r} has no patient-level variation.")
    standardized = (particles[:, :, feature_index] - center) / scale
    crossover_low = float(np.quantile(values, args.crossover_quantile_low))
    crossover_high = float(np.quantile(values, args.crossover_quantile_high))
    if crossover_high <= crossover_low:
        raise ValueError(f"Feature {feature!r} has no variation across its crossover prior.")

    with pm.Model(coords=model_coordinates(data, arrays)):
        baseline_log_rate, patient_effect = hierarchical_parameters(arrays)
        crossover = pm.Uniform("crossover", crossover_low, crossover_high)
        standardized_crossover = (crossover - center) / scale
        distance = pt.as_tensor_variable(standardized) - standardized_crossover
        activation = pm.math.sigmoid(distance / args.transition_scale_sd)
        pm.Deterministic(
            "patient_probability_high",
            pt.sum(activation * pt.as_tensor_variable(particle_weights), axis=1),
            dims="observation",
        )
        hazard = pt.exp(
            baseline_log_rate[:, None, :]
            + patient_effect[:, None, None] * activation[:, :, None]
        )
        conditional = interval_loglikelihood(hazard, arrays)
        patient_loglikelihood = marginalize_particles(conditional, particle_weights)
        pm.Deterministic("patient_loglikelihood", patient_loglikelihood, dims="observation")
        pm.Potential("survival_loglikelihood", pt.sum(patient_loglikelihood))
        started = time.time()
        idata = sample_model(args)
        elapsed = time.time() - started

    variables = [
        "global_effect", "dataset_effect_sd", "dataset_effect", "crossover",
        "baseline_log_rate_initial", "baseline_smoothness", "baseline_innovation",
    ]
    posterior_summary, diagnostic_values = diagnostics(idata, variables)
    result, dataset_results = summarize_effects(idata, arrays["datasets"])
    crossover_draws = np.asarray(idata.posterior.crossover).reshape(-1)
    crossover_median = float(np.median(crossover_draws))
    crossover_q025 = float(np.quantile(crossover_draws, 0.025))
    crossover_q975 = float(np.quantile(crossover_draws, 0.975))
    membership = np.asarray(idata.posterior.patient_probability_high)
    membership = membership.reshape(-1, len(data))
    result.update(diagnostic_values)
    result.update(
        {
            "feature": feature,
            "feature_kind": feature_kind(feature),
            "effect_unit": "high versus low feature regime plateaus",
            "center": center,
            "scale": scale,
            "crossover_prior_low": crossover_low,
            "crossover_prior_high": crossover_high,
            "crossover_median": crossover_median,
            "crossover_q025": crossover_q025,
            "crossover_q975": crossover_q975,
            "crossover_percentile": float(np.mean(values <= crossover_median)),
            "crossover_q025_percentile": float(np.mean(values <= crossover_q025)),
            "crossover_q975_percentile": float(np.mean(values <= crossover_q975)),
            "transition_scale_sd": args.transition_scale_sd,
            "patients": len(data),
            "events": int(data.event_observed.sum()),
            "datasets": len(arrays["datasets"]),
            "particles": particles.shape[1],
            "wall_time_seconds": elapsed,
        }
    )
    dataset_results.insert(0, "feature", feature)

    membership_results = data[["Dataset", "sample_id"]].copy()
    membership_results["feature_central"] = values
    membership_results["posterior_mean_probability_high"] = membership.mean(axis=0)
    membership_results["posterior_median_probability_high"] = np.median(membership, axis=0)

    pd.DataFrame([result]).to_csv(output_dir / "model_result.csv", index=False)
    dataset_results.to_csv(output_dir / "dataset_effects.csv", index=False)
    membership_results.to_csv(output_dir / "patient_regime_probabilities.csv", index=False)
    posterior_summary.to_csv(output_dir / "posterior_summary.csv")
    idata.to_netcdf(output_dir / "posterior.nc")
    manifest = {
        "model": "smooth uncertainty-marginalized TCAT survival regime",
        "feature": feature,
        "feature_kind": feature_kind(feature),
        "center": center,
        "scale": scale,
        "crossover_prior": {
            "distribution": "uniform",
            "quantile_low": args.crossover_quantile_low,
            "quantile_high": args.crossover_quantile_high,
            "lower": crossover_low,
            "upper": crossover_high,
        },
        "transition_scale_sd": args.transition_scale_sd,
    }
    (output_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    return result, dataset_results


def main():
    parser = argparse.ArgumentParser(
        description="Fit smooth uncertainty-marginalized survival regimes for individual TCAT features."
    )
    parser.add_argument("estimates_csv", type=Path)
    parser.add_argument("particles_npz", type=Path)
    parser.add_argument("outcomes_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-cells", type=int, default=1000)
    parser.add_argument("--feature", action="append", default=None)
    parser.add_argument("--feature-kind", choices=["all", "gep", "score", "cell_type"],
                        default="all")
    parser.add_argument("--maximum-features", type=int, default=None)
    parser.add_argument("--crossover-quantile-low", type=float, default=0.10)
    parser.add_argument("--crossover-quantile-high", type=float, default=0.90)
    parser.add_argument("--transition-scale-sd", type=float, default=0.10)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.crossover_quantile_low < args.crossover_quantile_high <= 1:
        raise ValueError("Crossover quantiles must satisfy 0 <= low < high <= 1.")
    if args.transition_scale_sd <= 0:
        raise ValueError("--transition-scale-sd must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, central, particles, particle_weights, feature_names = load_tcat_analysis(
        args.estimates_csv, args.particles_npz, args.outcomes_csv, args.minimum_cells
    )
    arrays = survival_arrays(data)
    selected = individual_features(
        feature_names, args.feature, args.feature_kind, args.maximum_features
    )
    lookup = {feature: index for index, feature in enumerate(feature_names)}
    data.to_csv(args.output_dir / "analysis_cohort.csv", index=False)
    print(
        f"Analysis cohort: {len(data):,} patients | {int(data.event_observed.sum()):,} events | "
        f"{len(arrays['datasets']):,} datasets | {particles.shape[1]:,} particles | "
        f"{len(selected):,} TCAT features",
        flush=True,
    )
    print(
        f"Crossover prior: Q{100 * args.crossover_quantile_low:.0f}-"
        f"Q{100 * args.crossover_quantile_high:.0f} | "
        f"transition scale={args.transition_scale_sd:.3f} SD",
        flush=True,
    )

    results = []
    dataset_results = []
    errors = []
    for number, feature in enumerate(selected, start=1):
        feature_dir = args.output_dir / "individual" / f"{lookup[feature]:03d}_{safe_name(feature)}"
        feature_dir.mkdir(parents=True, exist_ok=True)
        result_path = feature_dir / "model_result.csv"
        print("\n" + "=" * 100, flush=True)
        print(f"[{number}/{len(selected)}] {feature} ({feature_kind(feature)})", flush=True)
        print("=" * 100, flush=True)
        if result_path.exists() and not args.overwrite:
            result = pd.read_csv(result_path).iloc[0].to_dict()
            datasets = pd.read_csv(feature_dir / "dataset_effects.csv")
            print("RESUME: existing completed model", flush=True)
        else:
            try:
                result, datasets = regime_model(
                    feature, lookup[feature], data, central, particles, particle_weights,
                    arrays, args, feature_dir,
                )
            except Exception as error:
                errors.append(
                    {
                        "feature": feature,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                pd.DataFrame(errors).to_csv(args.output_dir / "regime_errors.csv", index=False)
                print(f"ERROR {type(error).__name__}: {error}", flush=True)
                continue
        results.append(result)
        dataset_results.append(datasets)
        print(
            f"HR={result['HR_median']:.3f} "
            f"[{result['HR_q025']:.3f}, {result['HR_q975']:.3f}] | "
            f"P(beta>0)={result['P_effect_gt_0']:.3f} | "
            f"P(all same direction)={result['P_all_datasets_same_direction']:.3f} | "
            f"crossover percentile={result['crossover_percentile']:.3f}",
            flush=True,
        )
        pd.DataFrame(results).to_csv(args.output_dir / "regime_results.csv", index=False)
        pd.concat(dataset_results, ignore_index=True).to_csv(
            args.output_dir / "regime_dataset_effects.csv", index=False
        )

    run_manifest = {
        "model": "independent smooth TCAT survival regimes",
        "features": selected,
        "minimum_cells": args.minimum_cells,
        "n_particles": particles.shape[1],
        "crossover_quantile_low": args.crossover_quantile_low,
        "crossover_quantile_high": args.crossover_quantile_high,
        "transition_scale_sd": args.transition_scale_sd,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "target_accept": args.target_accept,
        "random_state": args.random_state,
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
    if results:
        print("\nCompleted results:", args.output_dir / "regime_results.csv", flush=True)


if __name__ == "__main__":
    main()
