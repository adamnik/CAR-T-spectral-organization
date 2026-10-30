# Bayesian survival models for patient-level TCAT features

import argparse
import json
import time
import traceback
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from analysis_io import safe_name
from survival_model import alignment_summary, baseline_parameters, diagnostics, effect_summary
from survival_model import hierarchical_parameters, interval_loglikelihood, load_tcat_analysis
from survival_model import marginalize_particles, model_coordinates, summarize_effects, survival_arrays


JOINT_GLOBAL_SCALE = 0.30
JOINT_DATASET_SCALE = 0.20


def feature_kind(feature):
    if feature.startswith("TCAT_GEP_"):
        return "GEP"
    if "_proportion_" in feature:
        return "cell_type"
    return "score"


def individual_features(feature_names, requested, requested_kind, maximum):
    if requested:
        missing = sorted(set(requested).difference(feature_names))
        if missing:
            raise ValueError(f"Requested TCAT features were not found: {missing}")
        selected = list(dict.fromkeys(requested))
    else:
        selected = [
            feature for feature in feature_names
            if requested_kind == "all" or feature_kind(feature).lower() == requested_kind
        ]
    if maximum is not None:
        selected = selected[:maximum]
    if not selected:
        raise ValueError("No TCAT features were selected.")
    return selected


def joint_features(feature_names, mode):
    if mode == "joint-gep":
        selected = [feature for feature in feature_names if feature_kind(feature) == "GEP"]
        blocks = ["TCAT_GEP"] * len(selected)
    else:
        selected = [feature for feature in feature_names if feature_kind(feature) == "cell_type"]
        blocks = [feature.split("_proportion_", 1)[0] for feature in selected]
    if len(selected) < 2:
        raise ValueError(f"At least two features are required for {mode}.")

    ordered_features = []
    ordered_blocks = []
    for block in dict.fromkeys(blocks):
        for feature, feature_block in zip(selected, blocks):
            if feature_block == block:
                ordered_features.append(feature)
                ordered_blocks.append(block)
    return ordered_features, ordered_blocks


def block_slices(blocks):
    slices = []
    start = 0
    while start < len(blocks):
        stop = start + 1
        while stop < len(blocks) and blocks[stop] == blocks[start]:
            stop += 1
        slices.append((blocks[start], slice(start, stop)))
        start = stop
    return slices


def standardize_joint(central, particles, feature_names, blocks):
    standardized = np.empty_like(particles, dtype=float)
    scaling = {}
    for block, index in block_slices(blocks):
        block_central = central[:, index]
        center = block_central.mean(axis=0)
        scale = float(np.sqrt(np.mean(block_central.var(axis=0, ddof=1))))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"TCAT block {block!r} has no patient-level variation.")
        standardized[:, :, index] = (particles[:, :, index] - center[None, None, :]) / scale
        scaling[block] = {
            "scale": scale,
            "centers": dict(zip(feature_names[index], center.tolist())),
        }
    return standardized, scaling


def center_coefficients(values, blocks):
    pieces = []
    for _, index in block_slices(blocks):
        block = values[..., index]
        pieces.append(block - pt.mean(block, axis=-1, keepdims=True))
    return pt.concatenate(pieces, axis=-1)


def sample_model(args):
    return pm.sample(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        target_accept=args.target_accept,
        init="adapt_diag",
        random_seed=args.random_state,
        progressbar=True,
        return_inferencedata=True,
    )


def individual_model(feature, feature_index, data, central, particles, particle_weights,
                     arrays, args, output_dir):
    center = float(central[:, feature_index].mean())
    scale = float(central[:, feature_index].std(ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Feature {feature!r} has no patient-level variation.")
    standardized = (particles[:, :, feature_index] - center) / scale

    with pm.Model(coords=model_coordinates(data, arrays)):
        baseline_log_rate, patient_effect = hierarchical_parameters(arrays)
        hazard = pt.exp(
            baseline_log_rate[:, None, :]
            + patient_effect[:, None, None] * pt.as_tensor_variable(standardized)[:, :, None]
        )
        conditional = interval_loglikelihood(hazard, arrays)
        patient_loglikelihood = marginalize_particles(conditional, particle_weights)
        pm.Deterministic("patient_loglikelihood", patient_loglikelihood, dims="observation")
        pm.Potential("survival_loglikelihood", pt.sum(patient_loglikelihood))
        started = time.time()
        idata = sample_model(args)
        elapsed = time.time() - started

    variables = [
        "global_effect", "dataset_effect_sd", "dataset_effect",
        "baseline_log_rate_initial", "baseline_smoothness", "baseline_innovation",
    ]
    posterior_summary, diagnostic_values = diagnostics(idata, variables)
    result, dataset_results = summarize_effects(idata, arrays["datasets"])
    result.update(diagnostic_values)
    result.update(
        {
            "feature": feature,
            "feature_kind": feature_kind(feature),
            "effect_unit": "per global SD of the central patient estimate",
            "center": center,
            "scale": scale,
            "patients": len(data),
            "events": int(data.event_observed.sum()),
            "datasets": len(arrays["datasets"]),
            "particles": particles.shape[1],
            "wall_time_seconds": elapsed,
        }
    )
    dataset_results.insert(0, "feature", feature)

    pd.DataFrame([result]).to_csv(output_dir / "model_result.csv", index=False)
    dataset_results.to_csv(output_dir / "dataset_effects.csv", index=False)
    posterior_summary.to_csv(output_dir / "posterior_summary.csv")
    idata.to_netcdf(output_dir / "posterior.nc")
    return result, dataset_results


def summarize_joint(idata, features, datasets, mode):
    global_effect = idata.posterior.joint_global_effect.stack(sample=("chain", "draw"))
    global_effect = global_effect.transpose("sample", "feature").values
    dataset_effect = idata.posterior.joint_dataset_effect.stack(sample=("chain", "draw"))
    dataset_effect = dataset_effect.transpose("sample", "dataset", "feature").values
    global_scale = np.asarray(idata.posterior.joint_global_scale).reshape(-1)
    dataset_scale = np.asarray(idata.posterior.joint_dataset_scale).reshape(-1)

    feature_rows = []
    dataset_rows = []
    for feature_index, feature in enumerate(features):
        row = {"mode": mode, "feature": feature, **effect_summary(global_effect[:, feature_index])}
        row.update(alignment_summary(dataset_effect[:, :, feature_index]))
        row["joint_global_scale_median"] = float(np.median(global_scale))
        row["joint_dataset_scale_median"] = float(np.median(dataset_scale))
        feature_rows.append(row)
        for dataset_index, dataset in enumerate(datasets):
            dataset_rows.append(
                {
                    "mode": mode,
                    "feature": feature,
                    "Dataset": dataset,
                    **effect_summary(dataset_effect[:, dataset_index, feature_index]),
                }
            )
    return pd.DataFrame(feature_rows), pd.DataFrame(dataset_rows)


def joint_model(mode, data, central, particles, particle_weights, feature_names,
                arrays, args, output_dir):
    selected, blocks = joint_features(feature_names, mode)
    feature_lookup = {feature: index for index, feature in enumerate(feature_names)}
    indices = np.asarray([feature_lookup[feature] for feature in selected])
    selected_central = central[:, indices]
    selected_particles = particles[:, :, indices]
    standardized, scaling = standardize_joint(selected_central, selected_particles, selected, blocks)

    coords = model_coordinates(data, arrays)
    coords["feature"] = selected
    with pm.Model(coords=coords):
        baseline_log_rate = baseline_parameters(arrays)
        global_scale = pm.HalfNormal("joint_global_scale", JOINT_GLOBAL_SCALE)
        global_raw = pm.Normal("joint_global_raw", 0.0, 1.0, dims="feature")
        global_effect = pm.Deterministic(
            "joint_global_effect",
            global_scale * center_coefficients(global_raw, blocks),
            dims="feature",
        )
        dataset_scale = pm.HalfNormal("joint_dataset_scale", JOINT_DATASET_SCALE)
        dataset_raw = pm.Normal("joint_dataset_raw", 0.0, 1.0, dims=("dataset", "feature"))
        dataset_effect = pm.Deterministic(
            "joint_dataset_effect",
            global_effect[None, :] + dataset_scale * center_coefficients(dataset_raw, blocks),
            dims=("dataset", "feature"),
        )
        dataset_index = pt.as_tensor_variable(arrays["dataset_index"])
        patient_effect = dataset_effect[dataset_index]
        linear_predictor = pt.sum(
            pt.as_tensor_variable(standardized) * patient_effect[:, None, :],
            axis=2,
        )
        hazard = pt.exp(baseline_log_rate[:, None, :] + linear_predictor[:, :, None])
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
    posterior_summary, diagnostic_values = diagnostics(idata, variables)
    feature_results, dataset_results = summarize_joint(idata, selected, arrays["datasets"], mode)
    for key, value in diagnostic_values.items():
        feature_results[key] = value
    feature_results["patients"] = len(data)
    feature_results["events"] = int(data.event_observed.sum())
    feature_results["datasets"] = len(arrays["datasets"])
    feature_results["particles"] = particles.shape[1]
    feature_results["wall_time_seconds"] = elapsed

    feature_results.to_csv(output_dir / "joint_feature_results.csv", index=False)
    dataset_results.to_csv(output_dir / "joint_dataset_effects.csv", index=False)
    posterior_summary.to_csv(output_dir / "posterior_summary.csv")
    idata.to_netcdf(output_dir / "posterior.nc")
    manifest = {
        "mode": mode,
        "features": selected,
        "composition_blocks": blocks,
        "scaling": scaling,
        "coefficient_constraint": "sum to zero within each CLR composition block",
        "global_shrinkage_prior": f"HalfNormal({JOINT_GLOBAL_SCALE})",
        "dataset_deviation_prior": f"HalfNormal({JOINT_DATASET_SCALE})",
    }
    (output_dir / "joint_manifest.json").write_text(json.dumps(manifest, indent=2))
    return feature_results, dataset_results


def main():
    parser = argparse.ArgumentParser(description="Fit uncertainty-marginalized TCAT survival models.")
    parser.add_argument("estimates_csv", type=Path)
    parser.add_argument("particles_npz", type=Path)
    parser.add_argument("outcomes_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mode", choices=["individual", "joint-gep", "joint-cell-type"], required=True)
    parser.add_argument("--minimum-cells", type=int, default=50)
    parser.add_argument("--feature", action="append", default=None)
    parser.add_argument("--feature-kind", choices=["all", "gep", "score", "cell_type"], default="all")
    parser.add_argument("--maximum-features", type=int, default=None)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, central, particles, particle_weights, feature_names = load_tcat_analysis(
        args.estimates_csv, args.particles_npz, args.outcomes_csv, args.minimum_cells
    )
    arrays = survival_arrays(data)
    data.to_csv(args.output_dir / "analysis_cohort.csv", index=False)
    print(
        f"Analysis cohort: {len(data):,} patients | {int(data.event_observed.sum()):,} events | "
        f"{len(arrays['datasets']):,} datasets | {particles.shape[1]:,} particles | "
        f"{len(feature_names):,} TCAT features",
        flush=True,
    )

    if args.mode == "individual":
        selected = individual_features(
            feature_names, args.feature, args.feature_kind, args.maximum_features
        )
        lookup = {feature: index for index, feature in enumerate(feature_names)}
        results = []
        dataset_results = []
        errors = []
        print(f"Fitting {len(selected):,} independent TCAT models", flush=True)
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
                    result, datasets = individual_model(
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
                    pd.DataFrame(errors).to_csv(args.output_dir / "individual_errors.csv", index=False)
                    print(f"ERROR {type(error).__name__}: {error}", flush=True)
                    continue
            results.append(result)
            dataset_results.append(datasets)
            print(
                f"HR={result['HR_median']:.3f} "
                f"[{result['HR_q025']:.3f}, {result['HR_q975']:.3f}] | "
                f"P(beta>0)={result['P_effect_gt_0']:.3f} | "
                f"P(all same direction)={result['P_all_datasets_same_direction']:.3f}",
                flush=True,
            )
            pd.DataFrame(results).to_csv(args.output_dir / "individual_results.csv", index=False)
            pd.concat(dataset_results, ignore_index=True).to_csv(
                args.output_dir / "individual_dataset_effects.csv", index=False
            )
        if errors:
            print(f"Completed with {len(errors):,} model errors; see individual_errors.csv", flush=True)
    else:
        output_dir = args.output_dir / args.mode
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "joint_feature_results.csv"
        if result_path.exists() and not args.overwrite:
            print(f"RESUME: existing completed {args.mode} model", flush=True)
            feature_results = pd.read_csv(result_path)
        else:
            feature_results, _ = joint_model(
                args.mode, data, central, particles, particle_weights,
                feature_names, arrays, args, output_dir,
            )
        print("\nJoint posterior feature effects:", flush=True)
        print(
            feature_results.sort_values("P_effect_gt_0", ascending=False).to_string(index=False),
            flush=True,
        )

    run_manifest = {
        "mode": args.mode,
        "minimum_cells": args.minimum_cells,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "target_accept": args.target_accept,
        "random_state": args.random_state,
        "particle_definition": "patient cell-level Bayesian bootstrap",
        "uncertainty_integration": "likelihood marginalized across coherent patient particles",
    }
    (args.output_dir / f"{args.mode}_run_manifest.json").write_text(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    main()
