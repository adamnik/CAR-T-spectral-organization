# Leave-one-dataset-out TCAT clinical fields

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
from fit_tcat_survival import feature_kind, sample_model
from survival_model import add_adjustment_arguments, adjustment_design, adjustment_parameters
from survival_model import diagnostics, hierarchical_parameters, interval_loglikelihood
from survival_model import load_tcat_analysis, marginalize_particles, model_coordinates
from survival_model import parse_adjustments, summarize_adjustments
from survival_model import summarize_effects, survival_arrays


def fit_feature(feature, feature_index, data, central, particles, particle_weights,
                arrays, adjustment_matrix, adjustment_table, args, output_dir):
    center = float(central[:, feature_index].mean())
    scale = float(central[:, feature_index].std(ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Feature {feature!r} has no patient-level variation.")
    standardized = (particles[:, :, feature_index] - center) / scale

    coords = model_coordinates(data, arrays)
    if not adjustment_table.empty:
        coords["adjustment"] = adjustment_table.adjustment.tolist()
    with pm.Model(coords=coords):
        baseline_log_rate, patient_effect = hierarchical_parameters(arrays)
        clinical_adjustment = adjustment_parameters(
            adjustment_matrix, args.adjustment_effect_sd
        )
        hazard = pt.exp(
            baseline_log_rate[:, None, :]
            + clinical_adjustment[:, None, None]
            + patient_effect[:, None, None] * pt.as_tensor_variable(standardized)[:, :, None]
        )
        conditional = interval_loglikelihood(hazard, arrays)
        patient_loglikelihood = marginalize_particles(conditional, particle_weights)
        pm.Potential("survival_loglikelihood", pt.sum(patient_loglikelihood))
        started = time.time()
        idata = sample_model(args)
        elapsed = time.time() - started

    variables = [
        "global_effect", "dataset_effect_sd", "dataset_effect",
        "baseline_log_rate_initial", "baseline_smoothness", "baseline_innovation",
    ]
    if not adjustment_table.empty:
        variables.append("adjustment_effect")
    posterior_summary, diagnostic_values = diagnostics(idata, variables)
    result, dataset_results = summarize_effects(idata, arrays["datasets"])
    result.update(diagnostic_values)
    result.update({
        "feature": feature,
        "feature_kind": feature_kind(feature),
        "effect_unit": "per training-cohort SD of the central patient estimate",
        "center": center,
        "scale": scale,
        "patients": len(data),
        "events": int(data.event_observed.sum()),
        "datasets": len(arrays["datasets"]),
        "particles": particles.shape[1],
        "wall_time_seconds": elapsed,
    })
    dataset_results.insert(0, "feature", feature)
    pd.DataFrame([result]).to_csv(output_dir / "model_result.csv", index=False)
    dataset_results.to_csv(output_dir / "dataset_effects.csv", index=False)
    summarize_adjustments(idata, adjustment_table).to_csv(
        output_dir / "adjustment_effects.csv", index=False
    )
    posterior_summary.to_csv(output_dir / "posterior_summary.csv")
    if args.save_posteriors:
        idata.to_netcdf(output_dir / "posterior.nc")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Refit every TCAT GEP survival effect while holding out one dataset."
    )
    parser.add_argument("estimates_csv", type=Path)
    parser.add_argument("particles_npz", type=Path)
    parser.add_argument("outcomes_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-cells", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--save-posteriors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    add_adjustment_arguments(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adjustments = parse_adjustments(args.adjustment)

    data, central, particles, particle_weights, feature_names = load_tcat_analysis(
        args.estimates_csv, args.particles_npz, args.outcomes_csv, args.minimum_cells,
        adjustment_columns=adjustments,
    )
    gep_names = [feature for feature in feature_names if feature.startswith("TCAT_GEP_")]
    feature_lookup = {feature: index for index, feature in enumerate(feature_names)}
    datasets = sorted(data.Dataset.astype(str).unique())
    data.to_csv(args.output_dir / "analysis_cohort.csv", index=False)
    print(
        f"Cross-fitting {len(gep_names):,} GEP effects across {len(datasets):,} held-out datasets | "
        f"{len(data):,} patients",
        flush=True,
    )

    completed_rows = []
    errors = []
    for fold_number, heldout in enumerate(datasets, start=1):
        # Fit and standardize using the remaining datasets
        keep = data.Dataset.astype(str).ne(heldout).to_numpy()
        training_data = data.loc[keep].reset_index(drop=True)
        training_central = central[keep]
        training_particles = particles[keep]
        training_weights = particle_weights[keep]
        arrays = survival_arrays(training_data)
        adjustment_matrix, adjustment_table = adjustment_design(training_data, adjustments)
        fold_dir = args.output_dir / f"heldout_{safe_name(heldout)}"
        fold_dir.mkdir(exist_ok=True)
        training_data.to_csv(fold_dir / "training_cohort.csv", index=False)
        print("\n" + "#" * 108, flush=True)
        print(
            f"FOLD {fold_number}/{len(datasets)} | held out {heldout} | "
            f"training n={len(training_data):,}, events={int(training_data.event_observed.sum()):,}",
            flush=True,
        )
        print("#" * 108, flush=True)

        for feature_number, feature in enumerate(gep_names, start=1):
            feature_dir = fold_dir / "features" / (
                f"{feature_lookup[feature]:03d}_{safe_name(feature)}"
            )
            feature_dir.mkdir(parents=True, exist_ok=True)
            result_path = feature_dir / "model_result.csv"
            adjustment_path = feature_dir / "adjustment_effects.csv"
            print(f'[{feature_number}/{len(gep_names)}] {feature}', flush=True)
            can_resume = (
                result_path.exists()
                and not args.overwrite
                and (not adjustments or adjustment_path.exists())
            )
            if can_resume:
                result = pd.read_csv(result_path).iloc[0].to_dict()
                print("  RESUME", flush=True)
            else:
                try:
                    result = fit_feature(
                        feature, feature_lookup[feature],
                        training_data, training_central, training_particles, training_weights,
                        arrays, adjustment_matrix, adjustment_table, args, feature_dir,
                    )
                except Exception as error:
                    errors.append({
                        "heldout_dataset": heldout,
                        "feature": feature,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    })
                    pd.DataFrame(errors).to_csv(
                        args.output_dir / "crossfit_errors.csv", index=False
                    )
                    print(f"  ERROR {type(error).__name__}: {error}", flush=True)
                    continue
            result["heldout_dataset"] = heldout
            completed_rows.append(result)
            pd.DataFrame(completed_rows).to_csv(
                args.output_dir / "crossfit_feature_results.csv", index=False
            )
            print(
                f"  HR={float(result['HR_median']):.3f} "
                f"[{float(result['HR_q025']):.3f}, {float(result['HR_q975']):.3f}] | "
                f"Rhat={float(result['max_rhat']):.3f} | divergences={int(result['divergences'])}",
                flush=True,
            )

    manifest = {
        "model": "leave-one-dataset-out independent TCAT GEP survival effects",
        "heldout_datasets": datasets,
        "features": gep_names,
        "minimum_cells": args.minimum_cells,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "target_accept": args.target_accept,
        "random_state": args.random_state,
        "saved_posteriors": args.save_posteriors,
        "adjustments": adjustments,
        "adjustment_effect_prior": f"Normal(0, {args.adjustment_effect_sd})",
    }
    (args.output_dir / "crossfit_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nResults:", args.output_dir / "crossfit_feature_results.csv", flush=True)


if __name__ == "__main__":
    main()
