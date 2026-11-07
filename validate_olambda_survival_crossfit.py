# Leave-one-dataset-out validation of O-lambda survival models

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

from analysis_io import safe_name
from survival_model import add_adjustment_arguments, parse_adjustments


def run_command(command):
    print(" ".join(map(str, command)), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run([str(value) for value in command], check=True, env=environment)


def adjustment_arguments(args):
    arguments = ["--adjustment-effect-sd", args.adjustment_effect_sd]
    for specification in args.adjustment:
        arguments.extend(["--adjustment", specification])
    return arguments


def sampling_arguments(args):
    return [
        "--minimum-cells", args.minimum_cells,
        "--n-particles", args.n_particles,
        "--draws", args.draws,
        "--tune", args.tune,
        "--chains", args.chains,
        "--cores", args.cores,
        "--target-accept", args.target_accept,
        "--random-state", args.random_state,
        *adjustment_arguments(args),
    ]


def prepare_folds(asymptotes, outcomes, output_dir, minimum_cells):
    eligible = asymptotes.loc[
        asymptotes.n_cells_available >= minimum_cells,
        ["Dataset", "sample_id"],
    ].merge(
        outcomes[["Dataset", "sample_id", "event_observed"]],
        on=["Dataset", "sample_id"],
        how="inner",
        validate="one_to_one",
    )
    datasets = sorted(eligible.Dataset.astype(str).unique())
    rows = []
    for heldout in datasets:
        fold_dir = output_dir / "training_fits" / f"heldout_{safe_name(heldout)}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        training_outcomes = outcomes.loc[
            ~outcomes.Dataset.astype(str).eq(heldout)
        ].copy()
        training_path = fold_dir / "training_outcomes.csv"
        training_outcomes.to_csv(training_path, index=False)
        heldout_rows = eligible.Dataset.astype(str).eq(heldout)
        rows.append({
            "heldout_dataset": heldout,
            "training_outcomes_csv": str(training_path),
            "training_patients": int((~heldout_rows).sum()),
            "heldout_patients": int(heldout_rows.sum()),
            "heldout_events": int(eligible.loc[heldout_rows, "event_observed"].sum()),
        })
    folds = pd.DataFrame(rows)
    folds.to_csv(output_dir / "fold_manifest.csv", index=False)
    return folds


def run_training_fits(folds, script_dir, asymptotes_path,
                      particles_path, output_dir, args):
    for fold_number, fold in enumerate(folds.itertuples(index=False), start=1):
        fold_dir = (
            output_dir / "training_fits"
            / f"heldout_{safe_name(fold.heldout_dataset)}"
        )
        print("\n" + "=" * 108, flush=True)
        print(
            f"FOLD {fold_number}/{len(folds)} | held out {fold.heldout_dataset}",
            flush=True,
        )
        print("=" * 108, flush=True)

        continuous_dir = fold_dir / "continuous"
        continuous_complete = (
            (continuous_dir / "model_result.csv").exists()
            and (continuous_dir / "posterior.nc").exists()
            and not args.overwrite
        )
        if continuous_complete:
            print("Continuous: RESUME", flush=True)
        else:
            run_command([
                sys.executable, "-u",
                script_dir / "fit_olambda_survival_continuous.py",
                asymptotes_path, fold.training_outcomes_csv, continuous_dir,
                "--particles-csv", particles_path,
                *sampling_arguments(args),
            ])

        transition_dir = fold_dir / "smooth_transition_Q10_Q90"
        transition_complete = (
            (transition_dir / "model_result.csv").exists()
            and (transition_dir / "posterior.nc").exists()
            and not args.overwrite
        )
        if transition_complete:
            print("Smooth transition: RESUME", flush=True)
        else:
            run_command([
                sys.executable, "-u",
                script_dir / "fit_olambda_survival_regimes.py",
                asymptotes_path, fold.training_outcomes_csv, transition_dir,
                "--particles-csv", particles_path,
                *sampling_arguments(args),
                "--crossover-quantile-low", args.crossover_quantile_low,
                "--crossover-quantile-high", args.crossover_quantile_high,
                "--transition-scale-sd", args.transition_scale_sd,
            ])


def collect_training_results(folds, output_dir):
    rows = []
    for fold in folds.itertuples(index=False):
        fold_dir = (
            output_dir / "training_fits"
            / f"heldout_{safe_name(fold.heldout_dataset)}"
        )
        for model, directory in [
            ("Continuous", "continuous"),
            ("Smooth transition", "smooth_transition_Q10_Q90"),
        ]:
            result = pd.read_csv(
                fold_dir / directory / "model_result.csv"
            ).iloc[0].to_dict()
            rows.append({
                "heldout_dataset": fold.heldout_dataset,
                "model": model,
                "heldout_patients": fold.heldout_patients,
                "heldout_events": fold.heldout_events,
                **result,
            })
    results = pd.DataFrame(rows)
    results.to_csv(
        output_dir / "leave_one_dataset_out_training_results.csv", index=False
    )
    return results


def posterior_values(path, variable):
    posterior = az.from_netcdf(path).posterior
    return np.asarray(posterior[variable]).reshape(-1)


def build_cross_fitted_scores(folds, asymptotes, particles, outcomes,
                              output_dir, args):
    particle_columns = [
        column for column in particles if column.startswith("particle_")
    ]
    if len(particle_columns) != args.n_particles:
        raise ValueError(
            f"Expected {args.n_particles} O-lambda particles, "
            f"found {len(particle_columns)}."
        )
    adjustments = parse_adjustments(args.adjustment)
    rng = np.random.default_rng(args.random_state)
    score_rows = []
    audit_rows = []

    for fold_number, fold in enumerate(folds.itertuples(index=False), start=1):
        heldout = str(fold.heldout_dataset)
        fold_dir = (
            output_dir / "training_fits" / f"heldout_{safe_name(heldout)}"
        )
        heldout_data = (
            asymptotes.loc[
                asymptotes.Dataset.astype(str).eq(heldout)
                & (asymptotes.n_cells_available >= args.minimum_cells),
                ["Dataset", "sample_id", "n_cells_available"],
            ]
            .merge(
                particles[["Dataset", "sample_id", *particle_columns]],
                on=["Dataset", "sample_id"],
                how="inner",
                validate="one_to_one",
            )
            .merge(
                outcomes[["Dataset", "sample_id", *adjustments]],
                on=["Dataset", "sample_id"],
                how="inner",
                validate="one_to_one",
            )
        )
        if adjustments:
            complete = heldout_data[list(adjustments)].notna().all(axis=1)
            for column in adjustments:
                complete &= heldout_data[column].astype(str).str.strip().ne("")
            heldout_data = heldout_data.loc[complete].reset_index(drop=True)
        raw_particles = heldout_data[particle_columns].to_numpy(float)

        continuous_dir = fold_dir / "continuous"
        continuous_manifest = json.loads(
            (continuous_dir / "run_manifest.json").read_text()
        )
        continuous_beta = posterior_values(
            continuous_dir / "posterior.nc", "global_effect"
        )
        continuous_indices = rng.choice(
            len(continuous_beta),
            args.n_particles,
            replace=len(continuous_beta) < args.n_particles,
        )
        center = float(continuous_manifest["particle_scaling"]["center"])
        scale = float(continuous_manifest["particle_scaling"]["scale"])

        transition_dir = fold_dir / "smooth_transition_Q10_Q90"
        transition_manifest = json.loads(
            (transition_dir / "run_manifest.json").read_text()
        )
        transition_beta = posterior_values(
            transition_dir / "posterior.nc", "global_effect"
        )
        transition_crossover = posterior_values(
            transition_dir / "posterior.nc", "crossover_O_lambda"
        )
        transition_indices = rng.choice(
            len(transition_beta),
            args.n_particles,
            replace=len(transition_beta) < args.n_particles,
        )
        transition_center = float(
            transition_manifest["particle_scaling"]["center"]
        )
        transition_scale = float(
            transition_manifest["particle_scaling"]["scale"]
        )
        transition_width = float(transition_manifest["transition_scale_sd"])

        continuous_scores = np.empty_like(raw_particles)
        transition_scores = np.empty_like(raw_particles)
        for patient in range(len(heldout_data)):
            source_indices = rng.permutation(args.n_particles)
            patient_particles = raw_particles[patient, source_indices]
            continuous_scores[patient] = (
                continuous_beta[continuous_indices]
                * ((patient_particles - center) / scale)
            )
            activation = expit(
                (
                    (patient_particles - transition_crossover[transition_indices])
                    / transition_scale
                )
                / transition_width
            )
            transition_scores[patient] = (
                transition_beta[transition_indices] * activation
            )

        for patient, record in heldout_data.iterrows():
            score_rows.append({
                "Dataset": record.Dataset,
                "sample_id": record.sample_id,
                "n_cells": int(record.n_cells_available),
                "continuous": continuous_scores[patient],
                "transition": transition_scores[patient],
            })
        audit_rows.append({
            "heldout_dataset": heldout,
            "patients": len(heldout_data),
            "continuous_training_effect_median": float(
                np.median(continuous_beta)
            ),
            "transition_training_effect_median": float(
                np.median(transition_beta)
            ),
            "transition_training_crossover_median": float(
                np.median(transition_crossover)
            ),
            "continuous_training_center": center,
            "continuous_training_scale": scale,
            "transition_training_center": transition_center,
            "transition_training_scale": transition_scale,
        })
        print(
            f"[{fold_number}/{len(folds)}] {heldout}: "
            f"{len(heldout_data)} cross-fitted scores",
            flush=True,
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "cross_fitted_score_audit.csv", index=False)
    return score_rows


def save_cross_fitted_scores(score_rows, output_dir, n_particles):
    feature_names = np.asarray([
        "Cross_fitted_continuous_Olambda_score",
        "Cross_fitted_smooth_transition_Olambda_score",
    ])
    values = np.stack([
        np.column_stack([row["continuous"], row["transition"]])
        for row in score_rows
    ])
    weights = np.full((len(score_rows), n_particles), 1 / n_particles)
    estimates = pd.DataFrame({
        "Dataset": [row["Dataset"] for row in score_rows],
        "sample_id": [row["sample_id"] for row in score_rows],
        feature_names[0]: values[:, :, 0].mean(axis=1),
        feature_names[1]: values[:, :, 1].mean(axis=1),
    })
    score_dir = output_dir / "cross_fitted_scores"
    score_dir.mkdir(exist_ok=True)
    estimates_path = score_dir / "cross_fitted_score_estimates.csv"
    particles_path = score_dir / "cross_fitted_score_particles.npz"
    estimates.to_csv(estimates_path, index=False)
    np.savez_compressed(
        particles_path,
        values=values.astype(np.float32),
        particle_weights=weights.astype(np.float32),
        patient_ids=np.asarray([row["sample_id"] for row in score_rows]),
        datasets=np.asarray([row["Dataset"] for row in score_rows]),
        feature_names=feature_names,
        n_cells=np.asarray([row["n_cells"] for row in score_rows], dtype=int),
    )
    return feature_names, estimates_path, particles_path


def run_heldout_validation(feature_names, estimates_path, particles_path,
                           outcomes_path, script_dir, output_dir, args):
    validation_dir = output_dir / "heldout_score_validation"
    result_path = validation_dir / "individual_results.csv"
    if result_path.exists() and not args.overwrite:
        print("Held-out score validation: RESUME", flush=True)
    else:
        command = [
            sys.executable, "-u", script_dir / "fit_tcat_survival.py",
            estimates_path, particles_path, outcomes_path, validation_dir,
            "--mode", "individual",
            "--minimum-cells", args.minimum_cells,
            "--draws", args.draws,
            "--tune", args.tune,
            "--chains", args.chains,
            "--cores", args.cores,
            "--target-accept", args.target_accept,
            "--random-state", args.random_state,
            *adjustment_arguments(args),
        ]
        for feature in feature_names:
            command.extend(["--feature", feature])
        if args.overwrite:
            command.append("--overwrite")
        run_command(command)
    results = pd.read_csv(result_path)
    results.to_csv(
        output_dir / "cross_fitted_heldout_validation_results.csv", index=False
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Refit continuous and smooth O-lambda survival models while holding "
            "out each dataset, then validate their training-derived risk scores "
            "against the combined held-out outcomes."
        )
    )
    parser.add_argument("asymptotes_csv", type=Path)
    parser.add_argument("particles_csv", type=Path)
    parser.add_argument("outcomes_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-cells", type=int, default=1000)
    parser.add_argument("--n-particles", type=int, default=16)
    parser.add_argument("--crossover-quantile-low", type=float, default=0.10)
    parser.add_argument("--crossover-quantile-high", type=float, default=0.90)
    parser.add_argument("--transition-scale-sd", type=float, default=0.10)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    add_adjustment_arguments(parser)
    args = parser.parse_args()
    if not 0 <= args.crossover_quantile_low < args.crossover_quantile_high <= 1:
        raise ValueError("Crossover quantiles must satisfy 0 <= low < high <= 1.")
    if args.transition_scale_sd <= 0:
        raise ValueError("--transition-scale-sd must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    asymptotes = pd.read_csv(args.asymptotes_csv)
    particles = pd.read_csv(args.particles_csv)
    outcomes = pd.read_csv(args.outcomes_csv)
    if "event_observed" not in outcomes and "event" in outcomes:
        outcomes = outcomes.rename(columns={"event": "event_observed"})

    folds = prepare_folds(
        asymptotes, outcomes, args.output_dir, args.minimum_cells
    )
    print(
        f"Analysis cohort: {folds.heldout_patients.sum():,} patients | "
        f"{folds.heldout_events.sum():,} events | {len(folds)} datasets",
        flush=True,
    )
    run_training_fits(
        folds, script_dir, args.asymptotes_csv,
        args.particles_csv, args.output_dir, args,
    )
    training_results = collect_training_results(folds, args.output_dir)

    score_rows = build_cross_fitted_scores(
        folds, asymptotes, particles, outcomes, args.output_dir, args
    )
    feature_names, estimates_path, score_particles_path = save_cross_fitted_scores(
        score_rows, args.output_dir, args.n_particles
    )
    validation_results = run_heldout_validation(
        feature_names, estimates_path, score_particles_path,
        args.outcomes_csv, script_dir, args.output_dir, args,
    )

    manifest = {
        "analysis": "leave-one-dataset-out O-lambda survival validation",
        "datasets": folds.heldout_dataset.tolist(),
        "patients": int(folds.heldout_patients.sum()),
        "events": int(folds.heldout_events.sum()),
        "minimum_cells": args.minimum_cells,
        "particles": args.n_particles,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "target_accept": args.target_accept,
        "random_state": args.random_state,
        "adjustments": parse_adjustments(args.adjustment),
        "adjustment_effect_prior_sd": args.adjustment_effect_sd,
        "crossover_quantile_low": args.crossover_quantile_low,
        "crossover_quantile_high": args.crossover_quantile_high,
        "transition_scale_sd": args.transition_scale_sd,
        "heldout_score_definition": (
            "training global effect multiplied by the training-defined continuous "
            "or smooth-transition transformation of held-out O-lambda particles"
        ),
        "heldout_validation": (
            "hierarchical calibration of cross-fitted scores with dataset-specific "
            "baseline hazards; not frozen absolute-risk prediction"
        ),
    }
    (args.output_dir / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print("\nLeave-one-dataset-out training results", flush=True)
    print(training_results[[
        "heldout_dataset", "model", "HR_median", "HR_q025", "HR_q975",
        "P_effect_gt_0", "divergences", "max_rhat",
    ]].to_string(index=False), flush=True)
    print("\nCross-fitted held-out validation", flush=True)
    print(validation_results[[
        "feature", "HR_median", "HR_q025", "HR_q975",
        "P_effect_gt_0", "P_all_datasets_same_direction",
        "divergences", "max_rhat",
    ]].to_string(index=False), flush=True)
    print("\nResults:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
