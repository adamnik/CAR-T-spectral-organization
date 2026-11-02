# Held-out evaluation of cross-fitted TCAT clinical fields

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_tcat_predictive_field import posterior_median_correlations, posterior_states
from predictive_field import DESIRED_SIGN, METRICS, field_contrast
from predictive_field import permutation_summary, permute_within_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate O-lambda organization using TCAT effects fit without each dataset."
    )
    parser.add_argument("tcat_particles_npz", type=Path)
    parser.add_argument("olambda_regime_membership_csv", type=Path)
    parser.add_argument("crossfit_results_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--correlation-draws", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--minimum-effective-regime", type=float, default=3.0)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    archive = np.load(args.tcat_particles_npz, allow_pickle=False)
    feature_names = archive["feature_names"].astype(str)
    gep_names = np.asarray([
        feature for feature in feature_names if feature.startswith("TCAT_GEP_")
    ])
    feature_lookup = {feature: index for index, feature in enumerate(feature_names)}
    gep_indices = np.asarray([feature_lookup[feature] for feature in gep_names])
    patients = pd.DataFrame({
        "Dataset": archive["datasets"].astype(str),
        "sample_id": archive["patient_ids"].astype(str),
        "particle_row": np.arange(len(archive["patient_ids"])),
    })
    # Full-cohort regimes; only the clinical field is cross-fitted
    membership = pd.read_csv(args.olambda_regime_membership_csv)[[
        "Dataset", "sample_id", "posterior_mean_probability_high"
    ]]
    patients = patients.merge(
        membership, on=["Dataset", "sample_id"], how="inner", validate="one_to_one"
    )
    patients = patients.sort_values(['Dataset', 'sample_id']).reset_index(drop=True)
    rows = patients.particle_row.to_numpy(int)
    values = np.asarray(archive["values"][rows][:, :, gep_indices], dtype=float)
    particle_weights = np.asarray(archive["particle_weights"][rows], dtype=float)
    particle_weights /= particle_weights.sum(axis=1, keepdims=True)
    datasets = patients.Dataset.astype(str).to_numpy()
    states, bootstrap_weights = posterior_states(
        values, particle_weights, datasets,
        args.correlation_draws, args.random_state,
    )
    states = states["global"]

    crossfit = pd.read_csv(args.crossfit_results_csv)
    fold_rows = []
    fold_null = {}
    fold_result_path = args.output_dir / "crossfit_predictive_field_by_dataset.csv"
    null_dir = args.output_dir / "permutation_null_parts"
    null_dir.mkdir(exist_ok=True)
    for fold_number, heldout in enumerate(sorted(patients.Dataset.unique()), start=1):
        index = np.flatnonzero(datasets == heldout)
        high_probability = patients.loc[index, 'posterior_mean_probability_high'].to_numpy(float)
        effective_high = high_probability.sum()
        effective_low = len(index) - effective_high
        if min(effective_high, effective_low) < args.minimum_effective_regime:
            print(
                f"SKIP {heldout}: effective regimes {effective_low:.1f} low, "
                f"{effective_high:.1f} high", flush=True
            )
            continue

        keep = crossfit.heldout_dataset.astype(str).eq(heldout)
        effects = crossfit.loc[keep].set_index('feature')
        missing = sorted(set(gep_names).difference(effects.index.astype(str)))
        if missing:
            raise ValueError(f"Cross-fitted field for {heldout} is missing {len(missing)} GEPs.")
        beta = np.log(effects.loc[gep_names, "HR_median"].to_numpy(float))
        high, low = posterior_median_correlations(
            states, bootstrap_weights, high_probability, index
        )
        observed = field_contrast(high, low, beta)

        rng = np.random.default_rng(args.random_state + fold_number)
        null = np.empty((args.permutations, len(METRICS)))
        group = [np.arange(len(index))]
        for permutation in range(args.permutations):
            permuted = permute_within_dataset(high_probability, group, rng)
            permuted_high, permuted_low = posterior_median_correlations(
                states, bootstrap_weights, permuted, index
            )
            null[permutation] = field_contrast(permuted_high, permuted_low, beta)
        z, p, joint_z, joint_p, joint_null = permutation_summary(observed, null)
        fold_null[heldout] = (null, joint_null)
        row = {
            "heldout_dataset": heldout,
            "patients": len(index),
            "effective_high_patients": effective_high,
            "effective_low_patients": effective_low,
            "joint_oriented_z": joint_z,
            "joint_permutation_p": joint_p,
        }
        for metric_index, metric in enumerate(METRICS):
            row[f"contrast__{metric}"] = observed[metric_index]
            row[f"z__{metric}"] = z[metric_index]
            row[f"p__{metric}"] = p[metric_index]
        fold_rows.append(row)
        pd.DataFrame(fold_rows).to_csv(fold_result_path, index=False)
        np.savez_compressed(
            null_dir / f"{heldout}.npz", null=null, joint_null=joint_null
        )
        print(
            f"[{fold_number}/{patients.Dataset.nunique()}] {heldout} | n={len(index)} | "
            f"joint z={joint_z:.2f}, p={joint_p:.4g}", flush=True
        )

    folds = pd.DataFrame(fold_rows)
    if folds.empty:
        raise ValueError("No held-out dataset had sufficient effective regime size.")
    folds.to_csv(fold_result_path, index=False)

    combined_rows = []
    standardized_null = []
    for row in folds.itertuples(index=False):
        null, _ = fold_null[row.heldout_dataset]
        null_mean = null.mean(axis=0)
        null_sd = np.maximum(null.std(axis=0, ddof=1), 1e-12)
        standardized_null.append((null - null_mean) / null_sd * DESIRED_SIGN)
    # Give each held-out dataset equal weight in the combined test
    standardized_null = np.stack(standardized_null, axis=0)
    for metric_index, metric in enumerate(METRICS):
        observed = np.mean(folds[f'z__{metric}'].to_numpy(float) * DESIRED_SIGN[metric_index])
        null = standardized_null[:, :, metric_index].mean(axis=0)
        p = (1 + np.sum(null >= observed)) / (len(null) + 1)
        combined_rows.append({
            "metric": metric,
            "datasets": len(folds),
            "mean_oriented_z": observed,
            "permutation_p": p,
            "datasets_in_expected_direction": int(np.sum(
                folds[f"contrast__{metric}"].to_numpy(float)
                * DESIRED_SIGN[metric_index] > 0
            )),
        })
    oriented_z = []
    for _, row in folds.iterrows():
        for index, metric in enumerate(METRICS):
            oriented_z.append(row[f'z__{metric}'] * DESIRED_SIGN[index])
    observed_joint = np.mean(oriented_z)
    null_joint = standardized_null.mean(axis=(0, 2))
    combined_rows.append({
        "metric": "three_metric_joint",
        "datasets": len(folds),
        "mean_oriented_z": observed_joint,
        "permutation_p": (1 + np.sum(null_joint >= observed_joint)) / (len(null_joint) + 1),
        "datasets_in_expected_direction": int(np.sum(folds.joint_oriented_z.to_numpy(float) > 0)),
    })
    combined = pd.DataFrame(combined_rows)
    combined.to_csv(args.output_dir / "crossfit_predictive_field_combined.csv", index=False)

    manifest = {
        "validation": "dataset-held-out TCAT clinical field",
        "clinical_field": "posterior-median GEP log-HRs fit without the evaluated dataset",
        "regime_membership": "full-cohort O-lambda Bayesian change-point probabilities",
        "geometry": "within held-out dataset",
        "correlation_draws": args.correlation_draws,
        "permutations": args.permutations,
        "minimum_effective_regime": args.minimum_effective_regime,
        "random_state": args.random_state,
    }
    (args.output_dir / "crossfit_predictive_field_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print("\nCombined cross-fitted validation", flush=True)
    print(combined.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
