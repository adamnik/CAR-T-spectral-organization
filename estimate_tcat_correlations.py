# Bayesian correlations among TCAT features and spectral organization

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def feature_kind(feature):
    if feature.startswith("TCAT_GEP_"):
        return "GEP"
    if "_proportion_" in feature:
        return "cell_type"
    return "score"


def load_particles(tcat_path, olambda_path):
    tcat = np.load(tcat_path, allow_pickle=False)
    tcat_values = tcat["values"].astype(float)
    tcat_weights = tcat["particle_weights"].astype(float)
    feature_names = tcat["feature_names"].astype(str)
    patients = pd.DataFrame({
        "Dataset": tcat["datasets"].astype(str),
        "sample_id": tcat["patient_ids"].astype(str),
        "tcat_index": np.arange(tcat_values.shape[0]),
    })
    if patients.duplicated(["Dataset", "sample_id"]).any():
        raise ValueError("TCAT particle identities are not unique.")

    olambda = pd.read_csv(olambda_path)
    olambda_columns = [column for column in olambda if column.startswith("particle_")]
    if not olambda_columns:
        raise ValueError("No particle_* columns were found in the O-lambda particle file.")
    if olambda.duplicated(["Dataset", "sample_id"]).any():
        raise ValueError("O-lambda particle identities are not unique.")
    olambda = olambda[["Dataset", "sample_id", *olambda_columns]]

    data = patients.merge(
        olambda,
        on=["Dataset", "sample_id"],
        how="inner",
        validate="one_to_one",
    )
    if data.empty:
        raise ValueError("No patients were shared by the TCAT and O-lambda particle files.")
    indices = data.tcat_index.to_numpy(int)
    weights = tcat_weights[indices]
    weights = weights / weights.sum(axis=1, keepdims=True)
    return (
        data[["Dataset", "sample_id"]],
        feature_names,
        tcat_values[indices],
        weights,
        data[olambda_columns].to_numpy(float),
    )


def dataset_residuals(values, weights, datasets):
    residuals = np.asarray(values, dtype=float).copy()
    for dataset in np.unique(datasets):
        keep = datasets == dataset
        group_weights = weights[keep]
        group_weights = group_weights / group_weights.sum()
        residuals[keep] -= np.sum(values[keep] * group_weights[:, None], axis=0)
    return residuals


def weighted_correlation_matrix(values, weights):
    values = values - np.sum(weights[:, None] * values, axis=0)
    covariance = (values * weights[:, None]).T @ values
    variance = np.diag(covariance)
    denominator = np.sqrt(np.outer(variance, variance))
    correlation = covariance / np.maximum(denominator, 1e-14)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def weighted_vector_correlations(values, target, weights):
    values = values - np.sum(weights[:, None] * values, axis=0)
    target = target - np.sum(weights * target)
    covariance = np.sum(weights[:, None] * values * target[:, None], axis=0)
    value_variance = np.sum(weights[:, None] * values**2, axis=0)
    target_variance = np.sum(weights * target**2)
    return covariance / np.sqrt(np.maximum(value_variance * target_variance, 1e-14))


def correlation_draws(data, tcat_values, tcat_weights, olambda_values,
                      n_draws, random_state):
    rng = np.random.default_rng(random_state)
    datasets = data.Dataset.astype(str).to_numpy()
    n_patients, n_particles, n_features = tcat_values.shape
    pairwise = np.empty((n_draws, n_features, n_features), dtype=np.float32)
    olambda = np.empty((n_draws, n_features), dtype=np.float32)
    progress_step = max(1, n_draws // 10)

    for draw in range(n_draws):
        tcat_index = np.asarray([
            rng.choice(n_particles, p=tcat_weights[patient])
            for patient in range(n_patients)
        ])
        olambda_index = rng.integers(0, olambda_values.shape[1], size=n_patients)
        patient_weights = rng.dirichlet(np.ones(n_patients))
        tcat = tcat_values[np.arange(n_patients), tcat_index]
        spectral = olambda_values[np.arange(n_patients), olambda_index]

        tcat = dataset_residuals(tcat, patient_weights, datasets)
        spectral = dataset_residuals(spectral[:, None], patient_weights, datasets)[:, 0]
        pairwise[draw] = weighted_correlation_matrix(tcat, patient_weights)
        olambda[draw] = weighted_vector_correlations(tcat, spectral, patient_weights)

        if (draw + 1) % progress_step == 0 or draw + 1 == n_draws:
            print(f"Correlation draw {draw + 1:,}/{n_draws:,}", flush=True)
    return pairwise, olambda


def summarize_olambda(feature_names, draws):
    rows = []
    for index, feature in enumerate(feature_names):
        values = draws[:, index]
        rows.append({
            "feature": feature,
            "feature_kind": feature_kind(feature),
            "rho_median": np.median(values),
            "rho_q025": np.quantile(values, 0.025),
            "rho_q975": np.quantile(values, 0.975),
            "P_rho_gt_0": np.mean(values > 0),
            "P_rho_lt_0": np.mean(values < 0),
            "P_same_direction": max(np.mean(values > 0), np.mean(values < 0)),
        })
    return pd.DataFrame(rows)


def summarize_pairwise(feature_names, draws):
    rows = []
    for first in range(len(feature_names)):
        for second in range(first + 1, len(feature_names)):
            values = draws[:, first, second]
            rows.append({
                "feature_a": feature_names[first],
                "feature_b": feature_names[second],
                "feature_a_kind": feature_kind(feature_names[first]),
                "feature_b_kind": feature_kind(feature_names[second]),
                "rho_median": np.median(values),
                "rho_q025": np.quantile(values, 0.025),
                "rho_q975": np.quantile(values, 0.975),
                "P_rho_gt_0": np.mean(values > 0),
                "P_rho_lt_0": np.mean(values < 0),
                "P_same_direction": max(np.mean(values > 0), np.mean(values < 0)),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate uncertainty-aware, dataset-adjusted correlations among TCAT features "
            "and between TCAT features and asymptotic O-lambda."
        )
    )
    parser.add_argument("tcat_particles_npz", type=Path)
    parser.add_argument("olambda_particles_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--save-draws", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_paths = {
        "olambda": args.output_dir / "olambda_tcat_correlations.csv",
        "pairwise": args.output_dir / "tcat_pairwise_correlations.csv",
        "manifest": args.output_dir / "correlation_manifest.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output already exists: {existing[0]}. Use --overwrite to replace it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, feature_names, tcat_values, tcat_weights, olambda_values = load_particles(
        args.tcat_particles_npz,
        args.olambda_particles_csv,
    )
    print(
        f"Analysis cohort: {len(data):,} patients | {data.Dataset.nunique():,} datasets | "
        f"{len(feature_names):,} TCAT features",
        flush=True,
    )
    pairwise_draws, olambda_draws = correlation_draws(
        data,
        tcat_values,
        tcat_weights,
        olambda_values,
        args.draws,
        args.random_state,
    )
    summarize_olambda(feature_names, olambda_draws).to_csv(output_paths["olambda"], index=False)
    summarize_pairwise(feature_names, pairwise_draws).to_csv(output_paths["pairwise"], index=False)
    if args.save_draws:
        np.savez_compressed(
            args.output_dir / "correlation_draws.npz",
            pairwise=pairwise_draws,
            olambda=olambda_draws,
            feature_names=feature_names,
        )

    manifest = {
        "tcat_particles_npz": str(args.tcat_particles_npz),
        "olambda_particles_csv": str(args.olambda_particles_csv),
        "patients": len(data),
        "datasets": sorted(data.Dataset.astype(str).unique()),
        "features": feature_names.tolist(),
        "draws": args.draws,
        "random_state": args.random_state,
        "patient_uncertainty": "independent TCAT and O-lambda posterior-particle sampling",
        "cohort_uncertainty": "Bayesian bootstrap over patients",
        "dataset_adjustment": "weighted dataset-mean residualization within each draw",
    }
    output_paths["manifest"].write_text(json.dumps(manifest, indent=2))
    print("O-lambda correlations:", output_paths["olambda"], flush=True)
    print("Pairwise TCAT correlations:", output_paths["pairwise"], flush=True)


if __name__ == "__main__":
    main()
