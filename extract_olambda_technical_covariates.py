# Patient-level technical covariates and Bayesian-bootstrap particles

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

FEATURES = [
    "log_n_cells",
    "mean_log1p_total_counts",
    "mean_mito_fraction",
    "mean_log1p_detected_genes",
    "log_graph_nodes",
]


def matrix_to_memory(matrix):
    if hasattr(matrix, 'to_memory'):
        matrix = matrix.to_memory()
    if sp.issparse(matrix):
        return matrix.tocsr()
    return np.asarray(matrix)


def count_matrix(adata, layer):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Missing count layer {layer!r}.")
    return adata.layers[layer]


def cell_metrics(matrix, mitochondrial):
    total_counts = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    detected_genes = np.asarray((matrix > 0).sum(axis=1)).ravel().astype(float)
    if mitochondrial.any():
        mito_counts = np.asarray(matrix[:, mitochondrial].sum(axis=1)).ravel()
    else:
        mito_counts = np.zeros(len(total_counts), dtype=float)
    mito_fraction = np.divide(
        mito_counts,
        total_counts,
        out=np.zeros(len(total_counts), dtype=float),
        where=total_counts > 0,
    )
    return np.column_stack([
        np.log1p(total_counts),
        mito_fraction,
        np.log1p(detected_genes),
    ])


def patient_values(metrics, n_cells, graph_nodes, n_particles, rng):
    cell_means = metrics.mean(axis=0)
    # Shared cell weights preserve covariance among the three cell-level metrics
    weights = rng.exponential(size=(n_particles, n_cells))
    weights /= weights.sum(axis=1, keepdims=True)
    cell_draws = weights @ metrics

    central = np.concatenate([[np.log(n_cells)], cell_means, [np.log(graph_nodes)]])
    # Cell count and full-graph size are fixed across particles
    particles = np.column_stack([
        np.repeat(np.log(n_cells), n_particles),
        cell_draws,
        np.repeat(np.log(graph_nodes), n_particles),
    ])
    return central, particles


def full_graph_nodes(trajectory_path):
    trajectories = pd.read_csv(trajectory_path)
    required = {"Dataset", "sample_id", "is_full_reference", "n_nodes_used"}
    missing = required.difference(trajectories.columns)
    if missing:
        raise ValueError(f"Trajectory table is missing columns: {sorted(missing)}")

    full = trajectories.is_full_reference
    if full.dtype != bool:
        full = full.astype(str).str.lower().eq("true")
    full = trajectories.loc[full, ["Dataset", "sample_id", "n_nodes_used"]].copy()
    full["n_nodes_used"] = pd.to_numeric(full.n_nodes_used, errors="coerce")
    full = full.dropna(subset=["n_nodes_used"])
    full = full.loc[full.n_nodes_used > 1]
    graph_nodes = full.groupby(["Dataset", "sample_id"], as_index=False).agg(
        graph_nodes=("n_nodes_used", "median"),
        graph_node_minimum=("n_nodes_used", "min"),
        graph_node_maximum=("n_nodes_used", "max"),
        full_graph_rows=("n_nodes_used", "size"),
    )
    return graph_nodes


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract five patient-level technical covariates and coherent "
            "Bayesian-bootstrap particles from processed infusion datasets."
        )
    )
    parser.add_argument("dataset_manifest", type=Path)
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-key", default="sample_id")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--mito-prefix", default="MT-")
    parser.add_argument("--minimum-cells", type=int, default=1000)
    parser.add_argument("--n-particles", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_paths = {
        "estimates": args.output_dir / "olambda_technical_estimates.csv",
        "uncertainty": args.output_dir / "olambda_technical_uncertainty.csv",
        "particles": args.output_dir / "olambda_technical_particles.npz",
        "cohort": args.output_dir / "technical_extraction_cohort.csv",
        "manifest": args.output_dir / "technical_extraction_manifest.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output already exists: {existing[0]}. Use --overwrite to replace it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = pd.read_csv(args.dataset_manifest)
    required = {"Dataset", "h5ad_path"}
    missing = required.difference(datasets.columns)
    if missing:
        raise ValueError(f"Dataset manifest is missing columns: {sorted(missing)}")
    graph_nodes = full_graph_nodes(args.trajectory_csv)
    graph_lookup = graph_nodes.set_index(['Dataset', 'sample_id']).graph_nodes
    graph_lookup = graph_lookup.to_dict()
    rng = np.random.default_rng(args.random_state)

    central_rows = []
    uncertainty_rows = []
    particle_values = []
    patient_ids = []
    patient_datasets = []
    patient_cells = []
    cohort_rows = []

    for dataset_number, row in enumerate(datasets.itertuples(index=False), start=1):
        dataset = str(row.Dataset)
        path = Path(row.h5ad_path)
        print("=" * 100, flush=True)
        print(f"[{dataset_number}/{len(datasets)}] {dataset} | {path}", flush=True)
        data = ad.read_h5ad(path, backed="r")
        try:
            if args.sample_key not in data.obs:
                raise KeyError(f"Missing sample column {args.sample_key!r} in {path}.")
            if "timepoint" in data.obs:
                timepoints = data.obs.timepoint.dropna().astype(str).str.lower().unique()
                if len(timepoints) != 1 or timepoints[0] != "infusion":
                    raise ValueError(f"{path} must contain infusion cells only.")

            samples = data.obs[args.sample_key].astype(str).to_numpy()
            mitochondrial = np.asarray(
                data.var_names.astype(str).str.upper().str.startswith(args.mito_prefix.upper()),
                dtype=bool,
            )
            if not mitochondrial.any():
                print(f"  warning: no genes begin with {args.mito_prefix!r}", flush=True)
            matrix = count_matrix(data, args.layer)
            patients = sorted(pd.unique(samples))

            for patient_number, patient in enumerate(patients, start=1):
                rows = np.flatnonzero(samples == patient)
                key = (dataset, patient)
                if len(rows) < args.minimum_cells:
                    print(
                        f"  [{patient_number}/{len(patients)}] {patient}: "
                        f"skipped ({len(rows):,} cells)",
                        flush=True,
                    )
                    continue
                if key not in graph_lookup:
                    print(
                        f"  [{patient_number}/{len(patients)}] {patient}: "
                        "skipped (no full-graph node count)",
                        flush=True,
                    )
                    cohort_rows.append({
                        "Dataset": dataset,
                        "sample_id": patient,
                        "n_cells": len(rows),
                        "included": False,
                        "reason": "no full-graph node count",
                    })
                    continue

                patient_matrix = matrix_to_memory(matrix[rows, :])
                metrics = cell_metrics(patient_matrix, mitochondrial)
                graph_size = float(graph_lookup[key])
                central, particles = patient_values(
                    metrics, len(rows), graph_size,
                    args.n_particles, rng,
                )
                central_rows.append({
                    "Dataset": dataset,
                    "sample_id": patient,
                    "n_cells": len(rows),
                    "graph_nodes": graph_size,
                    **dict(zip(FEATURES, central)),
                })
                for feature_index, feature in enumerate(FEATURES):
                    uncertainty_rows.append({
                        "Dataset": dataset,
                        "sample_id": patient,
                        "feature": feature,
                        "central": central[feature_index],
                        "q025": np.quantile(particles[:, feature_index], 0.025),
                        "q975": np.quantile(particles[:, feature_index], 0.975),
                    })
                particle_values.append(particles)
                patient_ids.append(patient)
                patient_datasets.append(dataset)
                patient_cells.append(len(rows))
                cohort_rows.append({
                    "Dataset": dataset,
                    "sample_id": patient,
                    "n_cells": len(rows),
                    "included": True,
                    "reason": "",
                })
                print(
                    f"  [{patient_number}/{len(patients)}] {patient}: "
                    f"{len(rows):,} cells | {int(graph_size):,} graph nodes",
                    flush=True,
                )
        finally:
            if data.file:
                data.file.close()

    if not particle_values:
        raise ValueError("No patients have both eligible cells and a full-graph node count.")
    values = np.stack(particle_values)
    weights = np.full((len(values), args.n_particles), 1.0 / args.n_particles)

    pd.DataFrame(central_rows).to_csv(output_paths["estimates"], index=False)
    pd.DataFrame(uncertainty_rows).to_csv(output_paths["uncertainty"], index=False)
    pd.DataFrame(cohort_rows).to_csv(output_paths["cohort"], index=False)
    np.savez_compressed(
        output_paths["particles"],
        values=values,
        particle_weights=weights,
        patient_ids=np.asarray(patient_ids, dtype="U"),
        datasets=np.asarray(patient_datasets, dtype="U"),
        feature_names=np.asarray(FEATURES, dtype="U"),
        n_cells=np.asarray(patient_cells, dtype=np.int64),
    )
    manifest = {
        "dataset_manifest": str(args.dataset_manifest),
        "trajectory_csv": str(args.trajectory_csv),
        "sample_key": args.sample_key,
        "layer": args.layer,
        "mito_prefix": args.mito_prefix,
        "minimum_cells": args.minimum_cells,
        "n_particles": args.n_particles,
        "random_state": args.random_state,
        "particle_method": "patient-level Bayesian bootstrap with shared Dirichlet(1) cell weights",
        "features": FEATURES,
        "graph_size_definition": "n_nodes_used in the full-reference O-lambda graph",
        "patients": len(patient_ids),
    }
    output_paths["manifest"].write_text(json.dumps(manifest, indent=2))
    print("=" * 100, flush=True)
    print(
        f"Saved {len(patient_ids):,} patients x {args.n_particles:,} particles x "
        f"{len(FEATURES):,} technical covariates",
        flush=True,
    )


if __name__ == "__main__":
    main()
