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
    "cluster_composition_entropy",
    "pca_effective_rank",
    "mean_pairwise_cell_state_distance",
]


def matrix_to_memory(matrix):
    matrix = matrix.to_memory() if hasattr(matrix, "to_memory") else matrix
    return matrix.tocsr() if sp.issparse(matrix) else np.asarray(matrix)


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
        mitochondrial_counts = np.asarray(matrix[:, mitochondrial].sum(axis=1)).ravel()
    else:
        mitochondrial_counts = np.zeros(len(total_counts), dtype=float)
    mitochondrial_fraction = np.divide(
        mitochondrial_counts,
        total_counts,
        out=np.zeros(len(total_counts), dtype=float),
        where=total_counts > 0,
    )
    return np.column_stack([
        np.log1p(total_counts),
        mitochondrial_fraction,
        np.log1p(detected_genes),
    ])


def normalized_entropy(proportions):
    proportions = np.asarray(proportions, dtype=float)
    positive = proportions > 0
    entropy = -(proportions[positive] * np.log(proportions[positive])).sum()
    return entropy / np.log(len(proportions))


def effective_rank(embedding, weights):
    center = weights @ embedding
    centered = embedding - center
    covariance = (centered * weights[:, None]).T @ centered
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0, None)
    if eigenvalues.sum() <= 0:
        return 1.0
    spectrum = eigenvalues / eigenvalues.sum()
    spectrum = spectrum[spectrum > 0]
    return float(np.exp(-(spectrum * np.log(spectrum)).sum()))


def state_metrics(embedding, weights, pair_first, pair_second, pair_distance):
    rank = effective_rank(embedding, weights)
    pair_weights = weights[pair_first] * weights[pair_second]
    distance = np.sum(pair_weights * pair_distance) / pair_weights.sum()
    return rank, distance


def patient_values(metrics, labels, label_levels, embedding, n_cells,
                   graph_nodes, n_particles, distance_pairs, rng):
    central_cell_values = metrics.mean(axis=0)
    weights = rng.exponential(size=(n_particles, n_cells))
    weights /= weights.sum(axis=1, keepdims=True)
    particle_cell_values = weights @ metrics

    level_lookup = {level: index for index, level in enumerate(label_levels)}
    label_index = np.asarray([level_lookup[label] for label in labels], dtype=int)
    one_hot = np.eye(len(label_levels), dtype=float)[label_index]
    central_proportions = one_hot.mean(axis=0)
    particle_proportions = weights @ one_hot
    central_entropy = normalized_entropy(central_proportions)
    particle_entropy = np.asarray([
        normalized_entropy(proportions)
        for proportions in particle_proportions
    ])

    pair_first = rng.integers(0, n_cells, size=distance_pairs)
    pair_second = rng.integers(0, n_cells, size=distance_pairs)
    identical = pair_first == pair_second
    while identical.any():
        pair_second[identical] = rng.integers(0, n_cells, size=identical.sum())
        identical = pair_first == pair_second
    pair_distance = np.linalg.norm(
        embedding[pair_first] - embedding[pair_second], axis=1,
    )
    uniform_weights = np.full(n_cells, 1.0 / n_cells)
    central_rank, central_distance = state_metrics(
        embedding, uniform_weights, pair_first, pair_second, pair_distance,
    )
    particle_state = np.asarray([
        state_metrics(
            embedding, particle_weights,
            pair_first, pair_second, pair_distance,
        )
        for particle_weights in weights
    ])

    central = np.concatenate([
        [np.log(n_cells)],
        central_cell_values,
        [np.log(graph_nodes)],
        [central_entropy, central_rank, central_distance],
    ])
    particles = np.column_stack([
        np.repeat(np.log(n_cells), n_particles),
        particle_cell_values,
        np.repeat(np.log(graph_nodes), n_particles),
        particle_entropy,
        particle_state,
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
            "Extract patient-level technical and cell-state covariates with coherent "
            "Bayesian-bootstrap particles from processed infusion datasets."
        )
    )
    parser.add_argument("dataset_manifest", type=Path)
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-key", default="sample_id")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--mito-prefix", default="MT-")
    parser.add_argument("--cluster-key", default="functional_cart_atlas_label")
    parser.add_argument("--embedding-key", default="X_functional_cart_atlas")
    parser.add_argument("--pca-dimensions", type=int, default=20)
    parser.add_argument("--distance-pairs", type=int, default=50000)
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
    graph_lookup = graph_nodes.set_index(["Dataset", "sample_id"]).graph_nodes.to_dict()
    rng = np.random.default_rng(args.random_state)

    label_levels = set()
    for row in datasets.itertuples(index=False):
        data = ad.read_h5ad(Path(row.h5ad_path), backed="r")
        try:
            if args.cluster_key not in data.obs:
                raise KeyError(
                    f"Missing cluster column {args.cluster_key!r} in {row.h5ad_path}."
                )
            labels = data.obs[args.cluster_key]
            if labels.isna().any():
                raise ValueError(
                    f"Missing cluster labels in {row.h5ad_path}."
                )
            label_levels.update(labels.astype(str).unique())
            if args.embedding_key not in data.obsm:
                raise KeyError(
                    f"Missing cell-state embedding {args.embedding_key!r} in {row.h5ad_path}."
                )
        finally:
            if data.file:
                data.file.close()
    label_levels = sorted(label_levels)
    if len(label_levels) < 2:
        raise ValueError("At least two cluster levels are required for entropy.")
    print(f"Cluster levels ({len(label_levels)}): {label_levels}", flush=True)

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
                data.var_names.astype(str).str.upper().str.startswith(
                    args.mito_prefix.upper()
                ),
                dtype=bool,
            )
            if not mitochondrial.any():
                print(f"  warning: no genes begin with {args.mito_prefix!r}", flush=True)
            matrix = count_matrix(data, args.layer)
            labels = data.obs[args.cluster_key].astype(str).to_numpy()
            embedding = np.asarray(data.obsm[args.embedding_key], dtype=float)
            dimensions = min(args.pca_dimensions, embedding.shape[1])
            embedding = embedding[:, :dimensions]
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
                    metrics,
                    labels[rows],
                    label_levels,
                    embedding[rows],
                    len(rows),
                    graph_size,
                    args.n_particles,
                    args.distance_pairs,
                    rng,
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
        "cluster_key": args.cluster_key,
        "cluster_levels": label_levels,
        "embedding_key": args.embedding_key,
        "pca_dimensions": args.pca_dimensions,
        "distance_pairs": args.distance_pairs,
        "minimum_cells": args.minimum_cells,
        "n_particles": args.n_particles,
        "random_state": args.random_state,
        "particle_method": "patient-level Bayesian bootstrap with shared Dirichlet(1) cell weights",
        "features": FEATURES,
        "graph_size_definition": "n_nodes_used in the full-reference O-lambda graph",
        "cluster_entropy_definition": "Shannon entropy divided by log(number of global label levels)",
        "pca_effective_rank_definition": "exponential spectral entropy of the weighted within-patient cell-state covariance",
        "mean_pairwise_distance_definition": "weighted mean Euclidean cell-state distance over fixed random cell pairs",
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
