# Primary Bayesian-bootstrap pseudobulk estimator over a shared gene universe

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from extract_string_pseudobulk import METADATA_COLUMNS, read_count_block, source_genes


def logcpm(counts):
    library_size = counts.sum(axis=-1)
    if np.any(library_size <= 0):
        raise ValueError("At least one shared-gene library has zero counts.")
    cpm = counts / library_size[..., None] * 1_000_000.0
    return np.log2(cpm + 1.0), library_size


def dataset_particles(data, dataset, genes, sample_key, count_source,
                      n_particles, block_size, random_state):
    source = source_genes(data, count_source)
    columns = source.get_indexer(genes)
    if np.any(columns < 0):
        raise ValueError(f"{dataset}: at least one reference gene is absent.")
    column_order = np.argsort(columns)
    sorted_columns = columns[column_order]
    restore_order = np.argsort(column_order)

    sample_values = data.obs[sample_key].astype(str).to_numpy()
    patients = sorted(pd.unique(sample_values))
    patient_index = {patient: index for index, patient in enumerate(patients)}
    codes = np.asarray([patient_index[patient] for patient in sample_values], dtype=int)
    rows_by_patient = [np.flatnonzero(codes == index) for index in range(len(patients))]
    n_cells = np.asarray([len(rows) for rows in rows_by_patient], dtype=int)

    rng = np.random.default_rng(random_state)
    cell_weights = np.zeros((n_particles, data.n_obs), dtype=np.float32)
    for rows in rows_by_patient:
        cell_weights[:, rows] = (
            rng.dirichlet(np.ones(len(rows)), size=n_particles) * len(rows)
        ).astype(np.float32)

    central_counts = np.zeros((len(patients), len(genes)), dtype=np.float64)
    particle_counts = np.zeros(
        (len(patients), n_particles, len(genes)), dtype=np.float64
    )

    starts = range(0, data.n_obs, block_size)
    for block_number, start in enumerate(starts, start=1):
        stop = min(start + block_size, data.n_obs)
        block = read_count_block(data, count_source, slice(start, stop), sorted_columns)
        block = block[:, restore_order]
        block_codes = codes[start:stop]
        local_cells = np.arange(stop - start)
        indicator = sp.csr_matrix(
            (np.ones(stop - start), (block_codes, local_cells)),
            shape=(len(patients), stop - start),
        )
        central = indicator @ block
        central_counts += central.toarray() if sp.issparse(central) else central
        for particle in range(n_particles):
            weighted_indicator = sp.csr_matrix(
                (cell_weights[particle, start:stop], (block_codes, local_cells)),
                shape=(len(patients), stop - start),
            )
            weighted = weighted_indicator @ block
            particle_counts[:, particle] += (
                weighted.toarray() if sp.issparse(weighted) else weighted
            )
        if block_number % 10 == 0 or stop == data.n_obs:
            print(f"  cells {stop:,}/{data.n_obs:,}", flush=True)

    central, central_library = logcpm(central_counts)
    particles, particle_library = logcpm(particle_counts)
    return {
        "datasets": np.repeat(str(dataset), len(patients)),
        "patient_ids": np.asarray(patients, dtype=str),
        "n_cells": n_cells,
        "central_values": central.astype(np.float32),
        "values": particles.astype(np.float32),
        "central_library_size": central_library,
        "particle_library_size": particle_library,
    }


def save_part(path, result, genes, metadata):
    np.savez_compressed(
        path,
        **result,
        feature_names=np.asarray(genes, dtype=str),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create outcome-blind Bayesian-bootstrap pseudobulk particles for each "
            "patient over the gene universe of a reference pseudobulk table."
        )
    )
    parser.add_argument("dataset_manifest_csv", type=Path)
    parser.add_argument("reference_pseudobulk_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-key", default="sample_id")
    parser.add_argument("--count-source", default="X")
    parser.add_argument("--n-particles", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=10000)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.dataset_manifest_csv)
    missing = {"Dataset", "h5ad_path"}.difference(manifest.columns)
    if missing:
        raise ValueError(f"Dataset manifest is missing columns: {sorted(missing)}")
    if manifest.Dataset.duplicated().any():
        raise ValueError("The dataset manifest must contain one h5ad per dataset.")

    reference = pd.read_csv(args.reference_pseudobulk_csv, nrows=1)
    genes = [column for column in reference if column not in METADATA_COLUMNS]
    if not genes:
        raise ValueError("No reference pseudobulk genes were found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    part_dir = args.output_dir / "dataset_parts"
    part_dir.mkdir(exist_ok=True)
    part_paths = []
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        dataset = str(row.Dataset)
        part_path = part_dir / f"{dataset}.npz"
        part_paths.append(part_path)
        print("\n" + "=" * 100, flush=True)
        print(
            f"Dataset {number}/{len(manifest)} | {dataset} | {Path(row.h5ad_path).name}",
            flush=True,
        )
        print("=" * 100, flush=True)
        if part_path.exists() and not args.overwrite:
            print("  RESUME:", part_path, flush=True)
            continue

        data = ad.read_h5ad(row.h5ad_path, backed="r")
        try:
            result = dataset_particles(
                data, dataset, genes, args.sample_key, args.count_source,
                args.n_particles, args.block_size,
                args.random_state + number * 1000,
            )
        finally:
            if data.file is not None:
                data.file.close()
        save_part(part_path, result, genes, {
            "Dataset": dataset,
            "h5ad_path": str(row.h5ad_path),
            "n_particles": args.n_particles,
            "random_state": args.random_state + number * 1000,
        })
        print(
            f"  saved {len(result['patient_ids']):,} patients: {part_path}",
            flush=True,
        )

    parts = [np.load(path, allow_pickle=False) for path in part_paths]
    output = args.output_dir / "patient_pseudobulk_particles.npz"
    np.savez(
        output,
        datasets=np.concatenate([part["datasets"] for part in parts]),
        patient_ids=np.concatenate([part["patient_ids"] for part in parts]),
        n_cells=np.concatenate([part["n_cells"] for part in parts]),
        central_values=np.concatenate([part["central_values"] for part in parts]),
        values=np.concatenate([part["values"] for part in parts]),
        central_library_size=np.concatenate([
            part["central_library_size"] for part in parts
        ]),
        particle_library_size=np.concatenate([
            part["particle_library_size"] for part in parts
        ]),
        feature_names=np.asarray(genes, dtype=str),
        particle_weights=np.full(
            (sum(len(part["patient_ids"]) for part in parts), args.n_particles),
            1 / args.n_particles,
            dtype=np.float32,
        ),
    )
    (args.output_dir / "pseudobulk_particle_manifest.json").write_text(json.dumps({
        "dataset_manifest_csv": str(args.dataset_manifest_csv),
        "reference_pseudobulk_csv": str(args.reference_pseudobulk_csv),
        "patients": int(sum(len(part["patient_ids"]) for part in parts)),
        "datasets": manifest.Dataset.astype(str).tolist(),
        "genes": len(genes),
        "particles": args.n_particles,
        "sample_key": args.sample_key,
        "count_source": args.count_source,
        "random_state": args.random_state,
        "particle_definition": (
            "Dirichlet(1,...,1) cell weights multiplied by patient cell count; "
            "weighted shared-gene sums followed by log2(CPM+1)"
        ),
    }, indent=2))
    print("\nCombined particles:", output, flush=True)


if __name__ == "__main__":
    main()
