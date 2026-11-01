# Patient pseudobulk expression over the shared STRING gene universe

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


METADATA_COLUMNS = [
    "Dataset", "sample_id", "phase", "n_cells",
    "library_size_common_genes",
]


def source_genes(adata, count_source):
    if count_source == "X" or count_source in adata.layers:
        return pd.Index(adata.var_names.astype(str))
    if count_source == "raw":
        if adata.raw is None:
            raise ValueError("The requested raw count source is absent.")
        return pd.Index(adata.raw.var_names.astype(str))
    raise KeyError(f"Count source {count_source!r} is not X, raw, or an AnnData layer.")


def read_count_block(adata, count_source, row_slice, columns):
    if count_source == "X":
        values = adata[row_slice, columns].X
    elif count_source == "raw":
        values = adata.raw[row_slice, columns].X
    else:
        values = adata[row_slice, columns].layers[count_source]
    return values.tocsr() if sp.issparse(values) else np.asarray(values)


def common_string_genes(manifest, count_source):
    gene_lists = []
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        path = Path(row.h5ad_path)
        print(f"Gene scan {number}/{len(manifest)} | {row.Dataset} | {path.name}", flush=True)
        data = ad.read_h5ad(path, backed="r")
        try:
            genes = source_genes(data, count_source)
            if genes.has_duplicates:
                raise ValueError(f"{path.name}: STRING gene names are not unique.")
            gene_lists.append(genes)
        finally:
            if data.file is not None:
                data.file.close()

    shared = set(gene_lists[0])
    for genes in gene_lists[1:]:
        shared.intersection_update(genes)
    common = [gene for gene in gene_lists[0] if gene in shared]
    if not common:
        raise ValueError("The STRING-gene intersection is empty.")
    return common


def accumulate_patients(data, dataset, common_genes, sample_key, count_source, block_size):
    genes = source_genes(data, count_source)
    columns = genes.get_indexer(common_genes)
    if np.any(columns < 0):
        raise RuntimeError("The shared-gene lookup failed.")
    # Backed h5ad reads need sorted columns; restore the common gene order below
    column_order = np.argsort(columns)
    sorted_columns = columns[column_order]
    restore_order = np.argsort(column_order)

    sample_values = data.obs[sample_key].astype(str).to_numpy()
    patients = sorted(pd.unique(sample_values))
    patient_index = {patient: index for index, patient in enumerate(patients)}
    codes = np.asarray([patient_index[patient] for patient in sample_values], dtype=int)
    counts = np.zeros((len(patients), len(common_genes)), dtype=np.float64)
    n_cells = np.zeros(len(patients), dtype=np.int64)

    starts = range(0, data.n_obs, block_size)
    for block_number, start in enumerate(starts, start=1):
        stop = min(start + block_size, data.n_obs)
        block = read_count_block(data, count_source, slice(start, stop), sorted_columns)
        block = block[:, restore_order]
        block_codes = codes[start:stop]
        n_cells += np.bincount(block_codes, minlength=len(patients))

        if sp.issparse(block):
            # Map cells to patient rows before summing their counts
            cell_index = np.arange(len(block_codes))
            indicator = sp.csr_matrix(
                (np.ones(len(block_codes), dtype=float), (block_codes, cell_index)),
                shape=(len(patients), len(block_codes)),
            )
            counts += (indicator @ block).toarray()
        else:
            np.add.at(counts, block_codes, block)

        if block_number % 10 == 0 or stop == data.n_obs:
            print(f"  cells {stop:,}/{data.n_obs:,}", flush=True)

    # CPM denominator uses only the shared STRING genes
    library_size = counts.sum(axis=1)
    if np.any(library_size <= 0):
        failed = np.asarray(patients)[library_size <= 0].tolist()
        raise ValueError(f"Zero shared-STRING-gene library size: {failed[:10]}")
    cpm = counts / library_size[:, None] * 1_000_000.0
    logcpm = np.log2(cpm + 1.0)

    metadata = pd.DataFrame({
        "Dataset": dataset,
        "sample_id": patients,
        "phase": "infusion",
        "n_cells": n_cells,
        "library_size_common_genes": library_size,
    })
    return pd.concat([metadata, pd.DataFrame(logcpm, columns=common_genes)], axis=1)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sum infusion-product counts by patient and calculate log2(CPM + 1) "
            "over the exact STRING-gene intersection across datasets."
        )
    )
    parser.add_argument("dataset_manifest_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--sample-key", default="sample_id")
    parser.add_argument("--count-source", default="X")
    parser.add_argument("--block-size", type=int, default=20000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output_csv}")
    manifest = pd.read_csv(args.dataset_manifest_csv)
    missing = {"Dataset", "h5ad_path"}.difference(manifest.columns)
    if missing:
        raise ValueError(f"Dataset manifest is missing columns: {sorted(missing)}")
    if manifest.Dataset.duplicated().any():
        raise ValueError("The dataset manifest must contain one h5ad per dataset.")

    common_genes = common_string_genes(manifest, args.count_source)
    print(
        f"Shared STRING genes: {len(common_genes):,} across {len(manifest):,} datasets",
        flush=True,
    )

    frames = []
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        path = Path(row.h5ad_path)
        print("\n" + "=" * 100, flush=True)
        print(f'Dataset {number}/{len(manifest)} | {row.Dataset} | {path.name}', flush=True)
        print("=" * 100, flush=True)
        data = ad.read_h5ad(path, backed="r")
        try:
            frame = accumulate_patients(
                data, str(row.Dataset), common_genes,
                args.sample_key, args.count_source, args.block_size,
            )
            frames.append(frame)
            print(
                f"  completed: {len(frame):,} patients | "
                f"{int(frame.n_cells.sum()):,} cells",
                flush=True,
            )
        finally:
            if data.file is not None:
                data.file.close()

    pseudobulk = pd.concat(frames, ignore_index=True)
    pseudobulk = pseudobulk.sort_values(["Dataset", "sample_id"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pseudobulk.to_csv(args.output_csv, index=False)

    manifest_path = args.output_csv.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "dataset_manifest_csv": str(args.dataset_manifest_csv),
        "datasets": manifest.Dataset.astype(str).tolist(),
        "patients": len(pseudobulk),
        "common_string_genes": len(common_genes),
        "count_source": args.count_source,
        "sample_key": args.sample_key,
        "definition": "patient sum of counts over shared genes, followed by log2(CPM + 1)",
    }, indent=2))
    print("\nSaved pseudobulk:", args.output_csv, flush=True)
    print("Saved manifest:", manifest_path, flush=True)


if __name__ == "__main__":
    main()
