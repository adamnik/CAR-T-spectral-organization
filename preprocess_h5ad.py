"""Apply cell QC and batch-wise Scrublet filtering to one AnnData file."""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess one h5ad using cell QC followed by batch-wise Scrublet."
    )
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_h5ad", type=Path)
    parser.add_argument(
        "--count-source",
        default="X",
        help="Count matrix to use: X, raw, or the name of an AnnData layer (default: X).",
    )
    parser.add_argument("--batch-key", default="sample_id")
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--max-genes", type=int, default=None)
    parser.add_argument("--max-mito-pct", type=float, default=10.0)
    parser.add_argument("--mito-prefix", default="MT-")
    parser.add_argument("--min-batch-cells", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_count_source(adata, source):
    if source == "X":
        return adata

    if source == "raw":
        if adata.raw is None:
            raise ValueError("--count-source raw was requested, but adata.raw is empty.")
        selected = adata.raw.to_adata()
        selected.obs = adata.obs.copy()
        selected.obs_names = adata.obs_names.copy()
        return selected

    if source not in adata.layers:
        available = ", ".join(map(str, adata.layers.keys())) or "none"
        raise ValueError(
            f"Count layer {source!r} was not found. Available layers: {available}."
        )

    adata.X = adata.layers[source].copy()
    return adata


def collapse_duplicate_genes(adata):
    gene_names = pd.Index(adata.var_names.astype(str))
    if not gene_names.has_duplicates:
        return adata

    unique_names = pd.Index(pd.unique(gene_names))
    name_to_position = {name: i for i, name in enumerate(unique_names)}
    group_codes = np.asarray([name_to_position[name] for name in gene_names], dtype=np.int64)

    first_positions = np.full(len(unique_names), -1, dtype=np.int64)
    for source_position, target_position in enumerate(group_codes):
        if first_positions[target_position] == -1:
            first_positions[target_position] = source_position

    if sp.issparse(adata.X):
        projection = sp.csr_matrix(
            (
                np.ones(adata.n_vars, dtype=adata.X.dtype),
                (np.arange(adata.n_vars), group_codes),
            ),
            shape=(adata.n_vars, len(unique_names)),
        )
        collapsed_X = adata.X @ projection
    else:
        source_X = np.asarray(adata.X)
        collapsed_X = np.zeros((adata.n_obs, len(unique_names)), dtype=source_X.dtype)
        for source_position, target_position in enumerate(group_codes):
            collapsed_X[:, target_position] += source_X[:, source_position]

    collapsed_var = adata.var.iloc[first_positions].copy()
    collapsed_var.index = unique_names
    collapsed = ad.AnnData(X=collapsed_X, obs=adata.obs.copy(), var=collapsed_var)

    print(
        f"Collapsed {adata.n_vars - collapsed.n_vars:,} duplicate gene columns "
        f"({adata.n_vars:,} -> {collapsed.n_vars:,})."
    )
    return collapsed


def cell_qc_mask(adata, min_genes, max_genes, max_mito_pct, mito_prefix):
    if sp.issparse(adata.X):
        detected_genes = np.asarray((adata.X > 0).sum(axis=1)).ravel()
        total_counts = np.asarray(adata.X.sum(axis=1)).ravel()
    else:
        matrix = np.asarray(adata.X)
        detected_genes = np.sum(matrix > 0, axis=1)
        total_counts = np.sum(matrix, axis=1)

    gene_names = pd.Index(adata.var_names.astype(str)).str.upper()
    mitochondrial = np.asarray(gene_names.str.startswith(mito_prefix.upper()), dtype=bool)

    if mitochondrial.any():
        if sp.issparse(adata.X):
            mitochondrial_counts = np.asarray(adata.X[:, mitochondrial].sum(axis=1)).ravel()
        else:
            mitochondrial_counts = np.sum(np.asarray(adata.X)[:, mitochondrial], axis=1)
    else:
        mitochondrial_counts = np.zeros(adata.n_obs, dtype=float)
        print(f"Warning: no genes begin with mitochondrial prefix {mito_prefix!r}.")

    mitochondrial_pct = np.divide(
        100.0 * mitochondrial_counts,
        total_counts,
        out=np.zeros(adata.n_obs, dtype=float),
        where=total_counts != 0,
    )

    keep = detected_genes >= min_genes
    if max_genes is not None:
        keep &= detected_genes <= max_genes
    keep &= mitochondrial_pct < max_mito_pct
    return keep


def run_scrublet_by_batch(adata, batch_key, min_batch_cells, random_state):
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Batch column {batch_key!r} was not found in adata.obs.")
    if adata.obs[batch_key].isna().any():
        raise ValueError(f"Batch column {batch_key!r} contains missing values.")

    batch_values = adata.obs[batch_key].astype(str)
    batch_counts = batch_values.value_counts()
    retained_batches = batch_counts[batch_counts >= min_batch_cells].index
    removed_batches = batch_counts[batch_counts < min_batch_cells]

    if len(removed_batches):
        print(
            f"Removing {len(removed_batches):,} batches with fewer than "
            f"{min_batch_cells:,} cells ({int(removed_batches.sum()):,} cells)."
        )

    adata = adata[batch_values.isin(retained_batches)].copy()
    if adata.n_obs == 0:
        raise ValueError("No cells remain after the minimum batch-size requirement.")

    batch_values = adata.obs[batch_key].astype(str)
    doublet_scores = pd.Series(np.nan, index=adata.obs_names, dtype=float)
    predicted_doublets = pd.Series(False, index=adata.obs_names, dtype=bool)

    batches = pd.Index(pd.unique(batch_values))
    for batch_number, batch_name in enumerate(batches, start=1):
        batch_mask = batch_values.eq(batch_name).to_numpy()
        batch = adata[batch_mask].copy()
        print(
            f"Scrublet {batch_number}/{len(batches)} | {batch_name} | "
            f"{batch.n_obs:,} cells",
            flush=True,
        )

        sc.pp.scrublet(batch, random_state=random_state)
        scores = pd.to_numeric(batch.obs["doublet_score"], errors="raise")
        predictions = batch.obs["predicted_doublet"].fillna(False).astype(bool)
        doublet_scores.loc[batch.obs_names] = scores.to_numpy()
        predicted_doublets.loc[batch.obs_names] = predictions.to_numpy()
        print(f"  predicted doublets: {int(predictions.sum()):,}", flush=True)

    if doublet_scores.isna().any():
        raise RuntimeError("Scrublet did not score every retained cell.")

    adata.obs["doublet_score"] = doublet_scores.to_numpy()
    adata.obs["predicted_doublet"] = predicted_doublets.to_numpy()
    adata = adata[(~predicted_doublets).to_numpy()].copy()
    return adata


def write_h5ad(adata, output_path, overwrite):
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.stem + ".partial" + output_path.suffix)
    if temporary_path.exists():
        temporary_path.unlink()
    adata.write_h5ad(temporary_path)
    temporary_path.replace(output_path)


def main():
    args = parse_args()
    if not args.input_h5ad.exists():
        raise FileNotFoundError(args.input_h5ad)
    if args.min_genes < 0:
        raise ValueError("--min-genes must be nonnegative.")
    if args.max_genes is not None and args.max_genes < args.min_genes:
        raise ValueError("--max-genes must be greater than or equal to --min-genes.")
    if args.min_batch_cells < 1:
        raise ValueError("--min-batch-cells must be positive.")

    adata = ad.read_h5ad(args.input_h5ad)
    print(f"Loaded {args.input_h5ad}: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()
        print("Made duplicated observation names unique.")
    adata = select_count_source(adata, args.count_source)
    if not adata.obs_names.is_unique:
        raise RuntimeError("Observation names are not unique after count-source selection.")

    adata.var_names = pd.Index(adata.var_names.astype(str).str.strip())
    if (adata.var_names == "").any():
        raise ValueError("One or more genes have empty names.")
    adata = collapse_duplicate_genes(adata)
    if not adata.var_names.is_unique:
        raise RuntimeError("Gene names are not unique after duplicate-gene collapse.")
    adata.obs["timepoint"] = "infusion"

    n_before_qc = adata.n_obs
    keep = cell_qc_mask(
        adata,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
        max_mito_pct=args.max_mito_pct,
        mito_prefix=args.mito_prefix,
    )
    adata = adata[keep].copy()
    print(f"Cell QC: {n_before_qc:,} -> {adata.n_obs:,} cells")
    if adata.n_obs == 0:
        raise ValueError("No cells remain after cell QC.")

    n_before_scrublet = adata.n_obs
    adata = run_scrublet_by_batch(
        adata,
        batch_key=args.batch_key,
        min_batch_cells=args.min_batch_cells,
        random_state=args.random_state,
    )
    print(f"Batch filtering and Scrublet: {n_before_scrublet:,} -> {adata.n_obs:,} cells")
    if adata.n_obs == 0:
        raise ValueError("No cells remain after Scrublet.")

    write_h5ad(adata, args.output_h5ad, args.overwrite)
    print(f"Saved {args.output_h5ad}: {adata.n_obs:,} cells x {adata.n_vars:,} genes")


if __name__ == "__main__":
    main()
