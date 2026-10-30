# Cell QC and doublet removal

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from analysis_io import write_h5ad


def collapse_duplicate_genes(adata):
    gene_names = pd.Index(adata.var_names.astype(str))
    if not gene_names.has_duplicates:
        return adata

    # Sum duplicate columns, keeping the first gene annotation
    group_codes, unique_names = pd.factorize(gene_names, sort=False)
    unique_names = unique_names.rename(None)
    first_positions = np.flatnonzero(~gene_names.duplicated())

    if sp.issparse(adata.X):
        projection = sp.csr_matrix(
            (np.ones(adata.n_vars, dtype=adata.X.dtype), (np.arange(adata.n_vars), group_codes)),
            shape=(adata.n_vars, len(unique_names)),
        )
        collapsed_X = adata.X @ projection
    else:
        source_X = np.asarray(adata.X)
        collapsed_X = np.zeros((adata.n_obs, len(unique_names)), dtype=source_X.dtype)
        for i, j in enumerate(group_codes):
            collapsed_X[:, j] += source_X[:, i]

    collapsed_var = adata.var.iloc[first_positions].copy()
    collapsed_var.index = unique_names
    collapsed = ad.AnnData(X=collapsed_X, obs=adata.obs.copy(), var=collapsed_var)

    print(
        f"Collapsed {adata.n_vars - collapsed.n_vars:,} duplicate gene columns "
        f"({adata.n_vars:,} -> {collapsed.n_vars:,})."
    )
    return collapsed


def cell_qc_mask(adata, min_genes, max_genes, max_mito_pct, mito_prefix):
    n_genes = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    total_counts = np.asarray(adata.X.sum(axis=1)).ravel()

    # Mitochondrial counts
    gene_names = adata.var_names.str.upper()
    mt = gene_names.str.startswith(mito_prefix.upper())
    mt_counts = np.asarray(adata.X[:, mt].sum(axis=1)).ravel()
    if not mt.any():
        print(f"Warning: no genes begin with mitochondrial prefix {mito_prefix!r}.")

    pct_mt = np.divide(
        100.0 * mt_counts,
        total_counts,
        out=np.zeros(adata.n_obs, dtype=float),
        where=total_counts != 0,
    )

    keep = n_genes >= min_genes
    if max_genes is not None:
        keep &= n_genes <= max_genes
    keep &= pct_mt < max_mito_pct
    return keep


def run_scrublet_by_batch(adata, batch_key, min_batch_cells, random_state):
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
        batch_mask = (batch_values == batch_name).to_numpy()
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


def main():
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
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input_h5ad)
    print(f"Loaded {args.input_h5ad}: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()
        print("Made duplicated observation names unique.")

    # Read counts from X, raw, or a named layer
    if args.count_source == 'raw':
        counts = adata.raw.to_adata()
        counts.obs = adata.obs.copy()
        adata = counts
    elif args.count_source != 'X':
        adata.X = adata.layers[args.count_source].copy()

    adata.var_names = pd.Index(adata.var_names.astype(str).str.strip())
    if (adata.var_names == "").any():
        raise ValueError("One or more genes have empty names.")
    adata = collapse_duplicate_genes(adata)
    adata.obs["timepoint"] = "infusion"

    # Filter cells by detected genes and mitochondrial fraction
    n_before_qc = adata.n_obs
    keep = cell_qc_mask(
        adata, min_genes=args.min_genes, max_genes=args.max_genes,
        max_mito_pct=args.max_mito_pct, mito_prefix=args.mito_prefix,
    )
    adata = adata[keep].copy()
    print(f"Cell QC: {n_before_qc:,} -> {adata.n_obs:,} cells")
    if adata.n_obs == 0:
        raise ValueError("No cells remain after cell QC.")

    # Run Scrublet separately for each sample
    n_before_scrublet = adata.n_obs
    adata = run_scrublet_by_batch(
        adata, batch_key=args.batch_key,
        min_batch_cells=args.min_batch_cells, random_state=args.random_state,
    )
    print(f"Batch filtering and Scrublet: {n_before_scrublet:,} -> {adata.n_obs:,} cells")
    if adata.n_obs == 0:
        raise ValueError("No cells remain after Scrublet.")

    write_h5ad(adata, args.output_h5ad, args.overwrite)
    print(f"Saved {args.output_h5ad}: {adata.n_obs:,} cells x {adata.n_vars:,} genes")


if __name__ == "__main__":
    main()
