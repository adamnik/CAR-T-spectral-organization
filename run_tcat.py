# TCAT gene programs for one cohort

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from starcat import starCAT

from analysis_io import safe_name, write_h5ad


def select_tcat_genes(adata, reference_genes, gene_list):
    available = set(adata.var_names.astype(str))

    if gene_list is None:
        requested = list(map(str, reference_genes))
    else:
        genes = pd.read_csv(gene_list)
        if "gene_symbol" in genes.columns:
            requested = genes["gene_symbol"].dropna().astype(str).tolist()
        elif genes.shape[1] == 1:
            requested = genes.iloc[:, 0].dropna().astype(str).tolist()
        else:
            raise ValueError('--gene-list must contain a gene_symbol column or exactly one column.')

        reference_set = set(map(str, reference_genes))
        outside_reference = sorted(set(requested).difference(reference_set))
        if outside_reference:
            raise ValueError(
                f"The gene list contains genes outside {len(reference_set):,} TCAT reference genes; "
                f"examples: {outside_reference[:10]}"
            )

    return [gene for gene in requested if gene in available]


def attach_tcat_results(adata, usage, scores, reference, genes, random_state):
    usage = usage.reindex(adata.obs_names)
    if usage.isna().any().any():
        raise RuntimeError("TCAT usage rows do not align with the input cells.")

    # Keep GEP order the same in obs and obsm
    original_usage_columns = list(usage.columns)
    usage_columns = list(map(str, original_usage_columns))
    safe_usage_columns = [safe_name(column) for column in usage_columns]
    if len(set(safe_usage_columns)) != len(safe_usage_columns):
        raise ValueError("TCAT GEP names are not unique after conversion to obs column names.")

    adata.obsm["TCAT_usage"] = usage.to_numpy(dtype=np.float32)
    adata.uns["TCAT_usage_columns"] = usage_columns
    for original, safe in zip(original_usage_columns, safe_usage_columns):
        adata.obs[f"TCAT_GEP_{safe}"] = usage[original].to_numpy(dtype=np.float32)

    score_columns = []
    if scores is not None:
        scores = scores.reindex(adata.obs_names)
        for score_name in scores.columns:
            output_column = f"TCAT_SCORE_{safe_name(score_name)}"
            series = scores[score_name]
            if pd.api.types.is_bool_dtype(series):
                adata.obs[output_column] = series.astype(bool).to_numpy()
            elif pd.api.types.is_numeric_dtype(series):
                adata.obs[output_column] = pd.to_numeric(series, errors="raise").to_numpy()
            else:
                adata.obs[output_column] = pd.Categorical(series.astype(str))
            score_columns.append(output_column)

    usage_sums = np.asarray(adata.obsm["TCAT_usage"].sum(axis=1)).ravel()
    maximum_error = float(np.max(np.abs(usage_sums - 1.0)))
    if maximum_error > 1e-4:
        raise ValueError(
            "TCAT usages do not sum to one for every cell; "
            f"maximum absolute error={maximum_error:.6g}."
        )

    adata.uns["TCAT"] = {
        "reference": reference,
        "n_input_genes": len(genes),
        "random_state": random_state,
        "score_columns": score_columns,
        "maximum_usage_sum_error": maximum_error,
    }


def main():
    parser = argparse.ArgumentParser(description='Run TCAT independently on one STRING-filtered h5ad.')
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_h5ad", type=Path)
    parser.add_argument("--reference", default="TCAT.V1")
    parser.add_argument("--cache-dir", type=Path, default=Path("starcat_cache"))
    parser.add_argument("--gene-list", type=Path, default=None)
    parser.add_argument("--min-genes", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input_h5ad)
    print(f"Loaded {args.input_h5ad}: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("The input h5ad is empty.")
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()
        print("Made duplicated observation names unique.")
    if not adata.var_names.is_unique:
        raise ValueError(
            "Input gene names must be unique before running TCAT. Do not suffix duplicated "
            "genes; collapse them during preprocessing and canonicalize them with STRING."
        )

    # Fit the reference programs to the available genes
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    tcat = starCAT(reference=args.reference, cachedir=str(args.cache_dir))
    tcat._nmf_kwargs["random_state"] = args.random_state
    reference_genes = pd.Index(tcat.ref.columns.astype(str))
    genes = select_tcat_genes(adata, reference_genes, args.gene_list)

    if len(genes) < args.min_genes:
        raise ValueError(
            f"Only {len(genes):,} TCAT reference genes are available; "
            f"--min-genes={args.min_genes:,}."
        )

    query = ad.AnnData(
        X=adata[:, genes].X.copy(),
        obs=pd.DataFrame(index=adata.obs_names.copy()),
        var=pd.DataFrame(index=pd.Index(genes, name="gene_symbol")),
    )
    print(f'Running {args.reference}: {query.n_obs:,} cells x {query.n_vars:,} reference genes')
    usage, scores = tcat.fit_transform(query)
    attach_tcat_results(
        adata, usage, scores,
        reference=args.reference, genes=genes, random_state=args.random_state,
    )

    write_h5ad(adata, args.output_h5ad, args.overwrite)
    print(f"Saved {args.output_h5ad}: {adata.n_obs:,} cells x {adata.n_vars:,} genes")


if __name__ == "__main__":
    main()
