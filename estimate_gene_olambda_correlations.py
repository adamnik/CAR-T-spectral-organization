# Bayesian partial rank correlations between pseudobulk genes and O-lambda

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PSEUDOBULK_METADATA_COLUMNS = [
    "Dataset", "sample_id", "phase", "n_cells",
    "library_size_common_genes",
]

def rank_columns(values):
    return pd.DataFrame(values).rank(axis=0, method="average").to_numpy(float)


def nuisance_design(metadata, treatment_column):
    dataset = metadata.Dataset.astype(str)
    dataset_columns = pd.get_dummies(dataset, prefix="dataset", drop_first=True, dtype=float)
    arrays = [np.ones(len(metadata), dtype=float)]
    names = ["Intercept"]
    for column in dataset_columns:
        arrays.append(dataset_columns[column].to_numpy(float))
        names.append(column)

    audit = []
    if treatment_column is not None:
        treatment = metadata[treatment_column].astype(str)
        for dataset_name, group in metadata.groupby("Dataset", observed=True, sort=True):
            levels = sorted(group[treatment_column].astype(str).unique())
            if len(levels) < 2:
                continue
            reference = levels[0]
            for level in levels[1:]:
                label = f"treatment__{dataset_name}__{level}_vs_{reference}"
                arrays.append(
                    (dataset.eq(str(dataset_name)) & treatment.eq(level)).to_numpy(float)
                )
                names.append(label)
                audit.append({
                    "contrast": label,
                    "level": level,
                    "reference": reference,
                    "datasets_with_contrast": str(dataset_name),
                    "included": True,
                })

    design = np.column_stack(arrays)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError(f"The nuisance design is rank deficient: {names}")
    return design, names, pd.DataFrame(audit)


def weighted_partial_correlations(expression, predictor, design, weights):
    # Weighted residual correlations, with dataset and treatment removed
    weighted_X = design * weights[:, None]
    XtWX = design.T @ weighted_X
    inv = np.linalg.inv(XtWX)

    XtWY = design.T @ (expression * weights[:, None])
    ss_genes = np.sum(expression * expression * weights[:, None], axis=0)
    ss_genes -= np.sum(XtWY * (inv @ XtWY), axis=0)

    XtWy = design.T @ (weights * predictor)
    residual_y = predictor - design @ (inv @ XtWy)
    ss_y = np.sum(weights * residual_y**2)
    cov = residual_y @ (expression * weights[:, None])

    denom = np.sqrt(np.maximum(ss_genes * ss_y, 1e-20))
    return cov / denom


def association_draws(expression, olambda_particles, design, n_draws, random_state):
    rng = np.random.default_rng(random_state)
    n_patients, n_genes = expression.shape
    draws = np.empty((n_draws, n_genes), dtype=np.float32)
    progress_step = max(1, n_draws // 20)

    for draw in range(n_draws):
        particle_index = rng.integers(0, olambda_particles.shape[1], size=n_patients)
        olambda = olambda_particles[np.arange(n_patients), particle_index]
        # Rank again after selecting this draw's O-lambda particles
        olambda = pd.Series(olambda).rank(method="average").to_numpy(float)
        weights = rng.dirichlet(np.ones(n_patients))
        draws[draw] = weighted_partial_correlations(expression, olambda, design, weights)

        if (draw + 1) % progress_step == 0 or draw + 1 == n_draws:
            print(f"Association draw {draw + 1:,}/{n_draws:,}", flush=True)
    return draws


def summarize_associations(genes, draws):
    mean = draws.mean(axis=0)
    median = np.median(draws, axis=0)
    sd = draws.std(axis=0, ddof=1)
    probability_positive = np.mean(draws > 0, axis=0)
    probability_negative = np.mean(draws < 0, axis=0)
    result = pd.DataFrame({
        "gene": genes,
        "rho_mean": mean,
        "rho_median": median,
        "rho_sd": sd,
        "rho_q025": np.quantile(draws, 0.025, axis=0),
        "rho_q975": np.quantile(draws, 0.975, axis=0),
        "P_rho_gt_0": probability_positive,
        "P_rho_lt_0": probability_negative,
        "P_same_direction": np.maximum(probability_positive, probability_negative),
        "signed_direction_probability": 2.0 * probability_positive - 1.0,
        "posterior_standardized_effect": mean / np.maximum(sd, 1e-6),
    })
    result = result.sort_values(
        ['posterior_standardized_effect', 'gene'],
        ascending=[False, True], kind='mergesort',
    )
    return result.reset_index(drop=True)


def load_analysis(pseudobulk_path, olambda_path, patient_metadata_path, minimum_cells,
                  treatment_column, drop_missing_treatment=False):
    pseudobulk = pd.read_csv(pseudobulk_path)
    olambda = pd.read_csv(olambda_path)
    patient_metadata = pd.read_csv(patient_metadata_path)
    particle_columns = [column for column in olambda if column.startswith("particle_")]
    if not particle_columns:
        raise ValueError("No particle_* columns were found in the O-lambda file.")

    gene_columns = [
        column for column in pseudobulk
        if column not in PSEUDOBULK_METADATA_COLUMNS
    ]
    metadata_columns = ["Dataset", "sample_id", "n_cells"]
    analysis = pseudobulk[metadata_columns + gene_columns].merge(
        olambda[["Dataset", "sample_id", *particle_columns]],
        on=["Dataset", "sample_id"],
        how="inner",
        validate="one_to_one",
    )
    outcome_columns = ["Dataset", "sample_id"]
    if treatment_column is not None:
        if treatment_column not in patient_metadata:
            raise KeyError(f"Patient metadata lack treatment column {treatment_column!r}.")
        outcome_columns.append(treatment_column)
    analysis = analysis.merge(
        patient_metadata[outcome_columns],
        on=["Dataset", "sample_id"],
        how="left",
        validate="one_to_one",
    )
    analysis = analysis.loc[analysis.n_cells >= minimum_cells].copy()
    analysis = analysis.sort_values(["Dataset", "sample_id"]).reset_index(drop=True)
    excluded_patients = pd.DataFrame(columns=['Dataset', 'sample_id', 'n_cells', 'reason'])
    if treatment_column is not None and analysis[treatment_column].isna().any():
        missing_treatment = analysis[treatment_column].isna()
        excluded_patients = analysis.loc[
            missing_treatment, ["Dataset", "sample_id", "n_cells"]
        ].copy()
        excluded_patients["reason"] = "missing treatment annotation"
        if not drop_missing_treatment:
            raise ValueError(
                "Treatment metadata are missing for one or more retained patients. "
                "Use --drop-missing-treatment for a complete-case adjusted analysis."
            )
        analysis = analysis.loc[~missing_treatment].reset_index(drop=True)

    raw_expression = analysis[gene_columns].apply(pd.to_numeric, errors='raise')
    raw_expression = raw_expression.to_numpy(float)
    valid = np.isfinite(raw_expression).all(axis=0) & (raw_expression.var(axis=0) > 0)
    excluded_genes = np.asarray(gene_columns, dtype=object)[~valid]
    genes = np.asarray(gene_columns, dtype=object)[valid]
    expression = rank_columns(raw_expression[:, valid])
    particles = analysis[particle_columns].to_numpy(float)
    return analysis, genes, expression, particles, excluded_genes, excluded_patients


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Bayesian-bootstrap, dataset- and treatment-adjusted partial "
            "Spearman correlations between shared STRING genes and latent O-lambda."
        )
    )
    parser.add_argument("pseudobulk_csv", type=Path)
    parser.add_argument("olambda_particles_csv", type=Path)
    parser.add_argument("patient_metadata_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-cells", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--treatment-column", default="Treatment")
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--drop-missing-treatment", action="store_true")
    parser.add_argument("--save-draws", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_paths = {
        "associations": args.output_dir / "olambda_gene_associations.csv",
        "ranking_csv": args.output_dir / "olambda_gene_GSEA_rank.csv",
        "ranking_rnk": args.output_dir / "olambda_gene_GSEA_rank.rnk",
        "cohort": args.output_dir / "analysis_cohort.csv",
        "treatment": args.output_dir / "treatment_contrast_audit.csv",
        "excluded": args.output_dir / "excluded_patients.csv",
        "manifest": args.output_dir / "gene_association_manifest.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output already exists: {existing[0]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    treatment_column = None if args.dataset_only else args.treatment_column
    analysis, genes, expression, particles, excluded_genes, excluded_patients = load_analysis(
        args.pseudobulk_csv, args.olambda_particles_csv, args.patient_metadata_csv,
        args.minimum_cells, treatment_column, args.drop_missing_treatment,
    )
    design, design_names, treatment_audit = nuisance_design(analysis, treatment_column)
    print(
        f"Analysis cohort: {len(analysis):,} patients | "
        f"{analysis.Dataset.nunique():,} datasets | {len(genes):,} variable STRING genes | "
        f"{particles.shape[1]:,} O-lambda particles",
        flush=True,
    )
    print("Nuisance design:", " | ".join(design_names), flush=True)
    draws = association_draws(expression, particles, design, args.draws, args.random_state)
    associations = summarize_associations(genes, draws)
    associations["n_patients"] = len(analysis)
    associations["n_datasets"] = analysis.Dataset.nunique()
    associations["adjustment"] = (
        "dataset_fixed_effects+corrected_treatment"
        if treatment_column is not None else "dataset_fixed_effects"
    )
    associations.to_csv(output_paths["associations"], index=False)

    ranking = associations[["gene", "posterior_standardized_effect"]].rename(
        columns={"posterior_standardized_effect": "score"}
    )
    ranking.to_csv(output_paths["ranking_csv"], index=False)
    ranking.to_csv(output_paths["ranking_rnk"], sep="\t", header=False, index=False)
    cohort_columns = ['Dataset', 'sample_id', 'n_cells']
    if treatment_column:
        cohort_columns.append(treatment_column)
    analysis[cohort_columns].to_csv(output_paths['cohort'], index=False)
    treatment_audit.to_csv(output_paths["treatment"], index=False)
    excluded_patients.to_csv(output_paths["excluded"], index=False)
    if args.save_draws:
        np.savez_compressed(
            args.output_dir / "olambda_gene_correlation_draws.npz",
            correlations=draws,
            genes=genes,
        )

    output_paths["manifest"].write_text(json.dumps({
        "pseudobulk_csv": str(args.pseudobulk_csv),
        "olambda_particles_csv": str(args.olambda_particles_csv),
        "patient_metadata_csv": str(args.patient_metadata_csv),
        "patients": len(analysis),
        "datasets": sorted(analysis.Dataset.astype(str).unique()),
        "common_string_genes_ranked": len(genes),
        "constant_or_nonfinite_genes_excluded": excluded_genes.astype(str).tolist(),
        "patients_excluded_for_missing_treatment": excluded_patients[
            ["Dataset", "sample_id"]
        ].to_dict(orient="records"),
        "draws": args.draws,
        "random_state": args.random_state,
        "olambda_uncertainty": "one equal-probability patient particle sampled per draw",
        "cohort_uncertainty": "Bayesian bootstrap Dirichlet(1,...,1) patient weights",
        "association": "weighted partial Spearman correlation",
        "nuisance_design": design_names,
        "gsea_ranking": "posterior mean partial rho divided by posterior SD",
    }, indent=2))
    print("Associations:", output_paths["associations"], flush=True)
    print("GSEA ranking:", output_paths["ranking_rnk"], flush=True)


if __name__ == "__main__":
    main()
