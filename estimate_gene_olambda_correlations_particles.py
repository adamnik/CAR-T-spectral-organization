# Primary gene associations with pseudobulk and O-lambda particles

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

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


def rank_columns(values):
    return rankdata(values, axis=0, method="average").astype(np.float32)


def load_analysis(pseudobulk_path, olambda_path, patient_metadata_path,
                  minimum_cells, treatment_column):
    archive = np.load(pseudobulk_path, allow_pickle=False)
    patients = pd.DataFrame({
        "Dataset": archive["datasets"].astype(str),
        "sample_id": archive["patient_ids"].astype(str),
        "n_cells": archive["n_cells"].astype(int),
        "expression_row": np.arange(len(archive["patient_ids"])),
    })
    if patients.duplicated(["Dataset", "sample_id"]).any():
        raise ValueError("Pseudobulk particle identities are not unique.")

    olambda = pd.read_csv(olambda_path)
    particle_columns = [column for column in olambda if column.startswith("particle_")]
    if not particle_columns:
        raise ValueError("No O-lambda particle columns were found.")
    olambda["olambda_row"] = np.arange(len(olambda))
    metadata = pd.read_csv(patient_metadata_path)
    metadata_columns = ["Dataset", "sample_id"]
    if treatment_column is not None:
        if treatment_column not in metadata:
            raise KeyError(f"Patient metadata lack {treatment_column!r}.")
        metadata_columns.append(treatment_column)

    analysis = patients.merge(
        olambda[["Dataset", "sample_id", "olambda_row"]],
        on=["Dataset", "sample_id"], how="inner", validate="one_to_one",
    ).merge(
        metadata[metadata_columns],
        on=["Dataset", "sample_id"], how="left", validate="one_to_one",
    )
    analysis = analysis.loc[analysis.n_cells >= minimum_cells].copy()
    analysis = analysis.sort_values(["Dataset", "sample_id"]).reset_index(drop=True)
    if analysis.empty:
        raise ValueError("No patients satisfy the analysis requirements.")
    if treatment_column is not None and analysis[treatment_column].isna().any():
        raise ValueError("Treatment metadata are missing for one or more patients.")

    expression = np.asarray(
        archive["values"][analysis.expression_row.to_numpy(int)], dtype=np.float32
    )
    central = np.asarray(
        archive["central_values"][analysis.expression_row.to_numpy(int)], dtype=float
    )
    genes = archive["feature_names"].astype(str)
    valid = (
        np.isfinite(expression).all(axis=(0, 1))
        & np.isfinite(central).all(axis=0)
        & (central.var(axis=0) > 0)
    )
    excluded_genes = genes[~valid]
    genes = genes[valid]
    expression = expression[:, :, valid]
    for particle in range(expression.shape[1]):
        print(
            f"Ranking pseudobulk particle {particle + 1}/{expression.shape[1]}",
            flush=True,
        )
        expression[:, particle, :] = rank_columns(expression[:, particle, :])

    olambda_particles = olambda.loc[
        analysis.olambda_row.to_numpy(int), particle_columns
    ].to_numpy(float)
    return analysis, genes, expression, olambda_particles, excluded_genes


def association_draws(expression_particles, olambda_particles, design,
                      n_draws, random_state):
    rng = np.random.default_rng(random_state)
    patients, expression_particle_count, genes = expression_particles.shape
    draws = np.empty((n_draws, genes), dtype=np.float32)
    expression_particle_indices = rng.integers(
        0, expression_particle_count, size=n_draws
    )
    progress_step = max(1, n_draws // 20)
    for draw in range(n_draws):
        expression = expression_particles[:, expression_particle_indices[draw], :]
        olambda_index = rng.integers(0, olambda_particles.shape[1], size=patients)
        olambda = olambda_particles[np.arange(patients), olambda_index]
        olambda = pd.Series(olambda).rank(method="average").to_numpy(float)
        weights = rng.dirichlet(np.ones(patients))
        draws[draw] = weighted_partial_correlations(
            expression, olambda, design, weights
        )
        if (draw + 1) % progress_step == 0 or draw + 1 == n_draws:
            print(f"Association draw {draw + 1:,}/{n_draws:,}", flush=True)
    return draws, expression_particle_indices


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate dataset-adjusted partial Spearman correlations while "
            "propagating pseudobulk and latent O-lambda particles."
        )
    )
    parser.add_argument("pseudobulk_particles_npz", type=Path)
    parser.add_argument("olambda_particles_csv", type=Path)
    parser.add_argument("patient_metadata_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-cells", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--treatment-column")
    parser.add_argument("--save-draws", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    association_path = args.output_dir / "olambda_gene_associations.csv"
    if association_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {association_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    analysis, genes, expression, olambda, excluded = load_analysis(
        args.pseudobulk_particles_npz, args.olambda_particles_csv,
        args.patient_metadata_csv, args.minimum_cells, args.treatment_column,
    )
    design, design_names, treatment_audit = nuisance_design(
        analysis, args.treatment_column
    )
    print(
        f"Analysis cohort: {len(analysis):,} patients | "
        f"{analysis.Dataset.nunique():,} datasets | {len(genes):,} genes | "
        f"{expression.shape[1]:,} pseudobulk particles | "
        f"{olambda.shape[1]:,} O-lambda particles",
        flush=True,
    )
    draws, expression_indices = association_draws(
        expression, olambda, design, args.draws, args.random_state
    )
    associations = summarize_associations(genes, draws)
    associations["n_patients"] = len(analysis)
    associations["n_datasets"] = analysis.Dataset.nunique()
    associations["adjustment"] = (
        "dataset_fixed_effects+corrected_treatment"
        if args.treatment_column else "dataset_fixed_effects"
    )
    associations.to_csv(association_path, index=False)
    ranking = associations[["gene", "posterior_standardized_effect"]].rename(
        columns={"posterior_standardized_effect": "score"}
    )
    ranking.to_csv(args.output_dir / "olambda_gene_GSEA_rank.csv", index=False)
    ranking.to_csv(
        args.output_dir / "olambda_gene_GSEA_rank.rnk",
        sep="\t", header=False, index=False,
    )
    cohort_columns = ["Dataset", "sample_id", "n_cells"]
    if args.treatment_column:
        cohort_columns.append(args.treatment_column)
    analysis[cohort_columns].to_csv(
        args.output_dir / "analysis_cohort.csv", index=False
    )
    pd.DataFrame(columns=["Dataset", "sample_id", "n_cells", "reason"]).to_csv(
        args.output_dir / "excluded_patients.csv", index=False
    )
    treatment_audit.to_csv(
        args.output_dir / "treatment_contrast_audit.csv", index=False
    )
    if args.save_draws:
        np.savez_compressed(
            args.output_dir / "olambda_gene_correlation_draws.npz",
            correlations=draws, genes=genes,
            expression_particle_indices=expression_indices,
        )
    (args.output_dir / "gene_association_manifest.json").write_text(json.dumps({
        "pseudobulk_particles_npz": str(args.pseudobulk_particles_npz),
        "olambda_particles_csv": str(args.olambda_particles_csv),
        "patient_metadata_csv": str(args.patient_metadata_csv),
        "patients": len(analysis),
        "datasets": sorted(analysis.Dataset.astype(str).unique()),
        "common_string_genes_ranked": len(genes),
        "constant_or_nonfinite_genes_excluded": excluded.astype(str).tolist(),
        "draws": args.draws,
        "random_state": args.random_state,
        "pseudobulk_particles": int(expression.shape[1]),
        "pseudobulk_uncertainty": (
            "one coherent whole-cohort cell-Bayesian-bootstrap pseudobulk "
            "particle sampled per association draw"
        ),
        "olambda_uncertainty": (
            "one equal-probability patient O-lambda particle sampled per draw"
        ),
        "cohort_uncertainty": "Dirichlet(1,...,1) patient weights",
        "association": "weighted partial Spearman correlation",
        "nuisance_design": design_names,
        "gsea_ranking": "posterior mean partial rho divided by posterior SD",
    }, indent=2))
    print("Associations:", association_path, flush=True)


if __name__ == "__main__":
    main()
