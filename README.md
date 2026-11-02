# CAR T-cell spectral organization

Code for preprocessing CAR T-cell expression data, extracting patient-level TCAT features, and relating spectral organization to survival.

## Setup

Use Python 3.11 and install the dependencies in a dedicated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The asymptote model requires R and `nlme` (`install.packages("nlme")`). The environment used for the synthetic checks is recorded in `requirements-checked.txt`.

## Patient-level TCAT extraction

`extract_tcat_patient_particles.py` reads one processed TCAT AnnData dataset and creates outcome-blind patient estimates and Bayesian-bootstrap particles. GEP usages and multinomial cell-label proportions are CLR-transformed. ASA and Proliferation are reconstructed from their predefined component CLR GEPs rather than estimated independently.

```bash
python extract_tcat_patient_particles.py \
  processed_dataset_TCAT.h5ad \
  results/dataset_tcat_particles \
  --dataset-name DatasetName \
  --sample-key sample_id
```

The script produces `tcat_patient_estimates.csv`, `tcat_patient_uncertainty.csv`, `tcat_patient_particles.npz`, and a complete extraction manifest. Run it independently for each dataset, using a common categorical-level schema when combining cohorts.

## TCAT survival models

`fit_tcat_survival.py` uses the combined patient estimates and coherent Bayesian-bootstrap particles in the same interval-censored survival framework used for spectral organization. Three modes are available:

```bash
python fit_tcat_survival.py \
  all_patient_TCAT_estimates.csv \
  all_patient_TCAT_particles.npz \
  clinical_outcomes.csv \
  results/tcat_individual \
  --mode individual \
  --adjustment Treatment=axicel \
  --adjustment Disease=lymphoma

python fit_tcat_survival.py \
  all_patient_TCAT_estimates.csv \
  all_patient_TCAT_particles.npz \
  clinical_outcomes.csv \
  results/tcat_joint_gep \
  --mode joint-gep

python fit_tcat_survival.py \
  all_patient_TCAT_estimates.csv \
  all_patient_TCAT_particles.npz \
  clinical_outcomes.csv \
  results/tcat_joint_cell_type \
  --mode joint-cell-type
```

The individual models estimate a global effect and hierarchically pooled dataset effects for every GEP, derived score, and cell proportion. Their outputs include the posterior probabilities that all dataset effects are adverse, all are protective, or all share either direction. Individual fits are checkpointed by feature and can be resumed.

The joint models use learned Gaussian shrinkage across all coefficients. GEPs are fit as one CLR block; cell-label proportions are constrained separately within each label composition. A common scale is used within each CLR block so that the compositional zero-sum geometry is retained. Each patient particle represents the complete TCAT landscape, so the joint likelihood propagates correlated finite-cell uncertainty across features.

## TCAT correlation geometry

`estimate_tcat_correlations.py` estimates dataset-adjusted correlations among TCAT features and between each TCAT feature and asymptotic O-lambda. Every draw independently samples one TCAT particle and one O-lambda particle per patient, draws Bayesian-bootstrap patient weights, removes weighted dataset means, and calculates the resulting correlations.

```bash
python estimate_tcat_correlations.py \
  all_patient_TCAT_particles.npz \
  patient_olambda_particles.csv \
  results/tcat_correlations \
  --draws 2000
```

The primary outputs are posterior summaries for all pairwise TCAT correlations and all O-lambda–TCAT correlations. Complete posterior draws can optionally be retained with `--save-draws`.

`plot_tcat_correlation_matrices.py` creates separate GEP and cell-type figures. Each Bayesian correlation matrix is followed by perfectly aligned rows showing correlation with asymptotic O-lambda and the continuous survival effect on the symmetric log2 hazard-ratio scale.

```bash
python plot_tcat_correlation_matrices.py \
  results/tcat_correlations/tcat_pairwise_correlations.csv \
  results/tcat_correlations/olambda_tcat_correlations.csv \
  results/tcat_individual/individual_results.csv \
  results/tcat_correlation_figures
```

Black markers denote posterior directional probability of at least 0.95 for O-lambda correlations and a 95% credible interval excluding one for survival effects. The matrix order is determined by average-linkage clustering of the posterior-median correlations.

## Regime-conditioned TCAT predictive fields

`analyze_tcat_predictive_field.py` asks whether Bayesian survival regimes organize the TCAT GEP landscape around its clinical-risk field. The graph contains only TCAT GEPs. Its fixed clinical field is the posterior-median vector of continuous GEP log hazard ratios. Three complementary quantities are evaluated:

- prognostic channeling, the covariance projected along the clinical-risk field;
- signed graph variation, the disagreement between that field and the signed GEP graph;
- signed-neighbor cosine, the alignment of each GEP with its clinical neighborhood.

The primary geometry removes weighted dataset means within every Bayesian-correlation draw; the global geometry is reported as a sensitivity analysis. Each draw samples one coherent TCAT particle per patient and Bayesian-bootstrap patient weights. High- and low-regime matrices use the posterior regime probabilities saved by the corresponding full Bayesian change-point model. Comparator contexts include every GEP with a 95% interval excluding one in either survival model, together with ASA and Proliferation. An individual contextual GEP is removed from its evaluated graph.

```bash
python analyze_tcat_predictive_field.py \
  all_patient_TCAT_particles.npz \
  results/tcat_individual/individual_results.csv \
  results/tcat_smooth_regimes/regime_results.csv \
  results/tcat_smooth_regimes \
  results/olambda_survival_regimes/patient_regime_probabilities.csv \
  results/olambda_survival_regimes/model_result.csv \
  results/tcat_predictive_field
```

Regime probabilities are permuted within dataset. The script reports each metric, a descriptive joint standardized contrast, observed regime-specific matrices, comparison-wide false-discovery rates, global-geometry sensitivity, and leave-one-dataset-out stability for O-lambda.

## Dataset-cross-fitted clinical fields

`fit_tcat_survival_crossfit.py` refits every independent continuous GEP survival model while holding out one complete dataset. It uses the same priors, interval-censored likelihood, patient particles, and dataset-specific baseline hazards as the primary TCAT analysis. The faster validation defaults are two chains with 1,000 tuning and 1,000 posterior iterations. Fits are checkpointed by held-out dataset and GEP; complete posterior objects are omitted unless `--save-posteriors` is requested.

```bash
python fit_tcat_survival_crossfit.py \
  all_patient_TCAT_estimates.csv \
  all_patient_TCAT_particles.npz \
  clinical_outcomes.csv \
  results/tcat_survival_crossfit \
  --adjustment Treatment=axicel \
  --adjustment Disease=lymphoma
```

`analyze_tcat_predictive_field_crossfit.py` then evaluates each held-out dataset with the GEP clinical field learned from the other datasets. It reports dataset-specific results and an empirical combined test across the held-out folds.

```bash
python analyze_tcat_predictive_field_crossfit.py \
  all_patient_TCAT_particles.npz \
  results/olambda_survival_regimes/patient_regime_probabilities.csv \
  results/tcat_survival_crossfit/crossfit_feature_results.csv \
  results/tcat_predictive_field_crossfit
```

The cross-fitted field is independent of the held-out outcomes. The primary cross-fitted analysis retains the full-cohort O-lambda regime probabilities, so that only the clinical-field transfer—not the regime boundary—is changed across folds.

## Pseudobulk gene associations and enrichment

`extract_string_pseudobulk.py` constructs a single comparable gene-expression matrix across cohorts. It takes one processed infusion h5ad per dataset, intersects their canonical STRING gene names, sums counts by patient over that exact gene universe, and calculates log2(CPM + 1).

```bash
python extract_string_pseudobulk.py \
  dataset_manifest.csv \
  results/string_pseudobulk_logCPM.csv
```

The manifest has columns `Dataset` and `h5ad_path`. The output contains every shared STRING gene in a common order; genes are not selected using their association with O-lambda.

`estimate_gene_olambda_correlations.py` estimates uncertainty-aware partial Spearman correlations between each shared STRING gene and asymptotic O-lambda. Gene expression and O-lambda are rank transformed. Every draw selects one equal-probability O-lambda particle per patient, draws Bayesian-bootstrap patient weights, and removes dataset effects together with treatment indicators estimated only within datasets containing more than one observed treatment.

```bash
python estimate_gene_olambda_correlations.py \
  results/string_pseudobulk_logCPM.csv \
  results/olambda_asymptotes/patient_olambda_particles.csv \
  patient_metadata_with_treatment.csv \
  results/olambda_gene_associations
```

The primary GSEA score is the posterior mean partial correlation divided by its posterior standard deviation. The complete variable STRING-gene ranking is written as both CSV and two-column `.rnk` files. No association threshold is applied before enrichment.

Patients lacking a treatment annotation are not assigned a treatment by inference. The default is to stop; `--drop-missing-treatment` performs a complete-case adjusted analysis and records every omitted patient in `excluded_patients.csv`.

`run_fgsea_rankings.R` runs `fgseaMultilevel` against MSigDB Hallmark and Reactome collections and saves both enrichment results and the exact gene-set membership tables used.

```bash
Rscript run_fgsea_rankings.R \
  results/olambda_gene_associations/olambda_gene_GSEA_rank.rnk \
  results/olambda_gene_associations/fgsea
```

## Smooth TCAT survival regimes

`fit_tcat_survival_regimes.py` applies the O-lambda smooth-regime model independently to every TCAT feature. It uses the same outcome-blind Q10-Q90 crossover prior, 0.10-SD logistic transition, uncertainty-particle marginalization, interval-censored likelihood, dataset-specific baseline hazards, and hierarchically pooled dataset effects.

```bash
python fit_tcat_survival_regimes.py \
  all_patient_TCAT_estimates.csv \
  all_patient_TCAT_particles.npz \
  clinical_outcomes.csv \
  results/tcat_smooth_regimes \
  --adjustment Treatment=axicel \
  --adjustment Disease=lymphoma
```

Completed feature models are checkpointed independently. The combined results record the high-versus-low plateau hazard ratio, dataset-direction agreement, and the posterior crossover in both native CLR units and as a percentile of the observed patient distribution. These models are intended as a secondary nonlinear comparison with the O-lambda regime rather than a replacement for the primary continuous TCAT models.

## Finite-depth O-lambda trajectories

`extract_olambda_trajectories.py` reads one processed infusion-product AnnData dataset and measures spectral organization across nested, unique-cell subsamples. Each eligible patient contributes approximately \(2\sqrt{N}\) intermediate depths between 50 cells and the full cell count, including explicit 50-, 100-, 150-, and 200-cell anchors. At each depth, genes must occupy both binary states in at least 0.002 of cells. The script then constructs the raw binary plug-in mutual-information graph and calculates exact-center O-lambda.

```bash
python extract_olambda_trajectories.py \
  processed_dataset_TCAT.h5ad \
  results/dataset_olambda_trajectories \
  --dataset-name DatasetName \
  --sample-key sample_id
```

Patient-level checkpoints make the extraction resumable. Increasing `--point-multiplier` adds depths by filling the largest gaps while retaining all completed measurements. The primary output, `olambda_trajectory_points.csv`, can be concatenated across datasets and passed directly to the asymptote model below. Complete MI graphs are never saved. This calculation requires a CUDA-enabled PyTorch installation; Colab GPU runtimes provide one by default.

## O-lambda sampling sensitivities

`extract_olambda_sensitivity.py` generates fixed-depth products for three outcome-blind sensitivity analyses. The input manifest contains one row per processed infusion dataset with columns `Dataset` and `h5ad_path`.

```bash
python extract_olambda_sensitivity.py \
  dataset_manifest.csv \
  results/patient_bootstrap \
  --experiment patient-bootstrap

python extract_olambda_sensitivity.py \
  dataset_manifest.csv \
  results/cell_type_standardized \
  --experiment cell-type-standardized

python extract_olambda_sensitivity.py \
  dataset_manifest.csv \
  results/pooled_synthetic \
  --experiment pooled-synthetic
```

Each analysis draws 25 products of 10,000 cells by default. The patient bootstrap draws Dirichlet cell weights and then samples a posterior-predictive product. The cell-type-standardized analysis gives every patient the same cohort-average cell-type distribution; retained cell types must contribute at least 10 cells to every patient and at least 1% of the average product. The pooled analysis reproduces each target patient's native cell-type distribution from a leave-one-patient-out, within-dataset pool. Source donors are balanced within each cell type, and sampling with replacement is used when a requested quota exceeds the available cells.

The target patient's full-depth two-state gene set is fixed across all replicates. The script then uses the same raw binary plug-in MI graph and exact-center O-lambda calculation as the finite-depth trajectory analysis. Patient-replicate checkpoints make every mode resumable; increasing `--replicates` adds new draws without replacing completed results. Composition, effective-sample-size, duplicate-cell, and pooled-donor audits are saved beside the O-lambda estimates.

## Patient-level O-lambda inference

`fit_olambda_asymptotes.R` fits the latent-timescale spectral-tail model

\[
\log(\mathcal O_\lambda(n)-1)
=
\alpha_i+b_{1i}u_{\tau_i}(n)+b_{2i}u_{\tau_i}(n)^2,
\qquad
u_\tau(n)=\frac{\tau}{n/500+\tau}.
\]

The positive timescale has the shared prior

\[
\log \tau_i = \log 2 + 0.35T_3,
\]

represented by nine equal-probability quadrature nodes. Each component has patient-specific random coefficients and depth-dependent residual variance. Conditional slope deviations use a Student-t prior, while the asymptote hierarchy is shared across timescales. The script saves calibrated mixture intervals and equal-probability posterior particles for downstream uncertainty propagation.

```bash
Rscript fit_olambda_asymptotes.R \
  olambda_trajectory_points.csv \
  results/olambda_asymptotes \
  --minimum-cells 1000 \
  --n-particles 16
```

This script requires R and the `nlme` package. Its primary outputs are `patient_olambda_asymptotes.csv` and `patient_olambda_particles.csv`.

## Technical robustness of inferred O-lambda

`extract_olambda_technical_covariates.py` extracts five prespecified technical covariates from the processed infusion datasets: log cell count, mean log total counts per cell, mean mitochondrial fraction, mean log detected genes per cell, and log node count in the corresponding full-cell graph. Cell-level quantities use coherent Bayesian-bootstrap particles; cell count and graph size are exact.

```bash
python extract_olambda_technical_covariates.py \
  dataset_manifest.csv \
  olambda_trajectory_points.csv \
  results/olambda_technical_covariates
```

`fit_olambda_technical_regression.py` regresses latent asymptotic O-lambda on these measurements. Independent models use hierarchically pooled dataset slopes. The joint model uses weak shrinkage across all five coefficients, and the leave-one-dataset-out mode quantifies whether the technical measurements reconstruct O-lambda in unseen cohorts. Independent technical and O-lambda particles are integrated over their Cartesian product.

```bash
python fit_olambda_technical_regression.py \
  results/olambda_technical_covariates/olambda_technical_estimates.csv \
  results/olambda_technical_covariates/olambda_technical_particles.npz \
  results/olambda_asymptotes/patient_olambda_particles.csv \
  results/olambda_technical_regression \
  --mode all
```

## Continuous survival association

`fit_olambda_survival_continuous.py` fits a continuous interval-censored survival model. Uncertainty in each patient asymptote is propagated using posterior particles. Dataset effects are hierarchically pooled, while baseline hazard trajectories remain dataset-specific and share one temporal smoothness parameter.

```bash
python fit_olambda_survival_continuous.py \
  results/olambda_asymptotes/patient_olambda_asymptotes.csv \
  clinical_outcomes.csv \
  results/olambda_survival_continuous \
  --particles-csv results/olambda_asymptotes/patient_olambda_particles.csv \
  --adjustment Treatment=axicel \
  --adjustment Disease=lymphoma
```

## Smooth survival regimes

`fit_olambda_survival_regimes.py` fits the corresponding smooth-regime model. The crossover has an outcome-blind uniform prior spanning the 10th through 90th percentiles of the patient-level O-lambda distribution.

```bash
python fit_olambda_survival_regimes.py \
  results/olambda_asymptotes/patient_olambda_asymptotes.csv \
  clinical_outcomes.csv \
  results/olambda_survival_regimes \
  --particles-csv results/olambda_asymptotes/patient_olambda_particles.csv \
  --adjustment Treatment=axicel \
  --adjustment Disease=lymphoma
```

Both survival scripts use 16 posterior particles by default. This was the lowest evaluated resolution that reproduced the continuous hazard ratio, regime hazard ratio, and crossover estimate obtained with 21 and 64 particles.

Clinical outcomes must contain `Dataset`, `sample_id`, `event_observed` (or `event`), `t_start`, and `t_end`. Times are measured in months. An event with equal `t_start` and `t_end` is treated as exactly observed; an event with unequal bounds is interval censored; and a row with `event_observed = 0` is right censored at `t_end`.

All survival scripts accept repeated `--adjustment COLUMN=REFERENCE` arguments. Each requested categorical variable is represented by global nuisance coefficients with a `Normal(0, 0.35)` prior by default; use `--adjustment-effect-sd` to change that scale. Patients missing a requested adjustment are excluded, and posterior adjustment effects are written separately from the primary predictor results. Adjustment labels and reference levels are supplied by the analysis rather than hard-coded in the repository. Dataset-confounded adjustment coefficients should be treated as control terms rather than independently interpretable effects.


## Scientific checks and reproducibility

```bash
python -m unittest -v test_analysis
Rscript test_asymptotes.R
```

These small synthetic checks cover count aggregation, QC, STRING alias resolution, TCAT cell alignment, patient and technical bootstrap particles, and exact/interval/right-censored likelihoods. The R check covers the empirical-Bayes patient update. They do not require clinical data or rerun the full study.

The scripts assume the documented input columns and valid analysis parameters. Gene ambiguity, patient alignment, and mathematical-domain checks are retained. Results are written directly to the output files; h5ad replacement still requires `--overwrite`.
