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
  --mode individual

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

## Smooth TCAT survival regimes

`fit_tcat_survival_regimes.py` applies the O-lambda smooth-regime model independently to every TCAT feature. It uses the same outcome-blind Q10-Q90 crossover prior, 0.10-SD logistic transition, uncertainty-particle marginalization, interval-censored likelihood, dataset-specific baseline hazards, and hierarchically pooled dataset effects.

```bash
python fit_tcat_survival_regimes.py \
  all_patient_TCAT_estimates.csv \
  all_patient_TCAT_particles.npz \
  clinical_outcomes.csv \
  results/tcat_smooth_regimes
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

## Continuous survival association

`fit_olambda_survival_continuous.py` fits a continuous interval-censored survival model. Uncertainty in each patient asymptote is propagated using posterior particles. Dataset effects are hierarchically pooled, while baseline hazard trajectories remain dataset-specific and share one temporal smoothness parameter.

```bash
python fit_olambda_survival_continuous.py \
  results/olambda_asymptotes/patient_olambda_asymptotes.csv \
  clinical_outcomes.csv \
  results/olambda_survival_continuous \
  --particles-csv results/olambda_asymptotes/patient_olambda_particles.csv
```

## Smooth survival regimes

`fit_olambda_survival_regimes.py` fits the corresponding smooth-regime model. The crossover has an outcome-blind uniform prior spanning the 10th through 90th percentiles of the patient-level O-lambda distribution.

```bash
python fit_olambda_survival_regimes.py \
  results/olambda_asymptotes/patient_olambda_asymptotes.csv \
  clinical_outcomes.csv \
  results/olambda_survival_regimes \
  --particles-csv results/olambda_asymptotes/patient_olambda_particles.csv
```

Both survival scripts use 16 posterior particles by default. This was the lowest evaluated resolution that reproduced the continuous hazard ratio, regime hazard ratio, and crossover estimate obtained with 21 and 64 particles.

Clinical outcomes must contain `Dataset`, `sample_id`, `event_observed` (or `event`), `t_start`, and `t_end`. Times are measured in months. An event with equal `t_start` and `t_end` is treated as exactly observed; an event with unequal bounds is interval censored; and a row with `event_observed = 0` is right censored at `t_end`.


## Scientific checks and reproducibility

```bash
python -m unittest -v test_analysis
Rscript test_asymptotes.R
```

These small synthetic checks cover count aggregation, QC, STRING alias resolution, TCAT cell alignment, patient bootstrap particles, and exact/interval/right-censored likelihoods. The R check covers the empirical-Bayes patient update. They do not require clinical data or rerun the full study.

The scripts assume the documented input columns and valid analysis parameters. Gene ambiguity, patient alignment, and mathematical-domain checks are retained. Results are written directly to the output files; h5ad replacement still requires `--overwrite`.
