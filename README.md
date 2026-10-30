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

## Patient-level O-lambda inference

`fit_olambda_asymptotes.R` fits the second-order spectral-tail model

\[
\log(\mathcal O_\lambda(n)-1)
=
\theta_0
+
\frac{\theta_1}{n/500+2}
+
\frac{\theta_2}{(n/500+2)^2}.
\]

The population model is fit with patient-specific random coefficients and a depth-dependent residual variance. Patient asymptotes and Student-t uncertainty intervals are then estimated by robust empirical-Bayes updating.

```bash
Rscript fit_olambda_asymptotes.R \
  olambda_trajectory_points.csv \
  results/olambda_asymptotes \
  --minimum-cells 1000
```

This script requires R and the `nlme` package.

## Continuous survival association

`fit_olambda_survival_continuous.py` fits a continuous interval-censored survival model. Uncertainty in each patient asymptote is propagated using posterior particles. Dataset effects are hierarchically pooled, while baseline hazard trajectories remain dataset-specific and share one temporal smoothness parameter.

```bash
python fit_olambda_survival_continuous.py \
  results/olambda_asymptotes/patient_olambda_asymptotes.csv \
  clinical_outcomes.csv \
  results/olambda_survival_continuous
```

## Smooth survival regimes

`fit_olambda_survival_regimes.py` fits the corresponding smooth-regime model. The crossover has an outcome-blind uniform prior spanning the 10th through 90th percentiles of the patient-level O-lambda distribution.

```bash
python fit_olambda_survival_regimes.py \
  results/olambda_asymptotes/patient_olambda_asymptotes.csv \
  clinical_outcomes.csv \
  results/olambda_survival_regimes
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
