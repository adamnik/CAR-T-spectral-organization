# CAR T-cell spectral organization

This repository contains the preprocessing and statistical analysis code for the CAR T-cell spectral organization framework.

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
