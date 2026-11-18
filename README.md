# CAR T-cell spectral organization

Analysis code for spectral organization, cell states, and survival across CAR T-cell infusion cohorts.

| Analysis | Code |
|---|---|
| Expression preprocessing, STRING filtering, and TCAT patient features | [Preprocessing](preprocessing/) |
| Spectral trajectories, asymptote estimates, and validation | [Estimation](spectral_organization/estimation/) |
| Sampling sensitivity and technical covariates | [Robustness](spectral_organization/robustness/) |
| Survival models and cohort-held-out fits (Figs. 2–3) | [Survival](survival/) |
| Cell-state correlations (Fig. 4) and spectral reconstruction | [Cell-state relationships](cell_state_relationships/) |
| Pooled regime geometry (Fig. 5B) | [Analysis](clinical_fields/analyze_cross_system_predictive_field.py) · [Plot](clinical_fields/plot_regime_field_geometry.py) |
| Held-out field validation and surrogate specificity | [Table S4](clinical_fields/analyze_tcat_predictive_field_crossfit.py) · [Table S5](clinical_fields/compositional_surrogate_field.py) |
| Pseudobulk gene associations, GSEA, and Reactome networks | [Gene enrichment](gene_enrichment/) |

## Running the code

From the repository root, use Python 3.11 and run Python scripts as modules. For example, the Figure 5B plotting script takes the pooled field results directory.

```bash
python -m pip install -r requirements.txt
python -m clinical_fields.plot_regime_field_geometry results/pooled_field
```

Use `--help` for Python script arguments. Run R scripts with `Rscript`; the asymptote model requires `nlme`, and spectral trajectory extraction requires CUDA.

## Checks

```bash
python -m unittest discover -s tests
Rscript tests/test_asymptotes.R
```
