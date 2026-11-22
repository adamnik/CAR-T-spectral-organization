# Spectral analysis of gene-dependency networks reveals emergent organization across scales in CAR T biology

by Adam Nik and Keenan Hartert

preprint coming soon

Analysis code for spectral organization, cell states, and survival across CAR T-cell infusion cohorts.

| Analysis | Code |
|---|---|
| Expression preprocessing, STRING filtering, and TCAT patient features | [Preprocessing](preprocessing/) |
| Spectral trajectories, asymptote estimates, and validation | [Estimation](spectral_organization/estimation/) |
| Sampling sensitivity and technical covariates | [Robustness](spectral_organization/robustness/) |
| Survival models and cohort-held-out fits (Figs. 2–3) | [Survival](survival/) |
| Cell-state correlations (Fig. 4) and spectral reconstruction | [Cell-state relationships](cell_state_relationships/) |
| Regime geometry, held-out validation, and surrogate-field tests (Fig. 5, Tables S4–S5) | [Clinical fields](clinical_fields/) |
| Pseudobulk gene associations, GSEA, and Reactome networks | [Gene enrichment](gene_enrichment/) |

## Running the code

From the repository root, use Python 3.11 and run Python scripts as modules.

```bash
python -m pip install -r requirements.txt
```

Use `--help` for Python script arguments. Run R scripts with `Rscript`; the asymptote model requires `nlme`, and spectral trajectory extraction requires a CUDA-enabled PyTorch build.

To render the pooled compositional surrogate heatmap alongside Fig. 5B, use the final surrogate results and the pooled mean-geometry context results:

```bash
python -m clinical_fields.plot_surrogate_heatmap \
  path/to/pooled_surrogate_results.csv \
  path/to/mean_predictive_field_context_results.csv \
  path/to/figures
```

The command writes a PDF and a 600-DPI JPEG. Cells show nominal joint surrogate probabilities, with color on the shared `-log10(p)` scale.

## Checks

```bash
python -m unittest discover -s tests
Rscript tests/test_asymptotes.R
```
