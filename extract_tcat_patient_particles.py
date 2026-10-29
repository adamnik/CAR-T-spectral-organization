"""Extract patient-level TCAT estimates and Bayesian-bootstrap particles."""

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ASA_COMPONENTS = [
    "TCAT_GEP_OX40_EBI3",
    "TCAT_GEP_ICOS_CD38",
    "TCAT_GEP_CTLA4_CD38",
    "TCAT_GEP_TIMD4_TIM3",
]
PROLIFERATION_COMPONENTS = [
    "TCAT_GEP_CellCycle_G2M",
    "TCAT_GEP_CellCycle_Late_S",
    "TCAT_GEP_CellCycle_S",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract patient TCAT CLR features and Bayesian-bootstrap particles "
            "from one processed AnnData file."
        )
    )
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--sample-key", default="sample_id")
    parser.add_argument(
        "--categorical-column",
        action="append",
        default=None,
        help=(
            "Categorical TCAT column to convert to patient proportions. "
            "May be repeated. By default, TCAT score columns containing 'label' are used."
        ),
    )
    parser.add_argument(
        "--category-levels-json",
        type=Path,
        default=None,
        help="Optional JSON mapping categorical columns to a fixed list of levels.",
    )
    parser.add_argument("--minimum-cells", type=int, default=50)
    parser.add_argument("--n-particles", type=int, default=16)
    parser.add_argument("--category-pseudocount", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(value):
    name = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    return name or "unnamed"


def atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(value, path):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def atomic_npz(path, **values):
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def clr(values):
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("CLR inputs must be finite and strictly positive.")
    logged = np.log(values)
    return logged - logged.mean(axis=-1, keepdims=True)


def infer_dataset_name(adata, input_path, requested):
    if requested is not None:
        return str(requested)
    for column in ["Dataset", "dataset", "dataset_name", "cohort"]:
        if column not in adata.obs:
            continue
        values = adata.obs[column].dropna().astype(str).unique()
        if len(values) == 1:
            return values[0]
    return input_path.stem


def categorical_levels(series, fixed_levels):
    if fixed_levels is not None:
        levels = list(map(str, fixed_levels))
    elif isinstance(series.dtype, pd.CategoricalDtype):
        levels = list(map(str, series.cat.categories))
    else:
        levels = sorted(series.dropna().astype(str).unique())
    if not levels:
        raise ValueError(f"No categorical levels were found for {series.name!r}.")
    observed = set(series.dropna().astype(str))
    unknown = sorted(observed.difference(levels))
    if unknown:
        raise ValueError(
            f"Column {series.name!r} contains levels absent from its fixed schema: {unknown}"
        )
    return levels


def patient_features(
    gep_values,
    categorical_values,
    category_schemas,
    asa_index,
    proliferation_index,
    n_particles,
    category_pseudocount,
    rng,
):
    n_cells = gep_values.shape[0]
    cell_weights = rng.dirichlet(np.ones(n_cells), size=n_particles)

    central_gep = gep_values.mean(axis=0)
    particle_gep = cell_weights @ gep_values
    central_gep_clr = clr(central_gep)
    particle_gep_clr = clr(particle_gep)

    central_parts = [
        central_gep_clr,
        np.asarray(
            [
                central_gep_clr[asa_index].mean(),
                central_gep_clr[proliferation_index].mean(),
            ]
        ),
    ]
    particle_parts = [
        particle_gep_clr,
        np.column_stack(
            [
                particle_gep_clr[:, asa_index].mean(axis=1),
                particle_gep_clr[:, proliferation_index].mean(axis=1),
            ]
        ),
    ]

    for column, levels in category_schemas.items():
        values = categorical_values[column]
        one_hot = np.column_stack([values == level for level in levels]).astype(float)
        central_counts = one_hot.sum(axis=0) + category_pseudocount
        particle_counts = (
            (cell_weights @ one_hot) * n_cells + category_pseudocount
        )
        central_parts.append(clr(central_counts / central_counts.sum()))
        particle_parts.append(
            clr(particle_counts / particle_counts.sum(axis=1, keepdims=True))
        )

    return np.concatenate(central_parts), np.concatenate(particle_parts, axis=1)


def main():
    args = parse_args()
    if not args.input_h5ad.exists():
        raise FileNotFoundError(args.input_h5ad)
    if args.minimum_cells < 1:
        raise ValueError("--minimum-cells must be positive.")
    if args.n_particles < 2:
        raise ValueError("--n-particles must be at least two.")
    if args.category_pseudocount <= 0:
        raise ValueError("--category-pseudocount must be positive.")

    output_paths = {
        "estimates": args.output_dir / "tcat_patient_estimates.csv",
        "uncertainty": args.output_dir / "tcat_patient_uncertainty.csv",
        "particles": args.output_dir / "tcat_patient_particles.npz",
        "manifest": args.output_dir / "tcat_extraction_manifest.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {existing[0]}. Use --overwrite to replace it."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.input_h5ad)
    print(
        f"Loaded {args.input_h5ad}: {adata.n_obs:,} cells x {adata.n_vars:,} genes",
        flush=True,
    )
    if args.sample_key not in adata.obs:
        raise KeyError(f"Sample column {args.sample_key!r} was not found in adata.obs.")
    if adata.obs[args.sample_key].isna().any():
        raise ValueError(f"Sample column {args.sample_key!r} contains missing values.")
    if "timepoint" in adata.obs:
        timepoints = adata.obs["timepoint"].dropna().astype(str).str.lower().unique()
        if len(timepoints) != 1 or timepoints[0] != "infusion":
            raise ValueError(
                "The input must contain infusion cells only. "
                f"Observed timepoints: {sorted(timepoints)}"
            )

    gep_columns = sorted(
        column for column in adata.obs if str(column).startswith("TCAT_GEP_")
    )
    if not gep_columns:
        raise ValueError("No TCAT_GEP_* columns were found in adata.obs.")
    missing_components = sorted(
        (set(ASA_COMPONENTS) | set(PROLIFERATION_COMPONENTS))
        - set(gep_columns)
    )
    if missing_components:
        raise ValueError(
            "GEPs required for ASA or Proliferation are missing: "
            f"{missing_components}"
        )
    gep_values = adata.obs[gep_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(float)
    if np.any(~np.isfinite(gep_values)) or np.any(gep_values < 0):
        raise ValueError("TCAT GEP values must be finite and nonnegative.")
    if not np.allclose(gep_values.sum(axis=1), 1.0, atol=1e-4, rtol=0):
        raise ValueError("Cell-level TCAT GEP values do not sum to one.")

    if args.category_levels_json is not None:
        fixed_category_levels = json.loads(args.category_levels_json.read_text())
    else:
        fixed_category_levels = {}
    if args.categorical_column is None:
        categorical_columns = sorted(
            column
            for column in adata.obs
            if str(column).startswith("TCAT_SCORE_")
            and "label" in str(column).lower()
        )
    else:
        categorical_columns = list(dict.fromkeys(args.categorical_column))
    missing_categorical = [
        column for column in categorical_columns if column not in adata.obs
    ]
    if missing_categorical:
        raise KeyError(f"Categorical columns were not found: {missing_categorical}")

    categorical_frame = pd.DataFrame(index=adata.obs_names)
    category_schemas = {}
    for column in categorical_columns:
        series = adata.obs[column]
        if series.isna().any():
            raise ValueError(f"Categorical column {column!r} contains missing values.")
        levels = categorical_levels(series, fixed_category_levels.get(column))
        categorical_frame[column] = series.astype(str).to_numpy()
        category_schemas[column] = levels

    feature_names = [f"{column}_CLR" for column in gep_columns]
    feature_names.extend(["TCAT_SCORE_ASA_CLR", "TCAT_SCORE_Proliferation_CLR"])
    for column, levels in category_schemas.items():
        feature_names.extend(
            f"{column}_proportion_{safe_name(level)}_CLR" for level in levels
        )
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Feature names are not unique after sanitization.")

    component_index = {column: index for index, column in enumerate(gep_columns)}
    asa_index = np.asarray([component_index[x] for x in ASA_COMPONENTS])
    proliferation_index = np.asarray(
        [component_index[x] for x in PROLIFERATION_COMPONENTS]
    )
    dataset = infer_dataset_name(adata, args.input_h5ad, args.dataset_name)
    sample_values = adata.obs[args.sample_key].astype(str)
    patients = sorted(sample_values.unique())
    rng = np.random.default_rng(args.random_state)

    central_rows = []
    uncertainty_rows = []
    particle_values = []
    retained_patients = []
    retained_cell_counts = []

    for number, patient in enumerate(patients, start=1):
        indices = np.flatnonzero(sample_values.to_numpy() == patient)
        if len(indices) < args.minimum_cells:
            print(
                f"[{number}/{len(patients)}] {patient}: skipped ({len(indices):,} cells)",
                flush=True,
            )
            continue
        patient_categories = {
            column: categorical_frame[column].to_numpy()[indices]
            for column in categorical_columns
        }
        central, particles = patient_features(
            gep_values[indices],
            patient_categories,
            category_schemas,
            asa_index,
            proliferation_index,
            args.n_particles,
            args.category_pseudocount,
            rng,
        )
        central_rows.append(
            {
                "Dataset": dataset,
                "sample_id": patient,
                "n_cells": len(indices),
                **dict(zip(feature_names, central)),
            }
        )
        for index, feature in enumerate(feature_names):
            uncertainty_rows.append(
                {
                    "Dataset": dataset,
                    "sample_id": patient,
                    "feature": feature,
                    "central": central[index],
                    "q025": np.quantile(particles[:, index], 0.025),
                    "q975": np.quantile(particles[:, index], 0.975),
                }
            )
        particle_values.append(particles)
        retained_patients.append(patient)
        retained_cell_counts.append(len(indices))
        print(
            f"[{number}/{len(patients)}] {patient}: "
            f"{len(indices):,} cells | {len(feature_names):,} features",
            flush=True,
        )

    if not particle_values:
        raise ValueError("No patients satisfy the minimum cell requirement.")
    particle_values = np.stack(particle_values)
    particle_weights = np.full(
        (len(retained_patients), args.n_particles),
        1.0 / args.n_particles,
    )

    atomic_csv(pd.DataFrame(central_rows), output_paths["estimates"])
    atomic_csv(pd.DataFrame(uncertainty_rows), output_paths["uncertainty"])
    atomic_npz(
        output_paths["particles"],
        values=particle_values,
        particle_weights=particle_weights,
        patient_ids=np.asarray(retained_patients, dtype="U"),
        datasets=np.repeat(np.asarray(dataset, dtype="U"), len(retained_patients)),
        feature_names=np.asarray(feature_names, dtype="U"),
        n_cells=np.asarray(retained_cell_counts, dtype=np.int64),
    )
    atomic_json(
        {
            "input_h5ad": str(args.input_h5ad),
            "dataset": dataset,
            "sample_key": args.sample_key,
            "minimum_cells": args.minimum_cells,
            "n_particles": args.n_particles,
            "random_state": args.random_state,
            "particle_method": "cell-level Bayesian bootstrap with Dirichlet(1) weights",
            "GEP_transform": "CLR of patient-level weighted mean TCAT GEP composition",
            "ASA_components": ASA_COMPONENTS,
            "Proliferation_components": PROLIFERATION_COMPONENTS,
            "derived_score_aggregation": "mean of component CLR GEPs",
            "categorical_columns": categorical_columns,
            "category_levels": category_schemas,
            "category_pseudocount": args.category_pseudocount,
            "category_transform": (
                "Jeffreys-smoothed patient proportions followed by CLR within each label column"
            ),
            "feature_names": feature_names,
            "patients": len(retained_patients),
        },
        output_paths["manifest"],
    )
    print(
        f"Saved {len(retained_patients):,} patients x "
        f"{args.n_particles:,} particles x {len(feature_names):,} features",
        flush=True,
    )


if __name__ == "__main__":
    main()
