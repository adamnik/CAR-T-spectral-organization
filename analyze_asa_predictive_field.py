# ASA-conditioned organization of cell-type and non-ASA GEP clinical fields

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_io import safe_name
from analyze_cross_system_predictive_field import align_field
from analyze_tcat_predictive_field import context_test, load_regime_memberships
from analyze_tcat_predictive_field import posterior_states
from predictive_field import METRICS, benjamini_hochberg


ASA_FEATURE = "TCAT_SCORE_ASA_CLR"
ASA_COMPONENTS = [
    "TCAT_GEP_OX40_EBI3",
    "TCAT_GEP_ICOS_CD38",
    "TCAT_GEP_CTLA4_CD38",
    "TCAT_GEP_TIMD4_TIM3",
]
ASA_GEP_FEATURES = [f"{feature}_CLR" for feature in ASA_COMPONENTS]


def asa_context(regime_dir, continuous_results_csv):
    memberships = load_regime_memberships(regime_dir)
    if ASA_FEATURE not in memberships:
        raise FileNotFoundError(
            f"The completed smooth-transition result for {ASA_FEATURE} was not found in "
            f"{regime_dir}."
        )

    saved = memberships[ASA_FEATURE]
    result = saved["result"]
    continuous = pd.read_csv(continuous_results_csv).set_index("feature")
    continuous_hit = False
    if ASA_FEATURE in continuous.index:
        row = continuous.loc[ASA_FEATURE]
        continuous_hit = float(row.HR_q025) > 1 or float(row.HR_q975) < 1
    regime_hit = float(result.HR_q025) > 1 or float(result.HR_q975) < 1
    return {
        "context": ASA_FEATURE,
        "context_source": "TCAT_score",
        "context_kind": "score",
        "adverse_high": float(result.HR_median) >= 1,
        "HR_median": float(result.HR_median),
        "HR_q025": float(result.HR_q025),
        "HR_q975": float(result.HR_q975),
        "membership": saved["membership"],
        "membership_path": str(saved["path"]),
        "selected_by_continuous": continuous_hit,
        "selected_by_regime": regime_hit,
    }


def prepare_field(field, particles_path, continuous_results_csv, feature_prefix,
                  context, excluded_features):
    patients, features, values, particle_weights, beta, contexts = align_field(
        particles_path, continuous_results_csv, feature_prefix, [context]
    )
    excluded_features = set(excluded_features)
    keep = np.asarray([feature not in excluded_features for feature in features])
    excluded_present = features[~keep].tolist()
    missing_exclusions = sorted(excluded_features.difference(features))
    if missing_exclusions:
        raise ValueError(
            f"{field} is missing requested excluded nodes: {', '.join(missing_exclusions)}"
        )
    if keep.sum() < 3:
        raise ValueError(f"Only {int(keep.sum())} graph nodes remain for {field}.")
    return (
        patients,
        features[keep],
        values[:, :, keep],
        particle_weights,
        beta[keep],
        contexts[0],
        excluded_present,
    )


def correction_groups(frame):
    columns = ["geometry"]
    if "field" in frame:
        columns.insert(0, "field")
    return columns


def add_multiple_testing_columns(frame):
    frame = frame.copy()
    for _, indices in frame.groupby(correction_groups(frame), dropna=False).groups.items():
        indices = list(indices)
        frame.loc[indices, "joint_FDR"] = benjamini_hochberg(
            frame.loc[indices, "joint_permutation_p"]
        )
        frame.loc[indices, "joint_rank"] = frame.loc[
            indices, "joint_oriented_z"
        ].rank(ascending=False, method="min")
        for metric in METRICS:
            frame.loc[indices, f"FDR__{metric}"] = benjamini_hochberg(
                frame.loc[indices, f"p__{metric}"]
            )
    frame["joint_rank"] = frame.joint_rank.astype(int)
    return frame


def append_rows(path, new_rows, field=None):
    if path is None:
        return
    if not path.exists():
        raise FileNotFoundError(f"Existing results were not found: {path}")

    existing = pd.read_csv(path)
    additions = new_rows.copy()
    if field is not None and "field" in existing:
        additions = additions.loc[additions.field.eq(field)]
    if additions.empty:
        return

    replacement = existing.context.astype(str).eq(ASA_FEATURE)
    replacement &= existing.geometry.astype(str).isin(additions.geometry.astype(str))
    if "field" in existing and field is not None:
        replacement &= existing.field.astype(str).eq(field)
    existing = existing.loc[~replacement].copy()

    # Preserve each native results schema and add only the exclusion audit.
    columns = list(existing.columns)
    if "excluded_graph_nodes" not in columns:
        columns.append("excluded_graph_nodes")
    existing = existing.reindex(columns=columns)
    additions = additions.reindex(columns=columns)
    combined = pd.concat([existing, additions], ignore_index=True)
    combined = add_multiple_testing_columns(combined)
    combined.to_csv(path, index=False)
    print(f"Appended ASA to {path}", flush=True)


def analyze_one_field(field, particles_path, continuous_results_csv, feature_prefix,
                      context, excluded_features, geometries, args, completed_rows, started):
    prepared = prepare_field(
        field, particles_path, continuous_results_csv, feature_prefix,
        context, excluded_features,
    )
    patients, features, values, weights, beta, context, excluded_present = prepared
    datasets = patients.Dataset.astype(str).to_numpy()
    dataset_indices = [
        np.flatnonzero(datasets == dataset) for dataset in np.unique(datasets)
    ]
    states, bootstrap_weights = posterior_states(
        values, weights, datasets, args.correlation_draws, args.random_state
    )
    print(
        f"\n{field}: {len(patients):,} patients | {len(np.unique(datasets)):,} datasets | "
        f"{len(features):,} graph nodes",
        flush=True,
    )
    if excluded_present:
        print("Excluded ASA component nodes:", ", ".join(excluded_present), flush=True)

    result_path = args.output_dir / "asa_predictive_field_results.csv"
    matrix_dir = args.output_dir / "observed_matrices" / field
    matrix_dir.mkdir(parents=True, exist_ok=True)
    completed = {
        (str(row["field"]), str(row["geometry"])) for row in completed_rows
    }
    field_number = 0 if field == "atlas_cell_type" else 1
    for geometry_number, geometry in enumerate(geometries):
        if (field, geometry) in completed and not args.overwrite:
            print(f"[RESUME] {field} | {geometry}", flush=True)
            continue
        result = context_test(
            states[geometry], bootstrap_weights,
            context["adverse_membership"], beta, np.arange(len(features)),
            dataset_indices, args.permutations,
            args.random_state + 1000 * (field_number + 1) + geometry_number,
        )
        high, low, observed, z, p, joint_z, joint_p, _, _ = result
        row = {
            "field": field,
            "context": ASA_FEATURE,
            "context_source": "TCAT_score",
            "context_kind": "score",
            "geometry": geometry,
            "patients": len(patients),
            "graph_nodes": len(features),
            "context_node_excluded": False,
            "excluded_graph_nodes": ";".join(excluded_present),
            "clinical_adverse_tail": "high" if context["adverse_high"] else "low",
            "effective_adverse_patients": float(context["adverse_membership"].sum()),
            "HR_median": context["HR_median"],
            "HR_q025": context["HR_q025"],
            "HR_q975": context["HR_q975"],
            "selected_by_continuous": context["selected_by_continuous"],
            "selected_by_regime": context["selected_by_regime"],
            "joint_oriented_z": joint_z,
            "joint_permutation_p": joint_p,
        }
        for index, metric in enumerate(METRICS):
            row[f"contrast__{metric}"] = observed[index]
            row[f"z__{metric}"] = z[index]
            row[f"p__{metric}"] = p[index]
        completed_rows.append(row)
        completed_rows[:] = [
            saved for saved in completed_rows
            if (saved["field"], saved["geometry"]) != (field, geometry) or saved is row
        ]
        pd.DataFrame(completed_rows).to_csv(result_path, index=False)

        name = safe_name(ASA_FEATURE)
        pd.DataFrame(high, index=features, columns=features).to_csv(
            matrix_dir / f"{name}__{geometry}__adverse.csv"
        )
        pd.DataFrame(low, index=features, columns=features).to_csv(
            matrix_dir / f"{name}__{geometry}__favorable.csv"
        )
        print(
            f"{field} | {geometry} | joint z={joint_z:.2f}, p={joint_p:.4g} | "
            f"elapsed={(time.time() - started) / 60:.1f} min",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Test ASA as a context for the atlas cell-type field and the TCAT GEP "
            "field after removing the GEPs used to construct ASA."
        )
    )
    parser.add_argument("atlas_particles_npz", type=Path)
    parser.add_argument("atlas_continuous_results_csv", type=Path)
    parser.add_argument("tcat_particles_npz", type=Path)
    parser.add_argument("tcat_continuous_results_csv", type=Path)
    parser.add_argument("tcat_regime_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--append-atlas-results", type=Path, default=None)
    parser.add_argument("--append-gep-results", type=Path, default=None)
    parser.add_argument("--correlation-draws", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--geometry", action="append", choices=["global", "dataset_adjusted"],
        help="Geometry to test. Repeat to run both; the default is both geometries.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometries = args.geometry or ["global", "dataset_adjusted"]

    context = asa_context(args.tcat_regime_dir, args.tcat_continuous_results_csv)
    manifest_context = {
        key: value for key, value in context.items() if key != "membership"
    }
    pd.DataFrame([manifest_context]).to_csv(
        args.output_dir / "asa_context_manifest.csv", index=False
    )
    print(
        f"ASA smooth-transition HR={context['HR_median']:.3f} "
        f"[{context['HR_q025']:.3f}, {context['HR_q975']:.3f}] | "
        f"adverse tail={'high' if context['adverse_high'] else 'low'}",
        flush=True,
    )

    result_path = args.output_dir / "asa_predictive_field_results.csv"
    if result_path.exists() and not args.overwrite:
        completed = pd.read_csv(result_path)
        completed = completed.loc[completed.geometry.astype(str).isin(geometries)]
        completed_rows = completed.to_dict("records")
    else:
        completed_rows = []
    started = time.time()
    analyze_one_field(
        "atlas_cell_type", args.atlas_particles_npz,
        args.atlas_continuous_results_csv, "ATLAS_proportion_",
        context, [], geometries, args, completed_rows, started,
    )
    analyze_one_field(
        "TCAT_GEP_excluding_ASA", args.tcat_particles_npz,
        args.tcat_continuous_results_csv, "TCAT_GEP_",
        context, ASA_GEP_FEATURES, geometries, args, completed_rows, started,
    )

    results = add_multiple_testing_columns(pd.DataFrame(completed_rows))
    results.to_csv(result_path, index=False)
    append_rows(args.append_atlas_results, results, field="atlas_cell_type")
    # The original GEP results do not use a field column. Rename the reduced
    # field to TCAT_GEP only when it is appended to that native table.
    gep_rows = results.loc[results.field.eq("TCAT_GEP_excluding_ASA")].copy()
    gep_rows["field"] = "TCAT_GEP"
    append_rows(args.append_gep_results, gep_rows, field="TCAT_GEP")

    manifest = {
        "context": ASA_FEATURE,
        "context_definition": "posterior mean adverse-regime probability from the full Bayesian smooth-transition model",
        "atlas_field": "all atlas cell types",
        "gep_field": "TCAT GEPs excluding the four components used to construct ASA",
        "excluded_ASA_components": ASA_GEP_FEATURES,
        "metrics": METRICS,
        "geometries": geometries,
        "patient_uncertainty": "particle sampling and Bayesian patient bootstrap",
        "permutation": "ASA adverse-regime probabilities shuffled within dataset",
        "correlation_draws": args.correlation_draws,
        "permutations": args.permutations,
        "random_state": args.random_state,
        "runtime_minutes": (time.time() - started) / 60,
    }
    (args.output_dir / "asa_predictive_field_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print("\nFocused results:", result_path, flush=True)


if __name__ == "__main__":
    main()
