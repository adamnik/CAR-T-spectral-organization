# Cross-system organization of TCAT GEP and atlas cell-type clinical fields

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_io import safe_name
from analyze_tcat_predictive_field import context_test, load_regime_memberships
from analyze_tcat_predictive_field import posterior_states
from predictive_field import METRICS, benjamini_hochberg


def significant_features(continuous_path, regime_path, prefix):
    continuous = pd.read_csv(continuous_path)
    regimes = pd.read_csv(regime_path)
    continuous_hit = continuous.HR_q025.gt(1) | continuous.HR_q975.lt(1)
    regime_hit = regimes.HR_q025.gt(1) | regimes.HR_q975.lt(1)
    continuous_selected = set(continuous.loc[continuous_hit, "feature"].astype(str))
    regime_selected = set(regimes.loc[regime_hit, "feature"].astype(str))
    selected = sorted(
        feature for feature in continuous_selected | regime_selected
        if feature.startswith(prefix)
    )
    return selected, continuous_selected, regime_selected


def context_from_membership(feature, source, saved, continuous_selected, regime_selected):
    result = saved["result"]
    return {
        "context": feature,
        "context_source": source,
        "adverse_high": float(result.HR_median) >= 1,
        "HR_median": float(result.HR_median),
        "HR_q025": float(result.HR_q025),
        "HR_q975": float(result.HR_q975),
        "membership": saved["membership"],
        "membership_path": str(saved["path"]),
        "selected_by_continuous": feature in continuous_selected,
        "selected_by_regime": feature in regime_selected,
    }


def selected_contexts(args):
    atlas_memberships = load_regime_memberships(args.atlas_regime_dir)
    tcat_memberships = load_regime_memberships(args.tcat_regime_dir)
    atlas, atlas_continuous, atlas_regime = significant_features(
        args.atlas_continuous_results_csv,
        args.atlas_regime_results_csv,
        "ATLAS_proportion_",
    )
    geps, gep_continuous, gep_regime = significant_features(
        args.tcat_continuous_results_csv,
        args.tcat_regime_results_csv,
        "TCAT_GEP_",
    )
    missing_atlas = sorted(set(atlas).difference(atlas_memberships))
    missing_geps = sorted(set(geps).difference(tcat_memberships))
    if missing_atlas or missing_geps:
        missing = [*missing_atlas, *missing_geps]
        raise FileNotFoundError(
            "Missing completed Bayesian regime memberships for: " + ", ".join(missing)
        )

    olambda_result = pd.read_csv(args.olambda_regime_result_csv).iloc[0]
    olambda = {
        "context": "O_lambda",
        "context_source": "system",
        "adverse_high": float(olambda_result.HR_median) >= 1,
        "HR_median": float(olambda_result.HR_median),
        "HR_q025": float(olambda_result.HR_q025),
        "HR_q975": float(olambda_result.HR_q975),
        "membership": pd.read_csv(args.olambda_regime_membership_csv),
        "membership_path": str(args.olambda_regime_membership_csv),
        "selected_by_continuous": True,
        "selected_by_regime": True,
    }
    atlas_contexts = [
        context_from_membership(
            feature, "atlas_cell_type", atlas_memberships[feature],
            atlas_continuous, atlas_regime,
        )
        for feature in atlas
    ]
    gep_contexts = [
        context_from_membership(
            feature, "TCAT_GEP", tcat_memberships[feature],
            gep_continuous, gep_regime,
        )
        for feature in geps
    ]
    return olambda, atlas_contexts, gep_contexts


def align_field(particles_path, continuous_path, feature_prefix, contexts):
    archive = np.load(particles_path, allow_pickle=False)
    feature_names = archive["feature_names"].astype(str)
    field_features = np.asarray([
        feature for feature in feature_names if feature.startswith(feature_prefix)
    ])
    if not len(field_features):
        raise ValueError(f"No features begin with {feature_prefix!r} in {particles_path}.")
    continuous = pd.read_csv(continuous_path)
    effects = continuous.set_index("feature")
    missing = sorted(set(field_features).difference(effects.index.astype(str)))
    if missing:
        raise ValueError("Continuous survival results are missing: " + ", ".join(missing))

    patients = pd.DataFrame({
        "Dataset": archive["datasets"].astype(str),
        "sample_id": archive["patient_ids"].astype(str),
        "particle_row": np.arange(len(archive["patient_ids"])),
    })
    membership_columns = []
    for number, context in enumerate(contexts):
        column = f"membership_{number:03d}"
        membership = context["membership"][[
            "Dataset", "sample_id", "posterior_mean_probability_high"
        ]].rename(columns={"posterior_mean_probability_high": column})
        patients = patients.merge(
            membership, on=["Dataset", "sample_id"], how="inner", validate="one_to_one"
        )
        membership_columns.append(column)
    patients = patients.sort_values(["Dataset", "sample_id"]).reset_index(drop=True)
    if patients.empty:
        raise ValueError("No patients are shared by the field particles and regime memberships.")

    lookup = {feature: index for index, feature in enumerate(feature_names)}
    indices = np.asarray([lookup[feature] for feature in field_features])
    rows = patients.particle_row.to_numpy(int)
    values = np.asarray(archive["values"][rows][:, :, indices], dtype=float)
    particle_weights = np.asarray(archive["particle_weights"][rows], dtype=float)
    particle_weights /= particle_weights.sum(axis=1, keepdims=True)
    beta = np.log(effects.loc[field_features, "HR_median"].to_numpy(float))

    aligned_contexts = []
    for number, context in enumerate(contexts):
        aligned = dict(context)
        membership = patients[membership_columns[number]].to_numpy(float)
        if not context["adverse_high"]:
            membership = 1 - membership
        aligned["adverse_membership"] = membership
        aligned_contexts.append(aligned)
    return patients, field_features, values, particle_weights, beta, aligned_contexts


def analyze_field(field_name, particles_path, continuous_path, feature_prefix,
                  contexts, args, result_rows, started):
    patients, features, values, particle_weights, beta, contexts = align_field(
        particles_path, continuous_path, feature_prefix, contexts
    )
    datasets = patients.Dataset.astype(str).to_numpy()
    dataset_indices = [
        np.flatnonzero(datasets == dataset) for dataset in np.unique(datasets)
    ]
    states, bootstrap_weights = posterior_states(
        values, particle_weights, datasets,
        args.correlation_draws, args.random_state,
    )
    print(
        f"\n{field_name}: {len(patients):,} patients | "
        f"{len(np.unique(datasets)):,} datasets | {len(features):,} nodes | "
        f"{len(contexts):,} contexts",
        flush=True,
    )

    result_path = args.output_dir / "cross_system_predictive_field_results.csv"
    matrix_dir = args.output_dir / "observed_matrices" / field_name
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for context_number, context in enumerate(contexts, start=1):
        graph_indices = np.flatnonzero(features != context["context"])
        graph_features = features[graph_indices]
        graph_beta = beta[graph_indices]
        for geometry_number, geometry in enumerate(["global", "dataset_adjusted"]):
            result = context_test(
                states[geometry], bootstrap_weights,
                context["adverse_membership"], graph_beta, graph_indices,
                dataset_indices, args.permutations,
                args.random_state + len(result_rows) * 100 + geometry_number,
            )
            high, low, observed, z, p, joint_z, joint_p, _, _ = result
            row = {
                "field": field_name,
                "context": context["context"],
                "context_source": context["context_source"],
                "geometry": geometry,
                "patients": len(patients),
                "graph_nodes": len(graph_features),
                "context_node_excluded": context["context"] in features,
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
            result_rows.append(row)
            pd.DataFrame(result_rows).to_csv(result_path, index=False)

            name = safe_name(context["context"])
            pd.DataFrame(high, index=graph_features, columns=graph_features).to_csv(
                matrix_dir / f"{name}__{geometry}__adverse.csv"
            )
            pd.DataFrame(low, index=graph_features, columns=graph_features).to_csv(
                matrix_dir / f"{name}__{geometry}__favorable.csv"
            )
            print(
                f"[{context_number}/{len(contexts)}] {context['context']} | {geometry} | "
                f"joint z={joint_z:.2f}, p={joint_p:.4g} | "
                f"elapsed={(time.time() - started) / 60:.1f} min",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Test cell-type and GEP contexts across cell-type and GEP clinical fields."
    )
    parser.add_argument("atlas_particles_npz", type=Path)
    parser.add_argument("atlas_continuous_results_csv", type=Path)
    parser.add_argument("atlas_regime_results_csv", type=Path)
    parser.add_argument("atlas_regime_dir", type=Path)
    parser.add_argument("tcat_particles_npz", type=Path)
    parser.add_argument("tcat_continuous_results_csv", type=Path)
    parser.add_argument("tcat_regime_results_csv", type=Path)
    parser.add_argument("tcat_regime_dir", type=Path)
    parser.add_argument("olambda_regime_membership_csv", type=Path)
    parser.add_argument("olambda_regime_result_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--correlation-draws", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    olambda, atlas_contexts, gep_contexts = selected_contexts(args)
    manifest_rows = []
    for context in [olambda, *atlas_contexts, *gep_contexts]:
        manifest_rows.append({
            key: value for key, value in context.items() if key != "membership"
        })
    pd.DataFrame(manifest_rows).to_csv(args.output_dir / "context_manifest.csv", index=False)
    print(
        f"Selected contexts: {len(atlas_contexts)} atlas cell types | "
        f"{len(gep_contexts)} TCAT GEPs | O-lambda",
        flush=True,
    )

    started = time.time()
    rows = []
    analyze_field(
        "atlas_cell_type", args.atlas_particles_npz,
        args.atlas_continuous_results_csv, "ATLAS_proportion_",
        [olambda, *atlas_contexts, *gep_contexts], args, rows, started,
    )
    analyze_field(
        "TCAT_GEP", args.tcat_particles_npz,
        args.tcat_continuous_results_csv, "TCAT_GEP_",
        atlas_contexts, args, rows, started,
    )

    results = pd.DataFrame(rows)
    for (field, geometry), indices in results.groupby(["field", "geometry"]).groups.items():
        indices = list(indices)
        results.loc[indices, "joint_FDR"] = benjamini_hochberg(
            results.loc[indices, "joint_permutation_p"]
        )
        results.loc[indices, "joint_rank"] = results.loc[
            indices, "joint_oriented_z"
        ].rank(ascending=False, method="min")
        for metric in METRICS:
            results.loc[indices, f"FDR__{metric}"] = benjamini_hochberg(
                results.loc[indices, f"p__{metric}"]
            )
    results["joint_rank"] = results.joint_rank.astype(int)
    results.to_csv(
        args.output_dir / "cross_system_predictive_field_results.csv", index=False
    )

    manifest = {
        "fields": ["atlas_cell_type", "TCAT_GEP"],
        "metrics": METRICS,
        "context_selection": "95% HR interval excludes one in continuous or smooth-transition model",
        "patient_uncertainty": "particle sampling and Bayesian patient bootstrap",
        "geometries": ["global", "dataset_adjusted"],
        "permutation": "adverse-regime probabilities shuffled within dataset",
        "correlation_draws": args.correlation_draws,
        "permutations": args.permutations,
        "random_state": args.random_state,
        "runtime_minutes": (time.time() - started) / 60,
    }
    (args.output_dir / "cross_system_predictive_field_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(
        "\nResults:", args.output_dir / "cross_system_predictive_field_results.csv",
        flush=True,
    )


if __name__ == "__main__":
    main()
