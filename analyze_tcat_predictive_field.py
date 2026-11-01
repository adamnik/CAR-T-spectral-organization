# Regime-conditioned organization of the TCAT GEP clinical field

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_io import safe_name
from predictive_field import METRICS, benjamini_hochberg, dataset_residuals
from predictive_field import field_contrast, permutation_summary, permute_within_dataset
from predictive_field import weighted_correlation


SCORE_CONTEXTS = ["TCAT_SCORE_ASA_CLR", "TCAT_SCORE_Proliferation_CLR"]


def load_regime_memberships(regime_dir):
    memberships = {}
    for path in sorted((regime_dir / "individual").glob("*/patient_regime_probabilities.csv")):
        result_path = path.parent / "model_result.csv"
        if not result_path.exists():
            continue
        result = pd.read_csv(result_path).iloc[0]
        feature = str(result.feature)
        membership = pd.read_csv(path)
        memberships[feature] = {
            "result": result,
            "membership": membership,
            "path": path,
        }
    return memberships


def selected_contexts(continuous_results, regime_results, regime_memberships,
                      olambda_membership, olambda_result):
    continuous = pd.read_csv(continuous_results)
    regimes = pd.read_csv(regime_results)
    continuous_hit = continuous.HR_q025.gt(1) | continuous.HR_q975.lt(1)
    continuous_significant = set(continuous.loc[continuous_hit, 'feature'].astype(str))
    regime_hit = regimes.HR_q025.gt(1) | regimes.HR_q975.lt(1)
    regime_significant = set(regimes.loc[regime_hit, 'feature'].astype(str))
    selected = sorted(continuous_significant | regime_significant | set(SCORE_CONTEXTS))
    selected = [
        feature for feature in selected
        if feature.startswith("TCAT_GEP_") or feature in SCORE_CONTEXTS
    ]
    missing = sorted(set(selected).difference(regime_memberships))
    if missing:
        raise FileNotFoundError(
            "Missing completed Bayesian regime memberships for: " + ", ".join(missing)
        )

    olambda_model = pd.read_csv(olambda_result).iloc[0]
    contexts = [{
        "context": "O_lambda",
        "context_kind": "system",
        "adverse_high": float(olambda_model.HR_median) >= 1,
        "HR_median": float(olambda_model.HR_median),
        "HR_q025": float(olambda_model.HR_q025),
        "HR_q975": float(olambda_model.HR_q975),
        "membership": pd.read_csv(olambda_membership),
        "membership_path": str(olambda_membership),
        "selected_by_continuous": True,
        "selected_by_regime": True,
    }]
    for feature in selected:
        saved = regime_memberships[feature]
        result = saved["result"]
        contexts.append({
            "context": feature,
            "context_kind": "GEP" if feature.startswith("TCAT_GEP_") else "score",
            "adverse_high": float(result.HR_median) >= 1,
            "HR_median": float(result.HR_median),
            "HR_q025": float(result.HR_q025),
            "HR_q975": float(result.HR_q975),
            "membership": saved["membership"],
            "membership_path": str(saved["path"]),
            "selected_by_continuous": feature in continuous_significant,
            "selected_by_regime": feature in regime_significant,
        })
    return contexts, continuous


def align_analysis(tcat_path, contexts, continuous):
    archive = np.load(tcat_path, allow_pickle=False)
    feature_names = archive["feature_names"].astype(str)
    gep_names = [feature for feature in feature_names if feature.startswith("TCAT_GEP_")]
    missing_effects = sorted(set(gep_names).difference(continuous.feature.astype(str)))
    if missing_effects:
        raise ValueError("Continuous TCAT results are missing GEPs: " + ", ".join(missing_effects))

    patients = pd.DataFrame({
        "Dataset": archive["datasets"].astype(str),
        "sample_id": archive["patient_ids"].astype(str),
        "particle_row": np.arange(len(archive["patient_ids"])),
    })
    membership_columns = []
    for number, context in enumerate(contexts):
        column = f"membership_{number:03d}"
        table = context['membership'][[
            'Dataset', 'sample_id', 'posterior_mean_probability_high'
        ]]
        table = table.rename(columns={'posterior_mean_probability_high': column})
        patients = patients.merge(
            table, on=["Dataset", "sample_id"], how="inner", validate="one_to_one"
        )
        membership_columns.append(column)
    patients = patients.sort_values(["Dataset", "sample_id"]).reset_index(drop=True)
    if patients.empty:
        raise ValueError("No patients are shared by the TCAT particles and all regime models.")

    rows = patients.particle_row.to_numpy(int)
    lookup = {feature: index for index, feature in enumerate(feature_names)}
    gep_indices = np.asarray([lookup[feature] for feature in gep_names])
    values = np.asarray(archive["values"][rows][:, :, gep_indices], dtype=float)
    particle_weights = np.asarray(archive["particle_weights"][rows], dtype=float)
    particle_weights /= particle_weights.sum(axis=1, keepdims=True)
    # Fixed clinical field from the continuous survival fits
    effects = continuous.set_index('feature').loc[gep_names, 'HR_median']
    beta = np.log(effects.to_numpy(float))

    for number, context in enumerate(contexts):
        membership = patients[membership_columns[number]].to_numpy(float)
        if not context["adverse_high"]:
            membership = 1 - membership
        context["adverse_membership"] = membership
    return patients, np.asarray(gep_names), values, particle_weights, beta


def posterior_states(values, particle_weights, datasets, draws, random_state):
    rng = np.random.default_rng(random_state)
    n_patients, n_particles, n_features = values.shape
    raw = np.empty((draws, n_patients, n_features), dtype=np.float32)
    adjusted = np.empty_like(raw)
    bootstrap_weights = np.empty((draws, n_patients), dtype=np.float32)
    for draw in range(draws):
        particle_index = []
        for patient in range(n_patients):
            particle_index.append(rng.choice(n_particles, p=particle_weights[patient]))
        particle_index = np.asarray(particle_index)
        weights = rng.dirichlet(np.ones(n_patients))
        state = values[np.arange(n_patients), particle_index]
        raw[draw] = state
        adjusted[draw] = dataset_residuals(state, weights, datasets)
        bootstrap_weights[draw] = weights
    return {"global": raw, "dataset_adjusted": adjusted}, bootstrap_weights


def posterior_median_correlations(states, bootstrap_weights, membership, indices):
    n_draws, _, n_features = states.shape
    high = np.empty((n_draws, n_features, n_features), dtype=np.float32)
    low = np.empty_like(high)
    for draw in range(n_draws):
        weights = bootstrap_weights[draw, indices]
        values = states[draw, indices]
        high_weights = weights * membership
        high[draw] = weighted_correlation(values, high_weights)
        low_weights = weights * (1 - membership)
        low[draw] = weighted_correlation(values, low_weights)
    return np.median(high, axis=0), np.median(low, axis=0)


def context_test(states, bootstrap_weights, membership, beta, graph_indices,
                 dataset_indices, permutations, random_state):
    patient_indices = np.arange(states.shape[1])
    high, low = posterior_median_correlations(
        states[:, :, graph_indices], bootstrap_weights, membership, patient_indices
    )
    observed = field_contrast(high, low, beta)
    rng = np.random.default_rng(random_state)
    null = np.empty((permutations, len(METRICS)), dtype=float)
    for permutation in range(permutations):
        permuted = permute_within_dataset(membership, dataset_indices, rng)
        permuted_high, permuted_low = posterior_median_correlations(
            states[:, :, graph_indices], bootstrap_weights, permuted, patient_indices
        )
        null[permutation] = field_contrast(permuted_high, permuted_low, beta)
    z, p, joint_z, joint_p, joint_null = permutation_summary(observed, null)
    return high, low, observed, z, p, joint_z, joint_p, null, joint_null


def olambda_leave_one_dataset_out(states, bootstrap_weights, membership, beta,
                                  datasets, permutations, random_state):
    rows = []
    unique_datasets = sorted(np.unique(datasets))
    for number, omitted in enumerate(unique_datasets, start=1):
        keep = np.flatnonzero(datasets != omitted)
        reduced_datasets = datasets[keep]
        groups = [
            np.flatnonzero(reduced_datasets == dataset)
            for dataset in np.unique(reduced_datasets)
        ]
        reduced_membership = membership[keep]
        high, low = posterior_median_correlations(
            states, bootstrap_weights, reduced_membership, keep
        )
        observed = field_contrast(high, low, beta)
        rng = np.random.default_rng(random_state + number)
        null = np.empty((permutations, len(METRICS)))
        for permutation in range(permutations):
            permuted = permute_within_dataset(reduced_membership, groups, rng)
            permuted_high, permuted_low = posterior_median_correlations(
                states, bootstrap_weights, permuted, keep
            )
            null[permutation] = field_contrast(permuted_high, permuted_low, beta)
        z, p, joint_z, joint_p, _ = permutation_summary(observed, null)
        row = {
            "omitted_dataset": omitted,
            "patients": len(keep),
            "effective_adverse_patients": float(reduced_membership.sum()),
            "joint_oriented_z": joint_z,
            "joint_permutation_p": joint_p,
        }
        for index, metric in enumerate(METRICS):
            row[f"contrast__{metric}"] = observed[index]
            row[f"z__{metric}"] = z[index]
            row[f"p__{metric}"] = p[index]
        rows.append(row)
        print(
            f"LODO {number}/{len(unique_datasets)} | omitted {omitted} | "
            f"joint z={joint_z:.2f}, p={joint_p:.4g}", flush=True
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare regime-conditioned organization of the TCAT GEP clinical field."
    )
    parser.add_argument("tcat_particles_npz", type=Path)
    parser.add_argument("continuous_results_csv", type=Path)
    parser.add_argument("tcat_regime_results_csv", type=Path)
    parser.add_argument("tcat_regime_dir", type=Path)
    parser.add_argument("olambda_regime_membership_csv", type=Path)
    parser.add_argument("olambda_regime_result_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--correlation-draws", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--lodo-permutations", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--save-null", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    regime_memberships = load_regime_memberships(args.tcat_regime_dir)
    contexts, continuous = selected_contexts(
        args.continuous_results_csv, args.tcat_regime_results_csv, regime_memberships,
        args.olambda_regime_membership_csv, args.olambda_regime_result_csv,
    )
    patients, features, values, particle_weights, beta = align_analysis(
        args.tcat_particles_npz, contexts, continuous
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
        f"Analysis cohort: {len(patients):,} patients | {len(np.unique(datasets)):,} datasets | "
        f"{len(features):,} GEPs | {len(contexts):,} contexts",
        flush=True,
    )

    context_rows = []
    null_rows = []
    matrix_dir = args.output_dir / "observed_matrices"
    matrix_dir.mkdir(exist_ok=True)
    started = time.time()
    for context_number, context in enumerate(contexts, start=1):
        # A contextual GEP does not contribute to its own evaluated graph
        graph_indices = np.flatnonzero(features != context["context"])
        graph_features = features[graph_indices]
        graph_beta = beta[graph_indices]
        for geometry_number, geometry in enumerate(["dataset_adjusted", "global"]):
            result = context_test(
                states[geometry], bootstrap_weights,
                context["adverse_membership"], graph_beta, graph_indices,
                dataset_indices, args.permutations,
                args.random_state + context_number * 100 + geometry_number,
            )
            high, low, observed, z, p, joint_z, joint_p, null, joint_null = result
            row = {
                "context": context["context"],
                "context_kind": context["context_kind"],
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
            context_rows.append(row)

            name = safe_name(context["context"])
            pd.DataFrame(high, index=graph_features, columns=graph_features).to_csv(
                matrix_dir / f"{name}__{geometry}__adverse.csv"
            )
            pd.DataFrame(low, index=graph_features, columns=graph_features).to_csv(
                matrix_dir / f"{name}__{geometry}__favorable.csv"
            )
            if args.save_null:
                for permutation in range(args.permutations):
                    null_rows.append({
                        "context": context["context"],
                        "geometry": geometry,
                        "permutation": permutation + 1,
                        "joint_oriented_null_z": joint_null[permutation],
                        **{
                            metric: null[permutation, index]
                            for index, metric in enumerate(METRICS)
                        },
                    })
            print(
                f"[{context_number}/{len(contexts)}] {context['context']} | {geometry} | "
                f"joint z={joint_z:.2f}, p={joint_p:.4g}", flush=True
            )

    results = pd.DataFrame(context_rows)
    for geometry, indices in results.groupby("geometry").groups.items():
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
    results.to_csv(args.output_dir / "predictive_field_context_results.csv", index=False)
    if null_rows:
        pd.DataFrame(null_rows).to_csv(
            args.output_dir / "predictive_field_permutation_null.csv", index=False
        )

    manifest_rows = []
    for context in contexts:
        manifest_rows.append({
            key: value for key, value in context.items()
            if key not in {"membership", "adverse_membership"}
        })
    pd.DataFrame(manifest_rows).to_csv(args.output_dir / "context_manifest.csv", index=False)

    olambda = contexts[0]
    lodo_path = args.output_dir / "olambda_leave_one_dataset_out.csv"
    if lodo_path.exists():
        lodo_path.unlink()
    for geometry in ["dataset_adjusted", "global"]:
        lodo = olambda_leave_one_dataset_out(
            states[geometry], bootstrap_weights, olambda["adverse_membership"], beta,
            datasets, args.lodo_permutations,
            args.random_state + (10000 if geometry == "global" else 0),
        )
        lodo.insert(0, "geometry", geometry)
        lodo.to_csv(lodo_path, mode="a", header=not lodo_path.exists(), index=False)

    manifest = {
        "field": "TCAT GEPs",
        "metrics": METRICS,
        "context_definition": "posterior mean adverse-regime probability from each full Bayesian change-point model",
        "comparator_selection": "95% HR interval excludes one in continuous or change-point model, plus ASA and Proliferation",
        "patient_uncertainty": "joint TCAT particle sampling and Bayesian bootstrap",
        "primary_geometry": "weighted within-dataset mean residualization",
        "sensitivity_geometry": "global TCAT values",
        "permutation": "adverse-regime probabilities shuffled within dataset",
        "correlation_draws": args.correlation_draws,
        "permutations": args.permutations,
        "lodo_permutations": args.lodo_permutations,
        "random_state": args.random_state,
        "runtime_minutes": (time.time() - started) / 60,
    }
    (args.output_dir / 'predictive_field_manifest.json').write_text(json.dumps(manifest, indent=2))
    print("\nResults:", args.output_dir / "predictive_field_context_results.csv", flush=True)


if __name__ == "__main__":
    main()
