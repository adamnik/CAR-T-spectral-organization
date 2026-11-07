# Fixed-depth O-lambda sensitivity analyses from processed infusion AnnData datasets

import argparse
import gc
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from analysis_io import safe_name
from extract_olambda_trajectories import atomic_csv, atomic_json, binary_plugin_mi
from extract_olambda_trajectories import matrix_to_memory, spectral_organization
from extract_olambda_trajectories import stable_seed, upsert_csv


EXPERIMENTS = ["patient-bootstrap", "pooled-synthetic"]


def integer_quotas(proportions, total):
    proportions = pd.Series(proportions, dtype=float)
    proportions = proportions.loc[proportions > 0]
    proportions = proportions / proportions.sum()
    expected = proportions * int(total)
    quotas = np.floor(expected).astype(int)
    remainder = int(total) - int(quotas.sum())
    if remainder:
        order = (expected - quotas).sort_values(ascending=False, kind="stable").index
        quotas.loc[order[:remainder]] += 1
    return quotas


def equal_quotas(values, total, rng):
    values = np.asarray(sorted(values), dtype=object)
    quotas = pd.Series(int(total) // len(values), index=values, dtype=int)
    remainder = int(total) - int(quotas.sum())
    if remainder:
        quotas.loc[rng.permutation(values)[:remainder]] += 1
    return quotas


def bayesian_weights(n_cells, rng):
    weights = rng.exponential(size=int(n_cells))
    weights /= weights.sum()
    return weights


def draw_block(indices, count, rng):
    indices = np.asarray(indices, dtype=int)
    weights = bayesian_weights(len(indices), rng)
    drawn = rng.choice(indices, size=int(count), replace=True, p=weights)
    return drawn, weights


def read_rows(matrix, rows):
    rows = np.asarray(rows, dtype=int)
    # Read each cell once, then restore repeats from sampling with replacement
    unique, inverse = np.unique(rows, return_inverse=True)
    values = matrix_to_memory(matrix[unique, :])
    return values[inverse]


def native_gene_indices(expression, gene_fraction, minimum_state_count):
    n_cells = expression.shape[0]
    detected = np.asarray((expression > 0).sum(axis=0)).ravel()
    k_min = max(int(minimum_state_count), int(math.ceil(gene_fraction * n_cells)))
    genes = np.flatnonzero((detected >= k_min) & (n_cells - detected >= k_min))
    if len(genes) < 2:
        raise ValueError(f"Only {len(genes)} genes pass the native two-state filter.")
    return genes, k_min


def weight_summary(blocks, target_cells):
    squared = 0.0
    maximum = 0.0
    block_ess = []
    for count, weights in blocks:
        mass = count / target_cells
        squared += mass * mass * float(np.square(weights).sum())
        maximum = max(maximum, mass * float(weights.max()))
        block_ess.append(1 / float(np.square(weights).sum()))
    return {
        "cell_weight_ess": 1 / squared,
        "minimum_block_ess": min(block_ess),
        "mean_block_ess": float(np.mean(block_ess)),
        "maximum_cell_weight": maximum,
    }


def patient_bootstrap(rows, target_cells, rng):
    drawn, weights = draw_block(np.arange(len(rows)), target_cells, rng)
    return rows[drawn], weight_summary([(target_cells, weights)], target_cells), {}


def pooled_synthetic(rows_by_patient, labels, patient, target, target_cells, rng):
    quotas = integer_quotas(target, target_cells)
    selected = []
    blocks = []
    donor_counts = {}

    for cell_type, count in quotas.items():
        donor_rows = {}
        for donor, rows in rows_by_patient.items():
            if donor == patient:
                continue
            cells = rows[labels[rows] == cell_type]
            if len(cells):
                donor_rows[donor] = cells
        if not donor_rows:
            raise ValueError(
                f"No leave-one-patient-out source cells exist for cell type {cell_type!r}."
            )
        # Equal donor contributions within each cell type
        donor_quotas = equal_quotas(donor_rows, count, rng)
        for donor, donor_count in donor_quotas.items():
            if donor_count == 0:
                continue
            drawn, weights = draw_block(donor_rows[donor], donor_count, rng)
            selected.append(drawn)
            blocks.append((donor_count, weights))
            donor_counts[donor] = donor_counts.get(donor, 0) + int(donor_count)

    return (np.concatenate(selected), weight_summary(blocks, target_cells), donor_counts)


def composition_rows(dataset, patient, replicate, experiment, target, realized,
                     native, target_cells):
    cell_types = sorted(set(target) | set(realized) | set(native))
    rows = []
    for cell_type in cell_types:
        rows.append({
            "experiment": experiment,
            "Dataset": dataset,
            "sample_id": patient,
            "replicate": replicate,
            "cell_type": cell_type,
            "native_proportion": native.get(cell_type, 0.0),
            "target_proportion": target.get(cell_type, 0.0),
            "realized_proportion": realized.get(cell_type, 0.0),
            "realized_cells": int(round(realized.get(cell_type, 0.0) * target_cells)),
        })
    return rows


def metadata_scan(manifest, sample_key, cell_type_key, minimum_patient_cells):
    patients = []
    compositions = []
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        print(f"[{number}/{len(manifest)}] scanning {row.Dataset}: {row.h5ad_path}", flush=True)
        data = ad.read_h5ad(row.h5ad_path, backed="r")
        try:
            if sample_key not in data.obs:
                raise KeyError(f"Missing sample column {sample_key!r} in {row.h5ad_path}.")
            if cell_type_key not in data.obs:
                raise KeyError(f"Missing cell-type column {cell_type_key!r} in {row.h5ad_path}.")
            if "timepoint" in data.obs:
                timepoints = data.obs.timepoint.dropna().astype(str).str.lower().unique()
                if len(timepoints) != 1 or timepoints[0] != "infusion":
                    raise ValueError(f"{row.h5ad_path} must contain infusion cells only.")

            samples = data.obs[sample_key].astype(str)
            labels = data.obs[cell_type_key].astype(str)
            if data.obs[cell_type_key].isna().any():
                raise ValueError(f"Cell-type column {cell_type_key!r} contains missing values.")
            for patient in sorted(samples.unique()):
                keep = samples.eq(patient).to_numpy()
                n_cells = int(keep.sum())
                if n_cells < minimum_patient_cells:
                    continue
                patients.append({
                    "Dataset": row.Dataset,
                    "sample_id": patient,
                    "n_cells": n_cells,
                    "h5ad_path": str(row.h5ad_path),
                })
                counts = labels.loc[keep].value_counts()
                for cell_type, count in counts.items():
                    compositions.append({
                        "Dataset": row.Dataset,
                        "sample_id": patient,
                        "cell_type": cell_type,
                        "native_cells": int(count),
                        "native_proportion": count / n_cells,
                    })
        finally:
            if data.file:
                data.file.close()
    return pd.DataFrame(patients), pd.DataFrame(compositions)


def save_graph(path, graph, gene_indices, gene_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    with temporary.open("wb") as file:
        np.savez_compressed(
            file,
            W=graph.detach().cpu().numpy().astype(np.float32, copy=False),
            gene_indices=np.asarray(gene_indices, dtype=np.int64),
            gene_names=np.asarray(gene_names, dtype="U"),
        )
    temporary.replace(path)


@torch.inference_mode()
def calculate_olambda(expression, gene_indices, gene_names, device, graph_path=None):
    started = time.perf_counter()
    graph = binary_plugin_mi(expression[:, gene_indices], device)
    graph_seconds = time.perf_counter() - started
    if graph_path is not None:
        save_graph(graph_path, graph, gene_indices, gene_names)
    spectral_started = time.perf_counter()
    values = spectral_organization(graph)
    values["graph_seconds"] = graph_seconds
    values["spectral_seconds"] = time.perf_counter() - spectral_started
    values["total_seconds"] = time.perf_counter() - started
    del graph
    gc.collect()
    torch.cuda.empty_cache()
    return values


def read_parts(directory):
    frames = [pd.read_csv(path) for path in sorted(directory.glob("*.csv"))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def rebuild_outputs(output_dir):
    outputs = [
        ("replicate_parts", "olambda_sensitivity_replicates.csv",
         ["Dataset", "sample_id", "replicate"]),
        ("composition_parts", "cell_type_composition_audit.csv",
         ["Dataset", "sample_id", "replicate", "cell_type"]),
        ("donor_parts", "donor_balance_audit.csv",
         ["Dataset", "sample_id", "replicate", "donor"]),
    ]
    for directory, filename, keys in outputs:
        frame = read_parts(output_dir / directory)
        if len(frame):
            frame = frame.sort_values(keys).reset_index(drop=True)
            atomic_csv(frame, output_dir / filename)


def main():
    parser = argparse.ArgumentParser(
        description="Run fixed-depth O-lambda bootstrap and synthetic-product sensitivity analyses."
    )
    parser.add_argument("dataset_manifest", type=Path,
                        help="CSV with Dataset and h5ad_path columns.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--sample-key", default="sample_id")
    parser.add_argument("--cell-type-key", default="TCAT_SCORE_Multinomial_Label")
    parser.add_argument("--layer", default=None, help="Count layer; default uses adata.X.")
    parser.add_argument("--minimum-patient-cells", type=int, default=1000)
    parser.add_argument("--target-cells", type=int, default=10000)
    parser.add_argument("--replicates", type=int, default=25)
    parser.add_argument("--gene-cell-fraction", type=float, default=0.002)
    parser.add_argument("--minimum-state-count", type=int, default=1)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--patient", action="append", default=None)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--save-graphs", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the complete MI graphs.")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False

    manifest = pd.read_csv(args.dataset_manifest)
    missing = {"Dataset", "h5ad_path"} - set(manifest.columns)
    if missing:
        raise ValueError(f"Dataset manifest is missing columns: {sorted(missing)}")
    manifest = manifest[["Dataset", "h5ad_path"]].drop_duplicates().copy()
    manifest["h5ad_path"] = manifest.h5ad_path.map(Path)
    if args.dataset is not None:
        manifest = manifest.loc[manifest.Dataset.isin(args.dataset)].copy()
    if not len(manifest):
        raise ValueError("The dataset manifest is empty after filtering.")
    if manifest.Dataset.duplicated().any():
        raise ValueError("The dataset manifest must contain one h5ad path per dataset.")
    for path in manifest.h5ad_path:
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output_dir / "replicate_parts"
    composition_dir = args.output_dir / "composition_parts"
    donor_dir = args.output_dir / "donor_parts"
    graph_dir = args.output_dir / "graphs_by_dataset"
    for directory in [parts_dir, composition_dir, donor_dir]:
        directory.mkdir(exist_ok=True)
    if args.save_graphs:
        graph_dir.mkdir(exist_ok=True)
    errors_path = args.output_dir / "errors.csv"

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Experiment:", args.experiment, flush=True)
    print("Target cells:", f"{args.target_cells:,}", flush=True)
    print("Replicates:", args.replicates, flush=True)

    patients, native_composition = metadata_scan(
        manifest, args.sample_key, args.cell_type_key, args.minimum_patient_cells
    )
    all_patients = patients.copy()
    if args.patient is not None:
        patients = patients.loc[patients.sample_id.isin(args.patient)].copy()
    if not len(patients):
        raise ValueError("No patients satisfy the requested cell threshold.")
    atomic_csv(patients, args.output_dir / "patient_manifest.csv")
    atomic_csv(all_patients, args.output_dir / "eligible_patient_manifest.csv")
    atomic_csv(native_composition, args.output_dir / "native_cell_type_composition.csv")

    configuration = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "minimum_patient_cells": args.minimum_patient_cells,
        "target_cells": args.target_cells,
        "replicates": args.replicates,
        "gene_cell_fraction": args.gene_cell_fraction,
        "minimum_state_count": args.minimum_state_count,
        "cell_type_key": args.cell_type_key,
        "graph_estimator": "raw binary plug-in mutual information",
        "spectral_statistic": "exact-center O_lambda with one Perron mode removed",
        "gene_universe": "target patient's native full-depth two-state gene set",
        "random_state": args.random_state,
        "save_graphs": args.save_graphs,
    }
    configuration_path = args.output_dir / "sensitivity_configuration.json"
    if configuration_path.exists():
        previous = json.loads(configuration_path.read_text())
        changing = [
            key for key in configuration
            if key not in {"created_at_utc", "replicates", "save_graphs"}
            and previous.get(key) != configuration[key]
        ]
        if changing:
            raise ValueError(f'Settings differ from this output directory: {changing}')
    atomic_json(configuration, configuration_path)

    total_jobs = len(patients) * args.replicates
    job_number = 0
    for dataset, dataset_patients in patients.groupby("Dataset", sort=False):
        path = manifest.loc[manifest.Dataset.eq(dataset), "h5ad_path"].iloc[0]
        data = ad.read_h5ad(path, backed="r")
        try:
            samples = data.obs[args.sample_key].astype(str).to_numpy()
            labels = data.obs[args.cell_type_key].astype(str).to_numpy()
            source = data.X if args.layer is None else data.layers[args.layer]
            rows_by_patient = {
                patient: np.flatnonzero(samples == patient).astype(int)
                for patient in sorted(pd.unique(samples))
            }

            for patient_number, patient_row in enumerate(
                dataset_patients.itertuples(index=False), start=1
            ):
                patient = patient_row.sample_id
                rows = rows_by_patient[patient]
                patient_labels = labels[rows]
                native_counts = pd.Series(patient_labels).value_counts()
                native_target = (native_counts / native_counts.sum()).to_dict()

                if args.experiment == "pooled-synthetic":
                    pool_blocks = [
                        donor_rows for donor, donor_rows in rows_by_patient.items()
                        if donor != patient
                    ]
                    pool_types = (
                        set(labels[np.concatenate(pool_blocks)]) if pool_blocks else set()
                    )
                    missing_types = sorted(set(native_target) - pool_types)
                else:
                    missing_types = []
                feasibility = {
                    "experiment": args.experiment,
                    "Dataset": dataset,
                    "sample_id": patient,
                    "feasible": not missing_types,
                    "missing_cell_types": json.dumps(missing_types),
                    "n_native_cell_types": len(native_target),
                    "n_source_donors": len(rows_by_patient) - 1,
                }
                upsert_csv(
                    args.output_dir / "feasibility_audit.csv", [feasibility],
                    ["experiment", "Dataset", "sample_id"],
                )
                if missing_types:
                    job_number += args.replicates
                    print(
                        f"\n{dataset} | {patient} | skipped; unavailable cell types "
                        f"{missing_types}",
                        flush=True,
                    )
                    continue

                pending = []
                for replicate in range(args.replicates):
                    name = f"{safe_name(dataset)}__{safe_name(patient)}__r{replicate}.csv"
                    graph_path = (
                        graph_dir / safe_name(dataset) /
                        f"{safe_name(patient)}__r{replicate}.npz"
                    )
                    complete = (parts_dir / name).exists()
                    if args.save_graphs:
                        complete = complete and graph_path.exists()
                    if not complete:
                        pending.append(replicate)
                job_number += args.replicates - len(pending)
                print("\n" + "=" * 108, flush=True)
                print(
                    f"{dataset} | {patient_number}/{len(dataset_patients)} | {patient} | "
                    f"native n={len(rows):,} | pending={len(pending)}",
                    flush=True,
                )
                if not pending:
                    continue

                patient_expression = matrix_to_memory(source[np.sort(rows), :])
                genes, native_k_min = native_gene_indices(
                    patient_expression, args.gene_cell_fraction,
                    args.minimum_state_count,
                )
                print(
                    f"  fixed native gene universe: {len(genes):,} genes "
                    f"(k_min={native_k_min})",
                    flush=True,
                )

                for replicate in pending:
                    job_number += 1
                    seed = stable_seed(
                        args.random_state, args.experiment, dataset, patient, replicate
                    )
                    rng = np.random.default_rng(seed)
                    name = f"{safe_name(dataset)}__{safe_name(patient)}__r{replicate}.csv"
                    graph_path = (
                        graph_dir / safe_name(dataset) /
                        f"{safe_name(patient)}__r{replicate}.npz"
                    )
                    print(
                        f"  [{job_number}/{total_jobs}] replicate {replicate + 1}/{args.replicates}",
                        flush=True,
                    )
                    started = time.perf_counter()
                    selected_expression = None
                    try:
                        if args.experiment == "patient-bootstrap":
                            selected_rows, weights, donor_counts = patient_bootstrap(
                                rows, args.target_cells, rng
                            )
                            target = native_target
                            selected_expression = read_rows(source, selected_rows)
                        else:
                            selected_rows, weights, donor_counts = pooled_synthetic(
                                rows_by_patient, labels, patient, native_target,
                                args.target_cells, rng,
                            )
                            target = native_target
                            selected_expression = read_rows(source, selected_rows)

                        selected_labels = labels[selected_rows]
                        realized_counts = pd.Series(selected_labels).value_counts()
                        realized = (realized_counts / args.target_cells).to_dict()
                        all_types = sorted(set(target) | set(realized))
                        differences = [
                            abs(realized.get(x, 0) - target.get(x, 0)) for x in all_types
                        ]
                        values = calculate_olambda(
                            selected_expression,
                            genes,
                            np.asarray(data.var_names.astype(str))[genes],
                            device,
                            graph_path if args.save_graphs else None,
                        )
                        unique_cells = len(np.unique(selected_rows))
                        donor_total = sum(donor_counts.values())
                        donor_fractions = (
                            np.asarray(list(donor_counts.values())) / donor_total
                            if donor_total else np.asarray([])
                        )
                        donor_entropy = (
                            float(-(donor_fractions * np.log(donor_fractions)).sum())
                            if len(donor_fractions) else np.nan
                        )
                        result = {
                            "experiment": args.experiment,
                            "Dataset": dataset,
                            "sample_id": patient,
                            "replicate": replicate,
                            "seed": seed,
                            "n_cells_native": len(rows),
                            "n_cells_target": args.target_cells,
                            "n_unique_source_cells": unique_cells,
                            "duplicate_fraction": 1 - unique_cells / args.target_cells,
                            **weights,
                            "cell_types_targeted": len(target),
                            "cell_type_L1_error": sum(differences),
                            "cell_type_max_error": max(differences),
                            "n_donors_available": len(rows_by_patient) - 1 if donor_total else 1,
                            "n_donors_used": len(donor_counts) if donor_total else 1,
                            "donor_entropy": donor_entropy,
                            "maximum_donor_fraction": (
                                float(donor_fractions.max()) if len(donor_fractions) else 1.0
                            ),
                            "gene_cell_fraction": args.gene_cell_fraction,
                            "native_k_min": native_k_min,
                            "n_genes_fixed": len(genes),
                            "source_path": str(path),
                            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
                            "mi_npz_path": str(graph_path) if args.save_graphs else "",
                            **values,
                        }
                        atomic_csv(pd.DataFrame([result]), parts_dir / name)
                        atomic_csv(pd.DataFrame(composition_rows(
                            dataset, patient, replicate, args.experiment,
                            target, realized, native_target, args.target_cells,
                        )), composition_dir / name)
                        if donor_counts:
                            donor_rows = [{
                                "experiment": args.experiment,
                                "Dataset": dataset,
                                "sample_id": patient,
                                "replicate": replicate,
                                "donor": donor,
                                "drawn_cells": count,
                                "drawn_fraction": count / args.target_cells,
                            } for donor, count in sorted(donor_counts.items())]
                            atomic_csv(pd.DataFrame(donor_rows), donor_dir / name)
                        print(
                            f"    O_lambda={values['O_lambda']:.7f} | "
                            f"nodes={values['n_nodes_used']:,} | "
                            f"unique source={unique_cells:,} | "
                            f"time={(time.perf_counter() - started) / 60:.2f} min",
                            flush=True,
                        )
                    except Exception as error:
                        error_row = pd.DataFrame([{
                            "experiment": args.experiment,
                            "Dataset": dataset,
                            "sample_id": patient,
                            "replicate": replicate,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }])
                        if errors_path.exists():
                            error_row = pd.concat(
                                [pd.read_csv(errors_path), error_row], ignore_index=True
                            ).drop_duplicates(
                                ["experiment", "Dataset", "sample_id", "replicate"],
                                keep="last",
                            )
                        atomic_csv(error_row, errors_path)
                        print(f"    ERROR {type(error).__name__}: {error}", flush=True)
                    finally:
                        if selected_expression is not None:
                            del selected_expression
                        gc.collect()
                        torch.cuda.empty_cache()

                del patient_expression
                gc.collect()
                torch.cuda.empty_cache()
                rebuild_outputs(args.output_dir)
        finally:
            if data.file:
                data.file.close()

    rebuild_outputs(args.output_dir)
    print("\nReplicates:", args.output_dir / "olambda_sensitivity_replicates.csv")
    print("Cell-type audit:", args.output_dir / "cell_type_composition_audit.csv")
    if args.experiment == "pooled-synthetic":
        print("Donor audit:", args.output_dir / "donor_balance_audit.csv")


if __name__ == "__main__":
    main()
