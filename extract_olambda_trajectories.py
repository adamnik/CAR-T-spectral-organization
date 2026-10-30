# Finite-depth O-lambda trajectories from one infusion-product AnnData dataset

import argparse
import gc
import hashlib
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from analysis_io import safe_name


RESULT_COLUMNS = [
    "Dataset",
    "sample_id",
    "trajectory_replicate",
    "target_depth",
    "n_cells_available",
    "depth_fraction",
    "schedule_rank",
    "is_full_reference",
    "cell_draw_seed",
    "nested_unique_cells",
    "gene_cell_fraction",
    "graph_estimator",
    "estimator_fingerprint",
    "source_path",
    "computed_at_utc",
    "O_lambda",
    "n_nodes_input",
    "n_nodes_used",
    "n_isolates_removed",
    "n_eigenvalues_near_one",
    "largest_eigenvalue",
    "lambda2_normalized_adjacency",
    "smallest_normalized_adjacency_eigenvalue",
    "zero_deviation_case",
    "k_min",
    "n_genes_after_filter",
    "graph_seconds_including_filter",
    "spectral_seconds",
    "total_seconds",
]


def atomic_csv(frame, path):
    temporary = Path(str(path) + ".partial")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(values, path):
    temporary = Path(str(path) + ".partial")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True))
    temporary.replace(path)


def stable_seed(random_state, *parts):
    token = "|".join(map(str, (random_state,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)


def normalized_id(value):
    return "".join(character for character in str(value).lower() if character.isalnum())


def canonical_sample_id(sample_id, dataset):
    sample_id = str(sample_id).strip()
    suffix = str(dataset).strip().lower()
    if normalized_id(sample_id).endswith(normalized_id(suffix)):
        return sample_id
    return f"{sample_id}_{suffix}"


def matrix_to_memory(matrix):
    matrix = matrix.to_memory() if hasattr(matrix, "to_memory") else matrix
    return matrix.tocsr() if sp.issparse(matrix) else np.asarray(matrix)


def upsert_csv(path, rows, keys):
    frame = pd.DataFrame(rows)
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True, sort=False)
    frame = frame.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    atomic_csv(frame, path)


def completed_depths(path):
    if not path.exists():
        return set()
    values = pd.to_numeric(pd.read_csv(path).target_depth, errors="coerce")
    return set(values.dropna().astype(int))


def adaptive_uniform_depths(n_cells, completed, target_count, minimum_depth,
                            anchors, random_state, patient, replicate):
    """Retain completed depths and fill the largest remaining depth gaps."""
    target_count = min(int(target_count), n_cells - minimum_depth)
    selected = {int(x) for x in completed if minimum_depth <= int(x) < n_cells}
    added = []

    for depth in anchors:
        depth = min(int(depth), n_cells - 1)
        if depth >= minimum_depth and depth not in selected:
            selected.add(depth)
            added.append(depth)

    rng = np.random.default_rng(
        stable_seed(random_state, "adaptive-depth-gap-fill", patient, replicate)
    )
    while len(selected) < max(target_count, len(selected)):
        ordered = np.asarray(sorted(selected | {n_cells}), dtype=int)
        gaps = np.diff(ordered)
        candidates = np.flatnonzero(gaps > 1)
        if not len(candidates):
            break
        largest = gaps[candidates].max()
        candidates = candidates[gaps[candidates] == largest]
        index = int(rng.choice(candidates))
        left, right = int(ordered[index]), int(ordered[index + 1])
        midpoint = (left + right) // 2
        if (right - left) % 2 and rng.random() < 0.5:
            midpoint += 1
        selected.add(midpoint)
        added.append(midpoint)

    ranks = {depth: rank for rank, depth in enumerate(added, start=1)}
    return sorted(selected), ranks


@torch.inference_mode()
def binary_plugin_mi(expression, device, epsilon=1e-12):
    binary = (expression > 0).astype(np.float32)
    if sp.issparse(binary):
        binary = binary.toarray()
    binary = torch.as_tensor(binary, dtype=torch.float32, device=device)

    n_cells = float(binary.shape[0])
    detected = binary.sum(dim=0)
    n11 = (binary.T @ binary).clamp_min_(0)
    n10 = (detected[:, None] - n11).clamp_min_(0)
    n01 = (detected[None, :] - n11).clamp_min_(0)
    n00 = (n_cells - n11 - n10 - n01).clamp_min_(0)
    p1 = detected / n_cells
    p0 = 1 - p1

    def term(counts, independent):
        joint = counts / n_cells
        ratio = (joint / independent.clamp_min(epsilon)).clamp_min(epsilon)
        return torch.where(joint > 0, joint * torch.log(ratio), 0)

    graph = (
        term(n11, p1[:, None] * p1[None, :])
        + term(n10, p1[:, None] * p0[None, :])
        + term(n01, p0[:, None] * p1[None, :])
        + term(n00, p0[:, None] * p0[None, :])
    )
    torch.nan_to_num_(graph, nan=0, posinf=math.log(2), neginf=0)
    graph.clamp_(0, math.log(2))
    graph.fill_diagonal_(0)
    return graph


@torch.inference_mode()
def spectral_organization(graph, spectral_epsilon=1e-7,
                          isolate_epsilon=1e-12, trivial_tolerance=1e-5):
    graph = (graph + graph.T) / 2
    graph.fill_diagonal_(0)
    negative_edges = int((graph < -1e-7).sum().item())
    if negative_edges:
        raise ValueError(f"{negative_edges} negative graph weights entered the spectral calculation.")
    graph.clamp_min_(0)

    degree = graph.sum(dim=1)
    keep = degree > isolate_epsilon
    n_nodes_input = int(graph.shape[0])
    n_nodes_used = int(keep.sum().item())
    if n_nodes_used < 2:
        raise ValueError("Fewer than two non-isolated nodes remain.")
    if n_nodes_used < n_nodes_input:
        graph = graph[keep][:, keep].contiguous()
        degree = graph.sum(dim=1)

    inverse_sqrt_degree = torch.rsqrt(degree.clamp_min(isolate_epsilon))
    graph.mul_(inverse_sqrt_degree[:, None])
    graph.mul_(inverse_sqrt_degree[None, :])
    eigenvalues = torch.linalg.eigvalsh((graph + graph.T) / 2)

    # Remove exactly one Perron mode and center the remaining relaxation times at one.
    nontrivial = eigenvalues[:-1]
    relative_time = (
        n_nodes_used / (n_nodes_used - 1)
    ) * (1 - nontrivial).clamp_min(spectral_epsilon).reciprocal()
    weights = (relative_time - 1).square()
    if float(weights.sum().item()) <= 1e-12:
        olambda = 1.0
        zero_deviation = True
    else:
        olambda = float((weights * relative_time).sum().div(weights.sum()).item())
        zero_deviation = False

    return {
        "O_lambda": olambda,
        "n_nodes_input": n_nodes_input,
        "n_nodes_used": n_nodes_used,
        "n_isolates_removed": n_nodes_input - n_nodes_used,
        "n_eigenvalues_near_one": int(
            (eigenvalues > 1 - trivial_tolerance).sum().item()
        ),
        "largest_eigenvalue": float(eigenvalues[-1].item()),
        "lambda2_normalized_adjacency": float(nontrivial[-1].item()),
        "smallest_normalized_adjacency_eigenvalue": float(eigenvalues[0].item()),
        "zero_deviation_case": zero_deviation,
    }


def trajectory_point(expression, order, depth, gene_fraction,
                     minimum_state_count, device):
    started = time.perf_counter()
    rows = np.sort(order[:depth])
    selected = expression[rows]
    detected = np.asarray((selected > 0).sum(axis=0)).ravel()
    k_min = max(minimum_state_count, int(math.ceil(gene_fraction * depth)))
    genes = np.flatnonzero((detected >= k_min) & (depth - detected >= k_min))
    if len(genes) < 2:
        raise ValueError(f"Only {len(genes)} genes pass the two-state filter.")

    graph = binary_plugin_mi(selected[:, genes], device)
    graph_seconds = time.perf_counter() - started
    spectral_started = time.perf_counter()
    values = spectral_organization(graph)
    spectral_seconds = time.perf_counter() - spectral_started
    del graph, selected
    gc.collect()
    torch.cuda.empty_cache()

    return {
        **values,
        "k_min": k_min,
        "n_genes_after_filter": len(genes),
        "graph_seconds_including_filter": graph_seconds,
        "spectral_seconds": spectral_seconds,
        "total_seconds": time.perf_counter() - started,
    }


def rebuild_outputs(parts_dir, results_path, summary_path, schedule):
    frames = [pd.read_csv(path) for path in sorted(parts_dir.glob("*.csv"))]
    if not frames:
        return
    keys = ["Dataset", "sample_id", "trajectory_replicate", "target_depth"]
    results = pd.concat(frames, ignore_index=True, sort=False)
    results = results.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    results = results.reindex(columns=RESULT_COLUMNS)
    atomic_csv(results, results_path)

    summary = results.groupby(
        ["Dataset", "sample_id", "trajectory_replicate"], as_index=False
    ).agg(
        n_cells_available=("n_cells_available", "max"),
        points_completed=("target_depth", "nunique"),
        minimum_depth=("target_depth", "min"),
        maximum_depth=("target_depth", "max"),
        O_lambda_min=("O_lambda", "min"),
        O_lambda_max=("O_lambda", "max"),
    )
    full = results.loc[
        results.is_full_reference.astype(str).str.lower().eq("true"),
        ["Dataset", "sample_id", "trajectory_replicate", "O_lambda"],
    ].rename(columns={"O_lambda": "O_lambda_full"})
    summary = summary.merge(
        full.drop_duplicates(keys[:3], keep="last"),
        on=keys[:3], how="left", validate="one_to_one",
    )
    requested = schedule.groupby(keys[:3]).size().rename("points_requested").reset_index()
    summary = summary.merge(requested, on=keys[:3], how="outer")
    summary["completion_fraction"] = summary.points_completed / summary.points_requested
    atomic_csv(summary, summary_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract resumable finite-depth O-lambda trajectories from one infusion AnnData file."
    )
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--sample-key", default="sample_id")
    parser.add_argument("--layer", default=None, help="Count layer; default uses adata.X.")
    parser.add_argument("--minimum-patient-cells", type=int, default=200)
    parser.add_argument("--minimum-depth", type=int, default=50)
    parser.add_argument("--point-multiplier", type=float, default=2.0)
    parser.add_argument("--gene-cell-fraction", type=float, default=0.002)
    parser.add_argument("--minimum-state-count", type=int, default=1)
    parser.add_argument("--replicate", type=int, action="append", default=None)
    parser.add_argument("--patient", action="append", default=None)
    parser.add_argument("--maximum-patients", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=20261024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the complete MI graphs.")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output_dir / "trajectory_parts"
    parts_dir.mkdir(exist_ok=True)
    results_path = args.output_dir / "olambda_trajectory_points.csv"
    summary_path = args.output_dir / "patient_trajectory_summary.csv"
    schedule_path = args.output_dir / "trajectory_schedule.csv"
    errors_path = args.output_dir / "errors.csv"
    config_path = args.output_dir / "estimator_configuration.json"

    estimator = {
        "graph_estimator": "raw binary plug-in mutual information",
        "spectral_statistic": "exact-center O_lambda with one Perron mode removed",
        "gene_cell_fraction": args.gene_cell_fraction,
        "minimum_state_count": args.minimum_state_count,
        "layer": args.layer,
    }
    fingerprint = hashlib.sha256(
        json.dumps(estimator, sort_keys=True).encode()
    ).hexdigest()
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing["estimator_fingerprint"] != fingerprint:
            raise ValueError("Estimator settings differ from this output directory.")
    else:
        atomic_json({"estimator_fingerprint": fingerprint, **estimator}, config_path)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Input:", args.input_h5ad, flush=True)
    print("Output:", args.output_dir, flush=True)

    adata = ad.read_h5ad(args.input_h5ad, backed="r")
    try:
        if args.sample_key not in adata.obs:
            raise KeyError(f"Missing sample column {args.sample_key!r}.")
        if "timepoint" in adata.obs:
            timepoints = adata.obs.timepoint.dropna().astype(str).str.lower().unique()
            if len(timepoints) != 1 or timepoints[0] != "infusion":
                raise ValueError("The input AnnData must contain infusion cells only.")

        samples = adata.obs[args.sample_key].astype(str).map(
            lambda value: canonical_sample_id(value, args.dataset_name)
        ).to_numpy()
        patients = []
        for patient in sorted(pd.unique(samples)):
            rows = np.flatnonzero(samples == patient).astype(int)
            if len(rows) >= args.minimum_patient_cells:
                patients.append((patient, rows))
        if args.patient is not None:
            requested = set(args.patient)
            patients = [(patient, rows) for patient, rows in patients if patient in requested]
        patients.sort(key=lambda item: (-len(item[1]), item[0]))
        if args.maximum_patients is not None:
            patients = patients[:args.maximum_patients]
        if not patients:
            raise ValueError("No patients satisfy the requested cell threshold.")

        manifest = pd.DataFrame([
            {
                "Dataset": args.dataset_name,
                "sample_id": patient,
                "n_cells": len(rows),
                "source_path": str(args.input_h5ad),
            }
            for patient, rows in patients
        ])
        atomic_csv(manifest, args.output_dir / "patient_manifest.csv")

        replicates = args.replicate or [0]
        anchors = [args.minimum_depth, 100, 150, 200]
        schedule_rows = []
        for patient, rows in patients:
            for replicate in replicates:
                part_path = parts_dir / f"{safe_name(args.dataset_name)}__{safe_name(patient)}__r{replicate}.csv"
                done = completed_depths(part_path)
                count = math.ceil(args.point_multiplier * math.sqrt(len(rows)))
                depths, ranks = adaptive_uniform_depths(
                    len(rows), done, count, args.minimum_depth, anchors,
                    args.random_state, patient, replicate,
                )
                for depth in depths:
                    schedule_rows.append({
                        "Dataset": args.dataset_name,
                        "sample_id": patient,
                        "trajectory_replicate": replicate,
                        "target_depth": depth,
                        "schedule_rank": ranks.get(depth, 0),
                        "is_full_reference": False,
                        "n_cells_available": len(rows),
                    })
                schedule_rows.append({
                    "Dataset": args.dataset_name,
                    "sample_id": patient,
                    "trajectory_replicate": replicate,
                    "target_depth": len(rows),
                    "schedule_rank": 0,
                    "is_full_reference": True,
                    "n_cells_available": len(rows),
                })
        schedule = pd.DataFrame(schedule_rows).drop_duplicates(
            ["Dataset", "sample_id", "trajectory_replicate", "target_depth"]
        )
        atomic_csv(schedule.sort_values(["sample_id", "trajectory_replicate", "target_depth"]), schedule_path)

        source = adata.X if args.layer is None else adata.layers[args.layer]
        for patient_number, (patient, rows) in enumerate(patients, start=1):
            print("\n" + "=" * 100, flush=True)
            print(
                f"[{patient_number}/{len(patients)}] {args.dataset_name} | {patient} | "
                f"{len(rows):,} infusion cells",
                flush=True,
            )

            remaining = 0
            for replicate in replicates:
                part_path = parts_dir / f"{safe_name(args.dataset_name)}__{safe_name(patient)}__r{replicate}.csv"
                requested_depths = set(
                    schedule.loc[
                        schedule.sample_id.eq(patient)
                        & schedule.trajectory_replicate.eq(replicate),
                        "target_depth",
                    ].astype(int)
                )
                remaining += len(requested_depths - completed_depths(part_path))
            if remaining == 0:
                print("  complete; no expression data loaded", flush=True)
                continue

            expression = matrix_to_memory(source[np.sort(rows), :])

            for replicate in replicates:
                part_path = parts_dir / f"{safe_name(args.dataset_name)}__{safe_name(patient)}__r{replicate}.csv"
                done = completed_depths(part_path)
                jobs = schedule.loc[
                    schedule.sample_id.eq(patient)
                    & schedule.trajectory_replicate.eq(replicate)
                ].sort_values(["is_full_reference", "target_depth"], ascending=[False, False])
                patient_key = f"{args.dataset_name}|{patient}"
                seed = stable_seed(
                    args.random_state, "cell-permutation", patient_key, replicate
                )
                order = np.random.default_rng(seed).permutation(len(rows))
                print(f"  replicate {replicate}: {len(jobs)} requested, {len(done)} completed", flush=True)

                for point_number, job in enumerate(jobs.itertuples(index=False), start=1):
                    depth = int(job.target_depth)
                    if depth in done:
                        continue
                    print(
                        f"    [{point_number}/{len(jobs)}] n={depth:,} "
                        f"full={bool(job.is_full_reference)}",
                        flush=True,
                    )
                    try:
                        values = trajectory_point(
                            expression, order, depth, args.gene_cell_fraction,
                            args.minimum_state_count, device,
                        )
                        result = {
                            "Dataset": args.dataset_name,
                            "sample_id": patient,
                            "trajectory_replicate": replicate,
                            "target_depth": depth,
                            "n_cells_available": len(rows),
                            "depth_fraction": depth / len(rows),
                            "schedule_rank": int(job.schedule_rank),
                            "is_full_reference": bool(job.is_full_reference),
                            "cell_draw_seed": seed,
                            "nested_unique_cells": True,
                            "gene_cell_fraction": args.gene_cell_fraction,
                            "graph_estimator": "raw_binary_plugin_MI",
                            "estimator_fingerprint": fingerprint,
                            "source_path": str(args.input_h5ad),
                            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
                            **values,
                        }
                        upsert_csv(
                            part_path, [result],
                            ["Dataset", "sample_id", "trajectory_replicate", "target_depth"],
                        )
                        done.add(depth)
                        print(
                            f"      O_lambda={values['O_lambda']:.7f} | "
                            f"genes={values['n_genes_after_filter']:,} | "
                            f"time={values['total_seconds'] / 60:.2f} min",
                            flush=True,
                        )
                    except Exception as error:
                        error_row = {
                            "Dataset": args.dataset_name,
                            "sample_id": patient,
                            "trajectory_replicate": replicate,
                            "target_depth": depth,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                        upsert_csv(
                            errors_path, [error_row],
                            ["Dataset", "sample_id", "trajectory_replicate", "target_depth"],
                        )
                        print(f"      ERROR {type(error).__name__}: {error}", flush=True)
                        gc.collect()
                        torch.cuda.empty_cache()

            del expression
            gc.collect()
            torch.cuda.empty_cache()
            rebuild_outputs(parts_dir, results_path, summary_path, schedule)

    finally:
        if adata.file:
            adata.file.close()

    rebuild_outputs(parts_dir, results_path, summary_path, schedule)
    print("\nTrajectory points:", results_path)
    print("Patient summary:", summary_path)


if __name__ == "__main__":
    main()
