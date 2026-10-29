"""Fit uncertainty-marginalized continuous O-lambda survival effects."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from survival_model import (
    atomic_csv,
    atomic_json,
    diagnostics,
    hierarchical_parameters,
    interval_loglikelihood,
    load_olambda_analysis,
    marginalize_particles,
    model_coordinates,
    sample_model,
    summarize_effects,
    survival_arrays,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit a continuous hierarchical survival model for inferred O-lambda."
    )
    parser.add_argument("asymptotes_csv", type=Path)
    parser.add_argument("outcomes_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-cells", type=int, default=1000)
    parser.add_argument("--n-particles", type=int, default=16)
    parser.add_argument("--draws", type=int, default=1500)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.minimum_cells < 1:
        raise ValueError("--minimum-cells must be positive.")
    if args.n_particles < 2:
        raise ValueError("--n-particles must be at least two.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, _, particles, particle_weights, scaling = load_olambda_analysis(
        args.asymptotes_csv,
        args.outcomes_csv,
        minimum_cells=args.minimum_cells,
        n_particles=args.n_particles,
    )
    arrays = survival_arrays(data)
    print(
        f"Analysis cohort: {len(data):,} patients | "
        f"{int(data.event_observed.sum()):,} events | "
        f"{len(arrays['datasets']):,} datasets",
        flush=True,
    )

    with pm.Model(coords=model_coordinates(data, arrays)):
        baseline_log_rate, patient_effect = hierarchical_parameters(arrays)
        hazard = pt.exp(
            baseline_log_rate[:, None, :]
            + patient_effect[:, None, None]
            * pt.as_tensor_variable(particles)[:, :, None]
        )
        conditional_loglikelihood = interval_loglikelihood(hazard, arrays)
        patient_loglikelihood = marginalize_particles(
            conditional_loglikelihood,
            particle_weights,
        )
        pm.Deterministic(
            "patient_loglikelihood",
            patient_loglikelihood,
            dims="observation",
        )
        pm.Potential("survival_loglikelihood", pt.sum(patient_loglikelihood))

        started = time.time()
        idata = sample_model(
            args.draws,
            args.tune,
            args.chains,
            args.cores,
            args.target_accept,
            args.random_state,
        )
        elapsed = time.time() - started

    variables = [
        "global_effect",
        "dataset_effect_sd",
        "dataset_effect",
        "baseline_log_rate_initial",
        "baseline_smoothness",
        "baseline_innovation",
    ]
    posterior_summary, diagnostic_values = diagnostics(idata, variables)
    result, dataset_results = summarize_effects(idata, arrays["datasets"])
    result.update(diagnostic_values)
    result.update(
        {
            "effect_unit": "per global SD of inferred O_lambda_inf central estimates",
            "n_particles": args.n_particles,
            "minimum_cells": args.minimum_cells,
            "patients": len(data),
            "events": int(data.event_observed.sum()),
            "datasets": len(arrays["datasets"]),
            "wall_time_seconds": elapsed,
        }
    )

    atomic_csv(data, args.output_dir / "analysis_cohort.csv")
    atomic_csv(pd.DataFrame([result]), args.output_dir / "model_result.csv")
    atomic_csv(dataset_results, args.output_dir / "dataset_effects.csv")
    posterior_summary.to_csv(args.output_dir / "posterior_summary.csv")
    idata.to_netcdf(args.output_dir / "posterior.nc")
    atomic_json(
        {
            "model": "continuous uncertainty-marginalized O_lambda survival",
            "minimum_cells": args.minimum_cells,
            "n_particles": args.n_particles,
            "particle_definition": (
                "equal-probability midpoint particles from each patient's "
                "Student-t4 posterior in log(O_lambda_inf - 1)"
            ),
            "particle_scaling": scaling,
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "target_accept": args.target_accept,
            "random_state": args.random_state,
        },
        args.output_dir / "run_manifest.json",
    )
    print(pd.DataFrame([result]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
