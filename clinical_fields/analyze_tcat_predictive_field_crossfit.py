import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared.analysis_io import safe_name
from clinical_fields.predictive_field import combine_folds, field_test, particle_profiles, posterior_states


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate O-lambda organization using effects fit without each dataset."
    )
    parser.add_argument("tcat_particles_npz", type=Path)
    parser.add_argument("olambda_regime_membership_csv", type=Path)
    parser.add_argument("crossfit_results_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--correlation-draws", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--minimum-effective-regime", type=float, default=3.0)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--feature-kind", choices=["gep", "cell_type"], default="gep"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)


    representation, prefix = ('TCAT_GEP', 'TCAT_GEP_') if args.feature_kind == 'gep' else (
        'atlas_cell_type', 'ATLAS_proportion_'
    )
    membership = pd.read_csv(args.olambda_regime_membership_csv,
                             dtype={'Dataset': str, 'sample_id': str, 'heldout_dataset': str})
    if not membership.Dataset.eq(membership.heldout_dataset).all():
        raise ValueError('Memberships must come from the fold excluding the patient’s dataset')
    patients, features, values, weights = particle_profiles(args.tcat_particles_npz, membership, prefix)
    datasets = patients.Dataset.to_numpy()
    states, bootstrap = posterior_states(values, weights, datasets, args.correlation_draws, args.random_state)
    hazards = pd.read_csv(args.crossfit_results_csv)
    support, results, nulls = [], [], {}
    bundle_dir = args.output_dir / 'compositional_sensitivity_bundles'
    for number, heldout in enumerate(sorted(set(datasets)), start=1):
        indices = np.flatnonzero(datasets == heldout)
        high = patients.loc[indices, 'posterior_mean_probability_high'].to_numpy(float)
        if not np.all((0 <= high) & (high <= 1)):
            raise ValueError(f'Invalid held-out membership in {heldout}')
        low = 1 - high
        # Use membership mass to decide which cohorts enter the comparison
        high_mass = high.sum()
        low_mass = low.sum()
        eligible = min(high_mass, low_mass) >= args.minimum_effective_regime
        support.append({
            'Dataset': heldout, 'n': len(indices), 'sum_low': low_mass, 'sum_high': high_mass,
            'n_eff_low': low_mass**2 / (low @ low) if low.any() else 0.,
            'n_eff_high': high_mass**2 / (high @ high) if high.any() else 0.,
            'eligible': eligible,
        })
        if not eligible:
            continue
        effects = hazards.loc[hazards.heldout_dataset.eq(heldout)].set_index('feature').loc[features]
        result, nulls[heldout] = field_test(
            states, bootstrap, high, indices, [np.arange(len(indices))], effects, [],
            bundle_dir / f'{safe_name(heldout)}__O_lambda.npz',
            {'representation': representation, 'context': 'O_lambda', 'scope': 'fully_heldout',
             'heldout_dataset': heldout, 'fold_number': number},
            args.permutations, args.random_state + number,
        )
        results.append(dict(result, representation=representation, heldout_dataset=heldout,
                            patients=len(indices)))
        print(heldout, result['joint_oriented_z'], flush=True)
    pd.DataFrame(support).to_csv(args.output_dir / 'table_s4.csv', index=False)
    folds = pd.DataFrame(results)
    folds.to_csv(args.output_dir / 'mean_crossfit_predictive_field_by_dataset.csv', index=False)
    if len(folds):
        combined = combine_folds(folds, nulls)
    else:
        combined = pd.DataFrame(columns=['metric', 'datasets', 'mean_oriented_z',
                                         'permutation_p', 'datasets_in_expected_direction'])
    combined.insert(0, 'representation', representation)
    combined.to_csv(args.output_dir / 'mean_crossfit_predictive_field_combined.csv', index=False)


if __name__ == '__main__':
    main()
