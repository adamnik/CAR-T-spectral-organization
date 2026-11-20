import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from shared.tcat_features import ASA_GEP_FEATURES, SCORE_FEATURES

from shared.analysis_io import safe_name
from clinical_fields.predictive_field import METRICS, benjamini_hochberg, field_test, particle_profiles, posterior_states


CONTEXT_EXCLUDED_FEATURES = {"TCAT_SCORE_ASA_CLR": ASA_GEP_FEATURES}


def excluded_graph_features(context, features):
    features = set(map(str, features))
    excluded = set(CONTEXT_EXCLUDED_FEATURES.get(str(context), []))
    if str(context) in features:
        excluded.add(str(context))
    return sorted(excluded.intersection(features))


def significant_transition_features(path, accepted_prefixes, accepted_exact=()):
    frame = pd.read_csv(path)
    significant = frame.HR_q025.gt(1) | frame.HR_q975.lt(1)
    accepted_exact = set(map(str, accepted_exact))
    accepted = frame.feature.astype(str).map(
        lambda feature: (
            any(feature.startswith(prefix) for prefix in accepted_prefixes)
            or feature in accepted_exact
        )
    )
    return frame.loc[significant & accepted].copy()


def find_membership(regime_dir, feature):
    matches = []
    for result_path in (regime_dir / 'individual').glob('*/model_result.csv'):
        result = pd.read_csv(result_path, nrows=1)
        if len(result) and str(result.iloc[0].feature) == feature:
            membership_path = result_path.parent / 'patient_regime_probabilities.csv'
            if membership_path.exists():
                matches.append((result.iloc[0], membership_path))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one completed transition model for {feature}; found {len(matches)}')
    return matches[0]


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

    contexts = []
    olambda_result = pd.read_csv(args.olambda_regime_result_csv).iloc[0]
    contexts.append({
        'context': 'O_lambda', 'context_kind': 'system',
        'HR_median': float(olambda_result.HR_median),
        'HR_q025': float(olambda_result.HR_q025), 'HR_q975': float(olambda_result.HR_q975),
        'adverse_high': float(olambda_result.HR_median) >= 1,
        'membership_path': args.olambda_regime_membership_csv,
    })

    selected_tcat = significant_transition_features(
        args.tcat_regime_results_csv, ['TCAT_GEP_'],
        SCORE_FEATURES,
    )
    selected_atlas = significant_transition_features(
        args.atlas_regime_results_csv, ['ATLAS_proportion_']
    )
    for table, directory, kind in [
        (selected_tcat, args.tcat_regime_dir, 'TCAT'),
        (selected_atlas, args.atlas_regime_dir, 'cell_type'),
    ]:
        for row in table.itertuples(index=False):
            feature = str(row.feature)
            saved, membership_path = find_membership(directory, feature)
            contexts.append({
                'context': feature,
                'context_kind': 'score' if feature.startswith('TCAT_SCORE_') else kind,
                'HR_median': float(saved.HR_median),
                'HR_q025': float(saved.HR_q025), 'HR_q975': float(saved.HR_q975),
                'adverse_high': float(saved.HR_median) >= 1,
                'membership_path': membership_path,
            })


    pd.DataFrame(contexts).to_csv(args.output_dir / 'selected_contexts.csv', index=False)
    keys = ['Dataset', 'sample_id']
    membership = None
    for context in contexts:
        name = context['context']
        frame = pd.read_csv(context['membership_path'], dtype={'Dataset': str, 'sample_id': str})
        frame = frame[keys + ['posterior_mean_probability_high']].rename(
            columns={'posterior_mean_probability_high': name}
        )
        if membership is None:
            membership = frame
        else:
            membership = membership.merge(frame, on=keys, how='inner', validate='one_to_one')
    probabilities = membership.drop(columns=keys).to_numpy(float)
    if membership.empty or not np.all((0 <= probabilities) & (probabilities <= 1)):
        raise ValueError('Pooled contexts need shared patients with finite regime probabilities')

    representations = [
        ('TCAT_GEP', 'TCAT_GEP_', args.tcat_particles_npz, args.tcat_continuous_results_csv),
        ('atlas_cell_type', 'ATLAS_proportion_', args.atlas_particles_npz, args.atlas_continuous_results_csv),
    ]
    results = []
    for field, prefix, particles, continuous in representations:
        patients, features, values, weights = particle_profiles(particles, membership, prefix)
        effects = pd.read_csv(continuous).set_index('feature').loc[features]
        datasets = patients.Dataset.to_numpy()
        groups = [np.flatnonzero(datasets == dataset) for dataset in np.unique(datasets)]
        states, bootstrap = posterior_states(values, weights, datasets, args.correlation_draws, args.random_state)
        for number, context in enumerate(contexts, start=1):
            name = context['context']
            high = patients[name].to_numpy(float)
            adverse = high if context['adverse_high'] else 1 - high
            removed = excluded_graph_features(name, features)
            bundle = args.output_dir / 'compositional_sensitivity_bundles' / field / (safe_name(name) + '.npz')
            # These permutation probabilities are one-sided. The publication reports two-sided pooled tests.
            result, null = field_test(
                states, bootstrap, adverse, np.arange(len(patients)), groups, effects, removed,
                bundle, {'representation': field, 'context': name, 'scope': 'pooled'},
                args.permutations, args.random_state + 100 * number,
            )
            null_dir = args.output_dir / 'permutation_nulls'
            null_dir.mkdir(exist_ok=True)
            np.savez_compressed(null_dir / f'{field}__{safe_name(name)}.npz', null=null)
            results.append(dict(result, field=field, context=name, context_kind=context['context_kind'],
                                graph_nodes=len(features)-len(removed), geometry='dataset_adjusted',
                                aggregation='arithmetic_mean', patients=len(patients)))
            print(field, name, result['joint_oriented_z'], flush=True)

    results = pd.DataFrame(results)
    joint_scores = results.groupby('field').joint_oriented_z
    results['joint_rank'] = joint_scores.rank(ascending=False, method='min').astype(int)
    for field, index in results.groupby('field').groups.items():
        index = list(index)
        results.loc[index, 'joint_FDR'] = benjamini_hochberg(
            results.loc[index, 'joint_permutation_p']
        )
        for metric in METRICS:
            results.loc[index, f'FDR__{metric}'] = benjamini_hochberg(
                results.loc[index, f'p__{metric}']
            )
    results.to_csv(args.output_dir / 'mean_predictive_field_context_results.csv', index=False)


if __name__ == '__main__':
    main()
