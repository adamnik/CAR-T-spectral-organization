# Synthetic checks for the analysis
# Run: python -m unittest -v test_analysis

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.special import logsumexp
from scipy.stats import t

import extract_tcat_patient_particles as extraction
import extract_olambda_sensitivity as sensitivity
import extract_string_pseudobulk as pseudobulk
import estimate_gene_olambda_correlations as gene_correlations
import estimate_tcat_correlations as correlations
import extract_olambda_technical_covariates as technical_covariates
import filter_string as string
import fit_olambda_technical_regression as technical_regression
import fit_tcat_survival as tcat_survival
import preprocess_h5ad as preprocessing
import predictive_field as field
import reconstruct_olambda_from_cell_states as reconstruction
import run_tcat as tcat
import survival_model as survival


class AnalysisChecks(unittest.TestCase):
    def test_technical_covariates_and_particles(self):
        counts = np.array([
            [9., 1., 0.],
            [2., 0., 3.],
            [0., 0., 4.],
        ])
        mitochondrial = np.array([False, True, False])
        for matrix in [counts, sparse.csr_matrix(counts)]:
            metrics = technical_covariates.cell_metrics(matrix, mitochondrial)
            assert_allclose(metrics[:, 0], np.log1p([10., 5., 4.]))
            assert_allclose(metrics[:, 1], [0.1, 0., 0.])
            assert_allclose(metrics[:, 2], np.log1p([2., 2., 1.]))

        labels = np.asarray(["A", "A", "B"])
        embedding = np.asarray([[0., 0.], [1., 0.], [0., 2.]])
        central, particles = technical_covariates.patient_values(
            metrics, labels, ["A", "B"], embedding,
            3, 1000, .6, .2, 16, 20, np.random.default_rng(0),
        )
        self.assertEqual(central.shape, (10,))
        self.assertEqual(particles.shape, (16, 10))
        assert_allclose(central[[0, 4]], np.log([3, 1000]))
        assert_allclose(central[[5, 6]], [.6, .2])
        assert_allclose(particles[:, 0], np.log(3))
        assert_allclose(particles[:, 4], np.log(1000))
        assert_allclose(particles[:, 5], .6)
        assert_allclose(particles[:, 6], .2)

        central_x = np.column_stack([np.arange(4), np.arange(4) * 2 + 1])
        particle_x = np.repeat(central_x[:, None, :], 3, axis=1)
        central_y = np.arange(4) + 2
        particle_y = np.repeat(central_y[:, None], 3, axis=1)
        standardized, scaling = technical_regression.standardize(
            central_x, particle_x, central_y, particle_y
        )
        assert_allclose(standardized['central_x'].mean(axis=0), 0, atol=1e-14)
        assert_allclose(standardized['central_y'].mean(), 0, atol=1e-14)
        self.assertGreater(scaling['y_scale'], 0)

    def test_string_pseudobulk_and_gene_correlations(self):
        counts = np.array([
            [1., 5., 0.],
            [2., 4., 0.],
            [0., 3., 1.],
            [4., 2., 2.],
        ])
        for matrix in [counts, sparse.csr_matrix(counts)]:
            data = ad.AnnData(
                matrix,
                obs=pd.DataFrame({'sample_id': ['p1', 'p1', 'p2', 'p2']}),
                var=pd.DataFrame(index=['A', 'B', 'C']),
            )
            frame = pseudobulk.accumulate_patients(
                data, 'D', ['A', 'C'], 'sample_id', 'X', 2
            )
            expected_counts = np.array([[3., 0.], [4., 3.]])
            expected = np.log2(
                expected_counts / expected_counts.sum(axis=1, keepdims=True) * 1e6 + 1
            )
            assert_allclose(frame[['A', 'C']], expected)
            assert_array_equal(frame.n_cells, [2, 2])

        expression = np.array([
            [1., 6.], [2., 5.], [3., 4.], [4., 3.], [5., 2.], [6., 1.]
        ])
        predictor = np.array([1., 2., 3., 4., 5., 6.])
        design = np.column_stack([np.ones(6), [0., 0., 0., 1., 1., 1.]])
        weights = np.repeat(1 / 6, 6)
        actual = gene_correlations.weighted_partial_correlations(
            expression, predictor, design, weights
        )
        inverse = np.linalg.inv(design.T @ (design * weights[:, None]))
        residual_expression = expression - design @ (
            inverse @ (design.T @ (expression * weights[:, None]))
        )
        residual_predictor = predictor - design @ (
            inverse @ (design.T @ (weights * predictor))
        )
        expected = correlations.weighted_vector_correlations(
            residual_expression, residual_predictor, weights
        )
        assert_allclose(actual, expected)

    def test_predictive_field_metrics(self):
        correlation = np.array([
            [1., .5, -.5],
            [.5, 1., -.5],
            [-.5, -.5, 1.],
        ])
        beta = np.array([1., 1., -1.])
        metrics = field.field_metrics(correlation, beta)
        self.assertGreater(metrics[0], 1)
        assert_allclose(metrics[1], 0, atol=1e-14)
        assert_allclose(metrics[2], 1, atol=1e-14)

        membership = np.array([.1, .2, .8, .9])
        groups = [np.array([0, 1]), np.array([2, 3])]
        permuted = field.permute_within_dataset(
            membership, groups, np.random.default_rng(2)
        )
        assert_allclose(np.sort(permuted[:2]), np.sort(membership[:2]))
        assert_allclose(np.sort(permuted[2:]), np.sort(membership[2:]))

    def test_uncertainty_aware_correlations(self):
        values = np.array([
            [0., 0., 3.],
            [1., 2., 2.],
            [2., 4., 1.],
            [3., 6., 0.],
        ])
        weights = np.repeat(.25, 4)
        matrix = correlations.weighted_correlation_matrix(values, weights)
        assert_allclose(matrix, [[1., 1., -1.], [1., 1., -1.], [-1., -1., 1.]])
        actual = correlations.weighted_vector_correlations(values, values[:, 0], weights)
        assert_allclose(actual, [1., 1., -1.])

        datasets = np.array(['A', 'A', 'B', 'B'])
        residuals = correlations.dataset_residuals(values, weights, datasets)
        for dataset in np.unique(datasets):
            keep = datasets == dataset
            group_weights = weights[keep] / weights[keep].sum()
            assert_allclose(np.sum(residuals[keep] * group_weights[:, None], axis=0), 0.)

    def test_duplicate_genes_and_qc(self):
        counts = np.array([[2, 3, 0, 1], [0, 0, 4, 0], [0, 0, 0, 0]])
        for matrix in [counts, sparse.csr_matrix(counts)]:
            genes = pd.Index(['A', 'A', 'B', 'MT-C'], name='gene_symbol')
            data = ad.AnnData(matrix, var=pd.DataFrame(index=genes))
            collapsed = preprocessing.collapse_duplicate_genes(data)
            actual = collapsed.X.toarray() if sparse.issparse(collapsed.X) else collapsed.X
            assert_array_equal(actual, [[5, 0, 1], [0, 4, 0], [0, 0, 0]])
            self.assertEqual(list(collapsed.var_names), ['A', 'B', 'MT-C'])
            self.assertIsNone(collapsed.var_names.name)
            keep = preprocessing.cell_qc_mask(collapsed, 1, None, 10, 'MT-')
            assert_array_equal(keep, [False, True, False])

    def test_string_aliases(self):
        names = ['SEPT4', 'C17ORF47', 'abc', 'ABC', 'unmapped']
        counts = np.array([[1, 2, 3, 3, 9], [2, 4, 0, 0, 9]])
        preferred = {'SEPTIN4': 'SEPTIN4', 'ABC': 'ABC'}
        aliases = {'SEPT4': 'SEPTIN4', 'C17ORF47': 'SEPTIN4'}
        for matrix in [counts, sparse.csr_matrix(counts)]:
            selected = string.resolve_string_columns(matrix, names, preferred, set(preferred), aliases)
            result = string.build_filtered_matrix(matrix, selected)
            if sparse.issparse(result):
                result = result.toarray()
            assert_array_equal(result, [[3, 3], [0, 6]])
            self.assertEqual([x[0] for x in selected], ['ABC', 'SEPTIN4'])
        with self.assertRaises(ValueError):
            string.resolve_string_columns(np.array([[1, 2]]), ['ABC', 'abc'],
                                          {'ABC': 'ABC'}, {'ABC'}, {})

    def test_tcat_alignment_and_h5ad_roundtrip(self):
        data = ad.AnnData(np.ones((2, 2)), obs=pd.DataFrame(index=['a', 'b']))
        usage = pd.DataFrame([[.2, .8], [.7, .3]], index=['b', 'a'], columns=['GEP 1', 'GEP-2'])
        scores = pd.DataFrame({'label': ['CD4', 'CD8'], 'score': [2., 3.]}, index=['b', 'a'])
        tcat.attach_tcat_results(data, usage, scores, 'TCAT.V1', ['A', 'B'], 0)
        assert_allclose(data.obsm['TCAT_usage'], [[.7, .3], [.2, .8]])
        self.assertEqual(list(data.obs.TCAT_SCORE_label), ['CD8', 'CD4'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cells.h5ad'
            tcat.write_h5ad(data, path, False)
            assert_allclose(ad.read_h5ad(path).obsm['TCAT_usage'], data.obsm['TCAT_usage'])
            with self.assertRaises(FileExistsError):
                tcat.write_h5ad(data, path, False)

    def test_patient_bootstrap(self):
        gep = np.array([[.1, .2, .3, .4], [.4, .3, .2, .1], [.2, .2, .2, .4]])
        labels = np.array(['CD4', 'CD8', 'CD4'])
        center, particles = extraction.patient_features(gep, {'label': labels},
            {'label': ['CD4', 'CD8']}, np.array([0, 1]), np.array([2, 3]), 16, .5,
            np.random.default_rng(42))
        weights = np.random.default_rng(42).dirichlet(np.ones(3), size=16)
        logged = np.log(weights @ gep)
        expected = logged - logged.mean(axis=1, keepdims=True)
        assert_allclose(particles[:, :4], expected)
        assert_allclose(particles[:, 4], expected[:, :2].mean(axis=1))
        assert_allclose(particles[:, 5], expected[:, 2:].mean(axis=1))
        counts = weights @ np.column_stack([labels == 'CD4', labels == 'CD8']) * 3 + .5
        logged = np.log(counts / counts.sum(axis=1, keepdims=True))
        assert_allclose(particles[:, 6:], logged - logged.mean(axis=1, keepdims=True))
        assert_allclose(center[:4].sum(), 0, atol=1e-14)

    def test_sensitivity_cell_sampling(self):
        target = pd.Series({'CD4': .3333, 'CD8': .6667})
        quotas = sensitivity.integer_quotas(target, 10000)
        self.assertEqual(quotas.sum(), 10000)
        self.assertEqual(quotas.CD4, 3333)
        self.assertEqual(quotas.CD8, 6667)

        rows = np.arange(12)
        selected, weights, donors = sensitivity.patient_bootstrap(
            rows, 1000, np.random.default_rng(4)
        )
        self.assertEqual(len(selected), 1000)
        self.assertTrue(np.isin(selected, rows).all())
        self.assertGreater(weights['cell_weight_ess'], 0)
        self.assertEqual(donors, {})

    def test_sensitivity_pooled_draw(self):
        rows_by_patient = {
            'p1': np.arange(0, 6),
            'p2': np.arange(6, 12),
            'p3': np.arange(12, 18),
        }
        labels = np.asarray((['CD4'] * 3 + ['CD8'] * 3) * 3)
        selected, weights, donor_counts = sensitivity.pooled_synthetic(
            rows_by_patient, labels, 'p1', {'CD4': .5, 'CD8': .5}, 1000,
            np.random.default_rng(7),
        )
        self.assertEqual(len(selected), 1000)
        self.assertFalse(np.isin(selected, rows_by_patient['p1']).any())
        self.assertEqual(set(donor_counts), {'p2', 'p3'})
        self.assertLessEqual(abs(donor_counts['p2'] - donor_counts['p3']), 2)
        self.assertGreater(weights['cell_weight_ess'], 0)

    def test_reconstruction_dataset_adjustment(self):
        data = pd.DataFrame({
            'Dataset': ['A', 'A', 'B', 'B'],
            'feature': [1., 3., 10., 14.],
        })
        target = np.array([2., 4., 20., 24.])
        train_x, test_x, train_y, offsets = reconstruction.centered_geometry(
            data, data, ['feature'], target
        )
        assert_allclose(train_x[:, 0], [-1., 1., -2., 2.])
        assert_allclose(test_x, train_x)
        assert_allclose(train_y, [-1., 1., -2., 2.])
        assert_allclose(offsets, [3., 3., 22., 22.])

    def test_extraction_command(self):
        columns = sorted(extraction.ASA_COMPONENTS + extraction.PROLIFERATION_COMPONENTS)
        rng = np.random.default_rng(5)
        obs = pd.DataFrame(rng.dirichlet(np.ones(7), size=8), columns=columns,
                           index=[f'cell{i}' for i in range(8)])
        obs['sample_id'] = ['p1'] * 4 + ['p2'] * 4
        obs['timepoint'] = 'infusion'
        obs['TCAT_SCORE_label'] = ['CD4', 'CD8'] * 4
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / 'cells.h5ad'
            ad.AnnData(np.ones((8, 1)), obs=obs).write_h5ad(path)
            subprocess.run([sys.executable, str(Path(extraction.__file__)), str(path),
                            str(directory / 'results'), '--dataset-name', 'synthetic',
                            '--minimum-cells', '2'], check=True, capture_output=True)
            output = directory / 'results'
            frame = pd.read_csv(output / 'tcat_patient_estimates.csv')
            particles = np.load(output / 'tcat_patient_particles.npz')
            self.assertEqual(list(frame.sample_id), ['p1', 'p2'])
            self.assertEqual(particles['values'].shape, (2, 16, 11))
            assert_allclose(particles['particle_weights'].sum(axis=1), 1)
            manifest = json.loads((output / 'tcat_extraction_manifest.json').read_text())
            self.assertEqual(manifest['patients'], 2)

    def test_survival_particles_and_cohort(self):
        estimates = pd.DataFrame({'Dataset': ['B', 'A', 'A'], 'sample_id': ['p3', 'p2', 'p1'],
            'n_cells_available': [2000, 2000, 500], 'O_lambda_inf_central': [3., 2., 1.5],
            'log_O_minus_1_inf_location': [np.log(2), 0., np.log(.5)],
            'log_O_minus_1_inf_scale': [.1, .2, .1]})
        outcomes = estimates[['Dataset', 'sample_id']].copy()
        outcomes['event'] = [0, 1, 1]
        outcomes['t_start'] = [0., 3., 1.]
        outcomes['t_end'] = [12., 6., 2.]
        outcomes['Treatment'] = ['tisa', 'axi', 'axi']
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / 'estimates.csv', Path(directory) / 'outcomes.csv'
            estimates.to_csv(first, index=False)
            outcomes.to_csv(second, index=False)
            data, raw, standardized, weights, _ = survival.load_olambda_analysis(
                first, second, 1000, 16, adjustment_columns=['Treatment']
            )
        self.assertEqual(list(data.sample_id), ['p2', 'p3'])
        offsets = t.ppf((np.arange(16) + .5) / 16, df=4)
        location = np.array([0., np.log(2)])[:, None]
        scale = np.array([.2, .1])[:, None]
        assert_allclose(raw, 1 + np.exp(location + scale * offsets))
        assert_allclose(standardized, (raw - 2.5) / np.std([2., 3.], ddof=1))
        assert_allclose(weights.sum(axis=1), 1.)

    def test_saved_olambda_particles(self):
        estimates = pd.DataFrame({
            'Dataset': ['B', 'A'], 'sample_id': ['p2', 'p1'],
            'n_cells_available': [2000, 2000],
            'O_lambda_inf_central': [3., 2.],
        })
        outcomes = pd.DataFrame({
            'Dataset': ['A', 'B'], 'sample_id': ['p1', 'p2'],
            'event_observed': [1, 0], 't_start': [3., 0.], 't_end': [6., 12.],
        })
        particles = pd.DataFrame({
            'Dataset': ['B', 'A'], 'sample_id': ['p2', 'p1'],
            **{f'particle_{i:02d}': [3 + i / 100, 2 + i / 100]
               for i in range(1, 17)},
        })
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            estimates_path = directory / 'patient_olambda_asymptotes.csv'
            outcomes_path = directory / 'outcomes.csv'
            particles_path = directory / 'patient_olambda_particles.csv'
            estimates.to_csv(estimates_path, index=False)
            outcomes.to_csv(outcomes_path, index=False)
            particles.to_csv(particles_path, index=False)
            data, raw, standardized, weights, scaling = survival.load_olambda_analysis(
                estimates_path, outcomes_path, 1000, 16, particles_path
            )
        self.assertEqual(list(data.sample_id), ['p1', 'p2'])
        assert_allclose(raw[0], 2 + np.arange(1, 17) / 100)
        assert_allclose(raw[1], 3 + np.arange(1, 17) / 100)
        assert_allclose(standardized, (raw - 2.5) / np.std([2., 3.], ddof=1))
        assert_allclose(weights.sum(axis=1), 1.)
        self.assertEqual(scaling['particle_source'], str(particles_path))

    def test_clinical_adjustments(self):
        specifications = survival.parse_adjustments([
            'Treatment=axi', 'Disease=lymphoma',
        ])
        self.assertEqual(specifications, {'Treatment': 'axi', 'Disease': 'lymphoma'})
        data = pd.DataFrame({
            'Treatment': ['axi', 'tisa', 'tandem', 'axi'],
            'Disease': ['lymphoma', 'lymphoma', 'lymphoma', 'leukemia'],
        })
        matrix, table = survival.adjustment_design(data, specifications)
        self.assertEqual(
            list(table.adjustment),
            [
                'Treatment: tandem vs axi',
                'Treatment: tisa vs axi',
                'Disease: leukemia vs lymphoma',
            ],
        )
        assert_array_equal(
            matrix,
            [[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]],
        )
        with self.assertRaises(ValueError):
            survival.parse_adjustments(['Treatment'])
        with self.assertRaises(ValueError):
            survival.adjustment_design(data, {'Treatment': 'missing'})

    def test_effect_summaries(self):
        global_effect = np.array([[-.2, .1, .3, .5]])
        dataset_effect = np.stack([global_effect, global_effect - .1], axis=-1)
        idata = az.from_dict(posterior={
            'global_effect': global_effect, 'dataset_effect': dataset_effect,
            'dataset_effect_sd': np.full((1, 4), .2)},
            dims={'dataset_effect': ['dataset']}, coords={'dataset': ['A', 'B']})
        overall, datasets = survival.summarize_effects(idata, ['A', 'B'])
        assert_allclose(overall['HR_median'], np.median(np.exp(global_effect)))
        assert_allclose(overall['HR_q025'], np.quantile(np.exp(global_effect), .025))
        assert_allclose(overall['HR_q975'], np.quantile(np.exp(global_effect), .975))
        self.assertEqual(overall['P_effect_gt_0'], .75)
        self.assertEqual(overall['P_all_datasets_gt_0'], .5)
        self.assertEqual(overall['P_all_datasets_lt_0'], .25)
        self.assertEqual(overall['P_all_datasets_same_direction'], .75)
        assert_allclose(datasets.HR_median, np.median(np.exp(dataset_effect), axis=(0, 1)))

    def test_tcat_survival_inputs_and_joint_scaling(self):
        features = [
            'TCAT_GEP_A_CLR', 'TCAT_GEP_B_CLR',
            'TCAT_SCORE_label_proportion_CD4_CLR',
            'TCAT_SCORE_label_proportion_CD8_CLR',
        ]
        patients = ['p2', 'p1', 'p3']
        datasets = ['B', 'A', 'A']
        central = np.array([
            [.2, -.2, .3, -.3],
            [.1, -.1, .1, -.1],
            [-.2, .2, -.1, .1],
        ])
        delta = np.asarray([.01, -.01, .01, -.01])
        particles = np.stack([central - delta, central + delta], axis=1)
        weights = np.full((3, 2), .5)
        estimates = pd.DataFrame(central, columns=features)
        estimates.insert(0, 'sample_id', patients)
        estimates.insert(0, 'Dataset', datasets)
        outcomes = pd.DataFrame({
            'Dataset': datasets, 'sample_id': patients, 'event_observed': [1, 0, 1],
            't_start': [2., 0., 4.], 't_end': [3., 10., 4.],
        })
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            estimates_path = directory / 'estimates.csv'
            outcomes_path = directory / 'outcomes.csv'
            particles_path = directory / 'particles.npz'
            estimates.to_csv(estimates_path, index=False)
            outcomes.to_csv(outcomes_path, index=False)
            np.savez_compressed(
                particles_path, values=particles, particle_weights=weights,
                patient_ids=np.asarray(patients), datasets=np.asarray(datasets),
                feature_names=np.asarray(features), n_cells=np.asarray([500, 600, 700]),
            )
            data, loaded_central, loaded_particles, loaded_weights, loaded_features = (
                survival.load_tcat_analysis(estimates_path, particles_path, outcomes_path, 50)
            )
        self.assertEqual(list(data.sample_id), ['p1', 'p3', 'p2'])
        self.assertEqual(loaded_features, features)
        assert_allclose(loaded_central, central[[1, 2, 0]])
        assert_allclose(loaded_weights.sum(axis=1), 1.)

        selected, blocks = tcat_survival.joint_features(features, 'joint-gep')
        self.assertEqual(selected, features[:2])
        standardized, scaling = tcat_survival.standardize_joint(
            loaded_central[:, :2], loaded_particles[:, :, :2], selected, blocks
        )
        assert_allclose(standardized.sum(axis=2), 0., atol=1e-12)
        self.assertGreater(scaling['TCAT_GEP']['scale'], 0)

    def test_survival_likelihood(self):
        data = pd.DataFrame({'Dataset': ['A'] * 3, 'event_observed': [1, 1, 0],
                             't_start': [3., 3., 0.], 't_end': [3., 6., 6.]})
        arrays = survival.survival_arrays(data)
        hazard = np.broadcast_to(np.array([.1, .2])[None, :, None], (3, 2, 10)).copy()
        actual = survival.interval_loglikelihood(pt.as_tensor_variable(hazard), arrays).eval()
        rate = np.array([.1, .2])
        expected = np.vstack([np.log(rate) - rate * 3,
                              -rate * 3 + np.log1p(-np.exp(-rate * 3)), -rate * 6])
        assert_allclose(actual, expected)
        single = survival.interval_loglikelihood(pt.as_tensor_variable(hazard[:, 0, :]), arrays).eval()
        assert_allclose(single, expected[:, 0])
        weights = np.full((3, 2), .5)
        marginal = survival.marginalize_particles(pt.as_tensor_variable(actual), weights).eval()
        assert_allclose(marginal, logsumexp(expected, axis=1) - np.log(2))
        with pm.Model(coords=survival.model_coordinates(data, arrays)) as model:
            baseline, effect = survival.hierarchical_parameters(arrays)
            self.assertEqual(baseline.eval().shape, (3, 10))
            self.assertEqual(effect.eval().shape, (3,))
            self.assertTrue(np.isfinite(model.compile_logp()(model.initial_point())))

        adjustment_matrix = np.asarray([[0.], [1.], [0.]])
        coords = survival.model_coordinates(data, arrays)
        coords['adjustment'] = ['Treatment: tisa vs axi']
        with pm.Model(coords=coords) as model:
            adjustment = survival.adjustment_parameters(adjustment_matrix, .35)
            self.assertEqual(adjustment.eval().shape, (3,))
            self.assertTrue(np.isfinite(model.compile_logp()(model.initial_point())))


if __name__ == '__main__':
    unittest.main()
