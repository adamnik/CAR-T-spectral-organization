# Out-of-sample reconstruction of spectral organization from cellular states

import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODELS = ["Ridge", "Elastic net", "PLS", "Gradient boosting"]


def log_excess(values):
    return np.log(np.maximum(np.asarray(values, float) - 1.0, 1e-8))


def inverse_log_excess(values):
    return 1.0 + np.exp(np.asarray(values, float))


def load_data(tcat_path, atlas_path, asymptote_path, bootstrap_path):
    tcat = pd.read_csv(tcat_path)
    atlas = pd.read_csv(atlas_path)
    asymptotes = pd.read_csv(asymptote_path)
    bootstrap = pd.read_csv(bootstrap_path)

    tcat_features = [column for column in tcat if column.startswith("TCAT_GEP_")]
    atlas_features = [
        column for column in atlas if column.startswith("ATLAS_proportion_")
    ]
    if not tcat_features:
        raise ValueError("No TCAT_GEP_* columns were found in the TCAT estimates.")
    if not atlas_features:
        raise ValueError("No ATLAS_proportion_* columns were found in the Atlas estimates.")

    bootstrap = bootstrap.groupby(["Dataset", "sample_id"], as_index=False).agg(
        O_lambda_bootstrap_mean=("O_lambda", "mean"),
        bootstrap_replicates=("O_lambda", "size"),
    )
    data = (
        tcat[["Dataset", "sample_id", *tcat_features]]
        .merge(
            atlas[["Dataset", "sample_id", *atlas_features]],
            on=["Dataset", "sample_id"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            asymptotes[["Dataset", "sample_id", "O_lambda_inf_central"]],
            on=["Dataset", "sample_id"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            bootstrap,
            on=["Dataset", "sample_id"],
            how="inner",
            validate="one_to_one",
        )
        .sort_values(["Dataset", "sample_id"])
        .reset_index(drop=True)
    )
    if data.empty:
        raise ValueError("No patients were shared by all four inputs.")
    if data[tcat_features + atlas_features].isna().any().any():
        raise ValueError("The reconstruction feature matrix contains missing values.")
    if data[["O_lambda_inf_central", "O_lambda_bootstrap_mean"]].isna().any().any():
        raise ValueError("A reconstruction target contains missing values.")

    feature_groups = {
        "TCAT GEP": tcat_features,
        "Atlas cell type": atlas_features,
        "TCAT GEP + Atlas": tcat_features + atlas_features,
    }
    targets = {
        "O_lambda_inf": data.O_lambda_inf_central.to_numpy(float),
        "Bootstrap O_lambda": data.O_lambda_bootstrap_mean.to_numpy(float),
    }
    return data, feature_groups, targets


def centered_geometry(train, test, features, target):
    train_x = train[features].to_numpy(float)
    test_x = test[features].to_numpy(float)
    train_y = target[train.index.to_numpy()]

    x_means = pd.DataFrame(train_x, columns=features).assign(
        Dataset=train.Dataset.to_numpy()
    ).groupby("Dataset").mean()
    y_means = pd.Series(train_y).groupby(train.Dataset.to_numpy()).mean()

    centered_train_x = np.empty_like(train_x)
    centered_test_x = np.empty_like(test_x)
    centered_train_y = np.empty_like(train_y)
    test_offsets = np.empty(len(test))
    for dataset in train.Dataset.unique():
        train_rows = train.Dataset.eq(dataset).to_numpy()
        test_rows = test.Dataset.eq(dataset).to_numpy()
        centered_train_x[train_rows] = (
            train_x[train_rows] - x_means.loc[dataset].to_numpy()
        )
        centered_train_y[train_rows] = train_y[train_rows] - y_means.loc[dataset]
        if test_rows.any():
            centered_test_x[test_rows] = (
                test_x[test_rows] - x_means.loc[dataset].to_numpy()
            )
            test_offsets[test_rows] = y_means.loc[dataset]
    return centered_train_x, centered_test_x, centered_train_y, test_offsets


def parameter_grid(model, n_features, n_train, quick):
    if model == "Ridge":
        values = np.logspace(-3, 4, 4 if quick else 9)
        return [{"alpha": value} for value in values]
    if model == "Elastic net":
        alphas = np.logspace(-4, 0.5, 3 if quick else 6)
        ratios = [0.05, 0.60] if quick else [0.05, 0.25, 0.60, 1.0]
        return [
            {"alpha": alpha, "l1_ratio": ratio}
            for alpha, ratio in itertools.product(alphas, ratios)
        ]
    if model == "PLS":
        requested = [1, 3, 8] if quick else [1, 2, 3, 5, 8, 12, 16]
        return [
            {"n_components": value}
            for value in requested
            if value <= min(n_features, n_train - 2)
        ]
    if model == "Gradient boosting":
        estimators = [100] if quick else [100, 200]
        rates = [0.05] if quick else [0.03, 0.06]
        return [
            {
                "n_estimators": trees,
                "learning_rate": rate,
                "max_depth": depth,
                "min_samples_leaf": 8,
            }
            for trees, rate, depth in itertools.product(
                estimators, rates, [1, 2]
            )
        ]
    raise KeyError(model)


def make_estimator(model, parameters, random_state):
    if model == "Ridge":
        regression = Ridge(alpha=parameters["alpha"])
    elif model == "Elastic net":
        regression = ElasticNet(
            alpha=parameters["alpha"],
            l1_ratio=parameters["l1_ratio"],
            max_iter=50000,
            random_state=random_state,
        )
    elif model == "PLS":
        regression = PLSRegression(
            n_components=parameters["n_components"],
            scale=False,
            max_iter=2000,
        )
    elif model == "Gradient boosting":
        regression = GradientBoostingRegressor(
            **parameters,
            loss="huber",
            random_state=random_state,
        )
    else:
        raise KeyError(model)
    return Pipeline([("scale", StandardScaler()), ("model", regression)])


def predict(fitted, values):
    return np.asarray(fitted.predict(values)).reshape(-1)


def tune_model(model, train, features, target, inner_folds,
               split_seed, random_state, quick):
    splitter = StratifiedKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=split_seed,
    )
    splits = list(splitter.split(np.zeros(len(train)), train.Dataset))
    best_parameters = None
    best_error = np.inf

    for parameters in parameter_grid(model, len(features), len(train), quick):
        errors = []
        for train_rows, valid_rows in splits:
            training = train.iloc[train_rows].copy()
            validation = train.iloc[valid_rows].copy()
            x_train, x_valid, y_train, offsets = centered_geometry(
                training, validation, features, target
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = make_estimator(
                    model, parameters, random_state
                ).fit(x_train, y_train)
            estimate = predict(fitted, x_valid) + offsets
            observed = target[validation.index.to_numpy()]
            errors.append(mean_absolute_error(observed, estimate))

        error = float(np.mean(errors))
        if error < best_error:
            best_error = error
            best_parameters = parameters
    return best_parameters, best_error


def reconstruct(data, features, target_name, target_values, representation,
                outer_folds, outer_repeats, inner_folds,
                random_state, quick):
    target = log_excess(target_values)
    splitter = RepeatedStratifiedKFold(
        n_splits=outer_folds,
        n_repeats=outer_repeats,
        random_state=random_state,
    )
    predictions = []
    tuning = []

    for split_number, (train_rows, test_rows) in enumerate(
        splitter.split(np.zeros(len(data)), data.Dataset)
    ):
        repeat = split_number // outer_folds
        fold = split_number % outer_folds
        train = data.iloc[train_rows].copy()
        test = data.iloc[test_rows].copy()
        train.index = train_rows
        test.index = test_rows

        _, _, _, baseline_offsets = centered_geometry(
            train, test, features, target
        )
        for row, estimate in zip(
            test_rows, inverse_log_excess(baseline_offsets)
        ):
            predictions.append({
                "target": target_name,
                "representation": representation,
                "model": "Dataset mean",
                "repeat": repeat,
                "fold": fold,
                "row": row,
                "Dataset": data.loc[row, "Dataset"],
                "sample_id": data.loc[row, "sample_id"],
                "observed": target_values[row],
                "predicted": estimate,
            })

        for model in MODELS:
            parameters, inner_error = tune_model(
                model,
                train,
                features,
                target,
                inner_folds,
                random_state + 1000 * repeat + 10 * fold,
                random_state,
                quick,
            )
            x_train, x_test, y_train, offsets = centered_geometry(
                train, test, features, target
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = make_estimator(
                    model, parameters, random_state
                ).fit(x_train, y_train)
            estimates = inverse_log_excess(
                predict(fitted, x_test) + offsets
            )
            for row, estimate in zip(test_rows, estimates):
                predictions.append({
                    "target": target_name,
                    "representation": representation,
                    "model": model,
                    "repeat": repeat,
                    "fold": fold,
                    "row": row,
                    "Dataset": data.loc[row, "Dataset"],
                    "sample_id": data.loc[row, "sample_id"],
                    "observed": target_values[row],
                    "predicted": estimate,
                })
            tuning.append({
                "target": target_name,
                "representation": representation,
                "model": model,
                "repeat": repeat,
                "fold": fold,
                "inner_MAE_log_excess": inner_error,
                "best_parameters": json.dumps(parameters, sort_keys=True),
            })

        print(
            f"{target_name} | {representation} | "
            f"repeat {repeat + 1}/{outer_repeats} | "
            f"fold {fold + 1}/{outer_folds}",
            flush=True,
        )
    return pd.DataFrame(predictions), pd.DataFrame(tuning)


def average_patient_predictions(predictions):
    return predictions.groupby(
        [
            "target", "representation", "model", "row",
            "Dataset", "sample_id",
        ],
        as_index=False,
    ).agg(
        observed=("observed", "first"),
        predicted=("predicted", "mean"),
        prediction_SD=("predicted", "std"),
    )


def reconstruction_metrics(frame):
    observed = frame.observed.to_numpy(float)
    predicted = frame.predicted.to_numpy(float)
    return {
        "R2": r2_score(observed, predicted),
        "Spearman_r": spearmanr(observed, predicted).statistic,
    }


def summarize_predictions(patient_predictions, n_bootstraps, random_state):
    rng = np.random.default_rng(random_state)
    rows = []
    for keys, frame in patient_predictions.groupby(
        ["target", "representation", "model"], sort=False
    ):
        target, representation, model = keys
        frame = frame.reset_index(drop=True)
        point = reconstruction_metrics(frame)
        dataset_indices = {
            dataset: subset.index.to_numpy()
            for dataset, subset in frame.groupby("Dataset")
        }
        draws = {"R2": [], "Spearman_r": []}
        for _ in range(n_bootstraps):
            selected = np.concatenate([
                rng.choice(indices, len(indices), replace=True)
                for indices in dataset_indices.values()
            ])
            values = reconstruction_metrics(frame.iloc[selected])
            for metric in draws:
                draws[metric].append(values[metric])

        rows.append({
            "target": target,
            "representation": representation,
            "model": model,
            "patients": len(frame),
            "R2": point["R2"],
            "R2_q025": np.quantile(draws["R2"], 0.025),
            "R2_q975": np.quantile(draws["R2"], 0.975),
            "Spearman_r": point["Spearman_r"],
            "Spearman_r_q025": np.quantile(draws["Spearman_r"], 0.025),
            "Spearman_r_q975": np.quantile(draws["Spearman_r"], 0.975),
        })
    return pd.DataFrame(rows).sort_values(
        ["target", "representation", "R2"],
        ascending=[True, True, False],
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct asymptotic and patient-bootstrap O-lambda from "
            "TCAT GEPs and Atlas cell-type proportions using nested, "
            "dataset-adjusted patient-held-out validation."
        )
    )
    parser.add_argument("tcat_estimates_csv", type=Path)
    parser.add_argument("atlas_estimates_csv", type=Path)
    parser.add_argument("olambda_asymptotes_csv", type=Path)
    parser.add_argument("olambda_bootstrap_replicates_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--outer-folds", type=int, default=4)
    parser.add_argument("--outer-repeats", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--metric-bootstraps", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = {
        "repeated": args.output_dir / "repeated_cross_validated_predictions.csv",
        "patient": args.output_dir / "patient_cross_validated_predictions.csv",
        "tuning": args.output_dir / "nested_tuning_results.csv",
        "summary": args.output_dir / "reconstruction_summary.csv",
        "best": args.output_dir / "best_reconstruction_models.csv",
        "manifest": args.output_dir / "reconstruction_manifest.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {existing[0]}. Use --overwrite to replace it."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, feature_groups, targets = load_data(
        args.tcat_estimates_csv,
        args.atlas_estimates_csv,
        args.olambda_asymptotes_csv,
        args.olambda_bootstrap_replicates_csv,
    )
    if data.Dataset.value_counts().min() < args.outer_folds:
        raise ValueError(
            "At least one dataset has fewer patients than the number of outer folds."
        )
    print(
        f"Analysis cohort: {len(data):,} patients | "
        f"{data.Dataset.nunique():,} datasets",
        flush=True,
    )

    repeated_parts = []
    tuning_parts = []
    for representation, features in feature_groups.items():
        print("\n" + "=" * 100, flush=True)
        print(
            f"{representation} | {len(features):,} features",
            flush=True,
        )
        print("=" * 100, flush=True)
        for target_name, target_values in targets.items():
            predictions, tuning = reconstruct(
                data,
                features,
                target_name,
                target_values,
                representation,
                args.outer_folds,
                args.outer_repeats,
                args.inner_folds,
                args.random_state,
                args.quick,
            )
            repeated_parts.append(predictions)
            tuning_parts.append(tuning)

    repeated = pd.concat(repeated_parts, ignore_index=True)
    tuning = pd.concat(tuning_parts, ignore_index=True)
    patient = average_patient_predictions(repeated)
    summary = summarize_predictions(
        patient, args.metric_bootstraps, args.random_state
    )
    best = (
        summary.loc[~summary.model.eq("Dataset mean")]
        .sort_values("R2", ascending=False)
        .groupby(["target", "representation"], as_index=False)
        .first()
    )

    repeated.to_csv(outputs["repeated"], index=False)
    patient.to_csv(outputs["patient"], index=False)
    tuning.to_csv(outputs["tuning"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    best.to_csv(outputs["best"], index=False)

    manifest = {
        "tcat_estimates_csv": str(args.tcat_estimates_csv),
        "atlas_estimates_csv": str(args.atlas_estimates_csv),
        "olambda_asymptotes_csv": str(args.olambda_asymptotes_csv),
        "olambda_bootstrap_replicates_csv": str(
            args.olambda_bootstrap_replicates_csv
        ),
        "patients": len(data),
        "datasets": data.Dataset.value_counts().sort_index().to_dict(),
        "feature_groups": {
            name: len(features) for name, features in feature_groups.items()
        },
        "targets": list(targets),
        "models": MODELS,
        "outer_folds": args.outer_folds,
        "outer_repeats": args.outer_repeats,
        "inner_folds": args.inner_folds,
        "metric_bootstraps": args.metric_bootstraps,
        "random_state": args.random_state,
        "target_scale": "log(O_lambda - 1)",
        "dataset_adjustment": "training-fold dataset means only",
        "validation": "patient-held-out with every dataset retained in each fold",
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2))

    print("\nBest reconstruction models", flush=True)
    print(
        best[[
            "target", "representation", "model",
            "R2", "R2_q025", "R2_q975",
            "Spearman_r", "Spearman_r_q025", "Spearman_r_q975",
        ]].to_string(index=False),
        flush=True,
    )
    print("\nResults:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
