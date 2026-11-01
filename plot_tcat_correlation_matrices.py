# TCAT correlation-matrix figures with spectral and survival annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def feature_label(feature):
    label = (
        feature.replace("TCAT_GEP_", "")
        .replace("TCAT_SCORE_Multinomial_Label_proportion_", "")
        .replace("_CLR", "")
        .replace("_", " ")
    )
    replacements = {
        "Doublet Bcell": "B-cell-like",
        "Doublet Fibroblast": "Fibroblast-like",
        "Doublet Myeloid": "Myeloid-like",
        "Doublet Plasmablast": "Plasmablast-like",
        "Doublet Platelet": "Platelet-like",
        "Doublet RBC": "Erythroid-like",
        "Naive": "naive",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def correlation_matrix(features, pairwise):
    lookup = {feature: index for index, feature in enumerate(features)}
    matrix = np.eye(len(features))
    for row in pairwise.itertuples(index=False):
        if row.feature_a not in lookup or row.feature_b not in lookup:
            continue
        first = lookup[row.feature_a]
        second = lookup[row.feature_b]
        matrix[first, second] = row.rho_median
        matrix[second, first] = row.rho_median
    return matrix


def cluster_order(matrix):
    distance = np.clip(1 - matrix, 0, 2)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    distances = squareform(distance, checks=False)
    clusters = linkage(distances, method='average')
    return leaves_list(clusters)


def figure_annotations(features, olambda, survival):
    olambda = olambda.set_index("feature")
    survival = survival.set_index("feature")
    missing_olambda = sorted(set(features).difference(olambda.index))
    missing_survival = sorted(set(features).difference(survival.index))
    if missing_olambda:
        raise ValueError(f"Missing O-lambda correlations: {missing_olambda}")
    if missing_survival:
        raise ValueError(f"Missing continuous survival results: {missing_survival}")

    rows = []
    for feature in features:
        correlation = olambda.loc[feature]
        hazard = survival.loc[feature]
        rows.append({
            "feature": feature,
            "label": feature_label(feature),
            "O_lambda_rho_median": correlation.rho_median,
            "O_lambda_rho_q025": correlation.rho_q025,
            "O_lambda_rho_q975": correlation.rho_q975,
            "P_O_lambda_same_direction": correlation.P_same_direction,
            "HR_median": hazard.HR_median,
            "HR_q025": hazard.HR_q025,
            "HR_q975": hazard.HR_q975,
            "log2_HR": np.log2(hazard.HR_median),
            "HR_interval_excludes_1": bool(hazard.HR_q025 > 1 or hazard.HR_q975 < 1),
            "P_all_datasets_same_direction": hazard.P_all_datasets_same_direction,
        })
    return pd.DataFrame(rows)


def draw_figure(matrix, annotations, title, output_path, figure_size,
                label_size, marker_probability):
    fig = plt.figure(figsize=figure_size)
    if len(annotations) > 20:
        left, right, top, bottom = 0.17, 0.985, 0.94, 0.25
        annotation_height = 3.0
    else:
        left, right, top, bottom = 0.22, 0.97, 0.91, 0.30
        annotation_height = 2.4
    grid = fig.add_gridspec(
        2, 1, height_ratios=[len(annotations), annotation_height],
        left=left, right=right, top=top, bottom=bottom,
        hspace=0.025,
    )
    ax = fig.add_subplot(grid[0])
    ax_info = fig.add_subplot(grid[1], sharex=ax)
    edges = np.arange(len(annotations) + 1) - 0.5

    # Hide the diagonal, keeping both annotation rows aligned with the matrix
    M = matrix.copy()
    np.fill_diagonal(M, np.nan)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f3f3f3")
    matrix_image = ax.pcolormesh(
        edges, edges, M,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
        shading="flat",
    )
    ax.set_xlim(-0.5, len(annotations) - 0.5)
    ax.set_ylim(len(annotations) - 0.5, -0.5)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_yticks(np.arange(len(annotations)))
    ax.set_yticklabels(annotations.label, fontsize=label_size)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(length=0)

    olambda_image = ax_info.pcolormesh(
        edges,
        [-0.5, 0.5],
        annotations.O_lambda_rho_median.to_numpy()[None, :],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.4),
        shading="flat",
    )
    hr_image = ax_info.pcolormesh(
        edges,
        [0.5, 1.5],
        annotations.log2_HR.to_numpy()[None, :],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-0.75, vcenter=0, vmax=0.75),
        shading="flat",
    )
    ax_info.set_xlim(-0.5, len(annotations) - 0.5)
    ax_info.set_ylim(1.5, -0.5)
    ax_info.set_yticks([0, 1])
    ax_info.set_yticklabels(
        [r"Corr. with $\mathcal{O}_{\lambda}^{\infty}$", r"$\log_2$(HR)"],
        fontsize=8,
    )
    ax_info.set_xticks(np.arange(len(annotations)))
    ax_info.set_xticklabels(
        annotations.label, rotation=90, ha="center", va="top",
        fontsize=label_size,
    )
    ax_info.tick_params(length=0)

    marker_size = 5 if len(annotations) > 20 else 18
    for column, row in enumerate(annotations.itertuples(index=False)):
        if row.P_O_lambda_same_direction >= marker_probability:
            ax_info.scatter(column, 0, s=marker_size, color="black")
        if row.HR_interval_excludes_1:
            ax_info.scatter(column, 1, s=marker_size, color="black")
    for axis in [ax, ax_info]:
        for spine in axis.spines.values():
            spine.set_visible(False)

    available_width = right - left
    colorbar_width = available_width * 0.27
    gap = available_width * 0.095
    starts = [left, left + colorbar_width + gap, left + 2 * (colorbar_width + gap)]
    colorbars = [
        (matrix_image, "TCAT–TCAT correlation"),
        (olambda_image, r"Correlation with $\mathcal{O}_{\lambda}^{\infty}$"),
        (hr_image, r"$\log_2$(HR)"),
    ]
    for start, (image, label) in zip(starts, colorbars):
        colorbar = fig.colorbar(
            image,
            cax=fig.add_axes([start, 0.055, colorbar_width, 0.014]),
            orientation="horizontal",
        )
        colorbar.set_label(label, fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot separate GEP and cell-type correlation matrices with aligned "
            "O-lambda-correlation and continuous-survival annotation rows."
        )
    )
    parser.add_argument("pairwise_correlations_csv", type=Path)
    parser.add_argument("olambda_correlations_csv", type=Path)
    parser.add_argument("continuous_survival_results_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--marker-probability", type=float, default=0.95)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairwise = pd.read_csv(args.pairwise_correlations_csv)
    olambda = pd.read_csv(args.olambda_correlations_csv)
    survival = pd.read_csv(args.continuous_survival_results_csv)
    groups = {
        "gep": olambda.loc[olambda.feature_kind.eq("GEP"), "feature"].tolist(),
        "cell_type": olambda.loc[olambda.feature_kind.eq("cell_type"), "feature"].tolist(),
    }
    for group, features in groups.items():
        if group == 'gep':
            title = 'GEP correlation structure, spectral organization, and survival effects'
            size = (15.5, 16.5)
            label_size = 6.4
        else:
            title = 'Cell-type correlation structure, spectral organization, and survival effects'
            size = (8.5, 9.5)
            label_size = 9

        matrix = correlation_matrix(features, pairwise)
        order = cluster_order(matrix)
        ordered_features = [features[index] for index in order]
        matrix = matrix[np.ix_(order, order)]
        annotations = figure_annotations(ordered_features, olambda, survival)
        stem = args.output_dir / f"{group}_correlation_matrix"
        draw_figure(
            matrix, annotations, title, stem, size, label_size, args.marker_probability,
        )
        annotations.to_csv(args.output_dir / f"{group}_ordered_annotations.csv", index=False)
        pd.DataFrame(matrix, index=ordered_features, columns=ordered_features).to_csv(
            args.output_dir / f"{group}_ordered_correlation_matrix.csv"
        )
        print(f"Saved {group} matrix: {stem.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
