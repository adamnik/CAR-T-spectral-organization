# TCAT correlation-matrix figures with spectral and survival annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


DISPLAY_NAMES = {
    "TCAT_GEP_Doublet_Plasmablast_CLR": "Plasmablast-like",
    "TCAT_GEP_Mito_CLR": "Mitochondrial",
    "TCAT_GEP_Doublet_Myeloid_CLR": "Myeloid-like",
    "TCAT_GEP_Doublet_Bcell_CLR": "B Cell-like",
    "TCAT_GEP_Doublet_Fibroblast_CLR": "Fibroblast-like",
    "TCAT_GEP_Metallothionein_CLR": "Metallothionein",
    "TCAT_GEP_Doublet_RBC_CLR": "Erythroid-like",
    "TCAT_GEP_HLA_CLR": "HLA",
    "TCAT_GEP_NME1_FABP5_CLR": "NME1/FABP5",
    "TCAT_GEP_CellCycle_S_CLR": "Cell Cycle S",
    "TCAT_GEP_CellCycle_G2M_CLR": "Cell Cycle G2M",
    "TCAT_GEP_CellCycle_Late_S_CLR": "Cell Cycle Late S",
    "TCAT_GEP_Tfh_1_CLR": "TFH 1",
    "TCAT_GEP_IL10_IL19_CLR": "IL10/IL19",
    "TCAT_GEP_SOX4_TOX2_CLR": "SOX4/TOX2",
    "TCAT_GEP_IEG_CLR": "IEG",
    "TCAT_GEP_IEG2_CLR": "IEG2",
    "TCAT_GEP_Exhaustion_CLR": "Exhaustion",
    "TCAT_GEP_Multi_Cytokine_CLR": "Multi-cytokine",
    "TCAT_GEP_OX40_EBI3_CLR": "OX40/EBI3",
    "TCAT_GEP_Tph_CLR": "TPH",
    "TCAT_GEP_Doublet_Platelet_CLR": "Platelet-like",
    "TCAT_GEP_Tfh_2_CLR": "TFH 2",
    "TCAT_GEP_RGCC_MYADM_CLR": "RGCC/MYADM",
    "TCAT_GEP_Heatshock_CLR": "Heat Shock",
    "TCAT_GEP_IEG3_CLR": "IEG3",
    "TCAT_GEP_ISG_CLR": "ISG",
    "TCAT_GEP_BCL2_FAM13A_CLR": "BCL2/FAM13A",
    "TCAT_GEP_CD4_Naive_CLR": "CD4 Naive",
    "TCAT_GEP_CD8_Naive_CLR": "CD8 Naive",
    "TCAT_GEP_MAIT_CLR": "MAIT",
    "TCAT_GEP_Th17_Resting_CLR": "TH17 Resting",
    "TCAT_GEP_CTLA4_CD38_CLR": "CTLA4/CD38",
    "TCAT_GEP_Th17_Activated_CLR": "TH17 Activated",
    "TCAT_GEP_Poor_Quality_CLR": "Poor Quality",
    "TCAT_GEP_CD4_CM_CLR": "CD4 Central Memory",
    "TCAT_GEP_CD40LG_TXNIP_CLR": "CD40LG/TXNIP",
    "TCAT_GEP_Treg_CLR": "Treg",
    "TCAT_GEP_Th2_Activated_CLR": "TH2 Activated",
    "TCAT_GEP_Th2_Resting_CLR": "TH2 Resting",
    "TCAT_GEP_Th22_CLR": "TH22",
    "TCAT_GEP_Cytoskeleton_CLR": "Cytoskeleton",
    "TCAT_GEP_ICOS_CD38_CLR": "ICOS/CD38",
    "TCAT_GEP_CD172a_MERTK_CLR": "CD172a/MERTK",
    "TCAT_GEP_Translation_CLR": "Translation",
    "TCAT_GEP_TEMRA_CLR": "TEMRA",
    "TCAT_GEP_CD8_Trm_CLR": "CD8 TRM",
    "TCAT_GEP_TIMD4_TIM3_CLR": "TIMD4/TIM3",
    "TCAT_GEP_Th1_Like_CLR": "TH1-like",
    "TCAT_GEP_gdT_CLR": "γδ T",
    "TCAT_GEP_CD8_EM_CLR": "CD8 Effector Memory",
    "TCAT_GEP_Cytotoxic_CLR": "Cytotoxic",
    "TCAT_SCORE_Multinomial_Label_proportion_MAIT_CLR": "MAIT",
    "TCAT_SCORE_Multinomial_Label_proportion_CD4_EM_CLR": "CD4 Effector Memory",
    "TCAT_SCORE_Multinomial_Label_proportion_Treg_CLR": "Treg",
    "TCAT_SCORE_Multinomial_Label_proportion_CD4_CM_CLR": "CD4 Central Memory",
    "TCAT_SCORE_Multinomial_Label_proportion_CD4_Naive_CLR": "CD4 Naive",
    "TCAT_SCORE_Multinomial_Label_proportion_gdT_CLR": "γδ T",
    "TCAT_SCORE_Multinomial_Label_proportion_CD8_EM_CLR": "CD8 Effector Memory",
    "TCAT_SCORE_Multinomial_Label_proportion_CD8_TEMRA_CLR": "CD8 TEMRA",
    "TCAT_SCORE_Multinomial_Label_proportion_CD8_CM_CLR": "CD8 Central Memory",
    "TCAT_SCORE_Multinomial_Label_proportion_CD8_Naive_CLR": "CD8 Naive",
}


def feature_label(feature):
    if feature in DISPLAY_NAMES:
        return DISPLAY_NAMES[feature]
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
                label_size, marker_probability, dpi):
    fig = plt.figure(figsize=figure_size)
    if len(annotations) > 20:
        left, right, top, bottom = 0.17, 0.985, 0.95, 0.25
        annotation_height = 2.4
        spacer_height = 0.45
    else:
        left, right, top, bottom = 0.22, 0.97, 0.93, 0.31
        annotation_height = 1.7
        spacer_height = 0.20
    grid = fig.add_gridspec(
        3, 1,
        height_ratios=[len(annotations), spacer_height, annotation_height],
        left=left, right=right, top=top, bottom=bottom,
        hspace=0,
    )
    ax = fig.add_subplot(grid[0])
    ax_info = fig.add_subplot(grid[2], sharex=ax)
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
        norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
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
    colorbar_width = available_width * 0.39
    inset = available_width * 0.035
    starts = [left + inset, right - inset - colorbar_width]
    colorbars = [
        (matrix_image, "Correlation"),
        (hr_image, r"$\log_2$(HR)"),
    ]
    for start, (image, label) in zip(starts, colorbars):
        colorbar = fig.colorbar(
            image,
            cax=fig.add_axes([start, 0.06, colorbar_width, 0.016]),
            orientation="horizontal",
        )
        colorbar.set_label(label, fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
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
    parser.add_argument("--dpi", type=int, default=1200)
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
            title = 'GEP Correlation Matrix'
            size = (15.5, 14.2)
            label_size = 6.4
        else:
            title = 'Cell-type Correlation Matrix'
            size = (8.5, 7.8)
            label_size = 9

        matrix = correlation_matrix(features, pairwise)
        order = cluster_order(matrix)
        ordered_features = [features[index] for index in order]
        matrix = matrix[np.ix_(order, order)]
        annotations = figure_annotations(ordered_features, olambda, survival)
        stem = args.output_dir / f"{group}_correlation_matrix"
        draw_figure(
            matrix, annotations, title, stem, size, label_size,
            args.marker_probability, args.dpi,
        )
        annotations.to_csv(args.output_dir / f"{group}_ordered_annotations.csv", index=False)
        pd.DataFrame(matrix, index=ordered_features, columns=ordered_features).to_csv(
            args.output_dir / f"{group}_ordered_correlation_matrix.csv"
        )
        print(f"Saved {group} matrix: {stem.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
