"""Plot pooled joint surrogate probabilities in a companion heatmap to Fig. 5B."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from clinical_fields.plot_regime_field_geometry import context_class
from shared.tcat_features import plot_label


GEOMETRIES = {
    "TCAT_GEP": "TCAT GEP geometry",
    "atlas_cell_type": "Atlas cell-type geometry",
}
FAMILIES = {
    "permuted_field": "Coefficient\npermutation",
    "norm_matched_isotropic_field": "Isotropic\nfields",
    "sign_randomized_field": "Sign\nrandomization",
}
CONTEXT_CLASS_ORDER = {"system": 0, "TCAT GEP": 1, "TCAT score": 2, "Atlas cell type": 3}


def load_heatmap(surrogate_csv, primary_csv):
    primary = pd.read_csv(primary_csv)
    primary = primary.loc[
        primary["geometry"].eq("dataset_adjusted")
        & primary["aggregation"].eq("arithmetic_mean")
        & primary["field"].isin(GEOMETRIES)
    ]
    contexts = primary.loc[primary["field"].eq("TCAT_GEP"), ["context", "HR_median"]].copy()
    contexts["class_order"] = contexts["context"].map(context_class).map(CONTEXT_CLASS_ORDER)
    contexts = contexts.sort_values(["class_order", "HR_median", "context"], ascending=[True, False, True])
    context_order = contexts["context"].tolist()
    if len(context_order) != len(set(context_order)) or "O_lambda" not in context_order:
        raise ValueError("Expected unique dataset-adjusted mean contexts including O_lambda.")

    results = pd.read_csv(surrogate_csv)
    results = results.loc[results["analysis"].eq("compositional_sensitivity")]
    if not results["adjustment_status"].eq("final").all():
        raise ValueError("Use the final pooled surrogate results after all selected contexts finish.")
    keys = ["representation", "context", "surrogate_family"]
    probabilities = results.set_index(keys, verify_integrity=True)["joint_surrogate_p"]
    expected = pd.MultiIndex.from_product([GEOMETRIES, context_order, FAMILIES], names=keys)
    if set(probabilities.index) != set(expected):
        raise ValueError("The pooled and surrogate results must contain the same contexts in both geometries.")
    probabilities = probabilities.reindex(expected).to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any((probabilities <= 0) | (probabilities > 1)):
        raise ValueError("Nominal joint probabilities must be finite and in (0, 1].")
    return contexts, probabilities.reshape(len(GEOMETRIES), len(context_order), len(FAMILIES))


def plot_heatmap(contexts, matrices):
    labels = contexts["context"].map(plot_label)
    # Separate spectral organization and, when present, atlas comparator rows.
    plot_groups = contexts["class_order"].replace({2: 1}).to_numpy()
    gaps = np.r_[False, plot_groups[1:] != plot_groups[:-1]]
    positions = np.arange(len(contexts), dtype=float) + 0.30 * gaps.cumsum()
    spectral_row = contexts["context"].tolist().index("O_lambda")
    palette = LinearSegmentedColormap.from_list(
        "surrogate_specificity", ["#fafbfd", "#bfd5e7", "#4b82aa", "#153b60"],
    )
    ceiling = max(4.0, -np.log10(matrices.min()))
    norm = Normalize(0, ceiling)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, max(4.6, 0.52 * len(contexts) + 0.95)), sharey=True)
    fig.subplots_adjust(left=0.225, right=0.864, top=0.885, bottom=0.175, wspace=0.075)
    for axis, title, matrix in zip(axes, GEOMETRIES.values(), matrices):
        axis.set_title(title, fontsize=12, weight="bold", pad=11)
        axis.set_xlim(-0.5, 2.5)
        axis.set_ylim(positions[-1] + 0.5, -0.5)
        axis.set_yticks(positions, labels, fontsize=9.7)
        axis.set_xticks(range(len(FAMILIES)), FAMILIES.values(), fontsize=9.2, weight="bold")
        axis.tick_params(length=0, pad=7)
        axis.spines[:].set_visible(False)
        for y, row in zip(positions, matrix):
            for column, probability in enumerate(row):
                strength = -np.log10(probability)
                axis.add_patch(Rectangle(
                    (column - 0.5, y - 0.5), 1, 1, facecolor=palette(norm(strength)),
                    edgecolor="white", linewidth=1.55,
                ))
                if probability < 0.00001:
                    label = f"{probability:.1e}"
                elif probability < 0.001:
                    label = f"{probability:.5f}"
                else:
                    label = f"{probability:.3f}"
                axis.text(
                    column, y, label, ha="center", va="center", fontsize=9.8,
                    color="white" if norm(strength) > 0.54 else "#16232c",
                )
        axis.add_patch(Rectangle(
            (-0.5, positions[spectral_row] - 0.5), len(FAMILIES), 1, fill=False,
            edgecolor="#161616", linewidth=1.65, clip_on=False,
        ))
    color_axis = fig.add_axes([0.897, 0.365, 0.017, 0.37])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=palette), cax=color_axis)
    colorbar.set_label(r"$-\log_{10}(\mathrm{nominal}\ p)$", fontsize=9.5, labelpad=8)
    colorbar.outline.set_visible(False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surrogate_csv", type=Path, help="Final pooled_surrogate_results.csv")
    parser.add_argument("primary_csv", type=Path, help="Pooled mean_predictive_field_context_results.csv")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    contexts, probabilities = load_heatmap(args.surrogate_csv, args.primary_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        fig = plot_heatmap(contexts, probabilities)
        stem = args.output_dir / "surrogate_joint_nominal_heatmap"
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
        fig.savefig(stem.with_suffix(".jpg"), dpi=600, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
