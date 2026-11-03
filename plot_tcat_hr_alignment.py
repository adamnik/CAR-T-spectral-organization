from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator
import pandas as pd


COLORS = {
    "GEP": "#858585",
    "score": "#2878B5",
    "cell_type": "#4C956C",
    "O_lambda": "#C23B3B",
}

DISPLAY_NAMES = {
    "TCAT_GEP_Th17_Activated_CLR": "TH17 Activated",
    "TCAT_GEP_Tph_CLR": "TPH",
    "TCAT_GEP_Exhaustion_CLR": "Exhaustion",
    "TCAT_SCORE_ASA_CLR": "Antigen-associated\nActivation",
    "TCAT_GEP_CD4_Naive_CLR": "CD4 Naive",
    "TCAT_GEP_CTLA4_CD38_CLR": "CTLA4/CD38",
    "TCAT_GEP_CD8_Naive_CLR": "CD8 Naive",
    "TCAT_SCORE_Multinomial_Label_proportion_CD4_EM_CLR": "CD4 Effector Memory",
    "TCAT_GEP_SOX4_TOX2_CLR": "SOX4/TOX2",
    "TCAT_SCORE_Multinomial_Label_proportion_CD8_Naive_CLR": "CD8 Naive Proportion",
    "TCAT_GEP_Translation_CLR": "Translation",
    "TCAT_GEP_Heatshock_CLR": "Heat Shock",
    "O_lambda": r"$\mathcal{O}_{\lambda}$",
}

LABEL_LAYOUT = {
    "continuous": {
        "TCAT_GEP_Th17_Activated_CLR": (2.40, 0.98, "right"),
        "TCAT_GEP_Tph_CLR": (2.40, 0.91, "right"),
        "TCAT_GEP_Exhaustion_CLR": (2.40, 0.84, "right"),
        "TCAT_SCORE_ASA_CLR": (2.40, 0.77, "right"),
        "TCAT_GEP_CD4_Naive_CLR": (0.43, 0.82, "left"),
        "TCAT_GEP_CTLA4_CD38_CLR": (1.12, 0.69, "left"),
        "TCAT_GEP_CD8_Naive_CLR": (0.43, 0.75, "left"),
        "TCAT_SCORE_Multinomial_Label_proportion_CD4_EM_CLR": (1.12, 0.61, "left"),
        "TCAT_GEP_SOX4_TOX2_CLR": (0.43, 0.67, "left"),
        "TCAT_SCORE_Multinomial_Label_proportion_CD8_Naive_CLR": (0.43, 0.59, "left"),
        "TCAT_GEP_Translation_CLR": (0.43, 0.51, "left"),
        "O_lambda": (1.18, 0.88, "left"),
    },
    "regime": {
        "TCAT_GEP_Tph_CLR": (1.43, 0.97, "left"),
        "TCAT_GEP_Th17_Activated_CLR": (2.42, 0.93, "right"),
        "TCAT_GEP_Heatshock_CLR": (1.43, 0.89, "left"),
        "TCAT_GEP_Exhaustion_CLR": (1.43, 0.84, "left"),
        "TCAT_SCORE_ASA_CLR": (2.42, 0.79, "right"),
        "TCAT_GEP_CTLA4_CD38_CLR": (1.43, 0.74, "left"),
        "TCAT_SCORE_Multinomial_Label_proportion_CD4_EM_CLR": (2.42, 0.66, "right"),
        "O_lambda": (2.42, 0.87, "right"),
    },
}

OUTSIDE_REGIME_LABELS = {
    "TCAT_GEP_Th17_Activated_CLR",
    "TCAT_SCORE_ASA_CLR",
    "TCAT_SCORE_Multinomial_Label_proportion_CD4_EM_CLR",
    "O_lambda",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot TCAT effect magnitude against cross-dataset alignment."
    )
    parser.add_argument("continuous_tcat")
    parser.add_argument("continuous_olambda")
    parser.add_argument("regime_tcat")
    parser.add_argument("regime_olambda")
    parser.add_argument("output_dir")
    parser.add_argument("--dpi", type=int, default=1200)
    return parser.parse_args()


def read_results(tcat_path, olambda_path):
    frame = pd.read_csv(tcat_path)
    required = {
        "feature", "feature_kind", "HR_median", "HR_q025", "HR_q975",
        "P_all_datasets_same_direction",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"TCAT results are missing columns: {sorted(missing)}")

    frame = frame.rename(
        columns={
            "HR_median": "hr",
            "HR_q025": "lower",
            "HR_q975": "upper",
            "P_all_datasets_same_direction": "alignment",
            "feature_kind": "kind",
        }
    )[["feature", "kind", "hr", "lower", "upper", "alignment"]]

    olambda = pd.read_csv(olambda_path).iloc[0]
    extra = pd.DataFrame([
        {
            "feature": "O_lambda",
            "kind": "O_lambda",
            "hr": olambda.HR_median,
            "lower": olambda.HR_q025,
            "upper": olambda.HR_q975,
            "alignment": olambda.P_all_datasets_same_direction,
        }
    ])
    frame = pd.concat([frame, extra], ignore_index=True)
    frame["supported"] = (frame.lower > 1) | (frame.upper < 1)
    return frame


def draw_panel(ax, frame, model):
    ax.axvline(1, color="#555555", linestyle="--", linewidth=1.0, zorder=0)
    ax.axhline(0.5, color="#AAAAAA", linestyle=":", linewidth=0.9, zorder=0)

    for kind in ["GEP", "score", "cell_type", "O_lambda"]:
        for row in frame.loc[frame.kind.eq(kind)].itertuples(index=False):
            filled = bool(row.supported) or kind == "O_lambda"
            ax.scatter(
                row.hr,
                row.alignment,
                s=120 if kind == "O_lambda" else 45,
                marker="o",
                facecolor=COLORS[kind] if filled else "white",
                edgecolor=COLORS[kind],
                linewidth=1.25,
                alpha=1.0 if filled else 0.85,
                zorder=5 if kind == "O_lambda" else 3,
            )

    supported = frame.loc[frame.supported]
    layout = LABEL_LAYOUT[model]
    expected = set(layout)
    observed = set(supported.feature)
    if expected != observed:
        raise ValueError(
            f"Label layout does not match supported features for {model}: "
            f"missing={sorted(observed - expected)}, extra={sorted(expected - observed)}"
        )

    for row in supported.itertuples(index=False):
        x_text, y_text, horizontal = layout[row.feature]
        outside = model == "regime" and row.feature in OUTSIDE_REGIME_LABELS
        color = COLORS[row.kind] if row.kind != "GEP" else "#606060"
        ax.annotate(
            DISPLAY_NAMES[row.feature],
            xy=(row.hr, row.alignment),
            xytext=(1.025, y_text) if outside else (x_text, y_text),
            textcoords=ax.get_yaxis_transform() if outside else "data",
            ha="left" if outside else horizontal,
            va="center",
            fontsize=10.5 if row.kind == "O_lambda" else 8.4,
            fontweight="bold" if row.kind == "O_lambda" else "normal",
            color=color,
            arrowprops={
                "arrowstyle": "-",
                "color": "#A7A7A7",
                "linewidth": 0.75,
                "shrinkA": 1,
                "shrinkB": 3,
            },
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 0.5},
            zorder=6,
            annotation_clip=False,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.4, 2.5)
    ax.set_ylim(-0.01, 1.03)
    ticks = [0.5, 0.75, 1.0, 1.5, 2.0]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.5", "0.75", "1.0", "1.5", "2.0"]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(color="#E6E6E6", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.set_title(
        "Continuous effects" if model == "continuous" else "Smooth transition model",
        loc="left",
        fontweight="bold",
        pad=10,
    )


def legend_handles():
    return [
        Line2D([], [], marker="o", linestyle="", markerfacecolor=COLORS["GEP"], markeredgecolor=COLORS["GEP"], label="GEP", markersize=6),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=COLORS["score"], markeredgecolor=COLORS["score"], label="Composite score", markersize=6),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=COLORS["cell_type"], markeredgecolor=COLORS["cell_type"], label="Cell proportion", markersize=6),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=COLORS["O_lambda"], markeredgecolor=COLORS["O_lambda"], label=r"$\mathcal{O}_{\lambda}$", markersize=8.5),
    ]


def save_figure(frame, model, output_dir, dpi):
    width = 8.5 if model == "regime" else 7.5
    right = 0.72 if model == "regime" else 0.80
    fig, ax = plt.subplots(figsize=(width, 5.5))
    draw_panel(ax, frame, model)
    ax.set_xlabel("Hazard ratio (log scale)")
    ax.set_ylabel("P(all datasets share one direction)")
    fig.legend(
        handles=legend_handles(), frameon=False, loc="lower center", ncol=4,
        bbox_to_anchor=(0.43, 0.02),
    )
    fig.text(
        0.11,
        0.01,
        "Adjusted for Treatment and Disease. Filled points have a 95% credible interval excluding HR = 1.",
        fontsize=8.2,
        color="#555555",
    )
    fig.subplots_adjust(left=0.11, right=right, top=0.91, bottom=0.23)

    name = "continuous_effects" if model == "continuous" else "smooth_transition_effects"
    png_path = output_dir / f"{name}_and_alignment.png"
    pdf_path = output_dir / f"{name}_and_alignment.pdf"
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print("Saved:", png_path)
    print("Saved:", pdf_path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )

    continuous = read_results(args.continuous_tcat, args.continuous_olambda)
    regime = read_results(args.regime_tcat, args.regime_olambda)
    print(f"Continuous effects: {len(continuous):,}")
    print(f"Smooth-transition effects: {len(regime):,}")
    print("Raster DPI:", args.dpi)
    save_figure(continuous, "continuous", output_dir, args.dpi)
    save_figure(regime, "regime", output_dir, args.dpi)


if __name__ == "__main__":
    main()
