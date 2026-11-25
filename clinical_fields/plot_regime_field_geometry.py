import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from shared.tcat_features import SCORE_FEATURES, plot_label

def context_class(context):
    context = str(context)
    if context == 'O_lambda':
        return 'system'
    if context.startswith('TCAT_GEP_'):
        return 'TCAT GEP'
    if context in SCORE_FEATURES:
        return 'TCAT score'
    if context.startswith('ATLAS_proportion_'):
        return 'Atlas cell type'
    return 'other'

def main():
    parser = argparse.ArgumentParser(description='Plot Figure 5B from the pooled mean geometry.')
    parser.add_argument('pooled_dir', type=Path)
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.titlesize': 11,
                         'axes.titleweight': 'bold', 'axes.linewidth': 0.7, 'pdf.fonttype': 42})
    combined = pd.read_csv(args.pooled_dir / 'mean_predictive_field_context_results.csv')
    selected_contexts = pd.read_csv(args.pooled_dir / 'selected_contexts.csv')
    selected_contexts['display_name'] = selected_contexts.context.map(plot_label)
    selected_contexts['context_class'] = selected_contexts.context.map(context_class)
    order = {'system': 0, 'TCAT GEP': 1, 'TCAT score': 2, 'Atlas cell type': 3}
    selected_contexts['class_order'] = selected_contexts.context_class.map(order)
    selected_contexts = selected_contexts.sort_values(['class_order', 'HR_median'], ascending=[True, False])
    METRICS = [
        ('z__prognostic_channeling', 1, 'Prognostic\nchanneling'),
        ('z__signed_graph_variation', -1, 'Signed graph\nvariation'),
        ('z__signed_neighbor_cosine', 1, 'Signed-neighbor\ncosine'),
        ('joint_oriented_z', 1, r'$Z_{\mathrm{joint}}$'),
    ]
    FIELDS = [
        ('TCAT_GEP', 'TCAT GEP geometry\n(dataset-adjusted)'),
        ('atlas_cell_type', 'Cell-type geometry\n(dataset-adjusted)'),
    ]
    context_order = selected_contexts.context.tolist()
    labels = selected_contexts.display_name.tolist()
    classes = selected_contexts.context_class.tolist()

    # Treat GEPs and composite scores as one TCAT block, with visible gaps around
    # the system-level O-lambda row and before the Atlas cell-type block.
    plot_groups = [
        'system' if value == 'system'
        else 'cell_type' if value == 'Atlas cell type'
        else 'TCAT'
        for value in classes
    ]
    GROUP_GAP = 0.70
    y_positions = []
    position = 0.0
    previous_group = None
    for group in plot_groups:
        if previous_group is not None and group != previous_group:
            position += GROUP_GAP
        y_positions.append(position)
        position += 1.0
        previous_group = group

    def field_matrix(field):
        subset = combined.loc[combined.field.eq(field)].set_index('context')
        return np.asarray([
            [float(subset.loc[context, column]) * sign
             for column, sign, _ in METRICS]
            for context in context_order
        ])

    matrices = [field_matrix(field) for field, _ in FIELDS]
    maximum = max(3.0, np.nanmax(np.abs(np.concatenate(matrices))))
    maximum = np.ceil(maximum * 2) / 2
    cmap = LinearSegmentedColormap.from_list(
        'oriented_z', ['#2166ac', '#ffffff', '#b2182b']
    )
    cmap.set_bad('#EEEEEE')
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)

    height = max(6.2, 0.48 * len(context_order) + 2.4)
    fig = plt.figure(figsize=(10.8, height))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.065], wspace=0.14)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    color_axis = fig.add_subplot(grid[0, 2])

    for field_index, ((field, title), axis, matrix) in enumerate(zip(FIELDS, axes, matrices)):
        for row, y_value in enumerate(y_positions):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                facecolor = '#EEEEEE' if not np.isfinite(value) else cmap(norm(value))
                axis.add_patch(plt.Rectangle(
                    (column - 0.5, y_value - 0.5), 1, 1,
                    facecolor=facecolor, edgecolor='white', linewidth=1.2,
                ))
                if np.isfinite(value):
                    axis.text(
                        column, y_value, f'{value:.1f}', ha='center', va='center',
                        color='white' if abs(value) > 0.58 * maximum else '#222222',
                        fontsize=9,
                    )

        axis.set_title(title, pad=10)
        axis.set_xlim(-0.5, len(METRICS) - 0.5)
        axis.set_ylim(y_positions[-1] + 0.5, -0.5)
        axis.set_xticks(range(len(METRICS)))
        axis.set_xticklabels(
            [label for _, _, label in METRICS],
            fontweight='bold', fontsize=8.2,
        )
        axis.set_yticks(y_positions)
        axis.set_yticklabels(labels if field_index == 0 else [])
        axis.tick_params(which='both', length=0, pad=6)
        for spine in axis.spines.values():
            spine.set_visible(False)

        # Outline O-lambda as the system context.
        axis.add_patch(plt.Rectangle(
            (-0.5, y_positions[0] - 0.5), len(METRICS), 1,
            fill=False, edgecolor='#111111', linewidth=2.1, clip_on=False,
        ))

    image = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.ax.set_title('Oriented z', pad=9, fontweight='normal')
    colorbar.outline.set_visible(False)
    fig.subplots_adjust(left=0.25, right=0.92, top=0.95, bottom=0.08)
    fig.savefig(args.pooled_dir / 'figure_5b.pdf', bbox_inches='tight')
    plt.close(fig)

    heatmap_rows = []
    for (field, title), matrix in zip(FIELDS, matrices):
        for row, context in enumerate(context_order):
            for column, (metric, sign, metric_label) in enumerate(METRICS):
                heatmap_rows.append({
                    'field': field, 'field_label': title,
                    'context': context, 'display_name': plot_label(context),
                    'metric': metric, 'metric_label': metric_label.replace('\n', ' '),
                    'orientation': sign, 'oriented_z': matrix[row, column],
                })
    pd.DataFrame(heatmap_rows).to_csv(args.pooled_dir / 'figure_5b_values.csv', index=False)


if __name__ == '__main__':
    main()
