import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

GEP_NAMES = {
    'BCL2_FAM13A': 'BCL2/FAM13A', 'CD172a_MERTK': 'CD172a/MERTK',
    'CD40LG_TXNIP': 'CD40LG/TXNIP', 'CD4_CM': 'CD4 Central Memory',
    'CD4_Naive': 'CD4 Naive', 'CD8_EM': 'CD8 Effector Memory',
    'CD8_Naive': 'CD8 Naive', 'CD8_Trm': 'CD8 TRM',
    'CTLA4_CD38': 'CTLA4/CD38', 'CellCycle_G2M': 'Cell Cycle G2M',
    'CellCycle_Late_S': 'Cell Cycle Late S', 'CellCycle_S': 'Cell Cycle S',
    'Cytoskeleton': 'Cytoskeleton', 'Cytotoxic': 'Cytotoxic',
    'Doublet_Bcell': 'B Cell-like', 'Doublet_Fibroblast': 'Fibroblast-like',
    'Doublet_Myeloid': 'Myeloid-like', 'Doublet_Plasmablast': 'Plasmablast-like',
    'Doublet_Platelet': 'Platelet-like', 'Doublet_RBC': 'Erythroid-like',
    'Exhaustion': 'Exhaustion', 'HLA': 'HLA', 'Heatshock': 'Heat Shock',
    'ICOS_CD38': 'ICOS/CD38', 'IEG': 'IEG', 'IEG2': 'IEG2', 'IEG3': 'IEG3',
    'IL10_IL19': 'IL10/IL19', 'ISG': 'ISG', 'MAIT': 'MAIT',
    'Metallothionein': 'Metallothionein', 'Mito': 'Mitochondrial',
    'Multi_Cytokine': 'Multi-cytokine', 'NME1_FABP5': 'NME1/FABP5',
    'OX40_EBI3': 'OX40/EBI3', 'Poor_Quality': 'Poor Quality',
    'RGCC_MYADM': 'RGCC/MYADM', 'SOX4_TOX2': 'SOX4/TOX2',
    'TEMRA': 'TEMRA', 'TIMD4_TIM3': 'TIMD4/TIM3', 'Tfh_1': 'TFH 1',
    'Tfh_2': 'TFH 2', 'Th17_Activated': 'TH17 Activated',
    'Th17_Resting': 'TH17 Resting', 'Th1_Like': 'TH1-like', 'Th22': 'TH22',
    'Th2_Activated': 'TH2 Activated', 'Th2_Resting': 'TH2 Resting',
    'Tph': 'TPH', 'Translation': 'Translation', 'Treg': 'Treg', 'gdT': 'γδ T',
}
SCORE_NAMES = {
    'TCAT_SCORE_ASA_CLR': 'Antigen-associated\nActivation',
    'TCAT_SCORE_Proliferation_CLR': 'Proliferation',
}
ATLAS_NAMES = {
    'Apoptotic_T_cells': 'Apoptotic T cells',
    'CD4_central_memory': 'CD4 Central Memory',
    'CD4_cytotoxic': 'CD4 Cytotoxic',
    'CD4_effector_memory': 'CD4 Effector Memory',
    'CD8_cytotoxic': 'CD8 Cytotoxic',
    'CD8_effector_memory': 'CD8 Effector Memory',
    'CD8_memory': 'CD8 Memory',
    'Monocyte_like_T_cells': 'Monocyte-like T cells',
    'Proliferative_T_cells': 'Proliferative T cells',
    'Regulatory_T_cells': 'Regulatory T cells',
    'Ribosomal_enriched': 'Ribosomal Enriched',
}

def context_class(context):
    context = str(context)
    if context == 'O_lambda':
        return 'system'
    if context.startswith('TCAT_GEP_'):
        return 'TCAT GEP'
    if context in SCORE_NAMES:
        return 'TCAT score'
    if context.startswith('ATLAS_proportion_'):
        return 'Atlas cell type'
    return 'other'

def display_name(context):
    context = str(context)
    if context == 'O_lambda':
        return r'$\mathcal{O}_{\lambda}$'
    if context in SCORE_NAMES:
        return SCORE_NAMES[context]
    if context.startswith('TCAT_GEP_'):
        key = context.removeprefix('TCAT_GEP_').removesuffix('_CLR')
        return GEP_NAMES.get(key, key.replace('_', ' '))
    if context.startswith('ATLAS_proportion_'):
        key = context.removeprefix('ATLAS_proportion_').removesuffix('_CLR')
        return ATLAS_NAMES.get(key, key.replace('_', ' '))
    return context

def main():
    parser = argparse.ArgumentParser(description='Plot Figure 5B from the pooled mean geometry.')
    parser.add_argument('pooled_dir', type=Path)
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.titlesize': 11,
                         'axes.titleweight': 'bold', 'axes.linewidth': 0.7, 'pdf.fonttype': 42})
    combined = pd.read_csv(args.pooled_dir / 'mean_predictive_field_context_results.csv')
    selected_contexts = pd.read_csv(args.pooled_dir / 'selected_contexts.csv')
    selected_contexts['display_name'] = selected_contexts.context.map(display_name)
    selected_contexts['context_class'] = selected_contexts.context.map(context_class)
    order = {'system': 0, 'TCAT GEP': 1, 'TCAT score': 2, 'Atlas cell type': 3}
    selected_contexts['class_order'] = selected_contexts.context_class.map(order)
    selected_contexts = selected_contexts.sort_values(['class_order', 'HR_median'], ascending=[True, False])
    METRICS = [
        ('z__prognostic_channeling', 1, 'Prognostic\nchanneling'),
        ('z__signed_graph_variation', -1, 'Signed graph\nvariation'),
        ('z__signed_neighbor_cosine', 1, 'Signed-neighbor\ncosine'),
        ('joint_oriented_z', 1, 'Joint'),
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
                    'context': context, 'display_name': display_name(context),
                    'metric': metric, 'metric_label': metric_label.replace('\n', ' '),
                    'orientation': sign, 'oriented_z': matrix[row, column],
                })
    pd.DataFrame(heatmap_rows).to_csv(args.pooled_dir / 'figure_5b_values.csv', index=False)


if __name__ == '__main__':
    main()
