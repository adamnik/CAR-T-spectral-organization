suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop(
    "Usage: Rscript plot_tcat_individual_effects.R individual_results.csv output_dir [dpi]"
  )
}

input_path <- args[[1]]
output_dir <- args[[2]]
dpi <- if (length(args) >= 3) as.numeric(args[[3]]) else 600
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

results <- read.csv(input_path, check.names = FALSE)
required <- c(
  "feature", "feature_kind", "HR_median", "HR_q025", "HR_q975",
  "P_all_datasets_same_direction"
)
if (!all(required %in% names(results))) {
  stop("The individual result file is missing required columns.")
}

gep_names <- c(
  "BCL2_FAM13A" = "BCL2/FAM13A",
  "CD172a_MERTK" = "CD172a/MERTK",
  "CD40LG_TXNIP" = "CD40LG/TXNIP",
  "CD4_CM" = "CD4 Central Memory",
  "CD4_Naive" = "CD4 Naive",
  "CD8_EM" = "CD8 Effector Memory",
  "CD8_Naive" = "CD8 Naive",
  "CD8_Trm" = "CD8 TRM",
  "CTLA4_CD38" = "CTLA4/CD38",
  "CellCycle_G2M" = "Cell Cycle G2M",
  "CellCycle_Late_S" = "Cell Cycle Late S",
  "CellCycle_S" = "Cell Cycle S",
  "Cytoskeleton" = "Cytoskeleton",
  "Cytotoxic" = "Cytotoxic",
  "Doublet_Bcell" = "B Cell-like",
  "Doublet_Fibroblast" = "Fibroblast-like",
  "Doublet_Myeloid" = "Myeloid-like",
  "Doublet_Plasmablast" = "Plasmablast-like",
  "Doublet_Platelet" = "Platelet-like",
  "Doublet_RBC" = "Erythroid-like",
  "Exhaustion" = "Exhaustion",
  "HLA" = "HLA",
  "Heatshock" = "Heat Shock",
  "ICOS_CD38" = "ICOS/CD38",
  "IEG" = "IEG",
  "IEG2" = "IEG2",
  "IEG3" = "IEG3",
  "IL10_IL19" = "IL10/IL19",
  "ISG" = "ISG",
  "MAIT" = "MAIT",
  "Metallothionein" = "Metallothionein",
  "Mito" = "Mitochondrial",
  "Multi_Cytokine" = "Multi-cytokine",
  "NME1_FABP5" = "NME1/FABP5",
  "OX40_EBI3" = "OX40/EBI3",
  "Poor_Quality" = "Poor Quality",
  "RGCC_MYADM" = "RGCC/MYADM",
  "SOX4_TOX2" = "SOX4/TOX2",
  "TEMRA" = "TEMRA",
  "TIMD4_TIM3" = "TIMD4/TIM3",
  "Tfh_1" = "TFH 1",
  "Tfh_2" = "TFH 2",
  "Th17_Activated" = "TH17 Activated",
  "Th17_Resting" = "TH17 Resting",
  "Th1_Like" = "TH1-like",
  "Th22" = "TH22",
  "Th2_Activated" = "TH2 Activated",
  "Th2_Resting" = "TH2 Resting",
  "Tph" = "TPH",
  "Translation" = "Translation",
  "Treg" = "Treg",
  "gdT" = "γδ T"
)

cell_type_names <- c(
  "CD4_CM" = "CD4 Central Memory",
  "CD4_EM" = "CD4 Effector Memory",
  "CD4_Naive" = "CD4 Naive",
  "CD8_CM" = "CD8 Central Memory",
  "CD8_EM" = "CD8 Effector Memory",
  "CD8_Naive" = "CD8 Naive",
  "CD8_TEMRA" = "CD8 TEMRA",
  "MAIT" = "MAIT",
  "Treg" = "Treg",
  "gdT" = "γδ T"
)

display_name <- function(feature, kind) {
  if (kind == "GEP") {
    key <- sub("^TCAT_GEP_", "", feature)
    key <- sub("_CLR$", "", key)
    return(unname(gep_names[[key]]))
  }
  if (kind == "score") {
    key <- sub("^TCAT_SCORE_", "", feature)
    key <- sub("_CLR$", "", key)
    if (key == "ASA") {
      return("Antigen-associated Activation")
    }
    return(key)
  }
  key <- sub("^TCAT_SCORE_Multinomial_Label_proportion_", "", feature)
  key <- sub("_CLR$", "", key)
  unname(cell_type_names[[key]])
}

results$display_name <- mapply(display_name, results$feature, results$feature_kind)
if (any(is.na(results$display_name))) {
  missing <- results$feature[is.na(results$display_name)]
  stop("Missing display names: ", paste(missing, collapse = ", "))
}

make_figure <- function(kind, label, height, output_name) {
  frame <- results[results$feature_kind == kind, ]
  frame <- frame[order(frame$HR_median, decreasing = TRUE), ]
  frame$display_name <- factor(frame$display_name, levels = rev(frame$display_name))
  frame$log2_hr <- log2(frame$HR_median)
  frame$log2_lower <- log2(frame$HR_q025)
  frame$log2_upper <- log2(frame$HR_q975)

  largest_effect <- max(abs(c(frame$log2_lower, frame$log2_upper)))
  effect_limit <- max(1.25, ceiling(largest_effect * 10) / 10)
  hr_breaks <- c(0.5, 0.75, 1, 1.5, 2, 2.5)

  shared_theme <- theme_minimal(base_size = 10.5) +
    theme(
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = "#E5E5E5", linewidth = 0.35),
      axis.title.y = element_blank(),
      axis.text.y = element_text(color = "#252525", size = 9.2),
      axis.text.x = element_text(color = "#252525"),
      plot.title = element_text(size = 12, face = "plain", hjust = 0.5),
      plot.margin = margin(5.5, 4, 5.5, 5.5),
      panel.border = element_rect(color = "#4D4D4D", fill = NA, linewidth = 0.55)
    )

  effect_plot <- ggplot(frame, aes(y = display_name)) +
    geom_segment(
      aes(x = log2_lower, xend = log2_upper, yend = display_name),
      color = "#999999", linewidth = 0.65
    ) +
    geom_point(aes(x = log2_hr, color = log2_hr), size = 2.5) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#555555", linewidth = 0.55) +
    scale_color_gradient2(
      low = "#356AC3", mid = "#D9D9D9", high = "#B2182B",
      midpoint = 0, limits = c(-effect_limit, effect_limit), guide = "none"
    ) +
    scale_x_continuous(
      limits = c(-effect_limit, effect_limit),
      breaks = log2(hr_breaks),
      labels = hr_breaks,
      expand = expansion(mult = c(0.02, 0.02))
    ) +
    labs(
      title = paste("Independent", label, "effects"),
      x = "Global hazard ratio per SD (95% CrI)"
    ) +
    shared_theme

  alignment_plot <- ggplot(frame, aes(x = P_all_datasets_same_direction, y = display_name)) +
    geom_col(width = 0.72, fill = "#5B8DB8") +
    scale_x_continuous(
      limits = c(0, 1), breaks = seq(0, 1, 0.2),
      expand = expansion(mult = c(0, 0.01))
    ) +
    labs(
      title = "Cross-dataset alignment",
      x = "P(all datasets share direction)"
    ) +
    shared_theme +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      plot.margin = margin(5.5, 5.5, 5.5, 0)
    )

  png(
    file.path(output_dir, output_name),
    width = 11.2,
    height = height,
    units = "in",
    res = dpi,
    bg = "white"
  )
  grid.newpage()
  layout <- grid.layout(1, 2, widths = unit(c(2.35, 1), "null"))
  pushViewport(viewport(layout = layout))
  print(effect_plot, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
  print(alignment_plot, vp = viewport(layout.pos.row = 1, layout.pos.col = 2))
  dev.off()
}

make_figure("GEP", "GEP", 15.5, "GEP_effects_and_alignment.png")
make_figure("score", "score", 2.7, "score_effects_and_alignment.png")
make_figure("cell_type", "cell-type", 5.2, "cell_type_effects_and_alignment.png")

write.csv(
  results[, c(
    "feature", "feature_kind", "display_name",
    "HR_median", "HR_q025", "HR_q975",
    "P_all_datasets_same_direction"
  )],
  file.path(output_dir, "plotted_results.csv"),
  row.names = FALSE
)

cat("Input:", input_path, "\n")
cat("GEP features:", sum(results$feature_kind == "GEP"), "\n")
cat("Scores:", sum(results$feature_kind == "score"), "\n")
cat("Cell types:", sum(results$feature_kind == "cell_type"), "\n")
cat("Output:", output_dir, "\n")
