#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(igraph)
    library(ggraph)
    library(ggrepel)
    library(ggplot2)
    library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4 || length(args) > 7) {
    stop(paste(
        "Usage: Rscript plot_reactome_enrichment_network_prior_layout.R",
        "nodes_with_layout.csv edges.csv modules.csv output_dir",
        "[displayed|significant|context] [output_stem]",
        "[current|refined|fr|drl]"
    ))
}

nodes_path <- args[[1]]
edges_path <- args[[2]]
modules_path <- args[[3]]
output_dir <- args[[4]]
display_mode <- if (length(args) >= 5) args[[5]] else "displayed"
output_stem <- if (length(args) >= 6) args[[6]] else "reactome_enrichment_network_prior_layout_ggraph"
layout_mode <- if (length(args) >= 7) args[[7]] else "current"
if (!display_mode %in% c("displayed", "significant", "context")) {
    stop("display_mode must be 'displayed', 'significant', or 'context'.")
}
if (!layout_mode %in% c("current", "refined", "fr", "drl")) {
    stop("layout_mode must be 'current', 'refined', 'fr', or 'drl'.")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

nodes <- read.csv(nodes_path, check.names = FALSE)
edges <- read.csv(edges_path, check.names = FALSE)
modules <- read.csv(modules_path, check.names = FALSE)

if (display_mode == "significant") {
    nodes <- nodes[nodes$significant & is.finite(nodes$x) & is.finite(nodes$y), , drop = FALSE]
} else {
    nodes <- nodes[nodes$displayed & is.finite(nodes$x) & is.finite(nodes$y), , drop = FALSE]
}
nodes$name <- nodes$pathway
nodes$significance <- pmin(-log10(pmax(nodes$padj, .Machine$double.xmin)), 25)
nodes$node_alpha <- ifelse(
    nodes$significant,
    0.96,
    0.08 + 0.22 * pmin(nodes$significance / -log10(0.05), 1)
)
nodes$node_size <- 1.05
if (any(nodes$significant)) {
    nodes$node_size[nodes$significant] <- rescale(
        nodes$significance[nodes$significant], to = c(1.35, 3.25)
    )
}
nodes$node_outline <- ifelse(
    nodes$significant, alpha("#333333", 0.62), alpha("#FFFFFF", 0.30)
)
edges <- edges[
    edges$source_pathway %in% nodes$pathway &
        edges$target_pathway %in% nodes$pathway,
    , drop = FALSE
]
edges$from <- edges$source_pathway
edges$to <- edges$target_pathway

graph <- graph_from_data_frame(
    edges[, c("from", "to", "similarity")],
    directed = FALSE,
    vertices = nodes
)
coordinates <- as.matrix(nodes[match(V(graph)$name, nodes$pathway), c("x", "y")])

normalize_layout <- function(value) {
    value <- scale(value, center = TRUE, scale = FALSE)
    rotation <- eigen(cov(value), symmetric = TRUE)$vectors
    value <- value %*% rotation
    value[, 1] <- rescale(value[, 1], to = c(0.04, 0.96))
    value[, 2] <- rescale(value[, 2], to = c(0.04, 0.96))
    value
}

set.seed(0)
if (layout_mode == "refined") {
    coordinates <- layout_with_fr(
        graph, coords = coordinates, weights = E(graph)$similarity,
        niter = 1800, grid = "nogrid"
    )
    coordinates <- normalize_layout(coordinates)
} else if (layout_mode == "fr") {
    coordinates <- layout_with_fr(
        graph, weights = E(graph)$similarity,
        niter = 2400, grid = "nogrid"
    )
    coordinates <- normalize_layout(coordinates)
} else if (layout_mode == "drl") {
    coordinates <- layout_with_drl(graph, weights = E(graph)$similarity)
    coordinates <- normalize_layout(coordinates)
}

# Module labels must follow the selected layout rather than the coordinates
# stored by the original packed-module arrangement.
nodes$x <- coordinates[, 1]
nodes$y <- coordinates[, 2]
write.csv(
    nodes,
    file.path(output_dir, paste0(output_stem, "_nodes.csv")),
    row.names = FALSE
)

if (display_mode %in% c("significant", "context")) {
    significant_counts <- table(nodes$cluster[nodes$significant])
    modules <- modules[
        modules$cluster %in% as.integer(names(significant_counts)[significant_counts >= 6]),
        , drop = FALSE
    ]
}

# Replace generic word-frequency labels with pathway-family descriptions.
label_overrides <- c(
    `1` = "Cell cycle and APC/C proteolysis",
    `4` = "RNA polymerase II and TP53 transcription",
    `7` = "Chromatin, meiosis, and Pol I transcription",
    `8` = "TLR and innate immune signaling",
    `9` = "Mitotic checkpoint and Golgi trafficking",
    `14` = "Extracellular matrix and collagen organization",
    `16` = "Translation and ribosome biogenesis",
    `17` = "Phospholipid biosynthesis",
    `22` = "Mitosis and mitochondrial biogenesis",
    `23` = "Complement, phagocytosis, and immune interactions",
    `25` = "Nuclear transport and RNA processing",
    `34` = "Glycosaminoglycan metabolism and biosynthesis",
    `47` = "Receptor tyrosine kinase and PI3K/MAPK signaling",
    `48` = "Rho-family GTPase cycles",
    `53` = "DNA replication and repair",
    `71` = "mRNA processing and viral-host interactions",
    `99` = "Interleukin signaling"
)
matched_labels <- match(as.character(modules$cluster), names(label_overrides))
replace_labels <- !is.na(matched_labels)
modules$display_label[replace_labels] <- label_overrides[matched_labels[replace_labels]]

module_labels <- do.call(rbind, lapply(seq_len(nrow(modules)), function(index) {
    group <- modules$cluster[index]
    members <- nodes[nodes$cluster == group, , drop = FALSE]
    if (display_mode %in% c("significant", "context") && any(members$significant)) {
        members <- members[members$significant, , drop = FALSE]
    }
    data.frame(
        cluster = group,
        x = mean(members$x),
        y = mean(members$y),
        label = paste(strwrap(modules$display_label[index], width = 26), collapse = "\n")
    )
}))

maximum_nes <- max(abs(nodes$NES))
plot <- ggraph(graph, layout = coordinates) +
    geom_edge_link(
        aes(width = similarity, alpha = similarity),
        colour = "#777777", show.legend = FALSE
    ) +
    geom_node_point(
        aes(size = node_size, fill = NES, alpha = node_alpha),
        shape = 21, colour = "#00000000", stroke = 0
    ) +
    geom_label_repel(
        data = module_labels,
        aes(x = x, y = y, label = label),
        inherit.aes = FALSE,
        size = 2.65, lineheight = 0.92, fontface = "bold",
        colour = "#111111", fill = alpha("white", 0.88),
        label.size = 0, label.padding = unit(0.10, "lines"),
        box.padding = 0.55, point.padding = 0.18,
        force = 3.0, force_pull = 0.25,
        max.overlaps = Inf, min.segment.length = 0,
        segment.colour = alpha("#555555", 0.55), segment.size = 0.25,
        seed = 0
    ) +
    scale_edge_width(range = c(0.10, 0.50)) +
    scale_edge_alpha(range = c(0.03, 0.22)) +
    scale_size_identity(guide = "none") +
    scale_alpha_identity(guide = "none") +
    scale_fill_gradient2(
        name = expression("Normalized enrichment score with " * O[lambda]),
        low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
        midpoint = 0, limits = c(-maximum_nes, maximum_nes), oob = squish,
        guide = guide_colourbar(
            direction = "horizontal", title.position = "bottom",
            barwidth = unit(45, "mm"), barheight = unit(3.2, "mm"),
            ticks = FALSE, label = FALSE
        )
    ) +
    coord_equal(xlim = c(0, 1), ylim = c(-0.015, 1), clip = "off") +
    theme_graph(base_family = "Helvetica") +
    theme(
        legend.position = "bottom",
        legend.title = element_text(size = 10, colour = "#555555"),
        plot.margin = margin(15, 20, 10, 20)
    )

ggsave(
    file.path(output_dir, paste0(output_stem, ".jpg")),
    plot, width = 11.5, height = 9, dpi = 1200, bg = "white"
)
ggsave(
    file.path(output_dir, paste0(output_stem, ".pdf")),
    plot, width = 11.5, height = 9, device = cairo_pdf, bg = "white"
)

# Matched module summary for the communities labeled in the network. Means and
# ranges use every pathway in each selected community so the summary is not
# conditioned on pathway-level significance.
module_summary <- do.call(rbind, lapply(seq_len(nrow(modules)), function(index) {
    group <- modules$cluster[index]
    values <- nodes$NES[nodes$cluster == group]
    if (!length(values)) return(NULL)
    data.frame(
        cluster = group,
        label = modules$display_label[index],
        pathways = length(values),
        mean_NES = mean(values),
        q10_NES = unname(quantile(values, 0.10)),
        q90_NES = unname(quantile(values, 0.90)),
        minimum_NES = min(values),
        maximum_NES = max(values)
    )
}))
module_summary <- module_summary[order(module_summary$mean_NES), , drop = FALSE]
module_summary$label <- factor(module_summary$label, levels = module_summary$label)
write.csv(
    module_summary,
    file.path(output_dir, paste0(output_stem, "_module_summary.csv")),
    row.names = FALSE
)

summary_limit <- max(abs(c(module_summary$minimum_NES, module_summary$maximum_NES)))
summary_plot <- ggplot(module_summary, aes(y = label)) +
    geom_vline(xintercept = 0, colour = "#777777", linewidth = 0.45, linetype = 2) +
    geom_segment(
        aes(x = minimum_NES, xend = maximum_NES, yend = label),
        colour = "#D2D2D2", linewidth = 0.55
    ) +
    geom_segment(
        aes(x = q10_NES, xend = q90_NES, yend = label),
        colour = "#777777", linewidth = 1.25, lineend = "round"
    ) +
    geom_point(
        aes(x = mean_NES, fill = mean_NES),
        shape = 23, size = 4.2, colour = "#333333", stroke = 0.35
    ) +
    scale_fill_gradient2(
        name = expression("Mean NES with " * O[lambda]),
        low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
        midpoint = 0, limits = c(-summary_limit, summary_limit), oob = squish
    ) +
    scale_x_continuous(expand = expansion(mult = c(0.04, 0.05))) +
    labs(
        x = expression("Normalized enrichment score with " * O[lambda]),
        y = NULL
    ) +
    theme_classic(base_family = "Helvetica", base_size = 11) +
    theme(
        axis.text.y = element_text(size = 10, colour = "#222222"),
        axis.text.x = element_text(colour = "#444444"),
        axis.title.x = element_text(size = 11, margin = margin(t = 8)),
        axis.line.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.position = "right",
        legend.title = element_text(size = 10),
        plot.margin = margin(12, 16, 12, 12)
    )

summary_height <- max(5.8, 0.36 * nrow(module_summary) + 1.4)
ggsave(
    file.path(output_dir, paste0(output_stem, "_module_summary.jpg")),
    summary_plot, width = 9.2, height = summary_height, dpi = 1200, bg = "white"
)
ggsave(
    file.path(output_dir, paste0(output_stem, "_module_summary.pdf")),
    summary_plot, width = 9.2, height = summary_height, device = cairo_pdf, bg = "white"
)
