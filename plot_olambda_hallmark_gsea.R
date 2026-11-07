#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript plot_olambda_hallmark_gsea.R fgsea_Hallmark.csv output_dir")
}

hallmark_path <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

clean_name <- function(x) {
  x <- sub("^HALLMARK_", "", x)
  x <- gsub("_", " ", x)
  x <- tools::toTitleCase(tolower(x))
  replacements <- c(
    "Dna" = "DNA", "Rna" = "RNA", "Myc" = "MYC", "E2f" = "E2F",
    "G2m" = "G2M", "Mtorc1" = "mTORC1", "Il6" = "IL6",
    "Jak" = "JAK", "Stat3" = "STAT3", "Tnfa" = "TNF-alpha",
    "Nfkb" = "NF-kappaB", "Tgf Beta" = "TGF-beta"
  )
  for (from in names(replacements)) {
    x <- gsub(from, replacements[[from]], x, fixed = TRUE)
  }
  x
}

hallmark <- read.csv(hallmark_path, check.names = FALSE)
hallmark <- hallmark[is.finite(hallmark$NES), , drop = FALSE]
hallmark$significant <- !is.na(hallmark$padj) & hallmark$padj < 0.05
hallmark <- hallmark[hallmark$significant, , drop = FALSE]
if (!nrow(hallmark)) stop("No Hallmark pathways have padj < 0.05.")

hallmark$label <- clean_name(hallmark$pathway)
hallmark$display_q <- pmin(-log10(pmax(hallmark$padj, 1e-75)), 25)
hallmark <- hallmark[order(hallmark$NES), , drop = FALSE]
hallmark$label <- factor(hallmark$label, levels = hallmark$label)
write.csv(hallmark, file.path(output_dir, "hallmark_significant_pathways.csv"), row.names = FALSE)

limit <- max(abs(hallmark$NES))
plot <- ggplot(hallmark, aes(x = NES, y = label)) +
  geom_vline(xintercept = 0, colour = "#777777", linewidth = 0.5) +
  geom_segment(
    aes(x = 0, xend = NES, yend = label, colour = NES),
    linewidth = 0.95
  ) +
  geom_point(aes(fill = NES, size = display_q), shape = 21, colour = NA) +
  scale_colour_gradient2(
    low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
    midpoint = 0, limits = c(-limit, limit), guide = "none"
  ) +
  scale_fill_gradient2(
    name = expression("NES with " * O[lambda]),
    low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
    midpoint = 0, limits = c(-limit, limit), oob = squish
  ) +
  scale_size_continuous(
    name = expression(-log[10](italic(q))), range = c(2.8, 7.0)
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.06))) +
  labs(
    x = expression("Normalized enrichment score with " * O[lambda]),
    y = NULL
  ) +
  theme_classic(base_family = "Helvetica", base_size = 11) +
  theme(
    axis.text.y = element_text(size = 9.5, colour = "#222222"),
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.title.x = element_text(margin = margin(t = 8)),
    legend.title = element_text(size = 10),
    plot.margin = margin(12, 16, 12, 12)
  )

height <- max(6.5, 0.35 * nrow(hallmark) + 1.5)
ggsave(
  file.path(output_dir, "hallmark_significant_enrichment.jpg"),
  plot, width = 9.5, height = height, dpi = 1200, bg = "white"
)
ggsave(
  file.path(output_dir, "hallmark_significant_enrichment.pdf"),
  plot, width = 9.5, height = height, device = cairo_pdf, bg = "white"
)

cat("Significant Hallmark pathways:", nrow(hallmark), "\n")
cat("Saved:", output_dir, "\n")
