#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop(paste(
    "Usage: Rscript build_reactome_enrichmentmap.R",
    "fgsea_Reactome.csv membership_Reactome.csv",
    "reactome_official_branch_mapping.csv output_dir",
    "[similarity_threshold] [inflation] [significant|all]"
  ))
}

fgsea_path <- args[[1]]
membership_path <- args[[2]]
branch_path <- args[[3]]
output_dir <- args[[4]]
similarity_threshold <- if (length(args) >= 5) as.numeric(args[[5]]) else 0.375
inflation <- if (length(args) >= 6) as.numeric(args[[6]]) else 1.8
cluster_universe <- if (length(args) >= 7) args[[7]] else "significant"
if (!cluster_universe %in% c("significant", "all")) {
  stop("cluster_universe must be either 'significant' or 'all'.")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

blue <- "#2166AC"
red <- "#B2182B"
dark <- "#222222"
gray <- "#A6A6A6"

fgsea <- read.csv(fgsea_path, check.names = FALSE)
membership <- read.csv(membership_path, check.names = FALSE)
branches <- read.csv(branch_path, check.names = FALSE)

if (cluster_universe == "all") {
  nodes <- fgsea[is.finite(fgsea$NES), ]
} else {
  nodes <- fgsea[!is.na(fgsea$padj) & fgsea$padj < 0.05, ]
}
nodes <- nodes[order(nodes$pathway), ]
nodes$significant <- !is.na(nodes$padj) & nodes$padj < 0.05
nodes$plot_padj <- ifelse(is.finite(nodes$padj), nodes$padj, 1)
nodes$node <- seq_len(nrow(nodes))
branches <- branches[match(nodes$pathway, branches$pathway), ]
nodes$top_topic <- branches$top_topic
nodes$official_branch <- branches$official_branch

membership <- membership[membership$pathway %in% nodes$pathway, ]
gene_sets <- split(membership$gene, membership$pathway)
gene_sets <- lapply(gene_sets, unique)

combined_similarity <- matrix(0, nrow(nodes), nrow(nodes))
diag(combined_similarity) <- 1
for (i in seq_len(nrow(nodes) - 1)) {
  a <- gene_sets[[nodes$pathway[[i]]]]
  for (j in (i + 1):nrow(nodes)) {
    b <- gene_sets[[nodes$pathway[[j]]]]
    intersection <- length(intersect(a, b))
    if (!intersection) next
    union <- length(a) + length(b) - intersection
    jaccard <- intersection / union
    overlap <- intersection / min(length(a), length(b))
    value <- 0.5 * (jaccard + overlap)
    combined_similarity[i, j] <- value
    combined_similarity[j, i] <- value
  }
}

network <- combined_similarity
network[network < similarity_threshold] <- 0
diag(network) <- 1
nodes$has_connection <- rowSums(network > 0) > 1

normalize_columns <- function(x) {
  sums <- Matrix::colSums(x)
  x %*% Diagonal(x = 1 / pmax(sums, .Machine$double.eps))
}

markov_clustering <- function(adjacency, inflation = 1.8, prune = 1e-5,
                              tolerance = 1e-6, max_iterations = 100) {
  matrix <- Matrix(adjacency, sparse = TRUE)
  diag(matrix) <- 1
  matrix <- normalize_columns(matrix)
  delta <- Inf
  iteration <- 0
  for (iteration in seq_len(max_iterations)) {
    previous <- matrix
    matrix <- matrix %*% matrix
    matrix@x <- matrix@x ^ inflation
    matrix@x[matrix@x < prune] <- 0
    matrix <- drop0(matrix)
    matrix <- normalize_columns(matrix)
    delta <- max(abs(as.matrix(matrix - previous)))
    if (delta < tolerance) break
  }

  support <- as.matrix(matrix) > 1e-3
  support <- support | t(support)
  visited <- rep(FALSE, nrow(support))
  cluster <- integer(nrow(support))
  cluster_id <- 0
  for (start in seq_len(nrow(support))) {
    if (visited[[start]]) next
    cluster_id <- cluster_id + 1
    queue <- start
    visited[[start]] <- TRUE
    while (length(queue)) {
      current <- queue[[1]]
      queue <- queue[-1]
      cluster[[current]] <- cluster_id
      neighbors <- which(support[current, ] & !visited)
      if (length(neighbors)) {
        visited[neighbors] <- TRUE
        queue <- c(queue, neighbors)
      }
    }
  }
  list(cluster = cluster, matrix = matrix, iterations = iteration, delta = delta)
}

mcl <- markov_clustering(network, inflation = inflation)
nodes$cluster <- mcl$cluster
cluster_sizes <- table(nodes$cluster)
seed_clusters <- as.integer(names(cluster_sizes[cluster_sizes >= 6]))
nodes$displayed <- nodes$cluster %in% seed_clusters
queue <- which(nodes$displayed)
while (length(queue)) {
  current <- queue[[1]]
  queue <- queue[-1]
  neighbors <- which(network[current, ] > 0 & seq_len(nrow(network)) != current)
  neighbors <- neighbors[!nodes$displayed[neighbors]]
  if (length(neighbors)) {
    nodes$displayed[neighbors] <- TRUE
    queue <- c(queue, neighbors)
  }
}

clean_label <- function(x) {
  x <- sub("^REACTOME_", "", x)
  x <- gsub("_", " ", x)
  x <- tools::toTitleCase(tolower(x))
  substitutions <- c(
    "Dna" = "DNA", "Rna" = "RNA", "Mrna" = "mRNA", "Rrna" = "rRNA",
    "Nf Kappa B" = "NF-kappaB", "Interleukin" = "Interleukin",
    "T Cell" = "T-cell", "B Cell" = "B-cell", "Cell Cycle" = "Cell cycle"
  )
  for (from in names(substitutions)) {
    x <- gsub(from, substitutions[[from]], x, fixed = TRUE)
  }
  x
}

stop_words <- c(
  "REACTOME", "PATHWAY", "PATHWAYS", "SIGNALING", "SIGNAL", "REGULATION",
  "REGULATES", "REGULATED", "PROCESS", "PROCESSES", "RESPONSE", "RESPONSES",
  "THE", "AND", "OR", "OF", "IN", "BY", "TO", "FROM", "WITH", "THROUGH",
  "DEPENDENT", "MEDIATED", "ACTIVATION", "SYSTEM", "HUMAN", "CELL", "CELLS"
)

recurring_terms <- function(pathways, maximum = 3) {
  documents <- lapply(pathways, function(pathway) {
    words <- strsplit(gsub("[^A-Z0-9 ]", " ", sub("^REACTOME_", "", pathway)), " +")[[1]]
    unique(words[nchar(words) >= 3 & !words %in% stop_words])
  })
  counts <- sort(table(unlist(documents)), decreasing = TRUE)
  if (!length(counts)) return("Related pathways")
  words <- names(counts)[seq_len(min(maximum, length(counts)))]
  tools::toTitleCase(tolower(paste(words, collapse = " / ")))
}

curate_module_label <- function(generated, branch) {
  text <- paste(generated, branch)
  replacements <- list(
    "Mitochondrial translation" = "Mitochondrial / Translation / Elongation",
    "Translation and amino-acid stress" = "Translation / Acid / Amino",
    "E2F-mediated DNA replication" = "Dna / E2f / Replication",
    "Nuclear transport and RNA processing" = "Proteins / Sumoylation / Host|Nuclear / Sumoylation",
    "mRNA splicing and processing" = "Mrna / Processing / Capped|Processing of Capped",
    "Respiratory electron transport" = "Complex / Assembly / Electron",
    "mRNA decay and stress response" = "Mrna / Binds / Destabilizes|Binds / Destabilizes",
    "DNA replication and repair" = "Dna / Strand / Repair",
    "G1/S transcriptional control" = "Transcription / Complex / Control",
    "rRNA modification" = "Cytosol / Modification / Nucleus",
    "Fanconi-anemia repair" = "Anemia / Fanconi",
    "Cristae formation and ATP synthesis" = "Formation / Atp / Chemiosmotic",
    "Cell cycle and checkpoint control" = "Mitotic / Apc / Degradation",
    "miRNA biogenesis" = "Biogenesis / Microrna / Mirna",
    "piRNA biogenesis" = "Biogenesis / Interacting / Pirna",
    "Transcription and mRNA capping" = "Hiv / Elongation / Transcription",
    "Chromatin, repair, and RNA Pol I" = "Rna / Polymerase / Expression|Rna / Polymerase / Dna",
    "Cytosolic tRNA aminoacylation" = "Aminoacylation / Cytosolic / Trna",
    "TP53 cell-cycle transcription" = "Cycle / Genes / Tp53",
    "GPCR and chemokine-receptor signaling" = "Receptors / Bind / Binding",
    "Secretory trafficking and glycosylation" = "Transport / Anterograde / Golgi",
    "MAPK--FGFR signaling" = "Cascades / Family / Fgfr1",
    "Golgi trafficking" = "Golgi / Intra / Traffic",
    "Synaptic adhesion and signaling" = "Interactions / Neurexins / Neuroligins",
    "PI3K--AKT and CTLA4 signaling" = "Pi3k / Akt / Cancer",
    "Death-receptor and TNF signaling" = "Death / Receptor / Signalling",
    "Rho-family GTPase cycles" = "Cycle / Gtpase / Cdc42|Signaling by Rho GTPases",
    "Endosomal sorting and autophagy" = "Endosomal / Autophagy / Complex",
    "Peptide-hormone metabolism" = "Deacylation / Ghrelin / Hormone",
    "NOTCH signaling" = "Notch1 / Activated / Notch2",
    "Phospholipid remodeling" = "Acyl / Chain / Metabolism",
    "ERBB4 signaling" = "Erbb4 / Nuclear",
    "Innate immune signaling" = "Cascade / Like / Receptor",
    "Platelet and integrin signaling" = "Platelet / Aggregation / Ca2",
    "LXR-dependent cholesterol transport" = "Nr1h2 / Nr1h3 / Cholesterol",
    "TGF-beta and BMP signaling" = "Bmp / Family / Members",
    "Interleukin signaling" = "Interleukin / Interleukins / Other",
    "Mechanosensing" = "Mechanical / Activates / Cellular",
    "MET-dependent motility signaling" = "Met / Developmental / Activates",
    "Immune-cell interactions and phagocytosis" = "Infection / Complement / Interactions",
    "Extrinsic apoptosis" = "Caspase / via / Apoptotic",
    "Extracellular matrix and collagen" = "Matrix / Collagen / Formation",
    "Bacterial attachment and entry" = "Infection / Attachment / Bacteria",
    "Circadian and adipogenic transcription" = "Expression / Clock / Npas2",
    "FLT3 signaling" = "Flt3 / Negative",
    "Glycosaminoglycan metabolism and viral entry" = "Gag / Attachment / Biosynthesis|Gag / Metabolism / Attachment",
    "Noncanonical NF-kappaB signaling" = "Canonical / Non / Bind",
    "Scavenger-receptor uptake" = "Receptors / Binding / Class",
    "Eph--ephrin signaling" = "Ephrin / Eph / Ephb",
    "Lipoprotein assembly and remodeling" = "Lipoprotein / Plasma / Assembly",
    "ZBP1 and type I interferon signaling" = "Zbp1 / Dai / Ifns",
    "Semaphorin signaling" = "Interactions / Semaphorin / Axon",
    "Secretory trafficking" = "Golgi / Transport / Anterograde",
    "Growth-factor and developmental signaling" = "Met / Developmental / Infection"
  )
  for (label in names(replacements)) {
    if (grepl(replacements[[label]], text, ignore.case = TRUE)) return(label)
  }
  if (!is.na(branch) && nchar(branch) <= 52) return(branch)
  generated
}

cluster_rows <- lapply(split(nodes, nodes$cluster), function(x) {
  branch_table <- sort(table(x$official_branch), decreasing = TRUE)
  topic_table <- sort(table(x$top_topic), decreasing = TRUE)
  branch <- if (length(branch_table)) names(branch_table)[[1]] else NA_character_
  topic <- if (length(topic_table)) names(topic_table)[[1]] else NA_character_
  branch_purity <- if (length(branch_table)) max(branch_table) / nrow(x) else NA_real_
  generated <- recurring_terms(x$pathway)
  automatic_label <- if (!is.na(branch) && branch_purity >= 0.40 && nchar(branch) <= 52) {
    branch
  } else {
    generated
  }
  label <- curate_module_label(generated, automatic_label)
  data.frame(
    cluster = x$cluster[[1]], pathways = nrow(x), median_NES = median(x$NES),
    q10_NES = unname(quantile(x$NES, 0.10)),
    q90_NES = unname(quantile(x$NES, 0.90)),
    positive = sum(x$NES > 0), negative = sum(x$NES < 0),
    dominant_topic = topic, dominant_branch = branch,
    branch_purity = branch_purity, generated_label = generated,
    display_label = label, stringsAsFactors = FALSE
  )
})
cluster_summary <- do.call(rbind, cluster_rows)
cluster_summary <- cluster_summary[order(cluster_summary$median_NES), ]
row.names(cluster_summary) <- NULL

edge_index <- which(upper.tri(network) & network >= similarity_threshold, arr.ind = TRUE)
edges <- data.frame(
  source = edge_index[, 1], target = edge_index[, 2],
  similarity = network[edge_index],
  source_pathway = nodes$pathway[edge_index[, 1]],
  target_pathway = nodes$pathway[edge_index[, 2]]
)

write.csv(nodes, file.path(output_dir, "reactome_enrichmentmap_nodes.csv"), row.names = FALSE)
write.csv(edges, file.path(output_dir, "reactome_enrichmentmap_edges.csv"), row.names = FALSE)
write.csv(cluster_summary, file.path(output_dir, "reactome_autoannotate_clusters.csv"), row.names = FALSE)

# A packed spectral layout keeps every retained pathway visible while making
# the MCL modules legible. Placement within a module follows graph topology;
# module packing is ordered only by enrichment direction and module size.
local_coordinates <- function(indices) {
  if (length(indices) == 1) return(matrix(c(0, 0), nrow = 1))
  if (length(indices) == 2) return(matrix(c(-0.5, 0, 0.5, 0), ncol = 2, byrow = TRUE))
  subnetwork <- network[indices, indices, drop = FALSE]
  degrees <- rowSums(subnetwork)
  normalized <- diag(1 / sqrt(pmax(degrees, 1e-8))) %*% subnetwork %*%
    diag(1 / sqrt(pmax(degrees, 1e-8)))
  decomposition <- eigen(normalized, symmetric = TRUE)
  dimensions <- min(3, ncol(decomposition$vectors))
  coordinates <- decomposition$vectors[, 2:dimensions, drop = FALSE]
  if (ncol(coordinates) == 1) coordinates <- cbind(coordinates, seq(-1, 1, length.out = nrow(coordinates)))
  coordinates <- scale(coordinates, center = TRUE, scale = FALSE)
  maximum <- max(sqrt(rowSums(coordinates ^ 2)))
  if (!is.finite(maximum) || maximum == 0) maximum <- 1
  coordinates / maximum
}

clusters <- split(seq_len(nrow(nodes)), nodes$cluster)
display_cluster_ids <- sort(unique(nodes$cluster[nodes$displayed]))
major_ids <- intersect(
  cluster_summary$cluster[cluster_summary$pathways >= 4],
  display_cluster_ids
)
minor_ids <- setdiff(display_cluster_ids, major_ids)
coordinates <- matrix(NA_real_, nrow(nodes), 2)

# Lay out the major modules from their overlap connections alone. The module
# meta-graph uses the strongest retained pathway overlap between each pair of
# modules. Classical scaling of shortest-path distances gives deterministic,
# outcome-free module positions. NES enters only through node color below.
major_count <- length(major_ids)
major_similarity <- matrix(0, major_count, major_count)
for (a in seq_len(major_count - 1)) {
  indices_a <- clusters[[as.character(major_ids[[a]])]]
  for (b in (a + 1):major_count) {
    indices_b <- clusters[[as.character(major_ids[[b]])]]
    value <- max(network[indices_a, indices_b, drop = FALSE])
    major_similarity[a, b] <- value
    major_similarity[b, a] <- value
  }
}

force_layout <- function(adjacency, iterations = 2500) {
  n <- nrow(adjacency)
  if (n == 1) return(matrix(c(0, 0), ncol = 2))
  if (n == 2) return(matrix(c(-1, 0, 1, 0), ncol = 2, byrow = TRUE))
  angles <- 2 * pi * (seq_len(n) - 1) / n
  position <- cbind(cos(angles), sin(angles))
  spring_length <- sqrt(4 / n)
  maximum_weight <- max(adjacency)
  if (!is.finite(maximum_weight) || maximum_weight == 0) maximum_weight <- 1

  for (iteration in seq_len(iterations)) {
    displacement <- matrix(0, n, 2)
    for (a in seq_len(n - 1)) {
      for (b in (a + 1):n) {
        difference <- position[a, ] - position[b, ]
        distance <- max(1e-4, sqrt(sum(difference ^ 2)))
        direction <- difference / distance
        repulsion <- spring_length ^ 2 / distance
        displacement[a, ] <- displacement[a, ] + repulsion * direction
        displacement[b, ] <- displacement[b, ] - repulsion * direction
        if (adjacency[a, b] > 0) {
          weight <- 0.25 + 0.75 * adjacency[a, b] / maximum_weight
          attraction <- weight * distance ^ 2 / spring_length
          displacement[a, ] <- displacement[a, ] - attraction * direction
          displacement[b, ] <- displacement[b, ] + attraction * direction
        }
      }
    }
    displacement <- displacement - 0.025 * position
    temperature <- 0.12 * (1 - iteration / (iterations + 1)) + 0.002
    magnitudes <- sqrt(rowSums(displacement ^ 2))
    steps <- pmin(magnitudes, temperature) / pmax(magnitudes, 1e-8)
    position <- position + displacement * steps
  }
  position
}

if (major_count >= 3) {
  major_centers <- force_layout(major_similarity)
} else if (major_count == 2) {
  major_centers <- matrix(c(-1, 0, 1, 0), ncol = 2, byrow = TRUE)
} else {
  major_centers <- matrix(c(0, 0), ncol = 2)
}
major_centers <- scale(major_centers, center = TRUE, scale = FALSE)
rotation <- eigen(cov(major_centers), symmetric = TRUE)$vectors
major_centers <- major_centers %*% rotation
if (major_centers[which.min(major_ids), 1] > 0) major_centers[, 1] <- -major_centers[, 1]
radius <- max(sqrt(rowSums(major_centers ^ 2)))
if (!is.finite(radius) || radius == 0) radius <- 1
major_centers <- major_centers / radius
major_centers[, 1] <- 0.50 + 0.36 * major_centers[, 1]
major_centers[, 2] <- 0.50 + 0.37 * major_centers[, 2]
major_anchors <- major_centers
major_radii <- vapply(major_ids, function(id) {
  0.020 + 0.006 * sqrt(length(clusters[[as.character(id)]]))
}, numeric(1))
for (iteration in seq_len(1600)) {
  force <- 0.014 * (major_anchors - major_centers)
  for (a in seq_len(major_count - 1)) {
    for (b in (a + 1):major_count) {
      difference <- major_centers[a, ] - major_centers[b, ]
      distance <- sqrt(sum(difference ^ 2))
      minimum <- major_radii[[a]] + major_radii[[b]] + 0.035
      if (distance < minimum) {
        if (distance < 1e-8) {
          difference <- c(1, 0)
          distance <- 1
        }
        displacement <- 0.12 * (minimum - distance) * difference / distance
        force[a, ] <- force[a, ] + displacement
        force[b, ] <- force[b, ] - displacement
      }
    }
  }
  major_centers <- major_centers + force
  major_centers[, 1] <- pmin(0.95, pmax(0.05, major_centers[, 1]))
  major_centers[, 2] <- pmin(0.95, pmax(0.05, major_centers[, 2]))
}
row.names(major_centers) <- as.character(major_ids)

for (position in seq_along(major_ids)) {
  id <- major_ids[[position]]
  indices <- clusters[[as.character(id)]]
  local <- local_coordinates(indices)
  radius <- major_radii[[position]]
  coordinates[indices, 1] <- major_centers[position, 1] + radius * local[, 1]
  coordinates[indices, 2] <- major_centers[position, 2] + radius * local[, 2]
}

if (length(minor_ids)) {
  # Small modules belong visually between the two dominant systems. Their
  # anchors are determined by the larger modules to which they retain overlap
  # edges; a constrained relaxation then prevents the small modules from
  # collapsing onto one another. The topology within each module is preserved.
  major_indices <- unlist(clusters[as.character(major_ids)], use.names = FALSE)
  minor_centers <- matrix(NA_real_, length(minor_ids), 2)
  minor_anchors <- matrix(NA_real_, length(minor_ids), 2)
  minor_radii <- numeric(length(minor_ids))
  row.names(minor_centers) <- row.names(minor_anchors) <- as.character(minor_ids)

  for (position in seq_along(minor_ids)) {
    id <- minor_ids[[position]]
    indices <- clusters[[as.character(id)]]
    minor_radii[[position]] <- 0.006 + 0.006 * sqrt(length(indices))
    similarities <- colSums(network[indices, major_indices, drop = FALSE])
    connected <- similarities > 0
    if (any(connected)) {
      neighbor_coordinates <- coordinates[major_indices[connected], , drop = FALSE]
      weights <- similarities[connected]
      anchor <- colSums(neighbor_coordinates * weights) / sum(weights)
      # Preserve which major system supplies the connection while reserving a
      # distinct central region for modules too small to receive a main label.
      anchor[[1]] <- 0.5 + 0.32 * (anchor[[1]] - 0.5)
      minor_anchors[position, ] <- anchor
    } else {
      minor_anchors[position, ] <- c(
        0.5,
        0.08 + 0.84 * (position - 1) / max(1, length(minor_ids) - 1)
      )
    }
  }

  set.seed(0)
  minor_centers <- minor_anchors + matrix(
    runif(length(minor_ids) * 2, -0.012, 0.012), ncol = 2
  )

  for (iteration in seq_len(700)) {
    force <- 0.030 * (minor_anchors - minor_centers)
    for (a in seq_len(length(minor_ids) - 1)) {
      for (b in (a + 1):length(minor_ids)) {
        difference <- minor_centers[a, ] - minor_centers[b, ]
        distance <- sqrt(sum(difference ^ 2))
        minimum <- minor_radii[[a]] + minor_radii[[b]] + 0.012
        if (distance < minimum) {
          if (distance < 1e-8) {
            difference <- c(1, 0)
            distance <- 1
          }
          displacement <- 0.10 * (minimum - distance) * difference / distance
          force[a, ] <- force[a, ] + displacement
          force[b, ] <- force[b, ] - displacement
        }
      }
    }
    minor_centers <- minor_centers + force
    minor_centers[, 1] <- pmin(0.70, pmax(0.30, minor_centers[, 1]))
    minor_centers[, 2] <- pmin(0.96, pmax(0.04, minor_centers[, 2]))
  }

  for (position in seq_along(minor_ids)) {
    id <- minor_ids[[position]]
    indices <- clusters[[as.character(id)]]
    local <- local_coordinates(indices)
    radius <- minor_radii[[position]]
    coordinates[indices, 1] <- minor_centers[position, 1] + radius * local[, 1]
    coordinates[indices, 2] <- minor_centers[position, 2] + radius * local[, 2]
  }
}

nodes$x <- coordinates[, 1]
nodes$y <- coordinates[, 2]
write.csv(nodes, file.path(output_dir, "reactome_enrichmentmap_nodes_with_layout.csv"), row.names = FALSE)

color_value <- function(value, limit = max(abs(nodes$NES))) {
  value <- pmax(-limit, pmin(limit, value)) / limit
  if (value < 0) {
    rgb(colorRamp(c(blue, "#F7F7F7"))(value + 1), maxColorValue = 255)
  } else {
    rgb(colorRamp(c("#F7F7F7", red))(value), maxColorValue = 255)
  }
}

wrap_label <- function(x, width = 27) paste(strwrap(x, width = width), collapse = "\n")

png(file.path(output_dir, "reactome_enrichmentmap_autoannotate.png"),
    width = 4200, height = 2600, res = 300, bg = "white")
layout(matrix(c(1, 2), nrow = 1), widths = c(1.75, 1))

par(mar = c(1.2, 1.2, 3.7, 0.8), family = "sans", xpd = NA)
plot.new()
plot.window(xlim = c(0, 1), ylim = c(0, 1), asp = 1)
title("Reactome EnrichmentMap", line = 1.3, cex.main = 1.25, font.main = 2)
mtext("Displayed pathways have at least one retained gene-set-overlap edge", side = 3,
      line = 0.05, cex = 0.82, col = "#555555")

for (row in seq_len(nrow(edges))) {
  i <- edges$source[[row]]
  j <- edges$target[[row]]
  alpha <- 0.05 + 0.20 * (edges$similarity[[row]] - similarity_threshold) /
    max(1e-8, 1 - similarity_threshold)
  segments(nodes$x[[i]], nodes$y[[i]], nodes$x[[j]], nodes$y[[j]],
           col = adjustcolor("#777777", alpha.f = alpha), lwd = 0.35)
}

point_sizes <- 0.45 + 0.45 * sqrt(pmin(-log10(pmax(nodes$plot_padj, 1e-50)), 20) / 20)
display_nodes <- nodes$displayed & is.finite(nodes$x) & is.finite(nodes$y)
points(nodes$x[display_nodes], nodes$y[display_nodes], pch = 21,
       bg = vapply(nodes$NES[display_nodes], color_value, character(1)),
       col = adjustcolor("white", 0.75), lwd = 0.35,
       cex = point_sizes[display_nodes])

label_clusters <- cluster_summary[cluster_summary$pathways >= 6, ]
for (row in seq_len(nrow(label_clusters))) {
  id <- label_clusters$cluster[[row]]
  indices <- clusters[[as.character(id)]]
  x <- mean(nodes$x[indices])
  y <- max(nodes$y[indices]) + 0.026
  text(x, y, wrap_label(label_clusters$display_label[[row]], 24),
       cex = 0.52, font = 2, col = dark)
}

legend_x <- seq(0.38, 0.62, length.out = 101)
legend_y <- rep(0.035, length(legend_x))
points(legend_x, legend_y, pch = 15, cex = 0.62,
       col = vapply(seq(-max(abs(nodes$NES)), max(abs(nodes$NES)), length.out = 101),
                    color_value, character(1)))
text(0.36, 0.035, "lower", adj = 1, cex = 0.62, col = "#555555")
text(0.64, 0.035, "higher", adj = 0, cex = 0.62, col = "#555555")
text(0.5, 0.012, expression("Normalized enrichment score with " * O[lambda]^infinity),
     cex = 0.67, col = "#555555")

par(mar = c(1.2, 1.0, 3.7, 1.2), family = "sans", xpd = NA)
plot.new()
plot.window(xlim = c(-1.15, 1.15), ylim = c(0.5, nrow(label_clusters) + 0.5))
title("AutoAnnotate-style module summary", line = 1.3, cex.main = 1.25, font.main = 2)
mtext("MCL modules labeled by official branch or recurring pathway terms",
      side = 3, line = 0.05, cex = 0.82, col = "#555555")
abline(v = 0, col = "#888888", lwd = 0.8)

label_clusters <- label_clusters[order(label_clusters$median_NES), ]
y <- seq_len(nrow(label_clusters))
for (row in seq_len(nrow(label_clusters))) {
  values <- c(label_clusters$q10_NES[[row]], label_clusters$median_NES[[row]],
              label_clusters$q90_NES[[row]])
  scaled <- values / max(abs(nodes$NES))
  segments(scaled[[1]], y[[row]], scaled[[3]], y[[row]], col = "#777777", lwd = 1.2)
  points(scaled[[2]], y[[row]], pch = 21,
         bg = color_value(label_clusters$median_NES[[row]]), col = "white",
         lwd = 0.7, cex = 1.2 + 0.19 * sqrt(label_clusters$pathways[[row]]))
  if (label_clusters$median_NES[[row]] < 0) {
    text(1.12, y[[row]], wrap_label(label_clusters$display_label[[row]], 29),
         adj = 1, cex = 0.62, col = dark)
  } else {
    text(-1.12, y[[row]], wrap_label(label_clusters$display_label[[row]], 29),
         adj = 0, cex = 0.62, col = dark)
  }
}
axis(1, at = c(-1, 0, 1), labels = c("lower", "0", "higher"),
     tick = FALSE, line = -0.5, cex.axis = 0.7, col.axis = "#555555")
dev.off()

# -----------------------------------------------------------------------------
# Standalone module-focused map and summary: clusters with >= 6 pathways
# -----------------------------------------------------------------------------

map_label_clusters <- cluster_summary[
  cluster_summary$cluster %in% display_cluster_ids &
    cluster_summary$pathways >= 6,
]

png(file.path(output_dir, "reactome_enrichmentmap_modules_ge6.png"),
    width = 4400, height = 3200, res = 300, bg = "white")
par(mar = c(1.3, 1.0, 3.7, 1.0), family = "sans", xpd = NA)
plot.new()
plot.window(xlim = c(0, 1), ylim = c(0, 1), asp = 1)
title("Reactome EnrichmentMap", line = 1.3, cex.main = 1.35, font.main = 2)
mtext("Modules with at least six pathways are labeled; connected smaller modules are retained",
      side = 3, line = 0.05, cex = 0.86, col = "#555555")

for (row in seq_len(nrow(edges))) {
  i <- edges$source[[row]]
  j <- edges$target[[row]]
  if (!nodes$displayed[[i]] || !nodes$displayed[[j]]) next
  alpha <- 0.06 + 0.24 * (edges$similarity[[row]] - similarity_threshold) /
    max(1e-8, 1 - similarity_threshold)
  segments(nodes$x[[i]], nodes$y[[i]], nodes$x[[j]], nodes$y[[j]],
           col = adjustcolor("#777777", alpha.f = alpha), lwd = 0.45)
}
points(nodes$x[display_nodes], nodes$y[display_nodes], pch = 21,
       bg = vapply(nodes$NES[display_nodes], color_value, character(1)),
       col = adjustcolor("white", 0.78), lwd = 0.40,
       cex = point_sizes[display_nodes])

for (row in seq_len(nrow(map_label_clusters))) {
  id <- map_label_clusters$cluster[[row]]
  indices <- clusters[[as.character(id)]]
  indices <- indices[nodes$displayed[indices]]
  x <- mean(nodes$x[indices])
  y <- max(nodes$y[indices]) + 0.024
  text(x, y, wrap_label(map_label_clusters$display_label[[row]], 26),
       cex = 0.57, font = 2, col = dark)
}

legend_x <- seq(0.39, 0.61, length.out = 101)
points(legend_x, rep(0.018, length(legend_x)), pch = 15, cex = 0.64,
       col = vapply(seq(-max(abs(nodes$NES)), max(abs(nodes$NES)), length.out = 101),
                    color_value, character(1)))
text(0.37, 0.018, "lower", adj = 1, cex = 0.65, col = "#555555")
text(0.63, 0.018, "higher", adj = 0, cex = 0.65, col = "#555555")
text(0.5, -0.010, expression("Normalized enrichment score with " * O[lambda]^infinity),
     cex = 0.69, col = "#555555")
dev.off()

map_label_clusters <- map_label_clusters[order(map_label_clusters$median_NES), ]
y <- seq_len(nrow(map_label_clusters))
png(file.path(output_dir, "reactome_module_annotation_summary_ge6.png"),
    width = 3400, height = 2800, res = 300, bg = "white")
par(mar = c(4.0, 18.0, 4.2, 1.5), family = "sans")
plot.new()
plot.window(
  xlim = c(min(map_label_clusters$q10_NES) - 0.15,
           max(map_label_clusters$q90_NES) + 0.15),
  ylim = c(0.4, nrow(map_label_clusters) + 0.6)
)
abline(v = 0, col = "#888888", lwd = 0.8)
segments(map_label_clusters$q10_NES, y, map_label_clusters$q90_NES, y,
         col = "#777777", lwd = 1.35)
points(map_label_clusters$median_NES, y, pch = 21,
       bg = vapply(map_label_clusters$median_NES, color_value, character(1)),
       col = "white", lwd = 0.7,
       cex = 1.0 + 0.18 * sqrt(map_label_clusters$pathways))
axis(1, cex.axis = 0.75, col.axis = "#555555")
axis(2, at = y,
     labels = vapply(map_label_clusters$display_label, wrap_label, character(1), width = 35),
     las = 1, tick = FALSE, cex.axis = 0.68, line = -0.5)
title("Overlap-defined Reactome modules", cex.main = 1.25, font.main = 2, line = 1.2)
mtext(if (cluster_universe == "all") {
        "Only modules containing at least six pathways"
      } else {
        "Only modules containing at least six significant pathways"
      },
      side = 3, line = 0.1, cex = 0.82, col = "#555555")
mtext("Median module NES", side = 1, line = 2.4, cex = 0.80)
dev.off()

write.csv(
  map_label_clusters,
  file.path(output_dir, "reactome_modules_ge6.csv"),
  row.names = FALSE
)

cat("Reactome pathways in clustering universe:", nrow(nodes), "\n")
cat("Similarity threshold:", similarity_threshold, "| edges:", nrow(edges), "\n")
cat("MCL inflation:", inflation, "| clusters:", nrow(cluster_summary),
    "| iterations:", mcl$iterations, "| delta:", mcl$delta, "\n")
cat("Clusters with at least 6 pathways:", nrow(label_clusters), "\n")
print(label_clusters[, c("cluster", "pathways", "median_NES", "dominant_branch",
                          "branch_purity", "generated_label", "display_label")],
      row.names = FALSE)

