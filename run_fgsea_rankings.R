# Preranked GSEA for complete Bayesian gene-association rankings

suppressPackageStartupMessages({
    library(data.table)
    library(fgsea)
    library(msigdbr)
})

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) < 2) {
    stop(paste(
        "Usage: Rscript run_fgsea_rankings.R ranking.rnk output_dir",
        "[min_size] [max_size] [n_perm_simple]"
    ))
}

ranking_path <- arguments[[1]]
output_dir <- arguments[[2]]
min_size <- if (length(arguments) >= 3) as.integer(arguments[[3]]) else 15L
max_size <- if (length(arguments) >= 4) as.integer(arguments[[4]]) else 500L
n_perm_simple <- if (length(arguments) >= 5) as.integer(arguments[[5]]) else 100000L
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

ranking <- fread(ranking_path, header = FALSE, col.names = c("gene", "score"))
ranking <- ranking[is.finite(score) & !duplicated(gene)]
# Keep the full ranking; gene names break ties consistently
setorder(ranking, -score, gene)
statistics <- ranking$score
names(statistics) <- ranking$gene

collections <- list(
    Hallmark = msigdbr(species = "Homo sapiens", collection = "H"),
    Reactome = msigdbr(
        species = "Homo sapiens", collection = "C2", subcollection = "CP:REACTOME"
    )
)

all_results <- list()
collection_versions <- list()
for (collection_name in names(collections)) {
    membership <- as.data.table(collections[[collection_name]])
    msigdb_version <- if ("db_version" %in% names(membership)) {
        paste(sort(unique(membership$db_version)), collapse = "|")
    } else {
        NA_character_
    }
    collection_versions[[collection_name]] <- data.table(
        collection = collection_name,
        msigdb_version = msigdb_version,
        pathways_before_size_filter = uniqueN(membership$gs_name),
        minimum_pathway_size = min_size,
        maximum_pathway_size = max_size,
        simple_permutations = n_perm_simple
    )
    pathways <- split(membership$gene_symbol, membership$gs_name)
    result <- as.data.table(fgseaMultilevel(
        pathways = pathways, stats = statistics,
        minSize = min_size, maxSize = max_size,
        eps = 0,
        nPermSimple = n_perm_simple,
        gseaParam = 1
    ))
    result[, collection := collection_name]
    result[, leadingEdge := vapply(leadingEdge, paste, collapse = "|", FUN.VALUE = character(1))]
    result <- result[order(padj, -abs(NES), na.last = TRUE)]
    fwrite(result, file.path(output_dir, paste0("fgsea_", collection_name, ".csv")))
    fwrite(
        membership[, .(collection = collection_name, pathway = gs_name, gene = gene_symbol)],
        file.path(output_dir, paste0("membership_", collection_name, ".csv"))
    )
    all_results[[collection_name]] <- result
    cat(collection_name, ":", nrow(result), "pathways\n")
}

combined <- rbindlist(all_results, use.names = TRUE, fill = TRUE)
# Also report FDR across both collections
combined[, padj_all_collections := p.adjust(pval, method = "BH")]
combined <- combined[order(padj, -abs(NES), na.last = TRUE)]
fwrite(combined, file.path(output_dir, "fgsea_all_collections.csv"))
fwrite(rbindlist(collection_versions), file.path(output_dir, "fgsea_collection_versions.csv"))

writeLines(capture.output(sessionInfo()), file.path(output_dir, "R_session_info.txt"))
cat("Simple permutations:", format(n_perm_simple, big.mark = ","), "\n")
cat("Saved:", output_dir, "\n")
