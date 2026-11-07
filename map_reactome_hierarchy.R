#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop(paste(
    "Usage: Rscript map_reactome_hierarchy.R fgsea_Reactome.csv",
    "LowerLevelPathway2Topic_Human.txt ReactomePathwaysRelation.txt",
    "ReactomePathways.txt output.csv"
  ))
}

fgsea_path <- args[[1]]
topic_path <- args[[2]]
relation_path <- args[[3]]
pathway_path <- args[[4]]
output_path <- args[[5]]
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

fgsea <- read.csv(fgsea_path, check.names = FALSE)
topics <- read.delim(topic_path, check.names = FALSE)
relations <- read.delim(
  relation_path, header = FALSE, col.names = c("parent", "child")
)
pathways <- read.delim(
  pathway_path, header = FALSE,
  col.names = c("stable_id", "pathway_name", "species"),
  quote = "", comment.char = "", check.names = FALSE
)

relations <- relations[
  grepl("^R-HSA-", relations$parent) & grepl("^R-HSA-", relations$child),
]
pathways <- pathways[pathways$species == "Homo sapiens", ]

normalize_name <- function(x) gsub("[^A-Z0-9]", "", toupper(x))
fgsea$key <- normalize_name(sub("^REACTOME_", "", fgsea$pathway))
topics$key <- normalize_name(topics$SubPathwayName)

name_to_id <- topics[
  !duplicated(topics$key),
  c("key", "SubPathwayID", "TopicID", "TopicName")
]
fgsea$stable_id <- name_to_id$SubPathwayID[match(fgsea$key, name_to_id$key)]
fgsea$top_topic_id <- name_to_id$TopicID[match(fgsea$key, name_to_id$key)]
fgsea$top_topic <- name_to_id$TopicName[match(fgsea$key, name_to_id$key)]

topic_names <- unique(topics[, c("TopicID", "TopicName")])
topic_names$key <- normalize_name(topic_names$TopicName)
missing <- is.na(fgsea$stable_id)
fgsea$stable_id[missing] <- topic_names$TopicID[match(fgsea$key[missing], topic_names$key)]
fgsea$top_topic_id[missing] <- topic_names$TopicID[match(fgsea$key[missing], topic_names$key)]
fgsea$top_topic[missing] <- topic_names$TopicName[match(fgsea$key[missing], topic_names$key)]

pathway_name <- setNames(pathways$pathway_name, pathways$stable_id)
parents_by_child <- split(relations$parent, relations$child)

first_branch_below_topic <- function(stable_id, topic_id) {
  if (is.na(stable_id) || is.na(topic_id)) return(NA_character_)
  if (stable_id == topic_id) return(topic_id)
  queue <- stable_id
  visited <- character()
  branch_candidates <- character()
  for (depth in seq_len(20)) {
    next_queue <- character()
    for (node in queue) {
      parents <- unique(unlist(parents_by_child[node], use.names = FALSE))
      parents <- setdiff(parents, visited)
      if (topic_id %in% parents) {
        branch_candidates <- c(branch_candidates, node)
      } else {
        next_queue <- c(next_queue, parents)
      }
    }
    if (length(branch_candidates)) break
    visited <- unique(c(visited, queue))
    queue <- unique(next_queue)
    if (!length(queue)) break
  }
  if (!length(branch_candidates)) return(NA_character_)
  sort(unique(branch_candidates))[[1]]
}

fgsea$official_branch_id <- mapply(
  first_branch_below_topic,
  fgsea$stable_id,
  fgsea$top_topic_id,
  USE.NAMES = FALSE
)
fgsea$official_branch <- unname(pathway_name[fgsea$official_branch_id])
same_as_topic <- fgsea$official_branch_id == fgsea$top_topic_id
fgsea$official_branch[same_as_topic & !is.na(same_as_topic)] <-
  fgsea$top_topic[same_as_topic & !is.na(same_as_topic)]

write.csv(fgsea, output_path, row.names = FALSE)
cat(
  "Mapped", sum(!is.na(fgsea$official_branch)), "of", nrow(fgsea),
  "Reactome pathways to an official branch.\n"
)
cat("Saved:", output_path, "\n")
