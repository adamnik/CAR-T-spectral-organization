# Plot finite-depth O-lambda validation fits for selected patients.
#
# Usage:
#   Rscript plot_olambda_validation.R trajectories.csv validation_dir figure_dir \
#     --sample-ids P032_cao,ac28_li,CAR130_yu \
#     [--cutoffs 1000,3000,5000,full] [--x-maximum 15000] [--dpi 1200]


parse_plot_arguments <- function() {
  arguments <- commandArgs(trailingOnly = TRUE)
  if (length(arguments) < 3L) {
    stop(
      "Usage: Rscript plot_olambda_validation.R trajectories.csv ",
      "validation_dir figure_dir --sample-ids id1,id2"
    )
  }
  values <- list(
    trajectories_csv = arguments[1],
    validation_dir = arguments[2],
    figure_dir = arguments[3],
    sample_ids = NULL,
    cutoffs = c("1000", "3000", "5000", "full"),
    x_maximum = 15000,
    dpi = 1200L,
    width = 16,
    height = 4.4
  )

  index <- 4L
  while (index <= length(arguments)) {
    option <- arguments[index]
    if (index == length(arguments)) stop("Incomplete argument: ", option)
    value <- arguments[index + 1L]
    if (option == "--sample-ids") {
      values$sample_ids <- strsplit(value, ",", fixed = TRUE)[[1]]
    } else if (option == "--cutoffs") {
      values$cutoffs <- strsplit(value, ",", fixed = TRUE)[[1]]
    } else if (option == "--x-maximum") {
      values$x_maximum <- as.numeric(value)
    } else if (option == "--dpi") {
      values$dpi <- as.integer(value)
    } else if (option == "--width") {
      values$width <- as.numeric(value)
    } else if (option == "--height") {
      values$height <- as.numeric(value)
    } else {
      stop("Unknown argument: ", option)
    }
    index <- index + 2L
  }
  if (is.null(values$sample_ids)) stop("--sample-ids is required")
  values
}


cutoff_title <- function(cutoff) {
  if (cutoff == "full") return("full")
  format(as.numeric(cutoff), big.mark = ",", scientific = FALSE)
}


plot_patient <- function(
  sample_id, trajectories, results, curves, arguments
) {
  patient <- trajectories[
    trajectories$sample_id == sample_id, , drop = FALSE
  ]
  patient <- patient[order(patient$target_depth), , drop = FALSE]
  patient_results <- results[
    results$sample_id == sample_id &
      results$cutoff %in% arguments$cutoffs,
    , drop = FALSE
  ]
  patient_curves <- curves[
    curves$sample_id == sample_id &
      curves$cutoff %in% arguments$cutoffs &
      curves$target_depth <= arguments$x_maximum,
    , drop = FALSE
  ]
  if (!nrow(patient) || !nrow(patient_results) || !nrow(patient_curves)) {
    stop("Validation data are incomplete for ", sample_id)
  }

  patient_results$cutoff <- factor(
    patient_results$cutoff, levels = arguments$cutoffs
  )
  patient_results <- patient_results[order(patient_results$cutoff), ]
  full_reference <- patient_results$full_O_lambda_inf[1]
  shared_y_range <- range(
    patient$O_lambda,
    patient_curves$lower,
    patient_curves$upper,
    full_reference,
    1,
    finite = TRUE
  )
  padding <- max(0.005, 0.04 * diff(shared_y_range))
  shared_y_limits <- shared_y_range + c(-padding, padding)

  path <- file.path(
    arguments$figure_dir,
    paste0(gsub("[^A-Za-z0-9_.-]+", "_", sample_id), "__validation_fits.jpg")
  )
  jpeg(
    path,
    width = arguments$width,
    height = arguments$height,
    units = "in",
    res = arguments$dpi,
    quality = 100
  )
  par(
    mfrow = c(1, length(arguments$cutoffs)),
    mar = c(4.3, 4.2, 3.5, 1.0),
    oma = c(0, 0, 2.0, 0)
  )

  for (cutoff in arguments$cutoffs) {
    result <- patient_results[
      as.character(patient_results$cutoff) == cutoff, , drop = FALSE
    ]
    curve <- patient_curves[
      patient_curves$cutoff == cutoff, , drop = FALSE
    ]
    curve <- curve[order(curve$target_depth), , drop = FALSE]
    prefix <- patient[
      patient$target_depth <= result$maximum_depth_used[1], , drop = FALSE
    ]
    displayed_lower <- pmax(curve$lower, 1)

    plot(
      patient$target_depth,
      patient$O_lambda,
      type = "n",
      xlim = c(0, arguments$x_maximum),
      ylim = shared_y_limits,
      xlab = "Infusion cells",
      ylab = expression(O[lambda]),
      main = sprintf(
        "Data through n = %s\nasymptote %.3f [%.3f, %.3f]",
        cutoff_title(cutoff),
        result$O_lambda_inf_central[1],
        result$O_lambda_inf_q025[1],
        result$O_lambda_inf_q975[1]
      )
    )
    grid(col = adjustcolor("#bdbdbd", alpha.f = 0.35))
    polygon(
      c(curve$target_depth, rev(curve$target_depth)),
      c(displayed_lower, rev(curve$upper)),
      border = NA,
      col = adjustcolor("#9ecae1", alpha.f = 0.45)
    )
    points(
      patient$target_depth, patient$O_lambda,
      pch = 16, cex = 0.55, col = "#bdbdbd"
    )
    points(
      prefix$target_depth, prefix$O_lambda,
      pch = 16, cex = 0.65, col = "#3f2771"
    )
    lines(
      curve$target_depth, curve$fitted,
      lwd = 2.2, col = "#2171b5"
    )
    if (cutoff != "full") {
      abline(
        h = result$O_lambda_inf_central[1],
        lty = 2, lwd = 1.5, col = "#2171b5"
      )
    }
    abline(
      h = full_reference,
      lty = 2, lwd = 1.5, col = "#cb181d"
    )
    abline(
      v = result$maximum_depth_used[1],
      lty = 3, col = "#636363"
    )
  }
  mtext(sample_id, outer = TRUE, font = 2, cex = 1.1)
  dev.off()
  path
}


main <- function() {
  arguments <- parse_plot_arguments()
  dir.create(arguments$figure_dir, recursive = TRUE, showWarnings = FALSE)
  trajectories <- read.csv(
    arguments$trajectories_csv,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  results <- read.csv(
    file.path(arguments$validation_dir, "heldout_cutoff_results.csv"),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  curves <- read.csv(
    file.path(arguments$validation_dir, "heldout_fitted_curves.csv"),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  paths <- vapply(
    arguments$sample_ids,
    plot_patient,
    character(1),
    trajectories = trajectories,
    results = results,
    curves = curves,
    arguments = arguments
  )
  cat("Saved figures\n")
  cat(paste(paths, collapse = "\n"), "\n")
}


if (sys.nframe() == 0L) main()
