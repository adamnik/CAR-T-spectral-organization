# Leave-one-patient-out validation of finite-depth O-lambda inference.
#
# Usage:
#   Rscript validate_olambda_asymptotes.R trajectories.csv output_dir \
#     [--minimum-cells 1000] [--validation-minimum-cells 7500] \
#     [--cutoffs 1000,3000,5000] [--curve-maximum-cells 15000] \
#     [--curve-particles 1200] [--random-state 0]


script_directory <- function() {
  arguments <- commandArgs(trailingOnly = FALSE)
  path <- sub("^--file=", "", grep("^--file=", arguments, value = TRUE)[1])
  dirname(normalizePath(path))
}


source(file.path(script_directory(), "fit_olambda_asymptotes.R"))


parse_validation_arguments <- function() {
  arguments <- commandArgs(trailingOnly = TRUE)
  if (length(arguments) < 2L) {
    stop(
      "Usage: Rscript validate_olambda_asymptotes.R trajectories.csv ",
      "output_dir [--minimum-cells 1000] ",
      "[--validation-minimum-cells 7500] [--cutoffs 1000,3000,5000]"
    )
  }

  values <- list(
    trajectories_csv = arguments[1],
    output_dir = arguments[2],
    minimum_cells = 1000L,
    validation_minimum_cells = 7500L,
    cutoffs = c(1000, 3000, 5000),
    curve_maximum_cells = 15000,
    curve_points = 350L,
    curve_particles = 1200L,
    random_state = 0L,
    sample_ids = NULL
  )

  index <- 3L
  while (index <= length(arguments)) {
    option <- arguments[index]
    if (index == length(arguments)) stop("Incomplete argument: ", option)
    value <- arguments[index + 1L]
    if (option == "--minimum-cells") {
      values$minimum_cells <- as.integer(value)
    } else if (option == "--validation-minimum-cells") {
      values$validation_minimum_cells <- as.integer(value)
    } else if (option == "--cutoffs") {
      values$cutoffs <- as.numeric(strsplit(value, ",", fixed = TRUE)[[1]])
    } else if (option == "--curve-maximum-cells") {
      values$curve_maximum_cells <- as.numeric(value)
    } else if (option == "--curve-points") {
      values$curve_points <- as.integer(value)
    } else if (option == "--curve-particles") {
      values$curve_particles <- as.integer(value)
    } else if (option == "--random-state") {
      values$random_state <- as.integer(value)
    } else if (option == "--sample-ids") {
      values$sample_ids <- strsplit(value, ",", fixed = TRUE)[[1]]
    } else {
      stop("Unknown argument: ", option)
    }
    index <- index + 2L
  }

  values$cutoffs <- sort(unique(values$cutoffs[is.finite(values$cutoffs)]))
  if (!length(values$cutoffs)) stop("At least one finite cutoff is required")
  values
}


atomic_csv <- function(frame, path) {
  temporary <- paste0(path, ".temporary")
  write.csv(frame, temporary, row.names = FALSE)
  if (file.exists(path)) file.remove(path)
  file.rename(temporary, path)
}


safe_name <- function(value) {
  gsub("[^A-Za-z0-9_.-]+", "_", value)
}


sample_coefficients <- function(posterior, number) {
  covariance <- posterior$covariance
  covariance[1, 1] <- covariance[1, 1] +
    calibrated_log_asymptote_floor^2
  decomposition <- chol(positive_definite(covariance))
  standardized <- matrix(rnorm(number * 3), number, 3)
  standardized <- standardized * sqrt(
    residual_student_df / rchisq(number, df = residual_student_df)
  )
  sweep(standardized %*% decomposition, 2, posterior$coefficients, "+")
}


summarize_fit <- function(fit, tau_values, seed) {
  posterior <- sample_asymptote_posterior(fit, particle_simulations, seed)
  quantiles <- quantile(
    posterior$O_lambda, c(0.025, 0.50, 0.975), names = FALSE
  )
  central <- sum(
    fit$weights * vapply(fit$components, `[[`, numeric(1), "central")
  )
  weighted_tau <- sum(fit$weights * tau_values)
  list(
    central = central,
    median = quantiles[2],
    lower = quantiles[1],
    upper = quantiles[3],
    weighted_tau = weighted_tau,
    tau_sd = sqrt(sum(fit$weights * (tau_values - weighted_tau)^2)),
    weight_entropy = -sum(fit$weights * log(pmax(fit$weights, 1e-12)))
  )
}


fit_curve <- function(
  fit, tau_values, maximum_cells, grid_points, curve_particles, seed
) {
  grid <- data.frame(
    target_depth = seq(50, maximum_cells, length.out = grid_points)
  )
  component_means <- matrix(NA_real_, nrow(grid), length(tau_values))
  particles <- matrix(NA_real_, curve_particles, nrow(grid))
  set.seed(seed)
  selected_component <- sample(
    seq_along(tau_values), curve_particles,
    replace = TRUE, prob = fit$weights
  )

  for (index in seq_along(tau_values)) {
    basis <- add_basis(grid, tau_values[index])
    design <- cbind("(Intercept)" = 1, f1 = basis$f1, f2 = basis$f2)
    posterior <- fit$components[[index]]
    component_means[, index] <- 1 + exp(
      as.vector(design %*% posterior$coefficients)
    )
    selected <- which(selected_component == index)
    if (!length(selected)) next
    coefficients <- sample_coefficients(posterior, length(selected))
    particles[selected, ] <- 1 + exp(coefficients %*% t(design))
  }

  data.frame(
    target_depth = grid$target_depth,
    fitted = as.vector(component_means %*% fit$weights),
    lower = apply(particles, 2, quantile, 0.025, na.rm = TRUE),
    upper = apply(particles, 2, quantile, 0.975, na.rm = TRUE)
  )
}


fit_validation_patient <- function(
  record, patient_index, trajectories, tau_values, arguments
) {
  patient <- trajectories[
    trajectories$patient_key == record$patient_key, , drop = FALSE
  ]
  patient <- patient[order(patient$target_depth), , drop = FALSE]
  training <- trajectories[
    trajectories$patient_key != record$patient_key, , drop = FALSE
  ]
  training <- thin_by_patient(training, training_maximum_points)

  populations <- vector("list", length(tau_values))
  for (tau_index in seq_along(tau_values)) {
    cat(sprintf(
      "  population component %d/%d | tau %.4f\n",
      tau_index, length(tau_values), tau_values[tau_index]
    ))
    populations[[tau_index]] <- fit_population_model(
      training, tau_values[tau_index]
    )
  }
  populations <- share_asymptote_hierarchy(populations, tau_values)

  requested_cutoffs <- c(arguments$cutoffs, Inf)
  labels <- c(format(arguments$cutoffs, scientific = FALSE, trim = TRUE), "full")
  fits <- vector("list", length(requested_cutoffs))
  summaries <- vector("list", length(requested_cutoffs))
  prefixes <- vector("list", length(requested_cutoffs))

  for (index in seq_along(requested_cutoffs)) {
    cutoff <- requested_cutoffs[index]
    prefix <- if (is.infinite(cutoff)) patient else {
      patient[patient$target_depth <= cutoff, , drop = FALSE]
    }
    if (nrow(prefix) < 3L) {
      stop(record$sample_id, " has fewer than three points through ", labels[index])
    }
    prefixes[[index]] <- prefix
    fits[[index]] <- fit_patient_components(populations, tau_values, prefix)
    summaries[[index]] <- summarize_fit(
      fits[[index]], tau_values,
      seed = arguments$random_state + patient_index * 100 + index
    )
  }

  full_summary <- summaries[[length(summaries)]]
  result_rows <- vector("list", length(requested_cutoffs))
  curve_rows <- vector("list", length(requested_cutoffs))

  for (index in seq_along(requested_cutoffs)) {
    prefix <- prefixes[[index]]
    estimate <- summaries[[index]]
    future <- patient[
      patient$target_depth > max(prefix$target_depth), , drop = FALSE
    ]
    future_prediction <- if (nrow(future)) {
      predict_patient_curve(fits[[index]], tau_values, future)
    } else NA_real_

    result_rows[[index]] <- data.frame(
      Dataset = record$Dataset,
      sample_id = record$sample_id,
      n_cells_available = record$n_cells_available,
      cutoff = labels[index],
      requested_cutoff = requested_cutoffs[index],
      maximum_depth_used = max(prefix$target_depth),
      n_trajectory_points_used = nrow(prefix),
      O_lambda_inf_central = estimate$central,
      O_lambda_inf_median = estimate$median,
      O_lambda_inf_q025 = estimate$lower,
      O_lambda_inf_q975 = estimate$upper,
      full_O_lambda_inf = full_summary$central,
      full_O_lambda_inf_q025 = full_summary$lower,
      full_O_lambda_inf_q975 = full_summary$upper,
      error_vs_full_fit = estimate$central - full_summary$central,
      absolute_error_vs_full_fit = abs(estimate$central - full_summary$central),
      interval_covers_full_fit =
        estimate$lower <= full_summary$central &
        estimate$upper >= full_summary$central,
      interval_width = estimate$upper - estimate$lower,
      weighted_tau = estimate$weighted_tau,
      posterior_tau_sd = estimate$tau_sd,
      timescale_weight_entropy = estimate$weight_entropy,
      future_RMSE = if (nrow(future)) {
        sqrt(mean((future_prediction - future$O_lambda)^2))
      } else NA_real_,
      future_MAE = if (nrow(future)) {
        mean(abs(future_prediction - future$O_lambda))
      } else NA_real_,
      future_bias = if (nrow(future)) {
        mean(future_prediction - future$O_lambda)
      } else NA_real_,
      stringsAsFactors = FALSE
    )

    curve <- fit_curve(
      fits[[index]], tau_values,
      arguments$curve_maximum_cells,
      arguments$curve_points,
      arguments$curve_particles,
      seed = arguments$random_state + patient_index * 1000 + index
    )
    curve_rows[[index]] <- data.frame(
      Dataset = record$Dataset,
      sample_id = record$sample_id,
      cutoff = labels[index],
      requested_cutoff = requested_cutoffs[index],
      maximum_depth_used = max(prefix$target_depth),
      curve,
      stringsAsFactors = FALSE
    )
  }

  list(
    results = do.call(rbind, result_rows),
    curves = do.call(rbind, curve_rows)
  )
}


validation_summary <- function(results, cutoff_order) {
  rows <- lapply(cutoff_order, function(cutoff) {
    frame <- results[results$cutoff == cutoff, , drop = FALSE]
    data.frame(
      cutoff = cutoff,
      patients = nrow(frame),
      MAE_vs_full_fit = mean(frame$absolute_error_vs_full_fit),
      RMSE_vs_full_fit = sqrt(mean(frame$error_vs_full_fit^2)),
      bias_vs_full_fit = mean(frame$error_vs_full_fit),
      spearman_vs_full_fit = if (nrow(frame) > 2L) {
        cor(
          frame$O_lambda_inf_central, frame$full_O_lambda_inf,
          method = "spearman"
        )
      } else NA_real_,
      interval_coverage_of_full_fit = mean(frame$interval_covers_full_fit),
      median_interval_width = median(frame$interval_width),
      future_RMSE = mean(frame$future_RMSE, na.rm = TRUE),
      future_MAE = mean(frame$future_MAE, na.rm = TRUE),
      future_bias = mean(frame$future_bias, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  summary <- do.call(rbind, rows)
  for (column in c("future_RMSE", "future_MAE", "future_bias")) {
    summary[[column]][!is.finite(summary[[column]])] <- NA_real_
  }
  summary
}


main_validation <- function() {
  arguments <- parse_validation_arguments()
  dir.create(arguments$output_dir, recursive = TRUE, showWarnings = FALSE)
  checkpoint_dir <- file.path(arguments$output_dir, "patient_checkpoints")
  dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)

  trajectories <- read_trajectories(
    arguments$trajectories_csv, arguments$minimum_cells
  )
  trajectories$patient_key <- paste(
    trajectories$Dataset, trajectories$sample_id, sep = "::"
  )
  manifest <- unique(
    trajectories[, c(
      "Dataset", "sample_id", "patient_key", "n_cells_available"
    )]
  )
  heldout <- manifest[
    manifest$n_cells_available >= arguments$validation_minimum_cells,
    , drop = FALSE
  ]
  if (!is.null(arguments$sample_ids)) {
    heldout <- heldout[heldout$sample_id %in% arguments$sample_ids, , drop = FALSE]
  }
  heldout <- heldout[
    order(-heldout$n_cells_available, heldout$Dataset, heldout$sample_id),
    , drop = FALSE
  ]
  if (!nrow(heldout)) stop("No patients meet the validation requirements")

  tau_values <- timescale_grid()
  cat(sprintf(
    "Validating %d patients with N >= %s\n",
    nrow(heldout),
    format(arguments$validation_minimum_cells, big.mark = ",")
  ))
  cat("Cutoffs: ", paste(arguments$cutoffs, collapse = ", "), ", full\n", sep = "")

  for (patient_index in seq_len(nrow(heldout))) {
    record <- heldout[patient_index, ]
    stem <- safe_name(paste(record$Dataset, record$sample_id, sep = "__"))
    result_path <- file.path(checkpoint_dir, paste0(stem, "__results.csv"))
    curve_path <- file.path(checkpoint_dir, paste0(stem, "__curves.csv"))
    cat("\n", paste(rep("=", 100), collapse = ""), "\n", sep = "")
    cat(sprintf(
      "[%d/%d] %s | %s | N=%s\n",
      patient_index, nrow(heldout), record$Dataset, record$sample_id,
      format(record$n_cells_available, big.mark = ",")
    ))
    if (file.exists(result_path) && file.exists(curve_path)) {
      cat("  completed checkpoint found\n")
      next
    }

    fit <- fit_validation_patient(
      record, patient_index, trajectories, tau_values, arguments
    )
    atomic_csv(fit$results, result_path)
    atomic_csv(fit$curves, curve_path)
    cat("  patient checkpoint saved\n")
  }

  result_files <- list.files(
    checkpoint_dir, pattern = "__results[.]csv$", full.names = TRUE
  )
  curve_files <- list.files(
    checkpoint_dir, pattern = "__curves[.]csv$", full.names = TRUE
  )
  results <- do.call(rbind, lapply(result_files, read.csv, check.names = FALSE))
  curves <- do.call(rbind, lapply(curve_files, read.csv, check.names = FALSE))
  results <- results[
    results$sample_id %in% heldout$sample_id,
    , drop = FALSE
  ]
  curves <- curves[curves$sample_id %in% heldout$sample_id, , drop = FALSE]
  cutoff_order <- c(
    format(arguments$cutoffs, scientific = FALSE, trim = TRUE), "full"
  )
  summary <- validation_summary(results, cutoff_order)

  atomic_csv(
    heldout,
    file.path(arguments$output_dir, "validation_patient_manifest.csv")
  )
  atomic_csv(
    results,
    file.path(arguments$output_dir, "heldout_cutoff_results.csv")
  )
  atomic_csv(
    curves,
    file.path(arguments$output_dir, "heldout_fitted_curves.csv")
  )
  atomic_csv(
    summary,
    file.path(arguments$output_dir, "validation_summary.csv")
  )
  atomic_csv(
    data.frame(
      minimum_cells = arguments$minimum_cells,
      validation_minimum_cells = arguments$validation_minimum_cells,
      cutoffs = paste(arguments$cutoffs, collapse = ","),
      curve_maximum_cells = arguments$curve_maximum_cells,
      curve_points = arguments$curve_points,
      curve_particles = arguments$curve_particles,
      random_state = arguments$random_state,
      patients = nrow(heldout),
      quadrature_nodes = quadrature_nodes,
      log_tau_scale = log_tau_scale,
      calibrated_log_asymptote_floor = calibrated_log_asymptote_floor
    ),
    file.path(arguments$output_dir, "validation_config.csv")
  )

  cat("\nValidation summary\n")
  print(summary, row.names = FALSE)
  cat("\nResults: ", file.path(
    arguments$output_dir, "heldout_cutoff_results.csv"
  ), "\n", sep = "")
}


if (sys.nframe() == 0L) main_validation()
