# Infer patient O-lambda asymptotes from finite-depth trajectories.
#
# Usage:
#   Rscript fit_olambda_asymptotes.R trajectories.csv output_dir [--minimum-cells 1000]

library(nlme)


tau <- 2
training_maximum_points <- 30L
posterior_maximum_points <- 40L
effective_points_per_patient <- 12
student_df <- 4
minimum_O_minus_one <- 1e-5


uniform_depth_subset <- function(frame, maximum) {
  frame <- frame[order(frame$target_depth), , drop = FALSE]
  if (nrow(frame) <= maximum) return(frame)

  target_depths <- seq(min(frame$target_depth), max(frame$target_depth), length.out = maximum)
  index <- unique(vapply(
    target_depths,
    function(value) which.min(abs(frame$target_depth - value)),
    integer(1)
  ))
  frame[sort(unique(c(1L, index, nrow(frame)))), , drop = FALSE]
}


thin_by_patient <- function(frame, maximum) {
  do.call(rbind, lapply(split(frame, frame$patient_key), uniform_depth_subset, maximum = maximum))
}


add_basis <- function(frame) {
  frame$scaled_depth <- frame$target_depth / 500
  frame$f1 <- 1 / (frame$scaled_depth + tau)
  frame$f2 <- frame$f1^2
  frame
}


fit_population_model <- function(training) {
  frame <- add_basis(training)
  fit <- lme(
    fixed = log_O_minus_one ~ f1 + f2,
    random = ~ f1 + f2 | patient_key,
    weights = varPower(form = ~ scaled_depth),
    data = frame,
    method = "ML",
    control = lmeControl(
      opt = "optim", maxIter = 250, msMaxIter = 250,
      tolerance = 1e-7, niterEM = 50, returnObject = TRUE
    )
  )

  coefficient_names <- c("(Intercept)", "f1", "f2")
  random_covariance <- as.matrix(getVarCov(fit, type = "random.effects"))
  random_covariance <- random_covariance[coefficient_names, coefficient_names, drop = FALSE]
  # Keep the estimated covariance positive definite
  decomposition <- eigen((random_covariance + t(random_covariance)) / 2, symmetric = TRUE)
  random_covariance <- decomposition$vectors %*%
    diag(pmax(decomposition$values, 1e-5)) %*%
    t(decomposition$vectors)

  residual_floor <- max(
    0.015,
    as.numeric(quantile(abs(residuals(fit, type = "normalized")), 0.10)) *
      fit$sigma * 0.25
  )

  list(
    fit = fit,
    fixed_effect = fixef(fit)[coefficient_names],
    random_covariance = random_covariance,
    variance_power = as.numeric(coef(fit$modelStruct$varStruct, unconstrained = FALSE)),
    residual_sigma = fit$sigma,
    residual_floor = residual_floor
  )
}


patient_posterior <- function(population, patient_frame) {
  frame <- uniform_depth_subset(patient_frame, posterior_maximum_points)
  if (nrow(frame) < 3L) return(NULL)

  frame <- add_basis(frame)
  design <- cbind("(Intercept)" = 1, f1 = frame$f1, f2 = frame$f2)
  residual <- frame$log_O_minus_one - as.vector(design %*% population$fixed_effect)
  residual_variance <- population$residual_sigma^2 *
    pmax(frame$scaled_depth, 1e-3)^(2 * population$variance_power) +
    population$residual_floor^2
  # Limit the weight of densely sampled trajectories
  information_weight <- min(1, effective_points_per_patient / nrow(frame))
  precision <- information_weight / pmax(residual_variance, 1e-7)

  prior_precision <- solve(population$random_covariance)
  random_mean <- rep(0, ncol(design))
  random_covariance <- population$random_covariance
  robust_weight <- rep(1, nrow(frame))

  # Student-t reweighting
  for (iteration in seq_len(5)) {
    effective_precision <- precision * robust_weight
    random_covariance <- solve(
      prior_precision + crossprod(design, design * effective_precision)
    )
    random_mean <- as.vector(
      random_covariance %*% crossprod(design, residual * effective_precision)
    )
    standardized_residual <- (residual - as.vector(design %*% random_mean)) / sqrt(residual_variance)
    robust_weight <- (student_df + 1) / (student_df + standardized_residual^2)
  }

  coefficients <- as.vector(population$fixed_effect + random_mean)
  fitted_O_lambda <- 1 + exp(as.vector(design %*% coefficients))
  location <- coefficients[1]
  scale <- sqrt(max(random_covariance[1, 1], 1e-10))

  list(
    location = location,
    scale = scale,
    central = 1 + exp(location),
    lower = 1 + exp(location + qt(0.025, df = student_df) * scale),
    upper = 1 + exp(location + qt(0.975, df = student_df) * scale),
    coefficients = coefficients,
    points_used = nrow(frame),
    information_weight = information_weight,
    robust_weight_minimum = min(robust_weight),
    robust_weight_median = median(robust_weight),
    RMSE = sqrt(mean((frame$O_lambda - fitted_O_lambda)^2)),
    MAE = mean(abs(frame$O_lambda - fitted_O_lambda))
  )
}


estimate_patients <- function(population, frame) {
  manifest <- unique(frame[, c("Dataset", "sample_id", "patient_key", "n_cells_available")])
  manifest <- manifest[order(manifest$Dataset, manifest$sample_id), , drop = FALSE]
  rows <- vector("list", nrow(manifest))

  for (index in seq_len(nrow(manifest))) {
    patient <- frame[frame$patient_key == manifest$patient_key[index], , drop = FALSE]
    posterior <- patient_posterior(population, patient)
    if (is.null(posterior)) next

    rows[[index]] <- data.frame(
      Dataset = manifest$Dataset[index],
      sample_id = manifest$sample_id[index],
      n_cells_available = manifest$n_cells_available[index],
      minimum_trajectory_depth = min(patient$target_depth),
      maximum_trajectory_depth = max(patient$target_depth),
      n_trajectory_points_available = nrow(patient),
      n_trajectory_points_used = posterior$points_used,
      O_lambda_inf_central = posterior$central,
      O_lambda_inf_q025 = posterior$lower,
      O_lambda_inf_q975 = posterior$upper,
      log_O_minus_1_inf_location = posterior$location,
      log_O_minus_1_inf_scale = posterior$scale,
      trajectory_intercept = posterior$coefficients[1],
      trajectory_f1 = posterior$coefficients[2],
      trajectory_f2 = posterior$coefficients[3],
      information_weight = posterior$information_weight,
      robust_weight_minimum = posterior$robust_weight_minimum,
      robust_weight_median = posterior$robust_weight_median,
      observed_fit_RMSE_O = posterior$RMSE,
      observed_fit_MAE_O = posterior$MAE,
      model = "second_order_spectral_tail_tau2",
      interval_type = "95% empirical-Bayes Student-t4 interval",
      stringsAsFactors = FALSE
    )
  }

  rows <- rows[!vapply(rows, is.null, logical(1))]
  do.call(rbind, rows)
}


parse_arguments <- function() {
  arguments <- commandArgs(trailingOnly = TRUE)
  if (length(arguments) < 2) {
    stop(
      "Usage: Rscript fit_olambda_asymptotes.R ",
      "trajectories.csv output_dir [--minimum-cells 1000]"
    )
  }

  values <- list(
    trajectories_csv = arguments[1],
    output_dir = arguments[2],
    minimum_cells = 1000L
  )
  if (length(arguments) > 2) {
    index <- 3L
    while (index <= length(arguments)) {
      if (arguments[index] == "--minimum-cells" && index < length(arguments)) {
        values$minimum_cells <- as.integer(arguments[index + 1L])
        index <- index + 2L
      } else {
        stop("Unknown or incomplete argument: ", arguments[index])
      }
    }
  }
  if (!is.finite(values$minimum_cells) || values$minimum_cells < 1L) {
    stop("--minimum-cells must be positive")
  }
  values
}


main <- function() {
  arguments <- parse_arguments()
  dir.create(arguments$output_dir, recursive = TRUE, showWarnings = FALSE)

  # Read the trajectory points
  trajectories <- read.csv(arguments$trajectories_csv, stringsAsFactors = FALSE, check.names = FALSE)
  if (!"n_cells_available" %in% names(trajectories)) {
    maximum_depth <- aggregate(target_depth ~ Dataset + sample_id, data = trajectories, FUN = max)
    names(maximum_depth)[3] <- "n_cells_available"
    trajectories <- merge(
      trajectories,
      maximum_depth,
      by = c("Dataset", "sample_id"),
      all.x = TRUE,
      sort = FALSE
    )
  }

  trajectories <- trajectories[
    is.finite(trajectories$target_depth) &
      is.finite(trajectories$O_lambda) &
      trajectories$O_lambda > 1 + minimum_O_minus_one &
      trajectories$n_cells_available >= arguments$minimum_cells,
    ,
    drop = FALSE
  ]
  if (!nrow(trajectories)) stop("No eligible trajectory points remain")

  trajectories$patient_key <- paste(trajectories$Dataset, trajectories$sample_id, sep = "::")
  trajectories$log_O_minus_one <- log(trajectories$O_lambda - 1)
  patients <- length(unique(trajectories$patient_key))
  cat(sprintf(
    "Fitting %d patients from %d trajectory points\n",
    patients,
    nrow(trajectories)
  ))

  # Fit the population model, then update each patient
  training <- thin_by_patient(trajectories, training_maximum_points)
  population <- fit_population_model(training)
  estimates <- estimate_patients(population, trajectories)

  # Save the fitted model and patient estimates
  saveRDS(population$fit, file.path(arguments$output_dir, "population_model.rds"))
  write.csv(estimates, file.path(arguments$output_dir, "patient_olambda_asymptotes.csv"),
            row.names = FALSE)
  write.csv(
    data.frame(
      minimum_cells = arguments$minimum_cells,
      patients = patients,
      trajectory_points = nrow(training),
      fixed_intercept = population$fixed_effect[1],
      fixed_f1 = population$fixed_effect[2],
      fixed_f2 = population$fixed_effect[3],
      residual_sigma = population$residual_sigma,
      variance_power = population$variance_power,
      residual_floor = population$residual_floor,
      AIC = AIC(population$fit),
      BIC = BIC(population$fit)
    ),
    file.path(arguments$output_dir, "population_model_summary.csv"),
    row.names = FALSE
  )
  cat("Saved ", nrow(estimates), " patient asymptotes\n", sep = "")
}


main()
