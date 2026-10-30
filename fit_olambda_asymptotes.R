# Infer patient O-lambda asymptotes from finite-depth trajectories.
#
# Usage:
#   Rscript fit_olambda_asymptotes.R trajectories.csv output_dir \
#     [--minimum-cells 1000] [--n-particles 16] [--random-state 0]

library(nlme)


training_maximum_points <- 30L
posterior_maximum_points <- 40L
effective_points_per_patient <- 12
residual_student_df <- 4
slope_student_df <- 10
quadrature_nodes <- 9L
log_tau_scale <- 0.35
calibrated_log_asymptote_floor <- 0.0365797495529331
minimum_O_minus_one <- 1e-5
particle_simulations <- 50000L


uniform_depth_subset <- function(frame, maximum) {
  frame <- frame[order(frame$target_depth), , drop = FALSE]
  if (nrow(frame) <= maximum) return(frame)

  target_depths <- seq(
    min(frame$target_depth), max(frame$target_depth), length.out = maximum
  )
  index <- unique(vapply(
    target_depths,
    function(value) which.min(abs(frame$target_depth - value)),
    integer(1)
  ))
  frame[sort(unique(c(1L, index, nrow(frame)))), , drop = FALSE]
}


thin_by_patient <- function(frame, maximum) {
  do.call(rbind, lapply(
    split(frame, frame$patient_key),
    uniform_depth_subset,
    maximum = maximum
  ))
}


timescale_grid <- function() {
  probability <- (seq_len(quadrature_nodes) - 0.5) / quadrature_nodes
  exp(log(2) + log_tau_scale * qt(probability, df = 3))
}


add_basis <- function(frame, tau) {
  frame$scaled_depth <- frame$target_depth / 500
  frame$f1 <- tau / (frame$scaled_depth + tau)
  frame$f2 <- frame$f1^2
  frame
}


positive_definite <- function(matrix, minimum = 1e-8) {
  decomposition <- eigen((matrix + t(matrix)) / 2, symmetric = TRUE)
  decomposition$vectors %*% diag(pmax(decomposition$values, minimum)) %*%
    t(decomposition$vectors)
}


fit_population_model <- function(training, tau) {
  frame <- add_basis(training, tau)
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
  random_covariance <- random_covariance[
    coefficient_names, coefficient_names, drop = FALSE
  ]
  random_covariance <- positive_definite(random_covariance, minimum = 1e-5)
  residual_floor <- max(
    0.015,
    as.numeric(quantile(abs(residuals(fit, type = "normalized")), 0.10)) *
      fit$sigma * 0.25
  )

  list(
    fit = fit,
    fixed_effect = fixef(fit)[coefficient_names],
    random_covariance = random_covariance,
    variance_power = as.numeric(
      coef(fit$modelStruct$varStruct, unconstrained = FALSE)
    ),
    residual_sigma = fit$sigma,
    residual_floor = residual_floor
  )
}


share_asymptote_hierarchy <- function(populations, tau_values) {
  baseline <- populations[[which.min(abs(tau_values - 2))]]
  common_mean <- baseline$fixed_effect[1]
  common_variance <- baseline$random_covariance[1, 1]

  lapply(populations, function(population) {
    covariance <- population$random_covariance
    alpha_variance <- covariance[1, 1]
    slope_alpha <- covariance[2:3, 1, drop = FALSE]
    conditional_coefficient <- slope_alpha / alpha_variance
    conditional_covariance <-
      covariance[2:3, 2:3, drop = FALSE] -
      slope_alpha %*% t(slope_alpha) / alpha_variance
    slope_mean <- population$fixed_effect[2:3] +
      as.vector(
        conditional_coefficient * (common_mean - population$fixed_effect[1])
      )
    shared_slope_alpha <- conditional_coefficient * common_variance
    shared_slope_covariance <- conditional_covariance +
      conditional_coefficient %*% t(conditional_coefficient) * common_variance

    population$fixed_effect <- c(common_mean, slope_mean)
    population$random_covariance <- rbind(
      c(common_variance, t(shared_slope_alpha)),
      cbind(shared_slope_alpha, shared_slope_covariance)
    )
    population$random_covariance <- positive_definite(
      population$random_covariance
    )
    population
  })
}


conditional_slope_prior <- function(covariance) {
  intercept_variance <- covariance[1, 1]
  slope_intercept <- covariance[2:3, 1, drop = FALSE]
  conditional_mean_coefficient <- slope_intercept / intercept_variance
  conditional_covariance <-
    covariance[2:3, 2:3, drop = FALSE] -
    slope_intercept %*% t(slope_intercept) / intercept_variance
  conditional_covariance <- positive_definite(conditional_covariance)

  list(
    intercept_variance = intercept_variance,
    conditional_mean_coefficient = conditional_mean_coefficient,
    conditional_covariance = conditional_covariance,
    conditional_precision = solve(conditional_covariance)
  )
}


patient_posterior <- function(population, patient_frame, tau) {
  frame <- uniform_depth_subset(patient_frame, posterior_maximum_points)
  if (nrow(frame) < 3L) return(NULL)

  frame <- add_basis(frame, tau)
  design <- cbind("(Intercept)" = 1, f1 = frame$f1, f2 = frame$f2)
  residual_variance <- population$residual_sigma^2 *
    pmax(frame$scaled_depth, 1e-3)^(2 * population$variance_power) +
    population$residual_floor^2
  information_weight <- min(1, effective_points_per_patient / nrow(frame))
  prior <- conditional_slope_prior(population$random_covariance)

  objective <- function(random_effect) {
    coefficients <- population$fixed_effect + random_effect
    fitted <- as.vector(design %*% coefficients)
    standardized_squared <-
      (frame$log_O_minus_one - fitted)^2 / residual_variance
    data_loss <- information_weight * sum(
      0.5 * log(residual_variance) +
        0.5 * (residual_student_df + 1) *
        log1p(standardized_squared / residual_student_df)
    )

    intercept <- random_effect[1]
    slope_residual <- random_effect[2:3] -
      as.vector(prior$conditional_mean_coefficient * intercept)
    slope_distance <- sum(
      slope_residual *
        as.vector(prior$conditional_precision %*% slope_residual)
    )
    data_loss +
      0.5 * intercept^2 / prior$intercept_variance +
      0.5 * (slope_student_df + 2) *
        log1p(slope_distance / slope_student_df)
  }

  fit <- optim(
    par = c(0, 0, 0), fn = objective, method = "BFGS",
    hessian = TRUE, control = list(maxit = 500, reltol = 1e-10)
  )
  hessian <- positive_definite(fit$hessian)
  covariance <- positive_definite(solve(hessian))
  coefficients <- as.vector(population$fixed_effect + fit$par)
  fitted_O <- 1 + exp(as.vector(design %*% coefficients))

  log_prior_normalizer <-
    -0.5 * log(prior$intercept_variance) -
    0.5 * as.numeric(
      determinant(prior$conditional_covariance, logarithm = TRUE)$modulus
    )
  log_laplace_adjustment <-
    -0.5 * as.numeric(determinant(hessian, logarithm = TRUE)$modulus)

  list(
    coefficients = coefficients,
    covariance = covariance,
    location = coefficients[1],
    scale = sqrt(max(covariance[1, 1], 1e-10)),
    central = 1 + exp(coefficients[1]),
    log_evidence = -fit$value + log_prior_normalizer +
      log_laplace_adjustment,
    fit_RMSE = sqrt(mean((frame$O_lambda - fitted_O)^2)),
    points_used = nrow(frame),
    convergence = fit$convergence
  )
}


log_sum_exp <- function(values) {
  maximum <- max(values)
  maximum + log(sum(exp(values - maximum)))
}


fit_patient_components <- function(populations, tau_values, patient_frame) {
  components <- lapply(seq_along(tau_values), function(index) {
    patient_posterior(populations[[index]], patient_frame, tau_values[index])
  })
  log_weights <- vapply(components, `[[`, numeric(1), "log_evidence") -
    log(length(tau_values))
  log_weights <- log_weights - log_sum_exp(log_weights)
  list(components = components, weights = exp(log_weights))
}


sample_asymptote_posterior <- function(fit, simulations, seed) {
  set.seed(seed)
  component <- sample(
    seq_along(fit$weights), simulations,
    replace = TRUE, prob = fit$weights
  )
  log_excess <- numeric(simulations)
  for (index in seq_along(fit$weights)) {
    selected <- which(component == index)
    if (!length(selected)) next
    posterior <- fit$components[[index]]
    scale <- sqrt(
      posterior$scale^2 + calibrated_log_asymptote_floor^2
    )
    log_excess[selected] <- posterior$location +
      scale * rt(length(selected), df = residual_student_df)
  }
  list(log_excess = log_excess, O_lambda = 1 + exp(log_excess))
}


predict_patient_curve <- function(fit, tau_values, frame) {
  predictions <- matrix(NA_real_, nrow(frame), length(tau_values))
  for (index in seq_along(tau_values)) {
    basis <- add_basis(frame, tau_values[index])
    design <- cbind("(Intercept)" = 1, f1 = basis$f1, f2 = basis$f2)
    predictions[, index] <- 1 + exp(
      as.vector(design %*% fit$components[[index]]$coefficients)
    )
  }
  as.vector(predictions %*% fit$weights)
}


estimate_patients <- function(
  populations, tau_values, frame, n_particles, random_state
) {
  manifest <- unique(
    frame[, c("Dataset", "sample_id", "patient_key", "n_cells_available")]
  )
  manifest <- manifest[order(manifest$Dataset, manifest$sample_id), ]
  probabilities <- (seq_len(n_particles) - 0.5) / n_particles
  estimates <- vector("list", nrow(manifest))
  particles <- vector("list", nrow(manifest))

  for (index in seq_len(nrow(manifest))) {
    record <- manifest[index, ]
    patient <- frame[frame$patient_key == record$patient_key, , drop = FALSE]
    patient <- patient[order(patient$target_depth), , drop = FALSE]
    fit <- fit_patient_components(populations, tau_values, patient)
    posterior <- sample_asymptote_posterior(
      fit, particle_simulations, 100000L + random_state + index
    )
    quantiles <- quantile(
      posterior$O_lambda, c(0.025, 0.50, 0.975), names = FALSE
    )
    survival_particles <- quantile(
      posterior$O_lambda, probabilities, names = FALSE
    )
    central <- sum(
      fit$weights * vapply(fit$components, `[[`, numeric(1), "central")
    )
    compatible_scale <- (
      quantile(posterior$log_excess, 0.975, names = FALSE) -
        quantile(posterior$log_excess, 0.025, names = FALSE)
    ) / (2 * qt(0.975, df = residual_student_df))
    prediction <- predict_patient_curve(fit, tau_values, patient)
    weighted_tau <- sum(fit$weights * tau_values)

    estimates[[index]] <- data.frame(
      Dataset = record$Dataset,
      sample_id = record$sample_id,
      n_cells_available = record$n_cells_available,
      minimum_trajectory_depth = min(patient$target_depth),
      maximum_trajectory_depth = max(patient$target_depth),
      n_trajectory_points_available = nrow(patient),
      n_trajectory_points_used = fit$components[[1]]$points_used,
      O_lambda_inf_central = central,
      O_lambda_inf_median = quantiles[2],
      O_lambda_inf_q025 = quantiles[1],
      O_lambda_inf_q975 = quantiles[3],
      log_O_minus_1_inf_location = log(pmax(central - 1, 1e-12)),
      log_O_minus_1_inf_scale = compatible_scale,
      weighted_tau = weighted_tau,
      posterior_tau_sd = sqrt(sum(
        fit$weights * (tau_values - weighted_tau)^2
      )),
      timescale_weight_entropy = -sum(
        fit$weights * log(pmax(fit$weights, 1e-12))
      ),
      observed_fit_RMSE_O = sqrt(mean((patient$O_lambda - prediction)^2)),
      maximum_component_convergence = max(vapply(
        fit$components, `[[`, numeric(1), "convergence"
      )),
      model = "latent_timescale_student_t3_scale_0.35",
      interval_type = "95% calibrated latent-timescale mixture interval",
      stringsAsFactors = FALSE
    )
    particles[[index]] <- data.frame(
      Dataset = record$Dataset,
      sample_id = record$sample_id,
      as.list(setNames(
        survival_particles,
        sprintf("particle_%02d", seq_len(n_particles))
      )),
      check.names = FALSE
    )
    cat(sprintf(
      "[%d/%d] %s | %s | N=%s | O_inf %.4f [%.4f, %.4f]\n",
      index, nrow(manifest), record$Dataset, record$sample_id,
      format(record$n_cells_available, big.mark = ","),
      central, quantiles[1], quantiles[3]
    ))
  }

  list(
    estimates = do.call(rbind, estimates),
    particles = do.call(rbind, particles)
  )
}


parse_arguments <- function() {
  arguments <- commandArgs(trailingOnly = TRUE)
  if (length(arguments) < 2) {
    stop(
      "Usage: Rscript fit_olambda_asymptotes.R trajectories.csv output_dir ",
      "[--minimum-cells 1000] [--n-particles 16] [--random-state 0]"
    )
  }

  values <- list(
    trajectories_csv = arguments[1],
    output_dir = arguments[2],
    minimum_cells = 1000L,
    n_particles = 16L,
    random_state = 0L
  )
  index <- 3L
  while (index <= length(arguments)) {
    option <- arguments[index]
    if (index == length(arguments)) stop("Incomplete argument: ", option)
    value <- arguments[index + 1L]
    if (option == "--minimum-cells") {
      values$minimum_cells <- as.integer(value)
    } else if (option == "--n-particles") {
      values$n_particles <- as.integer(value)
    } else if (option == "--random-state") {
      values$random_state <- as.integer(value)
    } else {
      stop("Unknown argument: ", option)
    }
    index <- index + 2L
  }
  if (!is.finite(values$minimum_cells) || values$minimum_cells < 1L) {
    stop("--minimum-cells must be positive")
  }
  if (!is.finite(values$n_particles) || values$n_particles < 2L) {
    stop("--n-particles must be at least 2")
  }
  values
}


read_trajectories <- function(path, minimum_cells) {
  frame <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("Dataset", "sample_id", "target_depth", "O_lambda")
  missing <- setdiff(required, names(frame))
  if (length(missing)) {
    stop("Trajectory table is missing: ", paste(missing, collapse = ", "))
  }
  if (!"n_cells_available" %in% names(frame)) {
    maximum_depth <- aggregate(
      target_depth ~ Dataset + sample_id, data = frame, FUN = max
    )
    names(maximum_depth)[3] <- "n_cells_available"
    frame <- merge(
      frame, maximum_depth,
      by = c("Dataset", "sample_id"), all.x = TRUE, sort = FALSE
    )
  }

  frame <- frame[
    is.finite(frame$target_depth) &
      is.finite(frame$O_lambda) &
      frame$O_lambda > 1 + minimum_O_minus_one &
      frame$n_cells_available >= minimum_cells,
    , drop = FALSE
  ]
  if (!nrow(frame)) stop("No eligible trajectory points remain")
  frame$patient_key <- paste(frame$Dataset, frame$sample_id, sep = "::")
  frame$log_O_minus_one <- log(frame$O_lambda - 1)
  frame
}


main <- function() {
  arguments <- parse_arguments()
  dir.create(arguments$output_dir, recursive = TRUE, showWarnings = FALSE)
  trajectories <- read_trajectories(
    arguments$trajectories_csv, arguments$minimum_cells
  )
  patients <- length(unique(trajectories$patient_key))
  cat(sprintf(
    "Fitting %d patients from %d trajectory points\n",
    patients, nrow(trajectories)
  ))

  training <- thin_by_patient(trajectories, training_maximum_points)
  tau_values <- timescale_grid()
  populations <- vector("list", length(tau_values))
  for (index in seq_along(tau_values)) {
    cat(sprintf(
      "Population component %d/%d | tau %.4f\n",
      index, length(tau_values), tau_values[index]
    ))
    populations[[index]] <- fit_population_model(training, tau_values[index])
  }
  populations <- share_asymptote_hierarchy(populations, tau_values)
  results <- estimate_patients(
    populations, tau_values, trajectories,
    arguments$n_particles, arguments$random_state
  )

  saveRDS(
    list(populations = populations, tau_values = tau_values),
    file.path(arguments$output_dir, "population_model.rds")
  )
  write.csv(
    results$estimates,
    file.path(arguments$output_dir, "patient_olambda_asymptotes.csv"),
    row.names = FALSE
  )
  write.csv(
    results$particles,
    file.path(arguments$output_dir, "patient_olambda_particles.csv"),
    row.names = FALSE
  )
  write.csv(
    data.frame(
      minimum_cells = arguments$minimum_cells,
      n_particles = arguments$n_particles,
      random_state = arguments$random_state,
      patients = patients,
      trajectory_points = nrow(trajectories),
      training_points = nrow(training),
      quadrature_nodes = quadrature_nodes,
      log_tau_scale = log_tau_scale,
      calibrated_log_asymptote_floor = calibrated_log_asymptote_floor,
      tau_minimum = min(tau_values),
      tau_median = median(tau_values),
      tau_maximum = max(tau_values)
    ),
    file.path(arguments$output_dir, "population_model_summary.csv"),
    row.names = FALSE
  )
  cat("Saved ", nrow(results$estimates), " patient asymptotes\n", sep = "")
}


main()
