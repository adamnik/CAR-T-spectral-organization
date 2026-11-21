# Usage: Rscript fit_direct_olambda.R old_trajectories.csv cohort.csv new_output_dir
# The legacy O_lambda trajectory column stores T_lambda. Cohort membership is
# defined only by Dataset and sample_id. All response values are computed here.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Expected trajectories.csv, cohort.csv, and a new output directory")
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
source(file.path(dirname(normalizePath(script)), "fit_olambda_asymptotes.R"), local = TRUE)
rational_basis <- add_basis
mixed_basis <- function(frame, tau) {
  frame <- rational_basis(frame, tau)
  frame$f2 <- exp(-frame$scaled_depth / tau)
  frame
}

fit_direct <- function(pop, noise, frame, start) {
  frame <- mixed_basis(uniform_depth_subset(frame, posterior_maximum_points), 2)
  if (nrow(frame) < 3L) stop("Fewer than three fitting depths for ", frame$sample_id[1])
  X <- cbind(1, frame$f1, frame$f2)
  variance <- noise$residual_sigma^2 * frame$scaled_depth^(2 * noise$variance_power) +
    noise$residual_floor^2
  prior <- conditional_slope_prior(pop$random_covariance)
  weight <- min(1, 6 / nrow(frame))
  objective <- function(b) {
    residual <- frame$O - plogis(-drop(X %*% b))
    u <- b - pop$fixed_effect
    slopes <- u[-1] - drop(prior$conditional_mean_coefficient * u[1])
    d2 <- drop(crossprod(slopes, prior$conditional_precision %*% slopes))
    weight * sum((residual_student_df + 1) / 2 *
      log1p(residual^2 / (residual_student_df * variance))) +
      0.5 * u[1]^2 / prior$intercept_variance +
      (slope_student_df + 2) / 2 * log1p(d2 / slope_student_df)
  }
  fits <- lapply(list(start, as.numeric(pop$fixed_effect)), function(b) {
    optim(b, objective, method = "BFGS", hessian = TRUE,
          control = list(maxit = 500, reltol = 1e-11))
  })
  best <- fits[[which.min(vapply(fits, `[[`, numeric(1), "value"))]]
  if (best$convergence != 0) stop("Patient fit failed for ", frame$sample_id[1])
  list(b = best$par, scale = sqrt(solve(positive_definite(best$hessian))[1,1]),
       points = nrow(frame))
}

out <- args[3]
if (file.exists(out)) stop("Choose a new output directory: ", out)
keys <- c("Dataset", "sample_id")
cohort <- read.csv(args[2], stringsAsFactors = FALSE)[keys]
stopifnot(nrow(cohort) > 0, !anyNA(cohort), !anyDuplicated(cohort))
d <- merge(read_trajectories(args[1], 1000L), cohort, by = keys)
patients <- unique(d[c(keys, "n_cells_available")])
stopifnot(nrow(patients) == nrow(cohort), !anyDuplicated(patients[keys]),
          all(d$target_depth > 0))
# Use the composite patient identity for nlme grouping.
d$O <- 1 / d$O_lambda
train <- d
train$sample_id <- as.character(as.integer(interaction(train$Dataset, train$sample_id, drop = TRUE)))
train <- thin_by_patient(train, training_maximum_points)
cat(sprintf("Training on %d patients, %d datasets, %d trajectory points\n",
            nrow(patients), length(unique(d$Dataset)), nrow(d)))
anchor <- fit_population_model(train, 2)
add_basis <- mixed_basis
pop <- share_asymptote_hierarchy(list(anchor, fit_population_model(train, 2)), c(2,2))[[2]]
raw <- train
raw$log_O_minus_one <- raw$O
noise <- fit_population_model(raw, 2)
noise$residual_floor <- noise$residual_floor * median(train$O * (1 - train$O))
effective_points_per_patient <- 6

estimates <- particles <- vector("list", nrow(patients))
p <- (seq_len(16) - 0.5) / 16
for (i in seq_len(nrow(patients))) {
  meta <- patients[i,,drop = FALSE]
  frame <- d[d$Dataset == meta$Dataset & d$sample_id == meta$sample_id,,drop = FALSE]
  start <- patient_posterior(pop, frame, 2)$coefficients
  fit <- fit_direct(pop, noise, frame, start)
  alpha <- fit$b[1]
  # Increasing O quantiles for O=logistic(-alpha); no additive intercept floor.
  quantiles <- function(p) plogis(-alpha + fit$scale * qt(p, df = residual_student_df))
  draws <- quantiles(p)
  stopifnot(all(is.finite(draws)), all(draws > 0 & draws < 1))
  estimates[[i]] <- data.frame(meta, O_lambda_inf_central = plogis(-alpha),
    O_lambda_inf_median = quantiles(.5), O_lambda_inf_q025 = quantiles(.025),
    O_lambda_inf_q975 = quantiles(.975), alpha = alpha, alpha_scale = fit$scale,
    b_rational = fit$b[2], b_exponential = fit$b[3], n_trajectory_points_available = nrow(frame),
    n_trajectory_points_used = fit$points, olambda_scale = "reciprocal",
    model = "bounded_direct_no_asymptote_floor")
  particles[[i]] <- data.frame(meta[keys],
    as.list(setNames(draws, sprintf("particle_%02d", seq_along(p)))), check.names = FALSE)
  cat(sprintf("[%d/%d] %s | %s | O_inf %.4f [%.4f, %.4f]\n",
    i, nrow(patients), meta$Dataset, meta$sample_id, plogis(-alpha), quantiles(.025), quantiles(.975)))
}
# Publish the pair together, after all fits succeed.
dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
stage <- tempfile("direct-fit-", tmpdir = dirname(out))
stopifnot(dir.create(stage))
write.csv(do.call(rbind, estimates), file.path(stage, "patient_olambda_asymptotes.csv"), row.names = FALSE)
write.csv(do.call(rbind, particles), file.path(stage, "patient_olambda_particles.csv"), row.names = FALSE)
stopifnot(file.rename(stage, out))
