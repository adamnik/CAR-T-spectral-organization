# Synthetic checks for latent-timescale O-lambda inference
expressions <- parse('fit_olambda_asymptotes.R')
for (expression in expressions[-length(expressions)]) eval(expression)

frame <- data.frame(target_depth = seq(500, 10000, length.out = 80))
basis <- add_basis(frame, tau = 2)
frame$log_O_minus_one <- -.3 + .7 * basis$f1 + .2 * basis$f2 +
  .01 * sin(seq_len(nrow(frame)))
frame$O_lambda <- 1 + exp(frame$log_O_minus_one)
population <- list(
  fixed_effect = c(-.2, .5, .1), random_covariance = diag(c(.1, .2, .2)),
  residual_sigma = .1, variance_power = -.2, residual_floor = .015
)
result <- patient_posterior(population, frame, tau = 2)
posterior <- sample_asymptote_posterior(
  list(components = list(result), weights = 1),
  simulations = 1000,
  seed = 0
)
grid <- timescale_grid()
stopifnot(
  length(grid) == 9L,
  isTRUE(all.equal(median(grid), 2, tolerance = 1e-12)),
  result$points_used == 40L,
  result$convergence == 0,
  abs(result$central - (1 + exp(-.3))) < .1,
  all(is.finite(posterior$O_lambda)),
  all(posterior$O_lambda > 1)
)
cat('Latent-timescale asymptote checks passed\n')
