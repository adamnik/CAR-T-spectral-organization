# Synthetic check for the patient update
expressions <- parse('fit_olambda_asymptotes.R')
for (expression in expressions[-length(expressions)]) eval(expression)

frame <- data.frame(target_depth = seq(500, 10000, length.out = 80))
frame$log_O_minus_one <- -.3 + .7 / (frame$target_depth / 500 + 2) +
  .2 / (frame$target_depth / 500 + 2)^2 + .01 * sin(seq_len(nrow(frame)))
frame$O_lambda <- 1 + exp(frame$log_O_minus_one)
population <- list(
  fixed_effect = c(-.2, .5, .1), random_covariance = diag(c(.1, .2, .2)),
  residual_sigma = .1, variance_power = -.2, residual_floor = .015
)
result <- patient_posterior(population, frame)
# Results from the original patient update
expected_coefficients <- c(-0.294450881383516, 0.646810298153956, 0.148758548947383)
expected_interval <- c(1.68819287519518, 1.74494053589147, 1.80636754900561)
interval <- c(result$lower, result$central, result$upper)
stopifnot(
  isTRUE(all.equal(result$coefficients, expected_coefficients, tolerance = 1e-10)),
  isTRUE(all.equal(interval, expected_interval, tolerance = 1e-10)),
  result$points_used == 40L,
  result$information_weight == .3
)
cat('Patient asymptote regression check passed\n')
