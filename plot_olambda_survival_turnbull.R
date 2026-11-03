suppressPackageStartupMessages({
  library(ggplot2)
  library(survival)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop(
    paste(
      "Usage: Rscript plot_olambda_survival_turnbull.R",
      "patient_regime_probabilities.csv clinical_outcomes.csv",
      "model_result.csv output.png [x_max] [dpi]"
    )
  )
}

membership_path <- args[[1]]
outcome_path <- args[[2]]
model_result_path <- args[[3]]
output_path <- args[[4]]
x_max <- if (length(args) >= 5) as.numeric(args[[5]]) else 60
dpi <- if (length(args) >= 6) as.numeric(args[[6]]) else 600

membership <- read.csv(membership_path, check.names = FALSE)
outcomes <- read.csv(outcome_path, check.names = FALSE)
model_result <- read.csv(model_result_path, check.names = FALSE)

required_membership <- c("Dataset", "sample_id", "O_lambda_inf_central")
required_outcomes <- c("Dataset", "sample_id", "event_observed", "t_start", "t_end")
required_results <- c("HR_median", "HR_q025", "HR_q975", "crossover_median")

if (!all(required_membership %in% names(membership))) {
  stop("The membership file is missing required columns.")
}
if (!all(required_outcomes %in% names(outcomes))) {
  stop("The outcome file is missing required columns.")
}
if (!all(required_results %in% names(model_result)) || nrow(model_result) != 1) {
  stop("The model result file must contain one complete change-point result.")
}

data <- merge(
  membership,
  outcomes,
  by = c("Dataset", "sample_id"),
  all = FALSE
)
if (nrow(data) != nrow(membership)) {
  stop("Not every patient in the membership file has a matching clinical outcome.")
}

crossover <- model_result$crossover_median[[1]]
data$regime <- ifelse(
  data$O_lambda_inf_central >= crossover,
  "High Oλ",
  "Low Oλ"
)
data$regime <- factor(data$regime, levels = c("Low Oλ", "High Oλ"))

# Preserve interval-censored events. Censored observations contribute (t_end, infinity).
data$interval_left <- ifelse(data$event_observed == 1, data$t_start, data$t_end)
data$interval_right <- ifelse(data$event_observed == 1, data$t_end, NA_real_)

fit <- survfit(
  Surv(interval_left, interval_right, type = "interval2") ~ regime,
  data = data,
  conf.type = "log-log"
)

fit_summary <- summary(fit)
curve <- data.frame(
  time = fit_summary$time,
  survival = fit_summary$surv,
  lower = fit_summary$lower,
  upper = fit_summary$upper,
  regime = sub("^[^=]+=", "", as.character(fit_summary$strata))
)
curve$regime <- factor(curve$regime, levels = levels(data$regime))

stepify <- function(frame, observed) {
  pieces <- lapply(split(frame, frame$regime), function(group) {
    group <- group[order(group$time), ]
    group <- group[group$time <= x_max, ]
    previous_survival <- c(1, head(group$survival, -1))
    previous_lower <- c(1, head(group$lower, -1))
    previous_upper <- c(1, head(group$upper, -1))
    regime_name <- as.character(group$regime[[1]])
    last_follow_up <- min(
      x_max,
      max(observed$t_end[as.character(observed$regime) == regime_name])
    )

    stepped <- data.frame(
      time = c(0, as.vector(rbind(group$time, group$time))),
      survival = c(1, as.vector(rbind(previous_survival, group$survival))),
      lower = c(1, as.vector(rbind(previous_lower, group$lower))),
      upper = c(1, as.vector(rbind(previous_upper, group$upper))),
      regime = group$regime[[1]]
    )
    if (last_follow_up > max(stepped$time)) {
      stepped <- rbind(
        stepped,
        transform(tail(stepped, 1), time = last_follow_up)
      )
    }
    stepped
  })
  do.call(rbind, pieces)
}

curve <- stepify(curve, data)
colors <- c("Low Oλ" = "#2B6CB0", "High Oλ" = "#C43D4D")
fills <- c("Low Oλ" = "#9EC5E8", "High Oλ" = "#E9A2AA")

annotation <- sprintf(
  "Adjusted Bayesian HR %.2f (95%% CrI %.2f–%.2f)\nCrossover Oλ∞ = %.3f",
  model_result$HR_median[[1]],
  model_result$HR_q025[[1]],
  model_result$HR_q975[[1]],
  crossover
)

figure <- ggplot(curve, aes(x = time, y = survival, color = regime, fill = regime)) +
  geom_ribbon(
    aes(ymin = lower, ymax = upper, group = regime),
    alpha = 0.18,
    color = NA
  ) +
  geom_path(aes(group = regime), linewidth = 1.15) +
  annotate(
    "text",
    x = x_max * 0.97,
    y = 0.97,
    label = annotation,
    hjust = 1,
    vjust = 1,
    size = 3.8,
    lineheight = 1.15,
    color = "#202020"
  ) +
  scale_color_manual(values = colors, name = NULL) +
  scale_fill_manual(values = fills, name = NULL) +
  scale_x_continuous(
    limits = c(0, x_max),
    breaks = pretty(c(0, x_max), n = 7),
    expand = expansion(mult = c(0.005, 0.015))
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2),
    labels = function(x) sprintf("%d%%", round(100 * x)),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    x = "Months after infusion",
    y = "Progression-free survival"
  ) +
  theme_classic(base_size = 12) +
  theme(
    axis.title = element_text(size = 11.5),
    axis.text = element_text(color = "#252525"),
    legend.position = "inside",
    legend.position.inside = c(0.80, 0.16),
    legend.background = element_rect(fill = "white", color = "#D0D0D0", linewidth = 0.3),
    legend.key.width = unit(1.3, "cm"),
    panel.grid.major.y = element_line(color = "#E8E8E8", linewidth = 0.35),
    panel.grid.major.x = element_blank()
  )

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
ggsave(
  output_path,
  figure,
  width = 8.4,
  height = 5.8,
  dpi = dpi,
  bg = "white"
)

curve_path <- sub("\\.[^.]+$", "_curve.csv", output_path)
write.csv(curve, curve_path, row.names = FALSE)

group_sizes <- table(data$regime)
cat("Patients:", nrow(data), "| events:", sum(data$event_observed), "\n")
cat("Low Oλ:", group_sizes[["Low Oλ"]], "| High Oλ:", group_sizes[["High Oλ"]], "\n")
cat("Figure:", output_path, "\n")
cat("Curve data:", curve_path, "\n")
