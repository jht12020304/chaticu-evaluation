suppressPackageStartupMessages({library(brms); library(rstan); library(dplyr)})
rstan_options(boost_lib = system.file("include", package="BH"), eigen_lib = system.file("include", package="RcppEigen"), auto_write = TRUE)
options(mc.cores = 4)
d1 <- read.csv("data/D1_ratings_long.csv", stringsAsFactors = FALSE)
d1$model <- factor(d1$model, levels = c("ChatICU", "ChatGPT", "OpenEvidence"))
d1$question_id <- factor(d1$question_id); d1$expert <- factor(d1$expert)
DOMS <- c("Accuracy", "Relevance", "Clarity", "Trust", "Comparison", "Confidence")
pri <- c(prior(normal(0, 2.5), class = "b"), prior(student_t(3, 0, 2.5), class = "sd"))
out <- list(); diag <- list()
summarise_fit <- function(fit, dm) {
  dr <- as_draws_df(fit)
  np <- nuts_params(fit); ndiv <- sum(subset(np, Parameter == "divergent__")$Value)
  rh <- rhat(fit); rh <- rh[!is.na(rh)]
  for (ct in c("ChatGPT", "OpenEvidence")) {
    b <- dr[[paste0("b_model", ct)]]
    out[[length(out) + 1]] <<- data.frame(domain = dm, contrast = ct, OR = exp(median(b)),
      cri_lo = exp(quantile(b, 0.025)), cri_hi = exp(quantile(b, 0.975)), P_OR_lt_1 = mean(b < 0),
      n_divergent = ndiv, max_rhat = max(rh), min_bulk_ess = min(summary(fit)$fixed$Bulk_ESS))
  }
  diag[[dm]] <<- data.frame(domain = dm, n_divergent = ndiv, max_rhat = max(rh), family = family(fit)$family)
}
for (dm in DOMS) {
  dd <- d1 %>% filter(domain == dm)
  t0 <- Sys.time()
  fit <- brm(score ~ model + (1 | question_id) + (1 + model | expert), data = dd,
             family = cumulative("logit"), prior = pri, chains = 4, iter = 2000, warmup = 1000,
             cores = 4, seed = 20260902, control = list(adapt_delta = 0.95, max_treedepth = 12), refresh = 0, silent = 2)
  summarise_fit(fit, dm)
  cat(dm, "done in", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min; divergences:", diag[[dm]]$n_divergent, "\n")
  write.csv(bind_rows(out), "data/bayes_results.csv", row.names = FALSE)
}
ds <- d1 %>% filter(domain == "Safety")
fit <- brm(score ~ model + (1 | question_id) + (1 | expert), data = ds, family = bernoulli("logit"), prior = pri,
           chains = 4, iter = 2000, warmup = 1000, cores = 4, seed = 20260902, control = list(adapt_delta = 0.95), refresh = 0, silent = 2)
summarise_fit(fit, "Safety")
write.csv(bind_rows(out), "data/bayes_results.csv", row.names = FALSE)
write.csv(bind_rows(diag), "data/bayes_diagnostics.csv", row.names = FALSE)
cat("brms", as.character(packageVersion("brms")), "rstan", as.character(packageVersion("rstan")), "\n")
print(bind_rows(out), digits = 3)
