# ===========================================================================
# 01 — CONFIRMATORY PRIMARY: crossed cumulative-link mixed model (proportional odds)
# Random intercepts for item AND rater; random model-by-rater slope tests whether
# raters rank the systems consistently. Reports cumulative ORs vs ChatICU and
# model-implied P(score>=4). This is the formally confirmatory primary analysis
# (replaces the question-clustered OrdinalGEE, which is retained as a sensitivity).
# ===========================================================================
source("R/00_common.R")
d <- load_long()
OUT <- "results_R"; dir.create(OUT, showWarnings = FALSE)

res <- list()
for (dom in LIKERT_DOMAINS) {
  s <- domain_df(d, dom, ordered = TRUE)
  cat(sprintf("\n================ %s (n=%d) ================\n", dom, nrow(s)))

  # Try the full model with random model-by-rater slope; fall back to intercepts-only.
  fit <- tryCatch(
    clmm(y ~ model + (1 | item) + (1 + model | rater), data = s,
         link = "logit", Hess = TRUE,
         control = clmm.control(maxIter = 200, gradTol = 1e-4)),
    error = function(e) NULL)
  slope_ok <- !is.null(fit)
  if (is.null(fit)) {
    cat("  [random-slope model failed to converge; using crossed random intercepts]\n")
    fit <- clmm(y ~ model + (1 | item) + (1 | rater), data = s, link = "logit", Hess = TRUE)
  }

  co <- summary(fit)$coefficients
  beta_rows <- grep("^model", rownames(co))
  est <- co[beta_rows, "Estimate"]; se <- co[beta_rows, "Std. Error"]
  # ordinal::clm(m) parameterises logit(P(Y<=j)) = theta_j - beta*x, so a positive beta
  # shifts mass to HIGHER categories. Cumulative OR (odds of a higher score) for a
  # comparator vs the ChatICU reference = exp(beta); OR<1 => comparator worse than ChatICU.
  or  <- exp(est); lo <- exp(est - 1.96*se); hi <- exp(est + 1.96*se)
  p   <- co[beta_rows, "Pr(>|z|)"]

  # model-implied P(score>=4) per system (population-level: random effects at 0).
  # clmm: P(Y<=j) = plogis(theta_j - eta); P(Y>=4) = 1 - plogis(theta_{3|4} - eta).
  pge4 <- tryCatch({
    th <- fit$alpha; theta34 <- th[grep("3\\|4", names(th))]
    lv <- levels(s$model); eta <- setNames(numeric(length(lv)), lv)
    eta[sub("^model", "", names(est))] <- est   # ChatICU (ref) eta = 0
    setNames(as.numeric(1 - plogis(theta34 - eta)), lv)
  }, error = function(e) setNames(rep(NA_real_, 3), levels(s$model)))

  tab <- data.frame(domain = dom,
                    contrast = sub("^model", "", names(est)),
                    cumOR_vs_ChatICU = round(or, 3),
                    ci_low = round(lo, 3),
                    ci_high = round(hi, 3),
                    p = signif(p, 3),
                    P_ge4_ChatICU = round(pge4["ChatICU"], 3),
                    P_ge4_comparator = round(pge4[sub("^model","",names(est))], 3),
                    random_slope = slope_ok)
  print(tab, row.names = FALSE)
  if (!all(is.na(pge4))) cat(sprintf("  P(score>=4): %s\n",
      paste(sprintf("%s=%.3f", names(pge4), pge4), collapse = "  ")))
  res[[dom]] <- tab
}

allres <- do.call(rbind, res)
write.csv(allres, file.path(OUT, "table2_crossed_clmm.csv"), row.names = FALSE)
cat(sprintf("\n[01] wrote %s/table2_crossed_clmm.csv  (OR<1 => system worse than ChatICU)\n", OUT))
