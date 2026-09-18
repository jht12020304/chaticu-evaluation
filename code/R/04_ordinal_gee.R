# ===========================================================================
# 04 — Robustness: ordinal GEE (multgee::ordLORgee), cumulative logit, independence
# working correlation, clustered by question; sandwich standard errors. One model
# per Likert domain. ChatICU reference. Reported in Supplementary Figure S2.
# Run from the code/ directory after make_input.R: Rscript R/04_ordinal_gee.R
# ===========================================================================
source("R/00_common.R")
suppressWarnings(suppressMessages(library(multgee)))
d <- load_long()
OUT <- "results_R"; dir.create(OUT, showWarnings = FALSE)

rows <- list()
for (dom in LIKERT_DOMAINS) {
  s <- domain_df(d, dom, ordered = TRUE)
  s$qid_int <- as.integer(s$item)
  s <- s[order(s$qid_int), ]
  fit <- ordLORgee(y ~ model, data = s, id = qid_int, link = "logit", LORstr = "independence")
  co <- summary(fit)$coefficients
  b <- grep("^model", rownames(co))
  # ordLORgee models logit P(Y <= j) = beta_j + x'beta, so a positive beta shifts mass to LOWER
  # categories; the cumulative OR of a HIGHER rating vs ChatICU is exp(-beta).
  est <- co[b, "Estimate"]; se <- co[b, "san.se"]
  rows[[dom]] <- data.frame(domain = dom, contrast = sub("^model", "", rownames(co)[b]),
                            beta = round(est, 5), se = round(se, 5), p = signif(co[b, "Pr(>|san.z|)"], 3),
                            conv = fit$convergence$conv,
                            logOR = round(-est, 5), OR = exp(-est), ci_lo = exp(-est - 1.96 * se), ci_hi = exp(-est + 1.96 * se))
}
res <- do.call(rbind, rows); rownames(res) <- NULL
print(res, row.names = FALSE)
write.csv(res, file.path(OUT, "gee_results.csv"), row.names = FALSE)
cat(sprintf("\n[04] wrote %s/gee_results.csv  (OR<1 => lower odds of a higher rating than ChatICU)\n", OUT))
