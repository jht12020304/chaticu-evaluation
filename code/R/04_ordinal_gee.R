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
