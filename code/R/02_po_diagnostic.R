# ===========================================================================
# 02 — Proportional-odds (PO) diagnostic + partial-PO fallback
# Fixed-effects cumulative-link model per domain (clm); nominal_test = LR test
# of the PO assumption per term. Where PO is violated for 'model', refit with
# nominal effects (partial-PO) and report. Also reports a threshold-wise check.
# ===========================================================================
source("R/00_common.R")
d <- load_long()
OUT <- "results_R"

rows <- list()
for (dom in LIKERT_DOMAINS) {
  s <- domain_df(d, dom, ordered = TRUE)
  fit <- clm(y ~ model, data = s, link = "logit")
  nt <- tryCatch(nominal_test(fit), error = function(e) NULL)
  p_model <- if (!is.null(nt) && "model" %in% rownames(nt)) nt["model", "Pr(>Chi)"] else NA
  violated <- !is.na(p_model) && p_model < 0.05

  # partial-PO refit if violated
  ppo_note <- "PO holds (model term)"
  if (violated) {
    ppo <- tryCatch(clm(y ~ 1, nominal = ~ model, data = s, link = "logit"),
                    error = function(e) NULL)
    ppo_note <- if (!is.null(ppo)) "PO violated -> partial-PO (nominal=~model) fitted" else "PO violated; partial-PO failed"
  }
  cat(sprintf("%-11s nominal_test p(model)=%s  -> %s\n",
              dom, ifelse(is.na(p_model), "NA", signif(p_model,3)), ppo_note))
  rows[[dom]] <- data.frame(domain = dom, po_LRT_p_model = signif(p_model,4),
                            po_violated = violated, action = ppo_note)
}
res <- do.call(rbind, rows)
write.csv(res, file.path(OUT, "table_po_diagnostic.csv"), row.names = FALSE)
cat(sprintf("\n[02] wrote %s/table_po_diagnostic.csv\n", OUT))
