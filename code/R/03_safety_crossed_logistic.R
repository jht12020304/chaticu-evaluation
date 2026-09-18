# ===========================================================================
# 03 — Safety: crossed mixed-effects logistic (glmer) with item + rater random
# intercepts, vs the population-averaged binomial GEE (geeglm, exchangeable,
# clustered by item). Outcome = no-risk (1) vs risk (0). ChatICU reference.
# ===========================================================================
source("R/00_common.R")
suppressWarnings(suppressMessages(library(geepack)))
d <- load_long()
OUT <- "results_R"
s <- domain_df(d, "Safety", ordered = FALSE)   # y = 0/1 (1 = no identifiable risk)

# ---- crossed mixed-effects logistic ----
glfit <- glmer(y ~ model + (1 | item) + (1 | rater), data = s, family = binomial,
               control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
gco <- summary(glfit)$coefficients
b <- grep("^model", rownames(gco))
glmm_tab <- data.frame(
  contrast = sub("^model", "", rownames(gco)[b]),
  OR_vs_ChatICU = round(exp(gco[b, "Estimate"]), 3),
  ci_low  = round(exp(gco[b, "Estimate"] - 1.96*gco[b, "Std. Error"]), 3),
  ci_high = round(exp(gco[b, "Estimate"] + 1.96*gco[b, "Std. Error"]), 3),
  p = signif(gco[b, "Pr(>|z|)"], 3),
  estimator = "glmer crossed (item+rater)")

# ---- population-averaged GEE (clustered by item) ----
s2 <- s[order(s$item), ]
gee <- geeglm(y ~ model, id = item, data = s2, family = binomial, corstr = "exchangeable")
ge <- summary(gee)$coefficients
b2 <- grep("^model", rownames(ge))
gee_tab <- data.frame(
  contrast = sub("^model", "", rownames(ge)[b2]),
  OR_vs_ChatICU = round(exp(ge[b2, "Estimate"]), 3),
  ci_low  = round(exp(ge[b2, "Estimate"] - 1.96*ge[b2, "Std.err"]), 3),
  ci_high = round(exp(ge[b2, "Estimate"] + 1.96*ge[b2, "Std.err"]), 3),
  p = signif(ge[b2, "Pr(>|W|)"], 3),
  estimator = "binomial GEE (item-clustered)")

out <- rbind(glmm_tab, gee_tab)
print(out, row.names = FALSE)
cat("\nrisk events: ", paste(sprintf("%s=%d/%d", levels(s$model),
    tapply(s$y==0, s$model, sum), tapply(s$y, s$model, length)), collapse="  "),
    "  (OR<1 => lower odds of no-risk => less safe than ChatICU)\n")
write.csv(out, file.path(OUT, "table5_safety_crossed.csv"), row.names = FALSE)
cat(sprintf("[03] wrote %s/table5_safety_crossed.csv\n", OUT))
