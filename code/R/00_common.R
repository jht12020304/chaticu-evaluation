# ===========================================================================
# ChatICU — common data loading & helpers for the confirmatory R re-runs
# Data: long_with_global_qid.csv  (11,424 ratings; 68 items x 3 systems x 8 raters x 7 domains)
# Columns: expert, question, model, domain, score, role, qid
# Model coding: ChatICU is the REFERENCE category for all contrasts.
# ===========================================================================
suppressWarnings(suppressMessages({
  library(ordinal); library(lme4); library(emmeans); library(geepack)
}))

DATA <- "long_with_global_qid.csv"
LIKERT_DOMAINS <- c("Accuracy","Relevance","Clarity","Trust","Comparison","Confidence")

load_long <- function(path = DATA) {
  d <- read.csv(path, stringsAsFactors = FALSE)
  d$model  <- relevel(factor(d$model), ref = "ChatICU")  # ChatICU = reference
  d$rater  <- factor(d$expert)
  d$item   <- factor(d$qid)                               # global 1..68
  d$role   <- factor(d$role)
  d
}

# subset one domain; build ordered outcome for Likert
domain_df <- function(d, dom, ordered = TRUE) {
  s <- d[d$domain == dom, ]
  s <- s[!is.na(s$score), ]
  if (ordered) s$y <- factor(round(s$score), levels = 1:5, ordered = TRUE)
  else         s$y <- s$score
  droplevels(s)
}

emm_to_df <- function(x) as.data.frame(x)

cat(sprintf("[00_common] loaded helpers; outcome=ordered 1-5 (Likert), ref=ChatICU\n"))
