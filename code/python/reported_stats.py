#!/usr/bin/env python3
"""Recomputes, from the released data, the descriptive and exploratory statistics reported in the manuscript:
Table S1 (means with multiway cluster bootstrap 95% CIs), the composite score and domain correlations (Figure S1),
Friedman / Holm-adjusted Wilcoxon tests, Safety proportions with Wilson CIs and citation counts (Figure 4),
Table S3 and the flagged-burden metrics of the LLM-based evaluation (Figures S3-S5) and the exploratory
convergence / threshold analysis (Figure S7). Run from the repository root:
    python3 code/python/reported_stats.py
Outputs are written to code/results_py/; a check table against the archived results/ files is printed.
Bootstrap CIs are seeded but not bit-identical to the archived values (different resampling implementation);
they agree to about 0.02.
"""
import json, os, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, cohen_kappa_score

SEED, B = 20260902, 2000
SYS = ["ChatICU", "ChatGPT", "OpenEvidence"]
QUAL = ["Accuracy", "Relevance", "Clarity", "Trust", "Comparison"]
DIMS = ["drug_choice", "deescalation_stop_rule", "interaction_contraindication", "route_formulation",
        "organ_function_adjustment", "titration_target", "monitoring_parameter"]
W_SEV = {"Mild": 1, "Moderate": 3, "Severe": 12}
OUT = "code/results_py"; os.makedirs(OUT, exist_ok=True)

d = pd.read_csv("data/expert_ratings_long.csv", dtype={"question_id": str})
a = pd.read_csv("data/llm_evaluation_per_answer.csv", dtype={"question_id": str})
m = pd.read_csv("data/llm_evaluation_per_mention.csv", dtype={"question_id": str})
checks = []
def chk(label, computed, archived): checks.append((label, computed, archived))

def multiway_ci(M, seed=SEED):
    """Percentile CI of the mean of a questions x raters matrix, resampling questions and raters simultaneously."""
    rng = np.random.default_rng(seed); nq, nr = M.shape
    boots = np.array([M[np.ix_(rng.integers(0, nq, nq), rng.integers(0, nr, nr))].mean() for _ in range(B)])
    return np.percentile(boots, [2.5, 97.5])

def qboot_ci(groups, stat, seed=SEED):
    """Percentile CI of stat(list of per-question records) resampling questions with replacement."""
    rng = np.random.default_rng(seed); n = len(groups)
    boots = np.array([stat([groups[i] for i in rng.integers(0, n, n)]) for _ in range(B)])
    return np.percentile(boots, [2.5, 97.5])

# ---------------- Table S1: domain means with multiway cluster bootstrap CIs ----------------
arch_s1 = pd.read_csv("results/TableS1_domain_means_CI.csv").set_index("Domain")
rows, acc = [], {}
for dom in QUAL + ["Confidence"]:
    r = {"Domain": dom, "Outcome type": "Rater confidence (not answer quality)" if dom == "Confidence" else "Quality domain"}
    for s in SYS:
        M = d[(d.domain == dom) & (d.system == s)].pivot(index="question_id", columns="rater", values="score").values
        lo, hi = multiway_ci(M); r[f"{s} mean [95% CI]"] = f"{M.mean():.2f} [{lo:.2f}, {hi:.2f}]"
        if dom == "Accuracy": acc[s] = (M.mean(), lo, hi, M.size)
        chk(f"TableS1 {dom} {s}", r[f"{s} mean [95% CI]"], arch_s1.loc[dom, f"{s} mean [95% CI]"])
    rows.append(r)
pd.DataFrame(rows).to_csv(f"{OUT}/TableS1_domain_means_CI.csv", index=False)

# ---------------- Composite score and domain correlations (Figure S1) ----------------
wide = d[d.domain.isin(QUAL)].pivot_table(index=["question_id", "system", "rater"], columns="domain", values="score")[QUAL]
comp = wide.mean(axis=1).rename("composite").reset_index()
comp_rows = []
for s in SYS:
    M = comp[comp.system == s].pivot(index="question_id", columns="rater", values="composite").values
    lo, hi = multiway_ci(M); comp_rows.append({"model": s, "mean": round(M.mean(), 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
pd.DataFrame(comp_rows).to_csv(f"{OUT}/figS1A_composite.csv", index=False)
chk("composite ChatICU (manuscript 4.273 [4.026, 4.507])", f"{comp_rows[0]['mean']:.3f} [{comp_rows[0]['ci_lo']:.3f}, {comp_rows[0]['ci_hi']:.3f}]", "4.273 [4.026, 4.507]")
corr = wide.corr().round(3); corr.to_csv(f"{OUT}/figS1B_corr.csv")
chk("r(Accuracy, Trust) (manuscript 0.86)", f"{corr.loc['Accuracy', 'Trust']:.2f}", "0.86")

# ---------------- Friedman and Holm-adjusted Wilcoxon tests (question-level means) ----------------
arch_fw = pd.read_csv("results/friedman_wilcoxon.csv").set_index("domain")
fw = []
pairs = [("ChatICU", "ChatGPT"), ("ChatICU", "OpenEvidence"), ("ChatGPT", "OpenEvidence")]
for dom in QUAL + ["Confidence"]:
    q = d[d.domain == dom].groupby(["question_id", "system"]).score.mean().unstack("system")
    fr = stats.friedmanchisquare(*[q[s].values for s in SYS])
    raw = [stats.wilcoxon(q[x].values, q[y].values).pvalue for x, y in pairs]
    order = np.argsort(raw); holm = [None] * 3
    for rank, i in enumerate(order): holm[i] = min(1.0, raw[i] * (3 - rank))
    rank_order = " > ".join(q.mean().sort_values(ascending=False).index)
    fw.append({"domain": dom, "friedman_chi2": fr.statistic, "friedman_p": fr.pvalue, "n_questions": len(q),
               **{f"wilcoxon_holm_{x}_vs_{y}": p for (x, y), p in zip(pairs, holm)}, "rank_order": rank_order})
    chk(f"Friedman chi2 {dom}", f"{fr.statistic:.3f}", f"{arch_fw.loc[dom, 'friedman_chi2']:.3f}")
    chk(f"Holm Wilcoxon {dom} ChatGPT vs OpenEvidence", f"{holm[2]:.3g}", f"{arch_fw.loc[dom, 'wilcoxon_holm_ChatGPT_vs_OpenEvidence']:.3g}")
pd.DataFrame(fw).to_csv(f"{OUT}/friedman_wilcoxon.csv", index=False)

# ---------------- Figure 4: citation counts, Accuracy means, Safety proportions with Wilson CIs ----------------
arch_f4 = pd.read_csv("results/fig4_stats.csv").set_index("model")
f4 = []
for s in SYS:
    aa = a[a.system == s]; sf = d[(d.domain == "Safety") & (d.system == s)].score
    lo, hi = proportion_confint(sf.sum(), len(sf), method="wilson")
    f4.append({"model": s, "mean_markers": aa.n_citation_markers.mean(), "n_with_citation": int((aa.n_citation_markers > 0).sum()), "n_answers": len(aa),
               "acc_mean": acc[s][0], "acc_ci_lo": acc[s][1], "acc_ci_hi": acc[s][2], "acc_n": acc[s][3],
               "saf_sum": int(sf.sum()), "saf_size": len(sf), "saf_prop_safe": sf.mean(), "saf_wilson_lo": lo, "saf_wilson_hi": hi, "saf_prop_risk": 1 - sf.mean(),
               "mean_references": aa.n_references.mean(), "n_with_reference_list": int((aa.n_references > 0).sum()), "mean_doi_url": aa.n_doi_url.mean()})
    for k in ["mean_markers", "saf_prop_safe", "saf_wilson_lo", "saf_wilson_hi", "mean_references", "mean_doi_url"]:
        chk(f"fig4 {s} {k}", f"{f4[-1][k]:.4f}", f"{arch_f4.loc[s, k]:.4f}")
pd.DataFrame(f4).to_csv(f"{OUT}/fig4_stats.csv", index=False)

# ---------------- LLM-based evaluation: Table S3, dose classification, flagged burden (Figures S3-S5) ----------------
arch_d4 = json.load(open("results/D4_summary_stats.json"))["per_system"]
wcc = a.core_correctness.map({"fully_correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0})
per_system, adequate = {}, {}
for s in SYS:
    aa = a[a.system == s].copy(); mm = m[m.system == s]; st = mm[mm.mention_type == "stated"]
    fid_groups = [[v] for v in aa.semantic_fidelity.values]              # one answer per question
    fid_ci = qboot_ci(fid_groups, lambda g: np.mean([x[0] for x in g]))
    cls = st["class"].value_counts().to_dict()
    TP = cls.get("correct", 0); FP = len(st) - TP; FN = int((mm.mention_type == "omitted").sum())
    def f1_of(groups):
        tp = sum(g[0] for g in groups); fp = sum(g[1] for g in groups); fn = sum(g[2] for g in groups)
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        return 2 * p * r / (p + r) if p + r else 0
    per_q = []
    for qid in aa.question_id:
        mq = mm[mm.question_id == qid]; sq = mq[mq.mention_type == "stated"]
        per_q.append((int((sq["class"] == "correct").sum()), int((sq["class"] != "correct").sum()), int((mq.mention_type == "omitted").sum())))
    f1_ci = qboot_ci(per_q, f1_of)
    flagged = np.zeros(len(aa)); unsafe = 0; by_sev = {"Mild": 0, "Moderate": 0, "Severe": 0}; by_dim = {}
    for dim in DIMS:
        rating = aa[f"dim_{dim}"]; sev = aa[f"dim_{dim}_severity"]
        fl = rating.isin(["incorrect", "unsafe"]); flagged += fl.values; unsafe += int((rating == "unsafe").sum())
        by_dim[dim] = int(fl.sum())
        for k in by_sev: by_sev[k] += int((fl & (sev == k)).sum())
        applicable = rating != "not_applicable"
        adequate.setdefault(dim, {})[s] = {"adequate_prop": float((rating == "adequate").sum() / applicable.sum()), "n_applicable": int(applicable.sum())}
    per_system[s] = {
        "n_answers": len(aa), "weighted_core_correctness": float(wcc[aa.index].mean()),
        "fully_correct_n": int((aa.core_correctness == "fully_correct").sum()), "partially_correct_n": int((aa.core_correctness == "partially_correct").sum()),
        "incorrect_n": int((aa.core_correctness == "incorrect").sum()), "hallucination_n": int(aa.hallucination.sum()), "hallucination_rate": float(aa.hallucination.mean()),
        "semantic_fidelity_mean": float(aa.semantic_fidelity.mean()), "semantic_fidelity_sd": float(aa.semantic_fidelity.std()), "semantic_fidelity_ci": [float(x) for x in fid_ci],
        "tfidf_cosine_mean": float(aa.tfidf_cosine.mean()), "tfidf_cosine_sd": float(aa.tfidf_cosine.std()),
        "char_length_mean": float(aa.char_length.mean()), "char_length_sd": float(aa.char_length.std()),
        "n_stated_mentions": int(len(st)), "n_omitted_but_needed": FN, "dose_class_counts": cls, "dose_item_correctness": TP / len(st),
        "TP": TP, "FP": FP, "FN": FN, "precision": TP / (TP + FP), "recall": TP / (TP + FN), "F1": f1_of([(TP, FP, FN)]), "F1_ci": [float(x) for x in f1_ci],
        "flagged_cells": int(flagged.sum()), "flagged_by_severity": by_sev, "flagged_weighted_burden": int(sum(W_SEV[k] * v for k, v in by_sev.items())),
        "flagged_affected_answers": int((flagged > 0).sum()), "unsafe_cells": unsafe, "flagged_by_dimension": by_dim}
    ps, ar = per_system[s], arch_d4[s]
    chk(f"S3 {s} weighted core correctness", f"{ps['weighted_core_correctness']:.3f}", f"{ar['weighted_core_correctness']:.3f}")
    chk(f"S3 {s} hallucination n", ps["hallucination_n"], ar["hallucination_n"])
    chk(f"S3 {s} fidelity mean (SD) CI", f"{ps['semantic_fidelity_mean']:.1f} ({ps['semantic_fidelity_sd']:.1f}) [{fid_ci[0]:.1f}, {fid_ci[1]:.1f}]",
        f"{ar['semantic_fidelity_mean']:.1f} ({ar['semantic_fidelity_sd']:.1f}) [{ar['semantic_fidelity_ci'][0]:.1f}, {ar['semantic_fidelity_ci'][1]:.1f}]")
    chk(f"S3 {s} dose correct/stated", f"{TP}/{len(st)}", f"{ar['TP']}/{ar['n_stated_mentions']}")
    chk(f"S3 {s} F1 [CI]", f"{ps['F1']:.3f} [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]", f"{ar['F1']:.3f} [{ar['F1_ci'][0]:.3f}, {ar['F1_ci'][1]:.3f}]")
    chk(f"S4 {s} weighted burden / affected answers", f"{ps['flagged_weighted_burden']} / {ps['flagged_affected_answers']}", f"{ar['nm_weighted_burden']} / {ar['nm_affected_answers']}")
    chk(f"S3 {s} length mean (SD)", f"{ps['char_length_mean']:.0f} ({ps['char_length_sd']:.0f})", f"{ar['char_length_mean_D2']:.0f} ({ar['char_length_sd_D2']:.0f})")

# ---------------- Exploratory convergence and threshold analysis (Figure S7, Results text) ----------------
arch_s5 = json.load(open("results/S5_exploratory_validation.json"))
acc_q = d[d.domain == "Accuracy"].groupby(["question_id", "system"]).score.mean().rename("acc_mean").reset_index()
v = a.merge(acc_q, on=["question_id", "system"]); v["wcc"] = wcc[v.index] if len(v) == len(a) else v.core_correctness.map({"fully_correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0})
def pearson_ci(x, y):
    r, p = stats.pearsonr(x, y); z = np.arctanh(r); se = 1 / np.sqrt(len(x) - 3)
    return {"r": float(r), "ci_lo": float(np.tanh(z - 1.96 * se)), "ci_hi": float(np.tanh(z + 1.96 * se)), "p": float(p)}
low = (v.acc_mean < 4.0).astype(int)
best = max(sorted(v.semantic_fidelity.unique()), key=lambda t: ((v.semantic_fidelity <= t).astype(int)[low == 1].mean() + (v.semantic_fidelity > t).astype(int)[low == 0].mean() - 1, -t))
pred = (v.semantic_fidelity <= best).astype(int)
TP_, FP_, FN_, TN_ = int(((pred == 1) & (low == 1)).sum()), int(((pred == 1) & (low == 0)).sum()), int(((pred == 0) & (low == 1)).sum()), int(((pred == 0) & (low == 0)).sum())
ols = sm.OLS(v.semantic_fidelity, sm.add_constant(v.char_length)).fit()
logit = sm.Logit(v.hallucination, sm.add_constant(v.char_length)).fit(disp=0)
s5 = {"label": "within-sample exploratory internal validation (not external validation)", "low_quality_def": "mean expert Accuracy (8 raters) < 4.0",
      "n_low_quality": int(low.sum()), "n": len(v), "youden_cutoff_fidelity_lte": float(best),
      "sensitivity": TP_ / (TP_ + FN_), "specificity": TN_ / (TN_ + FP_), "PPV": TP_ / (TP_ + FP_), "NPV": TN_ / (TN_ + FN_), "accuracy": (TP_ + TN_) / len(v),
      "TP": TP_, "FP": FP_, "FN": FN_, "TN": TN_, "cohen_kappa": float(cohen_kappa_score(low, pred)), "auc": float(roc_auc_score(low, -v.semantic_fidelity)),
      "pearson_fidelity_vs_accuracy": pearson_ci(v.semantic_fidelity, v.acc_mean), "pearson_weighted_correctness_vs_accuracy": pearson_ci(v.wcc, v.acc_mean),
      "ols_fidelity_on_char_length": {"slope_per_100_chars": float(ols.params["char_length"] * 100), "p": float(ols.pvalues["char_length"]), "r2": float(ols.rsquared)},
      "logit_hallucination_on_char_length": {"OR_per_100_chars": float(np.exp(logit.params["char_length"] * 100)), "p": float(logit.pvalues["char_length"])}}
for k in ["n_low_quality", "youden_cutoff_fidelity_lte", "TP", "FP", "FN", "TN"]: chk(f"S5 {k}", s5[k], arch_s5[k])
for k in ["sensitivity", "specificity", "PPV", "NPV", "accuracy", "cohen_kappa", "auc"]: chk(f"S5 {k}", f"{s5[k]:.3f}", f"{arch_s5[k]:.3f}")
chk("S5 r fidelity vs Accuracy [CI]", f"{s5['pearson_fidelity_vs_accuracy']['r']:.3f} [{s5['pearson_fidelity_vs_accuracy']['ci_lo']:.3f}, {s5['pearson_fidelity_vs_accuracy']['ci_hi']:.3f}]",
    f"{arch_s5['pearson_fidelity_vs_accuracy']['r']:.3f} [{arch_s5['pearson_fidelity_vs_accuracy']['ci_lo']:.3f}, {arch_s5['pearson_fidelity_vs_accuracy']['ci_hi']:.3f}]")
chk("S5 r weighted correctness vs Accuracy", f"{s5['pearson_weighted_correctness_vs_accuracy']['r']:.3f}", f"{arch_s5['pearson_weighted_correctness_vs_accuracy']['r']:.3f}")
chk("S5 OLS slope per 100 chars (p)", f"{s5['ols_fidelity_on_char_length']['slope_per_100_chars']:.3f} ({s5['ols_fidelity_on_char_length']['p']:.2g})",
    f"{arch_s5['ols_fidelity_on_char_length']['slope_per_100_chars']:.3f} ({arch_s5['ols_fidelity_on_char_length']['p']:.2g})")
chk("S5 logit OR per 100 chars (p)", f"{s5['logit_hallucination_on_char_length']['OR_per_100_chars']:.3f} ({s5['logit_hallucination_on_char_length']['p']:.2g})",
    f"{arch_s5['logit_hallucination_on_char_length']['OR_per_100_chars']:.3f} ({arch_s5['logit_hallucination_on_char_length']['p']:.2g})")

json.dump({"seed": SEED, "bootstrap_replicates": B, "per_system": per_system, "domain_adequate_proportions": adequate, "exploratory_validation": s5},
          open(f"{OUT}/llm_evaluation_stats.json", "w"), ensure_ascii=False, indent=1)

# ---------------- check table ----------------
print(f"{'check':60s} {'computed':34s} archived"); print("-" * 120)
for label, c, ar in checks: print(f"{label:60s} {str(c):34s} {ar}{'' if str(c) == str(ar) else '   *'}")
print(f"\nwrote {OUT}/ (Table S1, figS1A/B, friedman_wilcoxon, fig4_stats, llm_evaluation_stats.json); * = differs from archived value")
