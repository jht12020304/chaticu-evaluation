#!/usr/bin/env python3
"""Inter-rater reliability (Supplementary Figure S6): two-way random-effects, absolute-agreement, average-measures
ICC(2,k) with 95% CI (pingouin) and Gwet's AC1 (Safety) / quadratic-weighted AC2 (ordinal domains) with a
subject-level bootstrap 95% CI. Subjects = 204 responses (question x system), raters = 8.
Run from the repository root: python3 code/python/irr_icc_gwet.py  ->  code/results_py/figS4_irr.csv
"""
import os, numpy as np, pandas as pd, pingouin as pg
SEED, B = 20260902, 2000
DOMS = ["Accuracy", "Relevance", "Clarity", "Trust", "Comparison", "Confidence", "Safety"]
OUT = "code/results_py"; os.makedirs(OUT, exist_ok=True)
df = pd.read_csv("data/expert_ratings_long.csv", dtype={"question_id": str})
df["target"] = df.question_id + "|" + df.system

def gwet(matrix, weighted=False, cats=None):
    """Gwet's AC1 (unweighted) or AC2 with quadratic weights (Gwet 2014); rows = subjects, columns = raters.
    cats fixes the category set (needed for bootstrap resamples in which a rare category is absent)."""
    cats = cats or sorted({v for r in matrix for v in r if not np.isnan(v)}); q = len(cats); idx = {c: i for i, c in enumerate(cats)}
    W = np.eye(q) if not weighted else np.array([[1 - ((i - j) ** 2) / ((q - 1) ** 2) for j in range(q)] for i in range(q)])
    pa_num, nprime, pik = 0.0, 0, np.zeros(q)
    for r in matrix:
        vals = [v for v in r if not np.isnan(v)]; ni = len(vals)
        if ni < 2: continue
        nprime += 1; c = np.zeros(q)
        for v in vals: c[idx[v]] += 1
        pa_num += sum(W[k, l] * c[k] * (c[l] - (1 if k == l else 0)) for k in range(q) for l in range(q)) / (ni * (ni - 1))
        pik += c / ni
    pa = pa_num / nprime; pik = pik / nprime
    pe = (W.sum() / (q * (q - 1))) * sum(pik[k] * (1 - pik[k]) for k in range(q))
    return (pa - pe) / (1 - pe)
assert abs(gwet([[1, 1, 1], [2, 2, 2], [1, 1, 1]]) - 1.0) < 1e-9

rows = []
for dom in DOMS:
    sub = df[df.domain == dom]
    icc = pg.intraclass_corr(data=sub, targets="target", raters="rater", ratings="score").set_index("Type").loc["ICC2k"]
    wide = sub.pivot(index="target", columns="rater", values="score"); M = wide.values.astype(float)
    weighted = dom != "Safety"; cats = sorted(set(M.ravel().tolist())); point = gwet(M.tolist(), weighted, cats)
    rng = np.random.default_rng(SEED)
    boots = [gwet(M[rng.integers(0, len(M), len(M))].tolist(), weighted, cats) for _ in range(B)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rows.append({"domain": dom, "ICC2k": round(icc["ICC"], 3), "ICC_lo": round(icc["CI95"][0], 3), "ICC_hi": round(icc["CI95"][1], 3),
                 "gwet_type": "AC2 (quadratic)" if weighted else "AC1", "gwet": round(point, 3), "gwet_lo": round(lo, 3), "gwet_hi": round(hi, 3),
                 "n_subjects": len(M), "n_raters": M.shape[1]})
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/figS4_irr.csv", index=False)
arch = pd.read_csv("results/figS4_irr.csv").set_index("domain")
for _, r in res.iterrows():
    a = arch.loc[r.domain]
    print(f"{r.domain:11s} ICC {r.ICC2k:.3f} [{r.ICC_lo:.2f}, {r.ICC_hi:.2f}] (archived {a.ICC2k:.3f} [{a.ICC_lo:.2f}, {a.ICC_hi:.2f}])   "
          f"{r.gwet_type} {r.gwet:.3f} [{r.gwet_lo:.3f}, {r.gwet_hi:.3f}] (archived {a.gwet:.3f} [{a.gwet_lo:.3f}, {a.gwet_hi:.3f}])")
