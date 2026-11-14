import csv, os
from statsmodels.stats.multitest import multipletests
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "results_R", "table2_crossed_clmm.csv")
OUT = os.path.join(os.path.dirname(HERE), "results_R", "holm_families.csv")

rows = list(csv.DictReader(open(SRC)))
fam = {"quality": [r for r in rows if r["domain"] != "Confidence"], "confidence": [r for r in rows if r["domain"] == "Confidence"]}
assert len(fam["quality"]) == 10 and len(fam["confidence"]) == 2
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["family", "domain", "contrast", "cumOR_vs_ChatICU", "p", "p_holm"])
    for name, rs in fam.items():
        for r, a in zip(rs, multipletests([float(r["p"]) for r in rs], method="holm")[1]):
            w.writerow([name, r["domain"], r["contrast"], r["cumOR_vs_ChatICU"], r["p"], f"{a:.3g}"])
print("wrote", OUT)
