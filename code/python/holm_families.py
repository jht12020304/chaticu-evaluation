#!/usr/bin/env python3
"""Holm adjustment of the crossed-CLMM contrasts as reported in the manuscript:
the 10 clinical-quality contrasts (5 domains x 2 comparators) form one family and
the 2 Confidence contrasts a separate family. Input: code/results_R/table2_crossed_clmm.csv
(written by code/R/01_primary_ordinal_clmm.R). Output: code/results_R/holm_families.csv
"""
import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "results_R", "table2_crossed_clmm.csv")
OUT = os.path.join(os.path.dirname(HERE), "results_R", "holm_families.csv")

def holm(p):
    m = len(p); order = sorted(range(m), key=lambda i: p[i]); adj = [None] * m; running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj

rows = list(csv.DictReader(open(SRC)))
fam = {"quality": [r for r in rows if r["domain"] != "Confidence"], "confidence": [r for r in rows if r["domain"] == "Confidence"]}
assert len(fam["quality"]) == 10 and len(fam["confidence"]) == 2
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["family", "domain", "contrast", "cumOR_vs_ChatICU", "p", "p_holm"])
    for name, rs in fam.items():
        for r, a in zip(rs, holm([float(r["p"]) for r in rs])):
            w.writerow([name, r["domain"], r["contrast"], r["cumOR_vs_ChatICU"], r["p"], f"{a:.3g}"])
print("wrote", OUT)
