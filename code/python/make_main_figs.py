import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, os
os.makedirs("figs", exist_ok=True)
apply_figure_style(sizes=(8, 7, 6))
COL = {"ChatICU": "#374E55", "ChatGPT": "#DF8F44", "OpenEvidence": "#00A1D5"}
SYS = ["ChatICU", "ChatGPT", "OpenEvidence"]
DOMS = ["Accuracy", "Relevance", "Clarity", "Trust", "Comparison", "Confidence"]
LIKERT = ["#d9d9d9", "#bdbdbd", "#969696", "#636363", "#252525"]  # 1..5 single-hue ordinal ramp

def save(fig, name):
    fig.savefig(f"figs/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"figs/{name}.pdf", bbox_inches="tight")

def bbox_check(fig):
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
    n = sum(1 for i, (a, ba) in enumerate(texts) for b, bb in texts[i+1:] if ba.overlaps(bb))
    return n

# ---------------- Figure 2: 100% stacked Likert bars ----------------
dist = pd.read_csv("data/fig2_score_distribution.csv")
fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.6), sharex=True)
for ax, dm in zip(axes.flat, DOMS):
    sub = dist[dist.domain == dm].set_index("model").loc[SYS[::-1]]
    left = np.zeros(3)
    for k, s in enumerate(["1", "2", "3", "4", "5"]):
        vals = sub[s].values
        ax.barh(range(3), vals, left=left, color=LIKERT[k], edgecolor="white", linewidth=0.4, height=0.7)
        for i, v in enumerate(vals):
            if v >= 8:
                ax.text(left[i] + v/2, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if k >= 3 else "#252525")
        left += vals
    ax.set_yticks(range(3)); ax.set_yticklabels([f"{s}" for s in SYS[::-1]])
    for lbl, s in zip(ax.get_yticklabels(), SYS[::-1]): lbl.set_color(COL[s])
    ax.set_xlim(0, 100); ax.set_title("Confidence (rater confidence)" if dm == "Confidence" else dm, loc="left")
    ax.set_xticks([0, 25, 50, 75, 100])
    set_frame(ax, "open")
for ax in axes[1]: ax.set_xlabel("Ratings (%)")
handles = [mpl.patches.Patch(color=LIKERT[k], label=f"{k+1}") for k in range(5)]
fig.legend(handles=handles, title="Score (1 = very poor, 5 = very good)", ncol=5, loc="lower center",
           bbox_to_anchor=(0.5, -0.06), frameon=False, handlelength=1.2, columnspacing=1.0)
fig.text(0.99, 0.985, "n = 544 ratings per bar (68 questions × 8 raters)", ha="right", va="top", fontsize=6, color=META_GREY)
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
save(fig, "Fig2_score_distribution"); print("Fig2 overlaps:", bbox_check(fig)); plt.close(fig)

# ---------------- Figure 3: CLMM forest ----------------
cl = pd.read_csv("data/clmm_results.csv"); sf = pd.read_csv("data/safety_glmm_results.csv")
rows = []
for dm in DOMS + ["Safety"]:
    src = cl if dm != "Safety" else sf
    for c in ["ChatGPT", "OpenEvidence"]:
        r = src[(src.domain == dm) & (src.contrast == c)].iloc[0]
        rows.append(dict(domain=dm, contrast=c, OR=r.OR, lo=r.ci_lo, hi=r.ci_hi,
                         p_adj=(r.p_holm if dm != "Safety" else r.p)))
fr = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ypos = []; ylab = []; y = 0
for dm in DOMS + ["Safety"]:
    if dm in ("Confidence", "Safety"):
        y -= 0.9; ax.axhline(y + 0.8, color=META_GREY, lw=0.6, ls=":")
        lbl = ("Rater confidence: separate outcome (Holm within 2 contrasts)" if dm == "Confidence"
               else "Safety: independent endpoint (logistic GLMM, unadjusted p)")
        ax.text(1.35, y + 0.72, lbl, fontsize=5.8, color=META_GREY, va="top", ha="left")
    for k, c in enumerate(["ChatGPT", "OpenEvidence"]):
        r = fr[(fr.domain == dm) & (fr.contrast == c)].iloc[0]
        ax.errorbar(r.OR, y, xerr=[[r.OR - r.lo], [r.hi - r.OR]], fmt="o", color=COL[c], ms=4.5, capsize=2, lw=1.1)
        star = "***" if r.p_adj < 0.001 else "**" if r.p_adj < 0.01 else "*" if r.p_adj < 0.05 else "ns"
        ax.text(1.35, y, f"{r.OR:.2f} [{r.lo:.2f}, {r.hi:.2f}] {star}", va="center", ha="left", fontsize=6, color=COL[c])
        if k == 0: ypos.append(y - 0.5); ylab.append(dm)
        y -= 1
ax.axvline(1, color="#555555", lw=0.8, ls="--")
ax.set_xscale("log"); ax.set_xlim(0.03, 4.0)
ax.set_xticks([0.03, 0.1, 0.3, 1, 3]); ax.set_xticklabels(["0.03", "0.1", "0.3", "1", "3"])
ax.set_yticks(ypos); ax.set_yticklabels(ylab)
ax.set_xlabel("Odds ratio vs ChatICU (log scale)")
ax.text(0.9, ypos[0] + 1.2, "← favours ChatICU", fontsize=6, color=META_GREY, ha="right", va="bottom")
ax.text(1.1, ypos[0] + 1.2, "favours comparator →", fontsize=6, color=META_GREY, ha="left", va="bottom")
ax.set_ylim(y + 0.3, ypos[0] + 2.5)
ax.text(1.35, ypos[0] + 1.85, "OR [95% CI]; Holm-adjusted within the 10 quality contrasts", fontsize=6, color=META_GREY, ha="left", va="bottom")
handles = [mpl.lines.Line2D([], [], marker="o", ls="", color=COL[c], label=f"{c} vs ChatICU") for c in ["ChatGPT", "OpenEvidence"]]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, frameon=False)
set_frame(ax, "open")
fig.tight_layout()
save(fig, "Fig3_clmm_forest"); print("Fig3 overlaps:", bbox_check(fig)); plt.close(fig)

# ---------------- Figure 4: citation vs accuracy vs safety ----------------
f4 = pd.read_csv("data/fig4_stats.csv", index_col=0).loc[SYS]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
x = np.arange(3); ax = axes[0]
for i, s in enumerate(SYS):
    r = f4.loc[s]
    ax.bar(i, r.mean_references, width=0.6, color=COL[s])
    ax.text(i, r.mean_references + 0.15, f"{r.mean_references:.1f}", ha="center", va="bottom", fontsize=6)
    ax.text(i, 0.25, f"{int(r.n_with_reference_list)}/68", ha="center", va="bottom", fontsize=6, color="white")
ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 9.2); ax.set_ylabel("Reference-list entries per answer\n(mean; answers with a list / 68)")
ax.set_title("Citation availability", loc="left")
ax = axes[1]
for i, s in enumerate(SYS):
    ax.errorbar(i, f4.loc[s, "acc_mean"], yerr=[[f4.loc[s, "acc_mean"] - f4.loc[s, "acc_ci_lo"]], [f4.loc[s, "acc_ci_hi"] - f4.loc[s, "acc_mean"]]],
                fmt="o", color=COL[s], capsize=3, ms=5, lw=1.1)
ax.set_ylim(3.0, 4.9); ax.set_ylabel("Human-rated Accuracy (1–5)\nmean, 95% CI")
ax.set_title("Human accuracy rating", loc="left")
ax = axes[2]
for i, s in enumerate(SYS):
    p = f4.loc[s, "saf_prop_safe"] * 100
    ax.bar(i, p, color=COL[s], width=0.6)
    ax.errorbar(i, p, yerr=[[p - f4.loc[s, "saf_wilson_lo"]*100], [f4.loc[s, "saf_wilson_hi"]*100 - p]], color="#222222", capsize=3, lw=1)
    ax.text(i, f4.loc[s, "saf_wilson_hi"]*100 + 1.5, f"{p:.1f}%", ha="center", va="bottom", fontsize=6)
ax.set_ylim(60, 102); ax.set_ylabel("Ratings with no safety risk\nidentified (%; Wilson 95% CI)")
ax.set_title("Risk-free rate", loc="left")
axes[1].set_ylabel("Human-rated Accuracy (1–5)\nmean; 95% CI")
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(SYS, rotation=0); ax.set_xlim(-0.6, 2.6)
    for lbl, s in zip(ax.get_xticklabels(), SYS): lbl.set_color(COL[s])
    set_frame(ax, "open")
for ax, L in zip(axes, "abc"): panel_letter(ax, L)
fig.text(0.5, -0.03, "Panels use independent scales; compare rankings only. n = 68 answers per system (a); 544 ratings per system (b, c).",
         ha="center", fontsize=6, color=META_GREY)
fig.tight_layout()
save(fig, "Fig4_citation_accuracy_safety"); print("Fig4 overlaps:", bbox_check(fig)); plt.close(fig)
