import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
apply_figure_style(sizes=(8, 7, 6))
COL = {"ChatICU": "#374E55", "ChatGPT": "#DF8F44", "OpenEvidence": "#00A1D5"}
SYS = ["ChatICU", "ChatGPT", "OpenEvidence"]
DOMS = ["Accuracy", "Relevance", "Clarity", "Trust", "Comparison", "Confidence"]
def save(fig, name):
    fig.savefig(f"figs/{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"figs/{name}.pdf", bbox_inches="tight")

# ---------------- Fig S1: composite + correlation ----------------
comp = pd.read_csv("data/figS1A_composite.csv").set_index("model").loc[SYS]
corr = pd.read_csv("data/figS1B_corr.csv", index_col=0)
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1, 1.25]))
for i, s in enumerate(SYS):
    a.errorbar(i, comp.loc[s, "composite"], yerr=[[comp.loc[s, "composite"] - comp.loc[s, "ci_lo"]], [comp.loc[s, "ci_hi"] - comp.loc[s, "composite"]]],
               fmt="o", color=COL[s], ms=5, capsize=3, lw=1.1)
    a.text(i + 0.12, comp.loc[s, "composite"], f"{comp.loc[s,'composite']:.2f}", va="center", fontsize=6, color=COL[s])
a.set_xticks(range(3)); a.set_xticklabels(SYS); a.set_xlim(-0.5, 2.7)
for lbl, s in zip(a.get_xticklabels(), SYS): lbl.set_color(COL[s])
a.set_ylim(3.0, 4.7); a.set_ylabel("Five-domain composite score (1–5)\nmean; multiway cluster-bootstrap 95% CI")
a.set_title("Composite quality score", loc="left"); set_frame(a, "open")
im = b.imshow(corr.values, cmap="Blues", vmin=0, vmax=1)
b.set_xticks(range(5)); b.set_yticks(range(5)); b.set_xticklabels(corr.columns, rotation=30, ha="right"); b.set_yticklabels(corr.index)
for i in range(5):
    for j in range(5):
        v = corr.values[i, j]; b.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.6 else "#222222")
b.set_title("Between-domain Pearson correlation", loc="left")
cb = fig.colorbar(im, ax=b, fraction=0.046, pad=0.03); cb.set_label("Pearson r", fontsize=7); cb.ax.tick_params(labelsize=6)
for ax, L in zip((a, b), "ab"): panel_letter(ax, L)
fig.tight_layout(); save(fig, "FigS1_composite_correlation"); plt.close(fig)

# ---------------- Fig S4: inter-rater reliability ----------------
irr = pd.read_csv("data/figS4_irr.csv")
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ys = np.arange(len(irr))[::-1]
for y, (_, r) in zip(ys, irr.iterrows()):
    ax.plot([r.ICC_lo, r.ICC_hi], [y + 0.12, y + 0.12], color="#9e9e9e", lw=2.2, solid_capstyle="butt", zorder=1)
    ax.plot(r.ICC2k, y + 0.12, "o", color="#222222", ms=5, zorder=2)
    ax.plot([r.gwet_lo, r.gwet_hi], [y - 0.12, y - 0.12], color="#c9c9c9", lw=1.2, zorder=1)
    ax.plot(r.gwet, y - 0.12, "s", color="#7a7a7a", ms=4.5, zorder=2, mfc="white", mew=1.2)
ax.set_yticks(ys); ax.set_yticklabels(irr.domain)
ax.axhline(0.5, color=META_GREY, lw=0.6, ls=":"); ax.axhline(1.5, color=META_GREY, lw=0.6, ls=":")
ax.set_yticklabels([("Confidence\n(rater confidence)" if d == "Confidence" else d) for d in irr.domain])
ax.set_xlim(0, 1); ax.set_xlabel("Agreement coefficient (0–1)")
h = [mpl.lines.Line2D([], [], marker="o", ls="", color="#222222", label="ICC(2,k), two-way random, absolute agreement; grey bar = 95% CI"),
     mpl.lines.Line2D([], [], marker="s", ls="", color="#7a7a7a", mfc="white", label="Gwet AC2 (quadratic weights); Safety: Gwet AC1; light bar = 95% CI")]
ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, handlelength=1)
ax.text(0.01, 0.35, "n = 204 answers × 8 raters per domain", fontsize=6, color=META_GREY, va="top")
set_frame(ax, "open"); fig.tight_layout(); save(fig, "FigS4_interrater_reliability"); plt.close(fig)

# ---------------- Fig S6: ordinal GEE forest ----------------
g = pd.read_csv("data/gee_results.csv")
fig, ax = plt.subplots(figsize=(5.2, 3.7))
y = 0; ypos = []; ylab = []
for dm in DOMS:
    if dm == "Confidence":
        y -= 0.9; ax.axhline(y + 0.8, color=META_GREY, lw=0.6, ls=":")
        ax.text(1.35, y + 0.72, "Rater confidence: separate outcome", fontsize=5.8, color=META_GREY, va="top", ha="left")
    for k, c in enumerate(["ChatGPT", "OpenEvidence"]):
        r = g[(g.domain == dm) & (g.contrast == c)].iloc[0]
        ax.errorbar(r.OR, y, xerr=[[r.OR - r.ci_lo], [r.ci_hi - r.OR]], fmt="o", color=COL[c], ms=4.5, capsize=2, lw=1.1)
        ax.plot(r.OR_clmm, y, marker="|", color=COL[c], ms=7, mew=1.2, ls="")
        ax.text(1.35, y, f"{r.OR:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]", va="center", fontsize=6, color=COL[c])
        if k == 0: ypos.append(y - 0.5); ylab.append(dm)
        y -= 1
ax.axvline(1, color="#555555", lw=0.8, ls="--"); ax.set_xscale("log"); ax.set_xlim(0.06, 4)
ax.set_xticks([0.1, 0.3, 1, 3]); ax.set_xticklabels(["0.1", "0.3", "1", "3"])
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_yticks(ypos); ax.set_yticklabels(ylab); ax.set_ylim(y + 0.3, 1.9)
ax.set_xlabel("Cumulative odds ratio vs ChatICU (log scale)")
ax.text(0.9, 0.7, "← favours ChatICU", fontsize=6, color=META_GREY, ha="right", va="bottom")
ax.text(1.1, 0.7, "favours comparator →", fontsize=6, color=META_GREY, ha="left", va="bottom")
ax.text(1.35, 1.3, "GEE OR [95% CI]", fontsize=6, color=META_GREY, va="bottom")
h = [mpl.lines.Line2D([], [], marker="o", ls="", color=COL[c], label=f"{c} vs ChatICU") for c in ["ChatGPT", "OpenEvidence"]]
h.append(mpl.lines.Line2D([], [], marker="|", ls="", color="#555555", mew=1.2, label="crossed CLMM estimate (Fig. 3)"))
ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False, columnspacing=1.2)
set_frame(ax, "open"); fig.tight_layout(); save(fig, "FigS6_ordinal_GEE_forest"); plt.close(fig)

# ---------------- Fig 1: study design schematic ----------------
fig, ax = plt.subplots(figsize=(7.2, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
def box(x, y, w, h, text, fc="#f2f2f2", ec="#555555", fs=6, color="#222222", weight_first=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=color, linespacing=1.35, wrap=True)
def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color="#555555", lw=0.8, shrinkA=0, shrinkB=0))
top = 5.45; bh = 1.3
boxes = [(0.15, 1.75, "68 clinical\nquestions\n37 items + 2 cases\n(16 + 15)"),
         (2.15, 1.75, "3 systems\nChatICU\nChatGPT\nOpenEvidence"),
         (4.15, 1.5, "204 answers\n(68 × 3)"),
         (5.85, 2.0, "8 blinded raters\n2 physicians\n4 pharmacists\n1 nurse\n1 nurse practitioner"),
         (8.05, 1.85, "5 quality domains\n+ rater confidence\n(Likert 1–5)\n+ Safety (0/1)\n11,424 ratings")]
for x, w, t in boxes: box(x, top, w, bh, t)
for (x0, w0, _), (x1, _, _) in zip(boxes[:-1], boxes[1:]): arrow(x0 + w0 + 0.02, top + bh/2, x1 - 0.02, top + bh/2)
# bus line from the ratings box down to three tracks
xs = [1.65, 5.0, 8.35]; ybus = 4.7
ax.plot([8.05 + 1.85/2, 8.05 + 1.85/2], [top - 0.02, ybus], color="#555555", lw=0.8)
ax.plot([xs[0], 8.05 + 1.85/2], [ybus, ybus], color="#555555", lw=0.8)
for xt in xs: arrow(xt, ybus, xt, 4.12)
tracks = [(0.15, "Primary analysis\nHuman quality ratings\n8,160 ordinal ratings\n(5 domains × 544 × 3)\n\nCrossed CLMM\nscore ~ system + (1|question)\n+ (1+system|rater)\n10 contrasts, Holm-adjusted;\nrater confidence reported\nseparately (2 contrasts)\nProportional-odds test →\ncommon OR read as\ndirectional summary", "#e8ecee", COL["ChatICU"]),
          (3.5, "Safety endpoint\n(independent)\n1,632 binary ratings\n\nCrossed logistic GLMM\nsafety ~ system\n+ (1|question) + (1|rater)\nNot part of the Holm family\nRisk-free rate, Wilson CI", "#f5f5f5", "#444444"),
          (6.85, "Secondary / exploratory\nCitation availability\nfrom answer text (D2):\nreference-list entries\nper answer (all 204 answers\ncarry a reference list)\n(descriptive, Fig. 4)\n\nBlinded automated text\nanalysis (D4): LLM judge,\npharmacist review pending\n(Table S3, Figs S2–S15)", "#f5f5f5", "#7a7a7a")]
for x, t, fc, ec in tracks: box(x, 0.7, 3.0, 3.4, t, fc=fc, ec=ec, fs=5.6)
ax.text(5, 0.25, "Sensitivity and reliability analyses: ordinal GEE (Fig. S6) · ICC(2,k) and Gwet AC1/AC2 (Fig. S4) · multiway cluster bootstrap CIs (Table S1, Fig. S1)",
        ha="center", fontsize=6, color=META_GREY)
save(fig, "Fig1_study_design"); plt.close(fig)
print("done")
