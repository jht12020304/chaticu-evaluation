# ChatICU evaluation: data and analysis code

Data and code accompanying the manuscript *Evaluating Clinical Reliability Beyond Citations: A Source-Blinded Expert Comparison of Generative AI Systems for ICU Pharmacotherapy* (submitted to npj Digital Medicine, 2026).

Three generative AI systems (ChatICU, ChatGPT and OpenEvidence) answered 68 ICU pharmacotherapy questions (204 responses). Eight interprofessional experts rated every source-blinded response on five clinical quality domains, rater confidence and a binary safety judgement (11,424 ratings). A blinded LLM-based evaluation was performed as an exploratory secondary analysis.

## Contents

| Path | Description |
|---|---|
| `data/expert_ratings_long.csv` | Complete de-identified expert rating dataset, one row per rating (11,424 rows). Columns: `question_id` (68 identifiers: 1–37, C1–C16 and E1–E15; the same identifiers are used in the LLM evaluation files), `system`, `rater` (R1–R8), `rater_profession`, `domain` (Accuracy, Relevance, Clarity, Trust, Comparison, Confidence, Safety), `score` (1–5 for ordinal domains; Safety 1 = no potential safety risk identified, 0 = at least one potential risk). |
| `data/llm_evaluation_per_answer.csv` | Answer-level outputs of the blinded LLM-based evaluation (204 rows): core correctness, content hallucination flag, semantic fidelity (0–100), TF-IDF cosine similarity, response length, counts of in-text citation markers, reference-list entries and DOI/URL entries (Figure 4), seven clinical-domain judgements with severity, and dose-mention counts. Free-text rationales are not included. |
| `data/llm_evaluation_per_mention.csv` | Dose/infusion-rate mention-level classification (620 rows): `mention_type` = stated or omitted, `drug` (drug or drug-class label assigned by the LLM judge), `class` = correct, imprecise, incorrect, unsafe or omitted_but_needed, with severity label. |
| `code/R/` | R scripts for the crossed cumulative-link mixed models (`01_primary_ordinal_clmm.R`), proportional-odds diagnostics (`02_po_diagnostic.R`), the Safety crossed mixed-effects logistic model and binomial GEE (`03_safety_crossed_logistic.R`), the ordinal GEE robustness analysis (`04_ordinal_gee.R`, multgee) and the Bayesian sensitivity analysis (`run_bayes.R`, brms). `make_input.R` converts the released rating file into the layout the scripts expect. |
| `code/python/` | Python scripts for the descriptive statistics with multiway cluster bootstrap, composite score, Friedman and Holm-adjusted Wilcoxon tests, Safety proportions with Wilson CIs, citation counts and all summary statistics of the LLM-based evaluation including the exploratory convergence and threshold analysis (`reported_stats.py`), inter-rater reliability (`irr_icc_gwet.py`), Holm adjustment of the CLMM contrasts within the 10-contrast quality family and the 2-contrast Confidence family (`holm_families.py`), the LLM-based evaluation (`d4_coding.py`), TF-IDF similarity (`d4_tfidf.py`) and figures (`make_main_figs.py`, `make_supp_figs.py`). |
| `protocol/` | Coding protocol and verbatim prompts of the LLM-based evaluation. The protocol refers to internal working files (reference standard, blinding key, raw model responses) that are not released. |
| `results/` | Model outputs underlying the tables and figures of the manuscript (see mapping below). |

### Mapping of `results/` files to manuscript items

| File | Manuscript item |
|---|---|
| `TableS1_domain_means_CI.csv` | Supplementary Table S1 (means with multiway cluster bootstrap 95% CIs) |
| `table2_crossed_clmm.csv`, `clmm_results.csv`, `holm_families.csv` | Figure 3, Supplementary Table S2 (crossed CLMM, Holm-adjusted p) |
| `table_po_diagnostic.csv` | Proportional-odds tests (Results text) |
| `table5_safety_crossed.csv` | Safety mixed model and binomial GEE (Results text) |
| `gee_results.csv`, `friedman_wilcoxon.csv` | Supplementary Figure S2 (ordinal GEE) and the Friedman / Wilcoxon robustness analyses (Results text) |
| `fig4_stats.csv` | Figure 4 (citation and reference counts, Accuracy and Safety summaries) |
| `TableS3_machine_text_analysis.csv`, `D4_summary_stats.json` | Supplementary Table S3, Supplementary Figures S3–S5 (LLM-based evaluation) |
| `S5_exploratory_validation.json` | Supplementary Figure S7 and the exploratory threshold analysis (Results text) |
| `figS4_irr.csv` | Supplementary Figure S6 (ICC and Gwet's AC) |
| `bayes_results.csv`, `bayes_diagnostics.csv`, `bayes_session_info.txt` | Supplementary Figure S8 (Bayesian sensitivity analysis) |

## Not included

The 68 evaluation questions, the reference standard and the 204 system responses are not released (the question set and reference standard are intended for reuse in future comparative evaluations, and the responses of the commercial systems are subject to the platforms' terms of use); they are available from the corresponding author on reasonable request. The ChatICU pipeline code and knowledge base are not included (see the Code availability statement of the manuscript).

## Environment

- Primary and robustness models: R 4.5.2 with `ordinal` 2025.12.29, `lme4`, `geepack`, `multgee` (`emmeans` is loaded but not required).
- Bayesian models: R 4.5.3 with `brms` 2.23.0, `rstan` 2.32.7 and `dplyr` (`results/bayes_session_info.txt`).
- Python 3 with `pandas`, `numpy`, `scipy`, `statsmodels` 0.14, `pingouin` 0.6, `scikit-learn`, `matplotlib`; `jieba` for `d4_tfidf.py`.
- The LLM-based evaluation used a commercial large language model accessed through a hosted API (`d4_coding.py`); the sampling temperature could not be fixed, so re-running it will not reproduce the archived outputs exactly. The archived per-answer and per-mention outputs are provided in `data/`.

## Running the analyses

```
cd <repository root>
cd code && Rscript R/make_input.R      # writes code/long_with_global_qid.csv
Rscript R/01_primary_ordinal_clmm.R && Rscript R/02_po_diagnostic.R && Rscript R/03_safety_crossed_logistic.R && Rscript R/04_ordinal_gee.R
python3 python/holm_families.py         # Holm adjustment of code/results_R/table2_crossed_clmm.csv
cd ..
python3 code/python/reported_stats.py   # Table S1, Figure S1, Friedman/Wilcoxon, Figure 4 statistics, Table S3, Figures S3-S5, S7
python3 code/python/irr_icc_gwet.py     # ICC and Gwet's AC (Figure S6)
Rscript code/R/run_bayes.R              # Bayesian sensitivity analysis (slow); writes results/bayes_*.csv
```

R outputs are written to `code/results_R/`, Python outputs to `code/results_py/`; `reported_stats.py` and `irr_icc_gwet.py` print a check table against the archived `results/` files. ChatICU is the reference category for all contrasts; odds ratios < 1 indicate lower odds of a higher rating (or of a no-risk judgement) than ChatICU.

## Reproduction notes

- All six ordinal domains are reported with the random-slope model (`spec = slope` in `results/clmm_results.csv`, `random_slope = TRUE` in `results/table2_crossed_clmm.csv`), as in the manuscript. With R 4.5.2 and `ordinal` 2025.12.29 the scripts reproduce every odds ratio and 95% CI to the precision reported in the manuscript; p-values and CI bounds may differ in the third significant digit between runs (for example 0.613 versus 0.614) because the released file uses recoded rater labels (a different factor order) and the optimiser stops within tolerance.
- `04_ordinal_gee.R`, `reported_stats.py` and `irr_icc_gwet.py` reproduce every archived point estimate exactly (GEE coefficients and standard errors, Friedman statistics, Wilson CIs, ICC, Gwet's AC, all Table S3 values, flagged burden, correlations and the threshold analysis). Bootstrap CIs (Table S1, composite score, semantic fidelity, F1, Gwet's AC) are seeded but not bit-identical to the archived values because the resampling implementation differs; they agree to within about 0.02.
- `d4_coding.py` and `d4_tfidf.py` require the question, reference-standard and response texts, which are not released; they are provided to document the prompts, coding logic and similarity computation. `make_main_figs.py` and `make_supp_figs.py` document the plotting code and read intermediate tables produced by the pipeline; they are not runnable from this repository as-is.

## Licence

Code: MIT License (`LICENSE`). Data (`data/`, `results/`): Creative Commons Attribution 4.0 International (CC BY 4.0).

## Contact

Ching-Po Lin, Institute of Neuroscience, National Yang Ming Chiao Tung University (chingpolin@gmail.com).
