# ChatICU evaluation: data and analysis code

Data and code accompanying the manuscript *Evaluating Clinical Reliability Beyond Citations: A Source-Blinded Expert Comparison of Generative AI Systems for ICU Pharmacotherapy* (submitted to npj Digital Medicine, 2026).

Three generative AI systems (ChatICU, ChatGPT and OpenEvidence) answered 68 ICU pharmacotherapy questions (204 responses). Eight interprofessional experts rated every source-blinded response on five clinical quality domains, rater confidence and a binary safety judgement (11,424 ratings). A blinded LLM-based evaluation was performed as an exploratory secondary analysis.

## Contents

| Path | Description |
|---|---|
| `data/expert_ratings_long.csv` | Complete de-identified expert rating dataset, one row per rating (11,424 rows). Columns: `question_id` (1–68), `system`, `rater` (R1–R8), `rater_profession`, `domain` (Accuracy, Relevance, Clarity, Trust, Comparison, Confidence, Safety), `score` (1–5 for ordinal domains; Safety 1 = no potential safety risk identified, 0 = at least one potential risk). |
| `data/llm_evaluation_per_answer.csv` | Answer-level outputs of the blinded LLM-based evaluation (204 rows): core correctness, content hallucination flag, semantic faithfulness (0–100), TF-IDF cosine similarity, response length, seven clinical-domain judgements with severity, and dose-mention counts. Free-text rationales are not included. |
| `data/llm_evaluation_per_mention.csv` | Dose/infusion-rate mention-level classification (620 rows): `class` = correct, imprecise, incorrect, unsafe or omitted_but_needed, with severity label. |
| `code/R/` | R scripts for the crossed cumulative-link mixed models (`01_primary_ordinal_clmm.R`), proportional-odds diagnostics (`02_po_diagnostic.R`), the Safety crossed mixed-effects logistic model and binomial GEE (`03_safety_crossed_logistic.R`) and the Bayesian sensitivity analysis (`run_bayes.R`, brms). `make_input.R` converts the released rating file into the layout the scripts expect. |
| `code/python/` | Python scripts for the ordinal GEE, Friedman and Wilcoxon tests, multiway cluster bootstrap and descriptive statistics (`analysis_pipeline.py`), Holm adjustment of the CLMM contrasts within the 10-contrast quality family and the 2-contrast Confidence family (`holm_families.py`), inter-rater reliability (`patch_icc_gwet.py`), the LLM-based evaluation (`d4_coding.py`), TF-IDF similarity (`d4_tfidf.py`) and figures (`make_main_figs.py`, `make_supp_figs.py`). |
| `protocol/` | Coding protocol and verbatim prompts of the LLM-based evaluation. |
| `results/` | Model outputs underlying the tables and figures of the manuscript, for verification. |

## Not included

The 68 evaluation questions, the reference standard and the 204 system responses are not released (the question set and reference standard are intended for reuse in future comparative evaluations, and the responses of the commercial systems are subject to the platforms' terms of use); they are available from the corresponding author on reasonable request. The ChatICU pipeline code and knowledge base are not included (see the Code availability statement of the manuscript).

## Environment

- R 4.5.3 with `ordinal`, `lme4`, `geepack`, `emmeans`; `brms` 2.23.0 and `rstan` 2.32.7 for the Bayesian models (`results/bayes_session_info.txt`).
- Python 3 with `pandas`, `statsmodels` 0.14, `pingouin` 0.6, `scipy`, `scikit-learn`, `matplotlib`.
- The LLM-based evaluation used a commercial large language model accessed through a hosted API (`d4_coding.py`); the sampling temperature could not be fixed, so re-running it will not reproduce the archived outputs exactly. The archived per-answer and per-mention outputs are provided in `data/`.

## Running the primary models

```
cd <repository root>
cd code && Rscript R/make_input.R      # writes code/long_with_global_qid.csv
Rscript R/01_primary_ordinal_clmm.R && Rscript R/02_po_diagnostic.R && Rscript R/03_safety_crossed_logistic.R
```

Outputs are written to `code/results_R/`. ChatICU is the reference category for all contrasts; odds ratios < 1 indicate lower odds of a higher rating (or of a no-risk judgement) than ChatICU.

## Reproduction notes

- With R 4.5.2 and `ordinal` 2025.12.29, `01_primary_ordinal_clmm.R` run on `data/expert_ratings_long.csv` reproduces the manuscript estimates for Relevance, Clarity, Trust, Comparison and Confidence (random-slope models) and the proportional-odds tests; `03_safety_crossed_logistic.R` reproduces the Safety mixed-model and GEE odds ratios. For Accuracy, the manuscript reports the crossed random-intercept model (OR 0.338 and 0.118; `spec = intercept_only` in `results/clmm_results.csv`) because the random-slope model did not converge in the environment used for the primary analysis (R 4.5.3); in the environment above the random-slope model converges (OR 0.315 and 0.106; rater intercept–slope correlation 0.98), and the script therefore reports that model unless it fails.
- `analysis_pipeline.py` and `patch_icc_gwet.py` were run on the original rating workbook (columns `expert, question, model, domain, score`); `data/expert_ratings_long.csv` contains the same 11,424 records with rater codes recoded (`rater` → `expert`, `question_id` → `question`, `system` → `model`).
- `d4_coding.py` calls the LLM through a host API and is provided to document the prompts and coding logic; it is not runnable as a standalone script.

## Licence

Code: MIT License (`LICENSE`). Data (`data/`, `results/`): Creative Commons Attribution 4.0 International (CC BY 4.0).

## Contact

Ching-Po Lin, Institute of Neuroscience, National Yang Ming Chiao Tung University (chingpolin@gmail.com).
