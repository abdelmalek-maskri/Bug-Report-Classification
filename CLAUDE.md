# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

University coursework (ISE, Option 1: Tool Building Project — Lab 1 bug report classification). It is a small set of standalone research scripts, not a package: no tests, no build, no `setup.py`, no `requirements.txt` (dependencies are documented in `requirements.pdf`). Each script is run top-to-bottom as a program; there is no shared library module and duplication between scripts is intentional (each is a self-contained experiment for the report).

## Commands

```bash
pip install pandas numpy nltk scikit-learn xgboost gensim streamlit matplotlib joblib

python main.py                              # primary tool: XGBoost + TF-IDF, 10 runs
python comparatorModels/main_with_Glove.py  # baseline: XGBoost + GloVe
python comparatorModels/word2vec+NB.py      # baseline: GaussianNB + Word2Vec
streamlit run app.py                        # Streamlit GUI (see caveats below)
```

Comparator scripts resolve `datasets/{project}.csv` relative to the CWD, so run them from the repo root, not from `comparatorModels/`.

`main_with_Glove.py` additionally needs `glove.6B.100d.txt` in the repo root (not committed — download from https://nlp.stanford.edu/projects/glove/).

## Experiment structure

All three scripts share the same shape, and changes to one usually need mirroring in the others:

1. Dataset selected by editing a module-level `project` string (`pytorch`, `tensorflow`, `keras`, `incubator-mxnet`, `caffe`) — there is no CLI argument.
2. Load `datasets/{project}.csv`, shuffle with `random_state=999`, concatenate `Title` + `Body` into a `text` column, rename `class` → `sentiment` (1 = performance-related bug).
3. Text cleaning chain: HTML strip → emoji strip → stopwords → `clean_str` → stemming or lemmatization.
4. Vectorize (TF-IDF / Word2Vec / GloVe).
5. Loop `REPEAT = 10` times with `random_state=repeated_time` as the split seed, collect accuracy/precision/recall/F1/AUC, report the means. Precision/recall/F1 are macro-averaged; AUC falls back to 0.5 when the test split is single-class.

`main.py` diverges from the comparators in ways that are the actual contribution — keep them when editing: lemmatization instead of stemming, `boost_performance_keywords` (repeats domain terms to inflate their TF-IDF weight), `scale_pos_weight=5` for class imbalance, and `stratify=` on the split.

## Data and outputs

- `datasets/*.csv` — raw labelled bug reports, one file per project. Treat as read-only inputs.
- `Title+Body.csv` — intermediate file that `main.py` **writes then re-reads**; it is regenerated (overwritten) on every run and is checked into git.
- `{project}_NB.csv` — results are **appended** to the repo root, one row per run, header written only if the file is missing. The `_NB` suffix is legacy from the Naive Bayes original; it applies to XGBoost results too.
- `outputs/` — the copies of those result CSVs that were submitted with the report. Don't overwrite them with fresh runs unless asked; `main.py` writes to the root, and moving files into `outputs/` is a manual step.

## Known inconsistencies

Real defects, documented so they aren't "fixed" by accident or overlooked when they matter:

- [app.py](app.py) loads `xgboost_bug_report_model.pkl` and `tfidf_vectorizer.pkl` at import time, but the `joblib.dump` calls that would produce them are commented out at the bottom of [main.py:206-208](main.py#L206-L208). The app cannot start until those lines are uncommented and `main.py` is re-run.
- [app.py](app.py) preprocesses with **stemming** while `main.py` trains with **lemmatization**, and its `clean_text` regex differs from `clean_str`. Inference-time text therefore does not match the vectorizer's training distribution. Fixing this means making `app.py` reuse `main.py`'s exact chain.
- `main.py` writes `Title+Body.csv` with a `Number` column, but the comparator scripts build their frame in memory and never produce it — the two paths are not interchangeable.

## Report artifacts

`ISE_report.pdf`, `manual.pdf`, `replication.pdf`, `requirements.pdf` are graded submission documents. If a change alters how the tool is run, configured, or reproduced, the corresponding PDF is now stale — say so rather than silently letting the docs drift.
