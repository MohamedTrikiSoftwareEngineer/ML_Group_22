# ML_Group_22 — Repeat Purchase Prediction

Machine Learning homework, Group 22 · NOVA IMS · **"Everything & Then Some"** dataset.

A complete, reproducible classification pipeline: from raw transactions to a Kaggle submission,
organised as four notebooks that hand their output to the next one as CSV.

**Task** — predict whether a customer makes another purchase within 90 days (`repeat_purchase_90d`).
**Metric** — F1 on hard 0/1 labels. **Result** — nested cross-validated **F1 ≈ 0.758** on customers
the model has never seen, and **0.792** on a chronological holdout scored at the submitted threshold,
against an always-majority baseline of 0.718. Public Kaggle score of the submitted file: **0.78934**.

---

## For the marker

**You do not need to run anything to read the work.** All four notebooks were executed before they were
committed, so every printed line, table and figure is stored inside them and renders in the browser. The
quickest route is `01_notebooks/`, opened in order: GitHub shows the markdown, the code and the output of
each cell without a kernel, without the data, and without a single package installed.

**If you would rather run it**, the environment is one command away, `conda env create -f environment.yml`,
and the four input CSVs are already committed, so a clone has everything. Run the notebooks in order from
inside `01_notebooks/`; the working directory matters because each one resolves the project root as
`Path.cwd().parent`. The chain takes about 95 seconds, writes its intermediate tables back into `data/`,
its figures, logs and metrics into `artifacts/`, and the submission into `deliverables/`. Those folders are
committed as an empty skeleton on purpose, because the notebooks do not create directories themselves.

**What the libraries are.** The notebooks import pandas, numpy, matplotlib, seaborn, scikit-learn and
scipy, plus the standard library `json`, `math` and `pathlib`. That is a subset of what the course material
imports, so an environment that runs the class notebooks runs these. `environment.yml` and
`requirements.txt` are both pinned to the versions this project was actually tested on.

**Where the evidence lives.** Every number quoted in this file and in the handout comes from something the
notebooks write. The two places worth checking first are `artifacts/metrics/assessment.json`, which holds
the headline metrics, and `artifacts/logs/`, which records what cleaning and selection changed and how many
rows it affected. They are regenerated on every run, so they are not tracked in git.

**The submission itself** is `deliverables/ML_Group_22_Version04.csv`, scored 0.78934 on the public
Kaggle leaderboard, and the two-page report is `deliverables/Group_22_handout_text.pdf`. The Moodle upload
packages the four notebooks as `ML_Group_22_notebook.zip`.

---

## Repository layout

```
ML_Group_22/
├── 01_notebooks/                                      # the four stages, run in order
│   ├── Homework_Group22_01_Exploration.ipynb
│   ├── Homework_Group22_02_Cleaning-Preprocessing.ipynb
│   ├── Homework_Group22_03_Feature-Selection.ipynb
│   └── Homework_Group22_04_Modelling-Assessment.ipynb
├── data/
│   ├── train.csv · test.csv · clientes.csv · sample_submission.csv   # supplied inputs
│   └── *_raw.csv · *_clean.csv · feature_lists.csv · selected_*.csv  # generated, not tracked
├── artifacts/
│   ├── figures/     # every figure the notebooks draw
│   ├── metrics/     # model comparison, validation protocols, thresholds, confusion matrix
│   ├── features/    # fitted imputer/scaler/encoder parameters, selection stages and scores
│   └── logs/        # cleaning_log.csv and selection_log.csv — the audit trail
├── deliverables/
│   ├── Group_22_handout_text.pdf               # the graded handout, exactly 2 pages
│   └── ML_Group_22_Version04.csv               # Kaggle submission, 986 rows, public 0.78934
├── report/
│   ├── Group_22_handout_text.tex               # LaTeX source of the same text (pdfLaTeX or tectonic)
│   └── figures/nb04_*.png                      # the two figures that .tex includes (committed)
├── README.md · requirements.txt · environment.yml · .gitignore
```

**Committed:** the notebooks, the four supplied CSVs, both deliverables, the handout source together
with the two figures it includes, and the environment files.
**Not committed:** everything else a notebook can regenerate — intermediate CSVs, the other 13 figures,
metrics, logs.
The folder skeleton is still tracked via `.gitkeep`, so a fresh clone never fails on a missing
directory (the notebooks do not create folders).

---

## Setup

With conda:

```bash
conda env create -f environment.yml
conda activate ml_group_22
```

With pip:

```bash
python -m venv ml_env
source ml_env/bin/activate          # Windows: ml_env\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 (tested on 3.11.0). Dependencies: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy,
joblib, jupyter, and nothing else beyond the standard library `json`, `math` and `pathlib`. Both files
cap the major versions (`pandas<3`, `numpy<3`, `scikit-learn<2`) and record the exact version each pin
was tested against, because an open-ended `>=` now resolves to pandas 3.x, a breaking release this
project was never run on.

## Running the pipeline

1. Place the four supplied files in `data/`: `train.csv`, `test.csv`, `clientes.csv`, `sample_submission.csv`.
   They are committed, so a fresh clone already has them.
2. Run the notebooks **in order**, from inside `01_notebooks/`. Each one reads the previous one's CSVs
   and resolves the project root as `Path.cwd().parent`, so the working directory matters.
3. The submission is written to `deliverables/ML_Group_22_Version04.csv` — the course requires the `<team>_Version<NN>.csv` name, and Notebook 04 takes the version from a single constant.

No internet access, no model files to download, no manual steps in between. The whole chain runs in
95 s end to end (25 + 9 + 13 + 47), roughly half of it notebook 04's grid searches, and it is
byte-reproducible: re-running from wiped `data/` and `artifacts/` folders regenerates all 36 artefacts
identically, the 15 figures included.

---

## The four stages

| Notebook | What it does | Output |
|---|---|---|
| **01 · Explore** (3 pts) | Structure and column roles, duplicate and key integrity, target balance, missingness on the column and row axis, invalid values, distributions for every feature, association with the target (point-biserial r, Cramér's V, full Pearson matrix), `clientes` profile and join coverage, train-vs-test drift | `train_raw.csv`, `test_raw.csv`, `clientes_raw.csv` |
| **02 · Clean & preprocess** (5 pts) | Drop structurally empty training rows; invalid values to NaN with indicator flags; structural nulls to 0 with history flags; `clientes` joined 1:1; categorical nulls to an explicit `Missing` level and rare levels grouped; median imputation; `log1p`, a month feature and two ratios; standardisation of the **numeric columns only** (0/1 flags and dummies keep their natural scale) and one-hot encoding | `train_clean.csv`, `test_clean.csv`, `feature_lists.csv`, fitted parameters |
| **03 · Feature selection** (3 pts) | Five taught stages applied per *concept* rather than per dummy: `VarianceThreshold`, redundancy pruning at \|r\| ≥ 0.80, `SelectKBest` with a custom point-biserial scorer, `SelectFromModel` (tree) and `RFE` (logistic), then a choice between candidate sets by cross-validated F1 under `StratifiedKFold` as the criterion, with `GroupKFold` on `customer_id` printed beside it as the stricter check, and a parsimony rule stated in advance | `selected_columns.csv`, `selected_features.csv`, CV results and scores |
| **04 · Model & assess** (4 pts) | Validation protocol chosen by measurement and then **audited** (grouped and chronological, not only random); the scaling decision measured rather than assumed; `GridSearchCV` over logistic, decision tree, PCA + logistic and a dummy baseline; the F1 decision threshold tuned on out-of-fold probabilities and verified by nested validation; confusion matrix, PR and ROC; submission built and checked against the template | the submission CSV and `artifacts/metrics/` |

---

## What the data said, and what we did about it

| Finding | Consequence |
|---|---|
| 65 training rows have every transaction field null; the test set has none | Dropped from train only, after checking the target balance moves just 0.5592 → 0.5597. Test rows are never dropped — every ID must be predicted |
| 1e9 placeholder values and impossible negatives | Invalid, not missing: converted to NaN plus an indicator flag. Clipping them would invent a value and poison the medians, the scaler and the correlations |
| Nulls in the prior-purchase family are structural (1,048 first-time or single-purchase customers) | Filled with 0 plus a history flag, because a median history would describe customers who have none |
| `transport_method` has 80 levels, 59 under 1% of rows; a category at "100% repeat" rests on 3 rows | Levels below 1% of train are grouped before encoding, and every rate is reported with its n |
| 28 predictor pairs at \|r\| ≥ 0.80, 14 columns dropped | Pruned per concept — one-hot dummies of one category are not independent variables |
| `clientes` carries real signal (commercial region 38.5 points off the baseline) | Kept as candidate features and settled by ablation rather than assumed away |
| Train covers 2015–2025, test 2025–2026, and test customers are longer-tenured | Model choice uses `GroupKFold` on `customer_id`; feature sets are ranked with `StratifiedKFold` and verified with the grouped one; a chronological estimate is reported as the realistic leaderboard number; `purchase_year` is excluded because a coefficient cannot extrapolate |
| Classes are near balanced (56% positive), not imbalanced | No resampling; F1 is optimised because it is the scored metric, through the decision threshold |
| 94.5% of the test transactions belong to a customer already in train, and every test date is later than every train date | Verified, not assumed: `customer_id` and `ID` never enter the design matrix, and no test transaction is a copy of a training one (0 of 986 on the 37 recorded columns). `GroupKFold` is audited to keep 0 customers on both sides of every fold, so a validation customer's own transactions are never used to fit the model that scores them |
| Standardising the 0/1 flags and one-hot dummies hurts | Measured in notebook 04 under identical folds: re-scaling every column inside each fold costs ROC-AUC (0.7864 against 0.7901) because a rare dummy divided by its own standard deviation escapes the penalty. Only the numeric columns are standardised |

## Results

| Quantity | Value | | Quantity | Value |
|---|---|---|---|---|
| Model | logistic regression, `C = 0.03` | | Grouped CV F1 (selection) | 0.7489 |
| Features used | 125 columns (30 numeric, 13 flags, 82 one-hot) | | Out-of-fold F1 at threshold 0.426 | 0.7603 |
| **Nested honest F1** | **0.7585 ± 0.0088** | | Out-of-fold F1 at 0.50 | 0.7490 |
| Chronological holdout F1 | 0.7920 | | ROC-AUC | 0.7923 |
| Holdout F1, returning / new customer | 0.7984 / 0.7308 | | Accuracy | 0.7149 |
| Always-majority F1 | 0.7177 | | Submission positive rate | 75.1% |

Two numbers bracket the leaderboard. The grouped, nested one (0.758) is a floor: it scores every
customer as if the model had never seen them, while 94.5% of the test rows belong to customers who
*are* in train. The chronological holdout (0.7920) is the closer analogue, and it separates the two
populations: 0.7984 on returning customers against 0.7308 on first appearances.

Notebook 03's parsimonious set (18 concepts / 63 columns) is statistically tied with the full set —
0.7456 against 0.7459 under the stratified criterion, 0.7399 against 0.7477 under the grouped check,
both inside one fold standard deviation — so notebook 04 scores both and uses the winner. That is
stated in the notebook rather than left as a silent override.

---

## Principles the project holds to

**Course toolkit only.** Every library, function and plotting argument is restricted to what was
taught. In practice that means CSV instead of pickle, `.join` instead of `.merge`, `isna` instead of
`isnull`, `cross_validate` and `GridSearchCV` instead of `cross_val_score` and `RandomizedSearchCV`,
Pearson / point-biserial / Cramér's V instead of Spearman or mutual information, and no gradient
boosting, random forests, kNN or SVM.

**No leakage.** Imputers, scalers and encoders are fitted on training rows only, every model is
refitted inside each validation fold, and `GroupKFold` on `customer_id` is audited to prove that no
customer appears on both sides of a split. The test set is read for prediction and for descriptive
drift checks, never for fitting. Where a transform is deliberately *not* refitted per fold — the
standardisation of the numeric columns, which notebook 02 fits once on train — the choice is measured
in notebook 04 and printed there, not assumed.

**Nothing is hidden.** Every subset, threshold and cut is a stated rule printed next to its
rationale; tables and figures cover all features rather than a top-k slice; the cleaning and
selection logs record what changed, how many rows it affected and why.

**Reproducible.** A fixed seed (`RANDOM_STATE = 42`), deterministic transforms, ASCII-only output so
nothing fails on a Windows console, and no code that creates directories.

---

## Deliverables

- **`deliverables/Group_22_handout_text.pdf`** — the report the course asks for, named as the specification
  requires and exactly two pages long. Page one carries the objective, the pipeline schematic and why each
  stage suits this data; page two carries the results, the two charts the notebooks draw, and the annexes.
  It is committed as the finished document, so what you find here is what was handed in.
  `report/Group_22_handout_text.tex` is the LaTeX source of the same text, kept so the wording can be
  edited and rebuilt, and it compiles unchanged on Overleaf with pdfLaTeX; its two figures are committed
  beside it. Every number in either version comes from the artefacts the notebooks write.

- **`deliverables/ML_Group_22_Version04.csv`** — the selected Kaggle submission: 986 rows, columns `ID` and
  `repeat_purchase_90d`, IDs identical to `sample_submission.csv` and in the same order, public score 0.78934.
  Notebook 04 regenerates it byte for byte (md5 `28c931e3d071d1d25b7a808dafc02a21`).
- **The four notebooks** — they carry the full analysis and are the record of every number quoted here. For
  the Moodle upload they are packaged as `ML_Group_22_notebook.zip`: the four executed notebooks at the
  archive root, 1.4 MB against a 50 MB limit. The zip itself is not tracked, because it is a copy of files
  that are.

**Kaggle deadline:** 26 October 2026, 17:59.

## Team

| Branch | Owner |
|---|---|
| `gustavo` | Gustavo |
| `wael` | Wael |
| `vasco` | Vasco |
| `Mohamed` | Mohamed |

## License

Educational project, prepared for a machine learning course at NOVA IMS.
