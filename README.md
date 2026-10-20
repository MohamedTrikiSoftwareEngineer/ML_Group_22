# ML_Group_22 - Repeat Purchase Prediction

Machine Learning homework, Group 22, NOVA IMS, on the "Everything & Then Some" dataset.

We predict whether a customer buys again within 90 days and send the predictions to Kaggle. F1 on the
positive class is what counts. Nested cross-validation gives **0.758** on customers the model has never
seen, a chronological holdout gives **0.792**, and always answering yes gives 0.718. The file we
submitted scored **0.78934** on the public leaderboard.

## Reading it without running anything

The four notebooks in `01_notebooks/` were executed before we committed them, so every printed table and
every figure is stored inside them and GitHub renders it in the browser. Open them in order and the whole
analysis is there. No kernel, no data, nothing installed.

## Running it

One command builds the environment:

```bash
conda env create -f environment.yml
conda activate ml_group_22
python -m ipykernel install --sys-prefix --name ml_group_22 --display-name "Python (ml_group_22)"
```

Then open the notebooks from inside `01_notebooks/` and run 01, 02, 03, 04 in that order. Pick the
`Python (ml_group_22)` kernel; the default `Python 3` entry can belong to a different installation. The
working directory matters, because each notebook finds the project root as `Path.cwd().parent`. The four
input files are already in `data/`, so a clone has everything it needs.

The chain takes about 95 seconds. It writes its intermediate tables back into `data/`, its figures, logs
and metrics into `artifacts/`, and the submission into `deliverables/`. Those folders are committed empty
on purpose: the notebooks never create a directory, so the skeleton has to be there before the first cell
runs.

`requirements.txt` and `environment.yml` carry the same exact pins. We started with version ranges, but a
clean environment built today resolves to newer scikit-learn and numpy, and that quietly changes the
probabilities, the tuned threshold and the submission file. The notebooks import pandas, numpy,
matplotlib, seaborn, scikit-learn and scipy, plus `json`, `math` and `pathlib` from the standard library,
so an environment that runs the class notebooks runs these.

## What is in here

```
ML_Group_22/
├── 01_notebooks/          the four stages, run in order
├── data/                  train.csv, test.csv, clientes.csv, sample_submission.csv
│                          plus the tables the notebooks pass to each other
├── artifacts/
│   ├── figures/           the 15 charts the notebooks draw
│   ├── metrics/           model comparison, thresholds, confusion matrix, assessment.json
│   ├── features/          fitted imputer, scaler and encoder parameters, selection scores
│   └── logs/              cleaning_log.csv and selection_log.csv, the audit trail
├── deliverables/
│   ├── Group_22_handout_text.pdf      the graded handout, two pages
│   └── ML_Group_22_Version04.csv      the Kaggle submission, 986 rows
├── report/figures/        the two charts that appear in the handout
├── README.md · requirements.txt · environment.yml · .gitignore
```

Committed: the notebooks, the four supplied CSVs, both deliverables and the environment files. Not
committed: anything a notebook can regenerate, which means the intermediate CSVs, the figures, the
metrics and the logs. They are all rebuilt on a run, and `artifacts/metrics/assessment.json` is where
the headline numbers actually come from.

## The four notebooks

**01, explore.** What the columns are, whether the keys are clean, how the target is distributed, what
is missing and why, and how the training period differs from the test period. Every numeric and
categorical feature gets a look, not a top ten.

**02, clean and preprocess.** Empty rows dropped, placeholder values separated from genuine gaps,
structural nulls filled with zero plus a flag, the customer file joined on a key we checked first, rare
categories grouped, medians imputed, a log transform, and standardisation of the numeric columns only.

**03, feature selection.** Five taught stages applied per concept rather than per dummy column, because
one-hot indicators of one category are not independent variables. The candidate sets are then compared
by cross-validated F1, with a grouped check printed beside it.

**04, model and assess.** Grid searches over logistic regression, a decision tree, a PCA variant and a
dummy baseline; the decision threshold tuned on out-of-fold probabilities and re-checked under nested
validation; a confusion matrix, PR and ROC; and the submission written out and compared against the
template.

## What the data pushed back on

| Finding | What we did |
|---|---|
| 65 training rows have every transaction field null, and the test set has none | Dropped from train only, after checking the target balance barely moves. No test row is ever dropped, every ID must be predicted |
| 1e9 placeholders and impossible negative amounts | Treated as invalid rather than missing: set to NaN and flagged. Clipping them would invent a value and poison the medians and the correlations |
| Nulls in the prior-purchase fields are structural, 1,048 rows from customers with no history | Filled with zero plus a history flag, because a median history describes customers who have none |
| `transport_method` has 80 levels and a category at "100% repeat" resting on 3 rows | Levels under 1% of train are grouped before encoding, and every rate is reported with its n |
| Train runs to February 2025, test starts in May 2025 | Grouped folds on `customer_id` for model choice, a chronological estimate as the realistic leaderboard number, and no year variable, since a coefficient cannot extrapolate |
| 94.5% of test rows belong to a customer already in train | Both numbers are reported. The grouped score is the floor, the chronological one is the closer analogue |
| Classes are near balanced at 56% positive | No resampling. F1 is optimised through the threshold, because F1 is the scored metric |
| Standardising the 0/1 flags and dummies made things worse | Measured under identical folds, so only the numeric columns are standardised |

## Results

| Quantity | Value | | Quantity | Value |
|---|---|---|---|---|
| Model | logistic regression, `C = 0.03` | | Grouped CV F1 | 0.7489 |
| Features used | 125 columns (30 numeric, 13 flags, 82 one-hot) | | Out-of-fold F1 at threshold 0.426 | 0.7603 |
| **Nested honest F1** | **0.7585 ± 0.0088** | | Out-of-fold F1 at 0.50 | 0.7490 |
| Chronological holdout F1 | 0.7920 | | ROC-AUC | 0.7923 |
| Holdout F1, returning / new customer | 0.7984 / 0.7308 | | Accuracy | 0.7149 |
| Always-majority F1 | 0.7177 | | Submission positive rate | 75.1% |

The grouped number is the conservative one: it scores every customer as if the model had never seen
them, while most of the test set is people the company already knows. The chronological holdout is
closer to what the leaderboard will do, and it splits the two populations honestly, 0.7984 on returning
customers against 0.7308 on first appearances.

Notebook 03's smaller set, 18 concepts and 63 columns, is statistically tied with the full set under
both criteria, inside one fold standard deviation. Notebook 04 scores both and uses the winner, and says
so rather than overriding the rule quietly.

## How we worked

Only what was taught in class, which in practice means CSV instead of pickle, `.join` instead of
`.merge`, `cross_validate` and `GridSearchCV` instead of their shortcuts, and no boosting, random
forests, kNN or SVM. Nothing is fitted on the test set. Every transform is fitted on training rows,
every model is refitted inside its own fold, and the grouped split is audited to prove no customer
appears on both sides. Every threshold and cut is a stated rule printed next to the reason for it, and
the logs record what changed and how many rows it touched. The seed is fixed at 42, the output is
ASCII-only so nothing fails on a Windows console, and no code creates directories.

## What we hand in

- **`deliverables/Group_22_handout_text.pdf`**, the two-page report: objective, pipeline schematic and
  rationale on page one, results, the two charts and the annexes on page two. It is committed as the
  finished document, so what is here is what we submitted.
- **`deliverables/ML_Group_22_Version04.csv`**, the Kaggle submission, 986 rows in the template order,
  public score 0.78934. Notebook 04 regenerates it byte for byte on Windows.
- **`ML_Group_22_notebook.zip`**, the four executed notebooks for Moodle, at the archive root, 1.4 MB.
  It is not tracked here because it is a copy of files that are.

**Deadline:** 26 October 2026, 17:59.

## Team

| Branch | Owner |
|---|---|
| `gustavo` | Gustavo |
| `wael` | Wael |
| `Mohamed` | Mohamed |

Educational project, prepared for a machine learning course at NOVA IMS.
