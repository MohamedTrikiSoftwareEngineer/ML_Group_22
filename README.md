# ML_Group_22

This is our Machine Learning project for NOVA IMS, Group 22. The dataset is "Everything & Then Some",
and the question is whether a customer will buy again in the next 90 days. Kaggle scores F1, so F1 is
what we chased.

We ended with a logistic regression over 125 columns. It scores 0.761 on folds cut in time order, 0.759
in nested cross validation, 0.785 on a holdout from the later part of the data, and 0.795 on the public
leaderboard. Answering yes every time gives 0.718, so the model is doing something, but the problem is
not easy.

## Running it

We committed the notebooks after running them, so every table and figure is already inside the files and
GitHub shows the analysis without installing anything. If you want to run it yourself:

```bash
conda env create -f environment.yml
conda activate ml_group_22
python -m ipykernel install --sys-prefix --name ml_group_22 --display-name "Python (ml_group_22)"
```

Open the notebooks in `01_notebooks/`, pick the `Python (ml_group_22)` kernel, and run 01, 02, 03 and 04
in that order. The working directory has to be `01_notebooks/` because each notebook finds the project
root as `Path.cwd().parent`. The course CSVs are already in `data/`, and a full run takes a couple of
minutes. The intermediate tables land back in `data/`, the figures, metrics and logs in `artifacts/`, and
the submission in `deliverables/`. Those folders are tracked empty on purpose, since the notebooks never
create a directory.

The two environment files carry exact versions, not ranges, and they pin the environment that produced
the committed outputs and the Kaggle file: Python 3.14, pandas 3.0.6 and scikit-learn 1.9.1. We tried
ranges at first, but a clean environment built later pulled newer scikit-learn and numpy and changed the
probabilities, the tuned threshold and the submission. The pins are what make the Kaggle file
reproducible, and the earlier Python 3.11 set is recorded in a comment inside `environment.yml` because
it does not reproduce them.

## How we worked

Notebook 01 is where we got to know the data. We checked what each column means, whether the customer and
transaction keys are clean, how many customers repeat, and where the missing values come from. Two things
stood out. 65 training rows are empty from top to bottom, and the test set has none of them. And some
values are not missing at all but impossible, like a billion used as a placeholder or a negative amount,
so we treated them as invalid and flagged them instead of filling them with a median.

Notebook 02 turns those findings into a usable table. We drop the empty training rows, keep every test
row, fill the structural gaps in the purchase history with zero and a flag, join the customer file on a
key we checked first, group the rare levels of `transport_method` before encoding, impute the rest,
apply a log transform, and standardise only the numeric columns. The flags and the one hot columns keep
their natural scale because scaling them made the model worse when we tested it.

Notebook 03 applies the five selection stages from class per concept rather than per dummy column. The
one hot columns of one category are not independent variables, so treating each of them as a separate
candidate would have been misleading. The candidate sets are compared by cross validated F1 on folds cut
in time order, with a random fold check and a grouped check on `customer_id` printed beside them. The set
we keep gives up 0.0025 F1 against the best one and holds 63 columns instead of 125, which is inside the
tolerance the fold spread allows, and notebook 04 scores both before choosing.

Notebook 04 picks the protocol before the model. It scores the same logistic regression under a random
split, a customer grouped split and `TimeSeriesSplit`, and keeps the last one, because every row we are
scored on is later than every row we can fit on. The audit printed beside it is what decides: the grouped
split does keep customers on one side, but 99.8% of the rows it scores are earlier than the newest row it
was fitted on, so it predicts the past out of the future, while the time ordered split is 0% on that
measure. The grouped number stays in the table as the floor for the customers we have never met. On those
folds notebook 04 runs the grid searches over logistic regression, a decision tree, a PCA variant and a
dummy baseline, tunes the decision threshold on out of fold probabilities, and checks the result again
under nested validation. The imputer and the scaler are fitted on training rows only, and the test set is
never used to fit anything. We stayed inside the course toolkit throughout: CSV instead of pickle,
`cross_validate` and `GridSearchCV` instead of the shortcuts, and no boosting, random forests, kNN or SVM.

## What we got

The conservative number is the customer grouped one, 0.750, because it pretends every customer is new.
The headline is the time ordered 0.761, since no fold is ever fitted on the future, and the nested 0.759
sits just under it, which says the tuned threshold adds nothing of its own. Most of the test set is not
new, about 94.5% of the rows belong to customers already in train, and the chronological holdout reflects
that: 0.785 overall, 0.793 on returning customers and 0.710 on first timers. The submitted file scored
0.795 on the public leaderboard against the 0.718 majority baseline.

The final model uses 125 columns and a logistic regression with `C = 0.01`. The threshold that gave the
best out of fold F1 was 0.4651. We also scored the smaller set from notebook 03, 63 columns, but the full
set scores higher on the same folds, so the submission uses it.

## What we hand in

The graded report is `deliverables/Group_22_handout_text.pdf`, a cover and two pages. The Kaggle file is
`deliverables/ML_Group_22_Version05.csv`, 986 rows in the template order. For Moodle we pack the four
executed notebooks, `environment.yml`, a short `README.txt` with the run steps and the empty `data/`,
`artifacts/` and `deliverables/` folders into `submission/ML_Group22_Notebook.zip`. The course CSVs are
not in the zip because they belong to the course, and neither the archive nor the `submission/` folder it
is built in is tracked in git, because both are only copies of files that are.

## Team

Gustavo, Wael and Mohamed.
