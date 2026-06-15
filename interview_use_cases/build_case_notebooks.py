"""Generates the three case-study take-home notebooks, one per training CSV in data/.
Run once: python3 build_case_notebooks.py
"""
import nbformat as nbf

DATA_DIR = "data"


def nb(cells):
    n = nbf.v4.new_notebook()
    n["cells"] = cells
    n["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    }
    return n


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


# ---------------------------------------------------------------------------
# Case Study 1: Credit Approval (PD)
# ---------------------------------------------------------------------------
cells_1 = [
md("""# Case Study 1 — Credit Approval (PD Model)

## Business question
**Should we approve this customer?**

You are given `data/case1_credit_approval.csv.gz` — one row per applicant, with application-level
features (income, employment, demographics, EXT_SOURCE scores) plus bureau aggregates
(`BUREAU_*` — what other lenders know about this customer).

**Target:** `TARGET` — 1 = customer had serious payment difficulties (default), 0 = repaid normally.

## Deliverables
1. A leakage-free PD model with honest evaluation (ROC-AUC, PR-AUC).
2. SHAP-based explainability (global + at least one individual explanation).
3. A risk segmentation (risk bands) that could drive an approve/review/decline policy."""),

md("""## Task 1 — Load & Explore

1. Load `case1_credit_approval.csv.gz`.
2. Report the class balance of `TARGET`. Why is **PR-AUC** often preferred over ROC-AUC
   for problems like this, and why is **accuracy** misleading here?
3. Report missing values for `EXT_SOURCE_1/2/3` and the `BUREAU_*` columns. What does a
   missing `BUREAU_*` value likely mean (hint: not every applicant has bureau history)?"""),

code("""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv(f"{DATA_DIR}/case1_credit_approval.csv.gz")
# TODO: class balance, missingness report
"""),

md("""## Task 2 — Feature Engineering

Create at least 5 engineered features, e.g.:
- `AMT_CREDIT` / `AMT_INCOME_TOTAL` (credit-to-income ratio)
- `AMT_ANNUITY` / `AMT_INCOME_TOTAL` (annuity-to-income ratio)
- Age in years from `DAYS_BIRTH`
- Years employed from `DAYS_EMPLOYED` — **note**: this column has a well-known anomalous
  placeholder value (365243). How will you handle it, and why?
- A feature combining `BUREAU_AMT_CREDIT_SUM_DEBT_TOTAL` with `AMT_INCOME_TOTAL` (external
  debt-to-income)

Justify each feature in a sentence — what relationship with default risk are you hypothesizing?"""),

code("""# TODO: feature engineering
"""),

md("""## Task 3 — Preprocessing & Split

1. Create a stratified train/validation split.
2. Build a preprocessing pipeline (impute, encode categoricals) **fit only on the training split**.
3. In 1-2 sentences: what would go wrong if you fit the imputer/encoder before splitting?"""),

code("""from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# TODO: stratified split + preprocessing pipeline
"""),

md("""## Task 4 — Model Training & Evaluation

1. Train a baseline Logistic Regression.
2. Train a gradient boosting model (LightGBM, XGBoost, or CatBoost).
3. Evaluate both on the validation set with **ROC-AUC** and **PR-AUC**.
4. (Optional) Use stratified k-fold CV instead of a single split."""),

code("""from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# TODO: train + evaluate baseline and boosted model
"""),

md("""## Task 5 — Explainability (SHAP)

1. Compute SHAP values for your boosted model.
2. Show a global summary plot — what are the top 5 drivers of predicted default risk?
   Do they make business sense?
3. Pick one applicant predicted as high-risk and explain the prediction with a SHAP
   force/waterfall plot — what would you tell this applicant if they asked why they
   were declined?"""),

code("""# TODO: SHAP global + individual explanation
"""),

md("""## Task 6 — Risk Segmentation & Policy

1. Bucket predicted probabilities into risk bands, e.g. **Low / Medium / High / Very High**.
2. Report the observed default rate and applicant volume in each band.
3. Propose an approve/manual-review/decline policy based on these bands. What approval
   rate and portfolio default rate would this policy imply, roughly?
4. What's the business trade-off if you move the "decline" threshold lower (more
   declines)?"""),

code("""# TODO: risk bands + policy simulation
"""),
]

nbf.write(nb(cells_1), "case1_credit_approval.ipynb")


# ---------------------------------------------------------------------------
# Case Study 2: Early Warning System
# ---------------------------------------------------------------------------
cells_2 = [
md("""# Case Study 2 — Early Warning System

## Business question
**Can we detect deterioration before a customer defaults?**

You are given `data/case2_early_warning.csv.xz` — one row per *existing* customer (someone
already being serviced), with:
- Application + bureau features (same as Case 1)
- Behavioural features prefixed `POS_EARLY_*`, `CC_EARLY_*`, `INSTAL_EARLY_*` — these are
  all computed from a snapshot **at least ~7 months in the past** ("what we knew back then")

**Target:** `EARLY_WARNING_SEGMENT` — one of:
- `Low Risk`
- `Medium Risk`
- `High Risk`
- `Immediate Attention`

This label was derived from what **actually happened in the 6 months after** that snapshot
(delinquency severity, late payments) plus the loan's final outcome. In other words:
*using only what we knew a year ago, can you flag who was about to deteriorate?*

## Deliverables
1. A multi-class classifier for `EARLY_WARNING_SEGMENT`.
2. Analysis of which "early" signals are most predictive of later deterioration.
3. A discussion of how each segment should be actioned operationally."""),

md("""## Task 1 — Load & Explore

1. Load `case2_early_warning.csv.xz` and report the distribution of `EARLY_WARNING_SEGMENT`.
   Which classes are rare? Why does this matter for model evaluation (which metric(s)
   would you use, and why is overall accuracy a poor choice)?
2. Many `*_EARLY_*` columns will be missing for a given row (e.g. a customer with no
   credit card ever won't have `CC_EARLY_*` values). Is this missingness *random*, or
   informative? How should you encode it?"""),

code("""import pandas as pd
import numpy as np

df = pd.read_csv(f"{DATA_DIR}/case2_early_warning.csv.xz")
# TODO: class distribution, missingness analysis
"""),

md("""## Task 2 — Feature Preparation

1. Build a preprocessing pipeline: impute missing behavioural features (consider whether
   "no history" should be imputed differently from "had history, value was 0").
2. Encode categoricals.
3. Create a stratified train/validation split on `EARLY_WARNING_SEGMENT`."""),

code("""from sklearn.model_selection import train_test_split

# TODO: preprocessing + stratified split
"""),

md("""## Task 3 — Multi-Class Model & Evaluation

1. Train a multi-class classifier (e.g. LightGBM/XGBoost multiclass, or RandomForest).
2. Evaluate with a **confusion matrix** and **per-class precision/recall/F1** (e.g.
   `classification_report`).
3. From a business standpoint, which error is worse: a `Low Risk` customer misclassified
   as `Immediate Attention` (false alarm), or an `Immediate Attention` customer
   misclassified as `Low Risk` (missed warning)? How would you adjust the model or
   decision threshold to reflect this?"""),

code("""from sklearn.metrics import classification_report, confusion_matrix

# TODO: train multi-class model, evaluate
"""),

md("""## Task 4 — What Predicts Deterioration?

1. Compute feature importance for your model.
2. Which `*_EARLY_*` behavioural signals are most predictive of ending up in
   `High Risk` / `Immediate Attention`? Do application/bureau features (income,
   EXT_SOURCE, bureau debt) add meaningfully beyond behavioural signals, or are
   behavioural signals dominant?"""),

code("""# TODO: feature importance analysis
"""),

md("""## Task 5 — From Prediction to Action

For each segment, propose a concrete operational action a bank's collections/relationship
team could take **before** the customer actually defaults, e.g.:
- `Medium Risk` → ?
- `High Risk` → ?
- `Immediate Attention` → ?

Then: if your model's predicted segment disagreed with the "true" segment for a customer,
which direction of disagreement (predicted higher vs. lower risk than actual) is more
costly for the business, and how would that shape which probability threshold you'd pick
for triggering an intervention?"""),

md("""*(Write your action plan + threshold discussion here)*"""),

md("""## Task 6 — Stretch: Segment Stability

`Immediate Attention` and `Low Risk` dominate the label distribution while `Medium`/`High`
are rare and arguably the most actionable (a customer is not yet in `Immediate Attention`,
so there's still time to act).

1. Try training a model that specifically optimizes for distinguishing
   `Medium Risk` / `High Risk` from `Low Risk` (e.g. by collapsing classes, reweighting,
   or a two-stage model).
2. Does this change which features matter most?"""),

code("""# TODO: stretch analysis
"""),
]

nbf.write(nb(cells_2), "case2_early_warning.ipynb")


# ---------------------------------------------------------------------------
# Case Study 3: Collections Prioritization
# ---------------------------------------------------------------------------
cells_3 = [
md("""# Case Study 3 — Collections Prioritization

## Business question
**Who should collections contact first?**

You are given `data/case3_collections.csv.gz` — one row per customer who **already
defaulted** (`TARGET == 1` in the original data), with:
- Application + bureau features
- Full-history behavioural severity: `POS_FULL_MAX_DPD_DEF`, `CC_FULL_MAX_DPD_DEF`,
  `INSTAL_FULL_PCT_LATE`, `INSTAL_FULL_MAX_DAYS_LATE`
- `EXPOSURE` = current loan amount + external debt
- `DPD_SEVERITY`, `LATE_PCT` — summary severity measures
- **Targets:**
  - `RECOVERY_PRIORITY_SCORE` (0-100, continuous) — a weighted blend:
    `0.40 * percentile(EXPOSURE) + 0.35 * percentile(DPD_SEVERITY) + 0.25 * percentile(LATE_PCT)`
  - `PRIORITY_TIER` (`Tier 1 - Immediate` ... `Tier 4 - Low`) — quartiles of the above score

## Important framing
`RECOVERY_PRIORITY_SCORE` is **defined directly** from `EXPOSURE`, `DPD_SEVERITY`, and
`LATE_PCT` — so a model that uses those three columns as inputs will trivially reproduce
the formula. The interesting question is:

> **Can application/bureau/demographic features (known at origination, before any of this
> delinquency history existed) predict who will eventually become a high collections
> priority — *without* access to the current exposure/severity numbers?**

This matters operationally: those features are available from day one, while
`EXPOSURE`/`DPD_SEVERITY`/`LATE_PCT` only become known once the account is already
deteriorating.

## Deliverables
1. A model predicting `RECOVERY_PRIORITY_SCORE` (or `PRIORITY_TIER`) using **only**
   origination-time features (Task 2).
2. A second model using **all** features including the severity/exposure columns, used
   as a sanity check / upper bound (Task 4).
3. A discussion of how this would integrate into a collections workflow."""),

md("""## Task 1 — Load & Explore

1. Load `case3_collections.csv.gz`. Confirm `PRIORITY_TIER` is roughly balanced (it's
   built from quartiles — why would that be a deliberate design choice for this label?).
2. Look at the distributions of `EXPOSURE`, `DPD_SEVERITY`, `LATE_PCT`, and
   `RECOVERY_PRIORITY_SCORE`. Are `EXPOSURE` and `DPD_SEVERITY` correlated with each
   other? What would that imply about the weighting in the score formula?"""),

code("""import pandas as pd
import numpy as np

df = pd.read_csv(f"{DATA_DIR}/case3_collections.csv.gz")
# TODO: explore distributions and correlations
"""),

md("""## Task 2 — Origination-Only Model

1. Define a feature set using **only** columns that would have been known at origination:
   application + bureau features (everything *except* `POS_FULL_*`, `CC_FULL_*`,
   `INSTAL_FULL_*`, `EXPOSURE`, `DPD_SEVERITY`, `LATE_PCT`, `RECOVERY_PRIORITY_SCORE`,
   `PRIORITY_TIER`).
2. Train a regression model to predict `RECOVERY_PRIORITY_SCORE` from this feature set.
3. Evaluate with **R²** and, more importantly, **Spearman rank correlation** between
   predicted and actual score. Why might rank correlation matter more than R² for a
   *prioritization* task (where what matters is the ordering of accounts, not the exact
   score value)?"""),

code("""from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

# TODO: origination-only feature set, train, evaluate (R2 and Spearman)
"""),

md("""## Task 3 — How Good Is "Good Enough"?

1. Using your Task 2 model's predicted scores, derive a predicted `PRIORITY_TIER` (e.g.
   by quartile-binning the predictions) and compare against the true `PRIORITY_TIER`
   with a confusion matrix.
2. If collections capacity only allows contacting the top 25% of accounts by predicted
   priority, what fraction of the *true* `Tier 1 - Immediate` accounts would actually
   get contacted? Is this good enough to be useful in practice?"""),

code("""# TODO: predicted tiers vs true tiers, top-25% capture analysis
"""),

md("""## Task 4 — Full-Feature Model (Sanity Check)

1. Now train the same kind of model **including** `EXPOSURE`, `DPD_SEVERITY`, `LATE_PCT`
   (and the other `*_FULL_*` columns).
2. Confirm it near-perfectly reproduces `RECOVERY_PRIORITY_SCORE` (as expected, since
   the score is a deterministic function of these). What is the practical use of this
   "model" vs. just computing the formula directly?
3. Compare its feature importances to Task 2's — does this change your answer to
   Task 1's question about `EXPOSURE` vs. `DPD_SEVERITY` weighting?"""),

code("""# TODO: full-feature model + comparison
"""),

md("""## Task 5 — Stretch: Operational Design

Write a short design note (~150-250 words):

1. In production, would you score accounts for collections priority **at the moment
   they default** (using only origination-time features, per Task 2) or **periodically
   as new delinquency data accumulates** (using full features, per Task 4)? Could you
   do both, and how would they be used differently?
2. How would you validate that this prioritization actually improves recovery rates
   once deployed (e.g. an A/B test design)?
3. Are there fairness/compliance considerations in using demographic-adjacent features
   (e.g. `NAME_EDUCATION_TYPE`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`) to decide who
   gets contacted first for debt collection?"""),

md("""*(Write your design note here)*"""),
]

nbf.write(nb(cells_3), "case3_collections.ipynb")

print("Done: 3 case-study notebooks written.")
