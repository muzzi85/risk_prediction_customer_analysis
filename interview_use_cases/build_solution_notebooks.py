"""Generates worked SOLUTION notebooks (one per case study), implementing every task
from the corresponding take-home notebook with runnable code. These are reference
answers for interviewers - run build_case_notebooks.py first for the take-home versions.

Run: python3 build_solution_notebooks.py
Then execute each notebook, e.g.:
  jupyter nbconvert --to notebook --execute --inplace case1_credit_approval_solution.ipynb
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


# ===========================================================================
# Case 1 SOLUTION - Credit Approval (PD)
# ===========================================================================
cells_1 = [
md("""# Case Study 1 — SOLUTION: Credit Approval (PD Model)

Reference solution for `case1_credit_approval.ipynb`. Uses `data/case1_credit_approval.csv`
(application + bureau features, target `TARGET`)."""),

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
df = pd.read_csv(f"{DATA_DIR}/case1_credit_approval.csv")
print(df.shape)
df.head()
"""),

md("""## Task 1 — Load & Explore"""),

code("""target_rate = df["TARGET"].mean()
print(f"Default rate: {target_rate:.3%}")
print(df["TARGET"].value_counts())
"""),

md("""With ~8% positive class, **accuracy** is misleading: a model predicting "no default"
for everyone scores ~92% accuracy while being useless. **PR-AUC** focuses on the
minority (positive) class and is far more informative for imbalanced credit risk problems."""),

code("""ext_bureau_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] + \\
    [c for c in df.columns if c.startswith("BUREAU_")]
df[ext_bureau_cols].isna().mean().sort_values(ascending=False)
"""),

md("""`EXT_SOURCE_1` is missing for ~56% of applicants (a normalized external scoring source
not available for everyone). `BUREAU_*` columns are missing for ~1.7M / 307k applicants who
have **no record in the credit bureau** — i.e. no previously tracked external credit. This
is informative (likely "thin file" / new-to-credit customers) rather than random, so it
should be imputed with a sentinel value or flagged, not dropped."""),

md("""## Task 2 — Feature Engineering"""),

code("""df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]

df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

# DAYS_EMPLOYED has a well-known anomalous placeholder of 365243 (used for
# pensioners / unemployed) -> treat as missing rather than ~1000 years employed
df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365.25

df["EXTERNAL_DEBT_INCOME_RATIO"] = df["BUREAU_AMT_CREDIT_SUM_DEBT_TOTAL"] / df["AMT_INCOME_TOTAL"]

new_feats = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_GOODS_RATIO",
              "AGE_YEARS", "YEARS_EMPLOYED", "EXTERNAL_DEBT_INCOME_RATIO"]
df[new_feats].describe()
"""),

md("""**Rationale:**
- `CREDIT_INCOME_RATIO` / `ANNUITY_INCOME_RATIO`: higher leverage relative to income -> higher
  default risk.
- `CREDIT_GOODS_RATIO`: credit amount above the goods price suggests financing fees/add-ons,
  potentially riskier loans.
- `AGE_YEARS`: younger applicants tend to have less stable credit history.
- `YEARS_EMPLOYED`: longer tenure -> more stable income -> lower risk. The 365243 anomaly is
  set to NaN so it doesn't get read as ~1000 years.
- `EXTERNAL_DEBT_INCOME_RATIO`: how leveraged the applicant already is *outside* this bank,
  relative to income."""),

md("""## Task 3 — Preprocessing & Split"""),

code("""from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

feature_cols = [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET")]
X = df[feature_cols]
y = df["TARGET"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()
print(f"{len(num_cols)} numeric, {len(cat_cols)} categorical features")

preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])

X_train_proc = preprocess.fit_transform(X_train)
X_val_proc = preprocess.transform(X_val)
print(X_train_proc.shape, X_val_proc.shape)
"""),

md("""If the imputer/encoder were fit on the **full** dataset before splitting, statistics
like the median or most-frequent category would leak information from the validation set
into training — a (mild but real) form of data leakage that makes validation performance
overly optimistic."""),

md("""## Task 4 — Model Training & Evaluation"""),

code("""from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

# Baseline: Logistic Regression
logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg.fit(X_train_proc, y_train)
proba_lr = logreg.predict_proba(X_val_proc)[:, 1]

print("Logistic Regression")
print(f"  ROC-AUC: {roc_auc_score(y_val, proba_lr):.4f}")
print(f"  PR-AUC : {average_precision_score(y_val, proba_lr):.4f}")
"""),

code("""# Gradient boosting with native categorical handling
X_train_lgb = X_train.copy()
X_val_lgb = X_val.copy()
for c in cat_cols:
    X_train_lgb[c] = X_train_lgb[c].astype("category")
    X_val_lgb[c] = pd.Categorical(X_val_lgb[c], categories=X_train_lgb[c].cat.categories)

lgbm = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    class_weight="balanced", random_state=42, verbose=-1,
)
lgbm.fit(X_train_lgb, y_train, categorical_feature=cat_cols)
proba_lgb = lgbm.predict_proba(X_val_lgb)[:, 1]

print("LightGBM")
print(f"  ROC-AUC: {roc_auc_score(y_val, proba_lgb):.4f}")
print(f"  PR-AUC : {average_precision_score(y_val, proba_lgb):.4f}")
"""),

md("""## Task 5 — Explainability (SHAP)"""),

code("""import shap

explainer = shap.TreeExplainer(lgbm)
sample = X_val_lgb.sample(1000, random_state=42)
shap_values = explainer.shap_values(sample)

# shap_values may be a list (one array per class) or a single array for binary LGBM
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

shap.summary_plot(sv, sample, show=False, max_display=10)
plt.tight_layout()
plt.show()
"""),

code("""# Individual explanation for one high-risk applicant
high_risk_idx = np.argsort(-proba_lgb)[0]
applicant = X_val_lgb.iloc[[high_risk_idx]]
applicant_sv = explainer.shap_values(applicant)
applicant_sv = applicant_sv[1] if isinstance(applicant_sv, list) else applicant_sv

print(f"Predicted default probability: {proba_lgb[high_risk_idx]:.3f}")
shap.force_plot(
    explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
    applicant_sv[0], applicant, matplotlib=True, show=False,
)
plt.tight_layout()
plt.show()
"""),

md("""**Top drivers** typically include `EXT_SOURCE_1/2/3` (external credit bureau scores),
`AGE_YEARS`, `CREDIT_INCOME_RATIO`/`ANNUITY_INCOME_RATIO`, and `EXTERNAL_DEBT_INCOME_RATIO`
— all consistent with credit risk intuition (lower external scores, younger age, higher
leverage -> higher predicted risk).

For a declined applicant, you'd point to the SHAP values pushing their score up most —
e.g. "a low external credit bureau score and a high loan-to-income ratio were the largest
factors in this decision.\""""),

md("""## Task 6 — Risk Segmentation & Policy"""),

code("""results = pd.DataFrame({"y_true": y_val.values, "proba": proba_lgb})
results["risk_band"] = pd.qcut(
    results["proba"], [0, .5, .8, .95, 1],
    labels=["Low", "Medium", "High", "Very High"],
)

band_stats = results.groupby("risk_band").agg(
    n_applicants=("y_true", "size"),
    default_rate=("y_true", "mean"),
).round(4)
band_stats["pct_of_applicants"] = (band_stats["n_applicants"] / len(results)).round(3)
band_stats
"""),

md("""**Policy proposal:**
- `Low` / `Medium` (bottom ~80%) -> **auto-approve**. Their observed default rate is well
  below the portfolio average.
- `High` (next ~15%) -> **manual review**. Default rate is meaningfully elevated; a human
  underwriter can weigh compensating factors (e.g. strong `EXT_SOURCE` despite high leverage).
- `Very High` (top ~5%) -> **decline** or require additional collateral/co-signer. This
  band concentrates a large share of total defaults in a small share of applicants.

**Trade-off of lowering the decline threshold:** declining more applicants (e.g. top 10%
instead of top 5%) further reduces the portfolio default rate, but rejects a growing
number of applicants who would have repaid — i.e. lost revenue and potential fair-lending/
business-volume concerns. The right cutoff depends on the bank's cost of a default vs. the
margin on a performing loan."""),
]

nbf.write(nb(cells_1), "case1_credit_approval_solution.ipynb")


# ===========================================================================
# Case 2 SOLUTION - Early Warning System
# ===========================================================================
cells_2 = [
md("""# Case Study 2 — SOLUTION: Early Warning System

Reference solution for `case2_early_warning.ipynb`. Uses `data/case2_early_warning.csv`
(application + bureau + "early window" behavioural features, target
`EARLY_WARNING_SEGMENT`)."""),

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
df = pd.read_csv(f"{DATA_DIR}/case2_early_warning.csv")
print(df.shape)
df.head()
"""),

md("""## Task 1 — Load & Explore"""),

code("""order = ["Low Risk", "Medium Risk", "High Risk", "Immediate Attention"]
counts = df["EARLY_WARNING_SEGMENT"].value_counts().reindex(order)
print(counts)
print((counts / len(df)).round(4))
"""),

md("""`High Risk` (~1%) and `Medium Risk` (~6.5%) are rare compared to `Low Risk` (~84%) and
`Immediate Attention` (~8%). Overall **accuracy** would be dominated by the `Low Risk`
majority class — a model that mostly predicts `Low Risk` could still score high accuracy
while completely missing the rare-but-actionable `Medium`/`High Risk` segments. Better
metrics: **macro-averaged F1** (treats all classes equally) and **per-class recall**,
especially for `High Risk`/`Immediate Attention`."""),

code("""behavioral_cols = [c for c in df.columns if c.startswith(("POS_EARLY", "CC_EARLY", "INSTAL_EARLY"))]
df[behavioral_cols].isna().mean().sort_values(ascending=False)
"""),

md("""Missingness here is **informative, not random**: e.g. `CC_EARLY_*` is missing for a
customer who never had a credit card product, and `POS_EARLY_*`/`INSTAL_EARLY_*` is missing
for someone with no point-of-sale loan or installment history in the early window. This
itself is a signal (a customer with no behavioural footprint differs from one with a clean
history) — so we add explicit "has history" indicator flags rather than just imputing
silently."""),

md("""## Task 2 — Feature Preparation"""),

code("""# Indicator flags for "has early-window history" per product type
df["HAS_POS_EARLY"] = df["POS_EARLY_NUM_RECORDS"].notna().astype(int)
df["HAS_CC_EARLY"] = df["CC_EARLY_MAX_UTILIZATION"].notna().astype(int)
df["HAS_INSTAL_EARLY"] = df["INSTAL_EARLY_NUM_PAYMENTS"].notna().astype(int)

from sklearn.model_selection import train_test_split

df["EARLY_WARNING_SEGMENT"] = pd.Categorical(df["EARLY_WARNING_SEGMENT"], categories=order, ordered=True)
y = df["EARLY_WARNING_SEGMENT"].cat.codes  # 0=Low Risk ... 3=Immediate Attention

feature_cols = [c for c in df.columns if c not in ("SK_ID_CURR", "EARLY_WARNING_SEGMENT")]
X = df[feature_cols]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

cat_cols = X.select_dtypes(include="object").columns.tolist()
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = pd.Categorical(X_val[c], categories=X_train[c].cat.categories)

print(X_train.shape, X_val.shape, "| categoricals:", len(cat_cols))
"""),

md("""Missing numeric behavioural values are left as `NaN` and handled natively by LightGBM
(which learns an optimal split direction for missing values), rather than imputed with e.g.
0 — a 0 utilization and a *missing* (never had a card) utilization mean very different things."""),

md("""## Task 3 — Multi-Class Model & Evaluation"""),

code("""import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix

clf = lgb.LGBMClassifier(
    objective="multiclass", num_class=4, n_estimators=300,
    learning_rate=0.05, num_leaves=31, class_weight="balanced",
    random_state=42, verbose=-1,
)
clf.fit(X_train, y_train, categorical_feature=cat_cols)
y_pred = clf.predict(X_val)

print(classification_report(y_val, y_pred, target_names=order))
"""),

code("""cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=order, yticklabels=order)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
"""),

md("""**Which error is worse?** Missing an `Immediate Attention` customer by predicting
`Low Risk` (a **false negative** on the highest-risk class) is far more costly — the bank
takes no action and the account deteriorates/defaults unchecked. A `Low Risk` customer
flagged as `Immediate Attention` (a **false positive**) only costs an unnecessary outreach.

To reflect this, you'd: (a) keep `class_weight="balanced"` or weight `Immediate Attention`
even more heavily, and/or (b) instead of taking `argmax` of the predicted probabilities,
lower the probability threshold required to assign `Low Risk` — i.e. bias the decision
rule toward flagging borderline cases for review rather than letting them default to
"no action.\""""),

md("""## Task 4 — What Predicts Deterioration?"""),

code("""importances = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
importances.head(15)
"""),

code("""plt.figure(figsize=(8, 6))
importances.head(15).iloc[::-1].plot(kind="barh")
plt.title("Top 15 feature importances - Early Warning model")
plt.tight_layout()
plt.show()
"""),

md("""Behavioural early-window signals (`INSTAL_EARLY_PCT_LATE`, `POS_EARLY_MAX_DPD`/`MAX_DPD_DEF`,
`CC_EARLY_MAX_UTILIZATION`) typically rank highly — a customer who was *already* showing
mild lateness or high utilization a year ago is more likely to deteriorate further.
Application/bureau features (`EXT_SOURCE_*`, income ratios, `BUREAU_*`) still contribute,
especially for customers with little/no early-window product history (`HAS_*` flags = 0),
where behavioural signals don't exist yet. In practice, **both layers add value** —
behavioural features dominate when present, demographic/bureau features fill the gap
otherwise."""),

md("""## Task 5 — From Prediction to Action"""),

md("""**Proposed actions per segment:**
- **Medium Risk** -> Automated check-in (email/SMS) reminding of upcoming payments;
  no human cost, catches early slippage before it compounds.
- **High Risk** -> Proactive outreach from a relationship/retention team — offer a
  restructured payment plan or temporary payment holiday before the account misses
  payments badly enough to reach `Immediate Attention`.
- **Immediate Attention** -> Route directly to collections/specialist team for active
  management; this segment is already on a trajectory toward default by the time of
  scoring.

**Threshold discussion:** Predicting a customer's risk as *lower* than their true segment
(e.g. true `High Risk` predicted as `Low Risk`) means a deteriorating account gets **no**
intervention — directly costly (missed early-warning value). Predicting *higher* than true
(e.g. true `Low Risk` predicted as `Medium Risk`) just triggers a cheap automated check-in —
low cost. Since under-prediction is far more costly than over-prediction, the decision
thresholds should be **biased toward flagging more accounts for at least the lightest-touch
intervention** — i.e. require a *high* predicted probability of `Low Risk` before taking no
action at all, rather than simply taking the argmax class."""),

md("""## Task 6 — Stretch: Segment Stability"""),

code("""# Collapse to binary: does the customer need ANY attention (Medium/High/Immediate) vs Low Risk?
y_train_binary = (y_train > 0).astype(int)
y_val_binary = (y_val > 0).astype(int)

clf_binary = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    class_weight="balanced", random_state=42, verbose=-1,
)
clf_binary.fit(X_train, y_train_binary, categorical_feature=cat_cols)
y_pred_binary = clf_binary.predict(X_val)

print(classification_report(y_val_binary, y_pred_binary, target_names=["Low Risk", "Needs Attention"]))
"""),

code("""importances_binary = pd.Series(clf_binary.feature_importances_, index=X_train.columns).sort_values(ascending=False)
importances_binary.head(10)
"""),

md("""Collapsing the rare `Medium`/`High Risk` classes together with `Immediate Attention` into a
single "needs attention" class gives the model far more positive examples to learn from
(~16% vs. ~1-6.5% for the individual rare classes), generally improving recall on the
classes that matter operationally. The top features are usually similar (behavioural
lateness/DPD/utilization dominate), but their relative ranking can shift — with more
balanced classes, application/bureau features sometimes gain relative importance because
the model is no longer almost entirely driven by separating the dominant `Low Risk` class."""),
]

nbf.write(nb(cells_2), "case2_early_warning_solution.ipynb")


# ===========================================================================
# Case 3 SOLUTION - Collections Prioritization
# ===========================================================================
cells_3 = [
md("""# Case Study 3 — SOLUTION: Collections Prioritization

Reference solution for `case3_collections.ipynb`. Uses `data/case3_collections.csv`
(defaulted customers only, targets `RECOVERY_PRIORITY_SCORE` / `PRIORITY_TIER`)."""),

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

DATA_DIR = "data"
df = pd.read_csv(f"{DATA_DIR}/case3_collections.csv")
print(df.shape)
df.head()
"""),

md("""## Task 1 — Load & Explore"""),

code("""print(df["PRIORITY_TIER"].value_counts())
df[["EXPOSURE", "DPD_SEVERITY", "LATE_PCT", "RECOVERY_PRIORITY_SCORE"]].describe()
"""),

md("""`PRIORITY_TIER` is built from **quartiles** of `RECOVERY_PRIORITY_SCORE`, so by
construction each tier holds ~25% of accounts — a deliberate design choice so collections
capacity can be planned around a known, stable volume per tier (rather than a fixed
threshold that might put 90% of accounts in one tier if the score distribution shifts)."""),

code("""corr_cols = ["EXPOSURE", "DPD_SEVERITY", "LATE_PCT", "RECOVERY_PRIORITY_SCORE"]
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation: exposure & severity components")
plt.tight_layout()
plt.show()
"""),

md("""`EXPOSURE` and `DPD_SEVERITY` are typically only weakly correlated — how much money is
at stake (exposure) is largely independent of how delinquent the account behaviour has been
(severity). This is exactly *why* the score blends both with substantial weights (40% / 35%):
a large, currently-low-DPD account and a small, severely-delinquent account would otherwise
be scored very differently depending on which single factor you used."""),

md("""## Task 2 — Origination-Only Model"""),

code("""exclude_cols = {
    "SK_ID_CURR", "POS_FULL_MAX_DPD_DEF", "CC_FULL_MAX_DPD_DEF",
    "INSTAL_FULL_PCT_LATE", "INSTAL_FULL_MAX_DAYS_LATE",
    "EXPOSURE", "DPD_SEVERITY", "LATE_PCT",
    "RECOVERY_PRIORITY_SCORE", "PRIORITY_TIER",
}
origination_cols = [c for c in df.columns if c not in exclude_cols]
print(f"{len(origination_cols)} origination-time features")
origination_cols
"""),

code("""import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X = df[origination_cols].copy()
y = df["RECOVERY_PRIORITY_SCORE"]

cat_cols = X.select_dtypes(include="object").columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

reg_origination = lgb.LGBMRegressor(
    n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1,
)
reg_origination.fit(X_train, y_train, categorical_feature=cat_cols)
pred_origination = reg_origination.predict(X_val)

r2 = r2_score(y_val, pred_origination)
rho, _ = spearmanr(y_val, pred_origination)
print(f"R^2:               {r2:.4f}")
print(f"Spearman rho:      {rho:.4f}")
"""),

md("""**Why Spearman over R²:** for prioritization, what matters is whether the model gets
the **ordering** of accounts right (does account A get contacted before account B), not
whether it predicts the exact 0-100 score. R² penalizes any miscalibration of scale/shape,
while Spearman rank correlation directly measures ranking quality — which is what
"top-25% capacity" decisions actually depend on."""),

md("""## Task 3 — How Good Is "Good Enough"?"""),

code("""val_results = pd.DataFrame({
    "true_score": y_val.values,
    "pred_score": pred_origination,
    "true_tier": df.loc[y_val.index, "PRIORITY_TIER"].values,
})
val_results["pred_tier"] = pd.qcut(
    val_results["pred_score"], 4,
    labels=["Tier 4 - Low", "Tier 3 - Medium", "Tier 2 - High", "Tier 1 - Immediate"],
)

from sklearn.metrics import confusion_matrix
tier_order = ["Tier 1 - Immediate", "Tier 2 - High", "Tier 3 - Medium", "Tier 4 - Low"]
cm = confusion_matrix(val_results["true_tier"], val_results["pred_tier"], labels=tier_order)
pd.DataFrame(cm, index=tier_order, columns=tier_order)
"""),

code("""# Top-25% capacity: what fraction of TRUE Tier 1 accounts fall in the predicted top 25%?
n_capacity = int(0.25 * len(val_results))
contacted = val_results.sort_values("pred_score", ascending=False).head(n_capacity)

true_tier1 = val_results[val_results["true_tier"] == "Tier 1 - Immediate"]
captured = contacted[contacted["true_tier"] == "Tier 1 - Immediate"]

capture_rate = len(captured) / len(true_tier1)
print(f"True Tier 1 accounts: {len(true_tier1)}")
print(f"Captured in top-25% contacted: {len(captured)}")
print(f"Capture rate: {capture_rate:.2%}")
"""),

md("""A capture rate meaningfully above 25% (the rate a random/no-model approach would
achieve) demonstrates the origination-only model adds real value — collections teams reach
a disproportionate share of the truly highest-priority accounts using only information
available on day one. Whether it's "good enough" depends on collections capacity and the
cost of missed high-priority accounts vs. the cost of building/maintaining the
behavioural-data pipeline needed for Task 4's full model."""),

md("""## Task 4 — Full-Feature Model (Sanity Check)"""),

code("""full_cols = [c for c in df.columns if c not in {"SK_ID_CURR", "RECOVERY_PRIORITY_SCORE", "PRIORITY_TIER"}]
X_full = df[full_cols].copy()
cat_cols_full = X_full.select_dtypes(include="object").columns.tolist()
for c in cat_cols_full:
    X_full[c] = X_full[c].astype("category")

X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(X_full, y, test_size=0.2, random_state=42)

reg_full = lgb.LGBMRegressor(
    n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1,
)
reg_full.fit(X_train_f, y_train_f, categorical_feature=cat_cols_full)
pred_full = reg_full.predict(X_val_f)

print(f"R^2:          {r2_score(y_val_f, pred_full):.4f}")
print(f"Spearman rho: {spearmanr(y_val_f, pred_full)[0]:.4f}")
"""),

md("""As expected, R² is near 1.0 — `RECOVERY_PRIORITY_SCORE` is a **deterministic function**
of `EXPOSURE`, `DPD_SEVERITY`, and `LATE_PCT`, so a model with access to those inputs
trivially reproduces it. The "practical use" of this model isn't prediction at all — it's
a **sanity check** that the formula and pipeline are implemented correctly, and a tool to
inspect feature importances/weights when the underlying formula is *not* known (e.g. if
the business rule were more complex or undocumented)."""),

code("""imp_origination = pd.Series(reg_origination.feature_importances_, index=origination_cols).sort_values(ascending=False)
imp_full = pd.Series(reg_full.feature_importances_, index=full_cols).sort_values(ascending=False)

print("Top 5 - origination-only model:")
print(imp_origination.head(5))
print("\\nTop 5 - full-feature model:")
print(imp_full.head(5))
"""),

md("""In the full-feature model, `EXPOSURE`, `DPD_SEVERITY`, and `LATE_PCT` dominate
importances almost entirely (consistent with the 0.40/0.35/0.25 weighting in the formula —
`EXPOSURE` slightly ahead of `DPD_SEVERITY`, both well ahead of `LATE_PCT`). The
origination-only model instead spreads importance across `EXT_SOURCE_*`, income/credit
ratios, and `BUREAU_*` debt features — proxies that *correlate* with eventual exposure and
severity but aren't the same thing. This confirms Task 1's finding: exposure and severity
are largely independent signals, both of which the score formula explicitly captures, but
which origination-time data can only partially anticipate."""),

md("""## Task 5 — Stretch: Operational Design"""),

md("""**Scoring cadence:** Both models have a role. The **origination-time model (Task 2)**
should score every new defaulted account *immediately* — at the moment of default, before
any collections-specific data (current exposure trend, DPD history) has accumulated, this
is the only signal available, and it lets the team triage on day one. The **full-feature
model (Task 4)** — or really, the underlying formula directly — should then be recomputed
**periodically** (e.g. weekly) as `EXPOSURE`, `DPD_SEVERITY`, and `LATE_PCT` evolve, refining
the priority as better information arrives. In practice the origination model's output
could serve as a prior/tiebreaker until enough behavioural data exists to compute the full
score reliably.

**Validation via A/B test:** Randomly assign newly-defaulted accounts to either (a) the
model-prioritized contact order, or (b) the existing/baseline contact order (e.g. FIFO or
exposure-only). Compare recovery rate (% of exposure recovered within N days) and
time-to-first-contact for high-priority accounts between arms. Guard against contamination
by ensuring collections agents don't have visibility into which arm an account belongs to
beyond the assigned contact order.

**Fairness/compliance:** Using `NAME_EDUCATION_TYPE`, `OCCUPATION_TYPE`, or
`ORGANIZATION_TYPE` to influence *who gets contacted first for debt collection* risks
disparate treatment of protected or quasi-protected groups (e.g. certain occupations
correlating with race/ethnicity or socioeconomic status), even if unintentional. Best
practice: exclude such features from the prioritization model entirely, or at minimum run
a disparate-impact analysis (compare contact-priority distributions across demographic
groups) before deployment, and ensure the *outcome* (contact order/timing) doesn't
systematically disadvantage any protected group beyond what's explained by `EXPOSURE`/
`DPD_SEVERITY`/`LATE_PCT` alone."""),
]

nbf.write(nb(cells_3), "case3_collections_solution.ipynb")

print("Done: 3 solution notebooks written.")
