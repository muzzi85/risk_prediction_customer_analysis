"""Appends a 'Deliverables' checklist to each task's markdown cell in the take-home
(non-solution) case notebooks.

Run: python3 add_deliverables.py
"""
import nbformat

DELIVERABLES = {
    "case1_credit_approval.ipynb": {
        1: [  # Task 1
            "Printed/displayed class balance of `TARGET` (counts and %)",
            "1-2 sentence answer: why PR-AUC is preferred over accuracy here",
            "Missingness table for `EXT_SOURCE_1/2/3` and all `BUREAU_*` columns",
            "1-2 sentence interpretation of what a missing `BUREAU_*` value likely means",
        ],
        3: [  # Task 2
            "At least 5 new engineered columns added to the dataframe",
            "One-sentence rationale per new feature (markdown or code comment)",
            "Explicit handling of the `DAYS_EMPLOYED == 365243` anomaly, with justification",
        ],
        5: [  # Task 3
            "Stratified train/validation split (`X_train`, `X_val`, `y_train`, `y_val`)",
            "A preprocessing pipeline/`ColumnTransformer` fit only on the training split",
            "1-2 sentence answer on the leakage risk of fitting before splitting",
        ],
        7: [  # Task 4
            "Trained Logistic Regression baseline and trained boosted model",
            "ROC-AUC and PR-AUC for both models on the validation set",
            "(Optional) stratified k-fold CV results table",
        ],
        9: [  # Task 5
            "Global SHAP summary plot showing top features",
            "One individual SHAP explanation (force/waterfall) for a high-risk applicant",
            "2-3 sentence interpretation of the top drivers and what you'd tell a declined applicant",
        ],
        11: [  # Task 6
            "A table of risk bands with applicant counts and observed default rate per band",
            "A written approve / manual-review / decline policy proposal referencing the table",
            "1-2 sentence discussion of the trade-off from tightening the decline threshold",
        ],
    },
    "case2_early_warning.ipynb": {
        1: [  # Task 1
            "`EARLY_WARNING_SEGMENT` class distribution (counts and %)",
            "1-2 sentence discussion of why accuracy is misleading + which metric(s) you'll use",
            "Missingness table for all `*_EARLY_*` columns",
            "1-2 sentence discussion of whether this missingness is informative and how to encode it",
        ],
        3: [  # Task 2
            "Documented missing-value handling strategy (including any 'no history' indicator flags)",
            "Encoded categorical features",
            "Stratified train/validation split on `EARLY_WARNING_SEGMENT`",
        ],
        5: [  # Task 3
            "Trained multi-class classifier",
            "Confusion matrix and per-class precision/recall/F1 (`classification_report`)",
            "2-3 sentence discussion of which error type is costlier and how you'd adjust the model/threshold for it",
        ],
        7: [  # Task 4
            "Feature importance ranking and plot",
            "Written comparison of behavioural vs. application/bureau feature contributions",
        ],
        9: [  # Task 5
            "A short written action plan with one concrete action per segment (Medium / High / Immediate Attention)",
            "Written discussion of which direction of disagreement (over- vs under-predicting risk) is costlier, and how that should shape your decision threshold",
        ],
        11: [  # Task 6
            "A second model trained on the collapsed/reweighted target",
            "Comparison of evaluation results and feature importance vs. the original multi-class model",
        ],
    },
    "case3_collections.ipynb": {
        1: [  # Task 1
            "`PRIORITY_TIER` value counts confirming the quartile balance, with 1-2 sentences on why this design is deliberate",
            "Summary statistics/distributions for `EXPOSURE`, `DPD_SEVERITY`, `LATE_PCT`, `RECOVERY_PRIORITY_SCORE`",
            "Correlation analysis between exposure and severity components, with interpretation of the score's weighting",
        ],
        3: [  # Task 2
            "Explicit list of origination-only feature columns used (and why each exclusion was made)",
            "Trained regression model predicting `RECOVERY_PRIORITY_SCORE`",
            "R² and Spearman rank correlation on the validation set, with 1-2 sentences on why rank correlation matters more for this task",
        ],
        5: [  # Task 3
            "Predicted `PRIORITY_TIER` (quartile-binned from predictions) vs. true `PRIORITY_TIER` confusion matrix",
            "Top-25% capacity capture-rate calculation for `Tier 1 - Immediate` accounts",
            "1-2 sentence verdict on whether this is 'good enough' for practical use",
        ],
        7: [  # Task 4
            "Trained full-feature regression model with R² and Spearman rank correlation",
            "Feature importance comparison vs. the Task 2 model",
            "Written interpretation of what this model is/isn't useful for in practice",
        ],
        9: [  # Task 5
            "A written design note (~150-250 words) covering scoring cadence, an A/B test validation design, and fairness/compliance considerations",
        ],
    },
}


def append_deliverables(cell_source, items):
    bullets = "\n".join(f"- {item}" for item in items)
    return cell_source.rstrip() + f"\n\n**Deliverables:**\n{bullets}"


for fname, task_map in DELIVERABLES.items():
    nb = nbformat.read(fname, as_version=4)
    for idx, items in task_map.items():
        cell = nb.cells[idx]
        assert cell.cell_type == "markdown", f"{fname} cell {idx} is not markdown"
        cell.source = append_deliverables(cell.source, items)
    nbformat.write(nb, fname)
    print(f"Updated {fname}")

print("Done.")
