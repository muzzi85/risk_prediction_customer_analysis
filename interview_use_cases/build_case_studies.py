"""Builds three self-contained training CSVs, one per case study:

  1. case1_credit_approval.csv   - PD model: should we approve this customer?
  2. case2_early_warning.csv     - predict deterioration risk segment for existing customers
  3. case3_collections.csv       - recovery priority score/tier for defaulted customers

Each behavioural table (POS_CASH_balance, credit_card_balance, installments_payments) is
split into an EARLY window (>= 7 months ago) and a RECENT window (last 6 months).
Case 2's label is derived from the RECENT window + final TARGET, while its FEATURES come
only from the EARLY window + application/bureau data - i.e. "using what we knew a year ago,
flag who is about to deteriorate." Only EARLY-window features are kept in the output CSV.

Run once: python3 build_case_studies.py
"""
import gc
import numpy as np
import pandas as pd

RAW = "../data/raw"
OUT = "data"

import os
os.makedirs(OUT, exist_ok=True)

RECENT_MONTHS = 6          # MONTHS_BALANCE >= -6  -> "recent" window
RECENT_DAYS = RECENT_MONTHS * 30  # for installments' DAYS_INSTALMENT

# ---------------------------------------------------------------------------
# 1. Application (base table for all 3 case studies)
# ---------------------------------------------------------------------------
print("Loading application_train.csv ...")
app_cols = [
    "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY", "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "AMT_GOODS_PRICE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "OWN_CAR_AGE", "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT", "ORGANIZATION_TYPE",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
]
app = pd.read_csv(f"{RAW}/application_train.csv", usecols=app_cols)
print(f"  -> {len(app)} rows")

# ---------------------------------------------------------------------------
# 2. Bureau aggregates (External layer) - used by all 3 case studies
# ---------------------------------------------------------------------------
print("Loading bureau.csv ...")
bureau = pd.read_csv(f"{RAW}/bureau.csv")

bureau_agg = bureau.groupby("SK_ID_CURR").agg(
    BUREAU_COUNT=("SK_ID_BUREAU", "count"),
    BUREAU_ACTIVE_COUNT=("CREDIT_ACTIVE", lambda s: (s == "Active").sum()),
    BUREAU_AMT_CREDIT_SUM_TOTAL=("AMT_CREDIT_SUM", "sum"),
    BUREAU_AMT_CREDIT_SUM_DEBT_TOTAL=("AMT_CREDIT_SUM_DEBT", "sum"),
    BUREAU_AMT_CREDIT_SUM_OVERDUE_TOTAL=("AMT_CREDIT_SUM_OVERDUE", "sum"),
    BUREAU_CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
    BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
).reset_index()
print(f"  -> aggregated to {len(bureau_agg)} customers")
del bureau
gc.collect()

# ---------------------------------------------------------------------------
# 3. POS_CASH_balance: early vs recent window aggregates
# ---------------------------------------------------------------------------
print("Loading POS_CASH_balance.csv ...")
pos = pd.read_csv(f"{RAW}/POS_CASH_balance.csv")

pos_early = pos[pos["MONTHS_BALANCE"] < -RECENT_MONTHS]
pos_recent = pos[pos["MONTHS_BALANCE"] >= -RECENT_MONTHS]

pos_early_agg = pos_early.groupby("SK_ID_CURR").agg(
    POS_EARLY_NUM_RECORDS=("SK_ID_PREV", "count"),
    POS_EARLY_MAX_DPD=("SK_DPD", "max"),
    POS_EARLY_MEAN_DPD=("SK_DPD", "mean"),
    POS_EARLY_MAX_DPD_DEF=("SK_DPD_DEF", "max"),
    POS_EARLY_PCT_DPD_GT0=("SK_DPD", lambda s: (s > 0).mean()),
).reset_index()

pos_recent_agg = pos_recent.groupby("SK_ID_CURR").agg(
    POS_RECENT_MAX_DPD_DEF=("SK_DPD_DEF", "max"),
    POS_RECENT_PCT_DPD_GT0=("SK_DPD", lambda s: (s > 0).mean()),
).reset_index()

print(f"  -> early: {len(pos_early_agg)}, recent: {len(pos_recent_agg)} customers")
del pos, pos_early, pos_recent
gc.collect()

# ---------------------------------------------------------------------------
# 4. credit_card_balance: early vs recent window aggregates
# ---------------------------------------------------------------------------
print("Loading credit_card_balance.csv ...")
cc = pd.read_csv(f"{RAW}/credit_card_balance.csv")
cc["UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)

cc_early = cc[cc["MONTHS_BALANCE"] < -RECENT_MONTHS]
cc_recent = cc[cc["MONTHS_BALANCE"] >= -RECENT_MONTHS]

cc_early_agg = cc_early.groupby("SK_ID_CURR").agg(
    CC_EARLY_MAX_UTILIZATION=("UTILIZATION", "max"),
    CC_EARLY_MEAN_UTILIZATION=("UTILIZATION", "mean"),
    CC_EARLY_MEAN_DRAWINGS=("AMT_DRAWINGS_CURRENT", "mean"),
    CC_EARLY_MAX_DPD_DEF=("SK_DPD_DEF", "max"),
).reset_index()

cc_recent_agg = cc_recent.groupby("SK_ID_CURR").agg(
    CC_RECENT_MAX_DPD_DEF=("SK_DPD_DEF", "max"),
    CC_RECENT_MAX_UTILIZATION=("UTILIZATION", "max"),
).reset_index()

print(f"  -> early: {len(cc_early_agg)}, recent: {len(cc_recent_agg)} customers")
del cc, cc_early, cc_recent
gc.collect()

# ---------------------------------------------------------------------------
# 5. installments_payments: early vs recent window aggregates
# ---------------------------------------------------------------------------
print("Loading installments_payments.csv ...")
inst = pd.read_csv(f"{RAW}/installments_payments.csv")
inst["DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
inst["PAYMENT_RATIO"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"].replace(0, np.nan)

inst_early = inst[inst["DAYS_INSTALMENT"] < -RECENT_DAYS]
inst_recent = inst[inst["DAYS_INSTALMENT"] >= -RECENT_DAYS]

inst_early_agg = inst_early.groupby("SK_ID_CURR").agg(
    INSTAL_EARLY_NUM_PAYMENTS=("NUM_INSTALMENT_NUMBER", "count"),
    INSTAL_EARLY_PCT_LATE=("DAYS_LATE", lambda s: (s > 0).mean()),
    INSTAL_EARLY_MEAN_DAYS_LATE=("DAYS_LATE", "mean"),
    INSTAL_EARLY_MEAN_PAYMENT_RATIO=("PAYMENT_RATIO", "mean"),
).reset_index()

inst_recent_agg = inst_recent.groupby("SK_ID_CURR").agg(
    INSTAL_RECENT_PCT_LATE=("DAYS_LATE", lambda s: (s > 0).mean()),
    INSTAL_RECENT_MAX_DAYS_LATE=("DAYS_LATE", "max"),
).reset_index()

print(f"  -> early: {len(inst_early_agg)}, recent: {len(inst_recent_agg)} customers")
del inst, inst_early, inst_recent
gc.collect()

# ---------------------------------------------------------------------------
# 6. Full-history aggregates (used only for Case 3 - collections)
# ---------------------------------------------------------------------------
print("Computing full-history behavioural severity (Case 3) ...")
print("  re-reading POS_CASH_balance.csv and installments_payments.csv ...")
pos_full = pd.read_csv(f"{RAW}/POS_CASH_balance.csv", usecols=["SK_ID_CURR", "SK_DPD_DEF"])
pos_full_agg = pos_full.groupby("SK_ID_CURR")["SK_DPD_DEF"].max().rename("POS_FULL_MAX_DPD_DEF").reset_index()
del pos_full
gc.collect()

cc_full = pd.read_csv(f"{RAW}/credit_card_balance.csv", usecols=["SK_ID_CURR", "SK_DPD_DEF"])
cc_full_agg = cc_full.groupby("SK_ID_CURR")["SK_DPD_DEF"].max().rename("CC_FULL_MAX_DPD_DEF").reset_index()
del cc_full
gc.collect()

inst_full = pd.read_csv(f"{RAW}/installments_payments.csv", usecols=["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"])
inst_full["DAYS_LATE"] = inst_full["DAYS_ENTRY_PAYMENT"] - inst_full["DAYS_INSTALMENT"]
inst_full_agg = inst_full.groupby("SK_ID_CURR").agg(
    INSTAL_FULL_PCT_LATE=("DAYS_LATE", lambda s: (s > 0).mean()),
    INSTAL_FULL_MAX_DAYS_LATE=("DAYS_LATE", "max"),
).reset_index()
del inst_full
gc.collect()


def merge_all(base, *aggs):
    out = base.copy()
    for a in aggs:
        out = out.merge(a, on="SK_ID_CURR", how="left")
    return out


# ---------------------------------------------------------------------------
# Case 1 - Credit Approval (PD)
# population: all applicants. Features: application + bureau (External layer)
# available at application time. Target: TARGET.
# ---------------------------------------------------------------------------
print("\nBuilding case1_credit_approval.csv ...")
case1 = merge_all(app, bureau_agg)
case1.to_csv(f"{OUT}/case1_credit_approval.csv.gz", index=False, compression="gzip")
print(f"  -> {case1.shape}")

# ---------------------------------------------------------------------------
# Case 2 - Early Warning System
# population: customers with behavioural history (an existing/serviced loan).
# Features: application + bureau + EARLY-window behavioural aggregates only
# (everything we knew ~7+ months ago). Target: EARLY_WARNING_SEGMENT, derived
# from the RECENT window severity + final TARGET (what actually happened next).
# ---------------------------------------------------------------------------
print("Building case2_early_warning.csv ...")
case2 = merge_all(app, bureau_agg, pos_early_agg, cc_early_agg, inst_early_agg,
                   pos_recent_agg, cc_recent_agg, inst_recent_agg)

# population = customers with at least one behavioural record (early or recent)
has_history = (
    case2["POS_EARLY_NUM_RECORDS"].notna() | case2["INSTAL_EARLY_NUM_PAYMENTS"].notna()
    | case2["POS_RECENT_MAX_DPD_DEF"].notna() | case2["INSTAL_RECENT_PCT_LATE"].notna()
)
case2 = case2[has_history].copy()

recent_dpd = case2[["POS_RECENT_MAX_DPD_DEF", "CC_RECENT_MAX_DPD_DEF"]].max(axis=1).fillna(0)
recent_late_pct = case2["INSTAL_RECENT_PCT_LATE"].fillna(0)


def segment(row_target, dpd, late_pct):
    if row_target == 1 or dpd >= 60:
        return "Immediate Attention"
    if dpd >= 30 or late_pct >= 0.5:
        return "High Risk"
    if dpd >= 1 or late_pct >= 0.15:
        return "Medium Risk"
    return "Low Risk"


case2["EARLY_WARNING_SEGMENT"] = [
    segment(t, d, l) for t, d, l in zip(case2["TARGET"], recent_dpd, recent_late_pct)
]

# Drop columns that were only used to derive the label (avoid leakage) and TARGET
recent_cols = ["POS_RECENT_MAX_DPD_DEF", "POS_RECENT_PCT_DPD_GT0",
                "CC_RECENT_MAX_DPD_DEF", "CC_RECENT_MAX_UTILIZATION",
                "INSTAL_RECENT_PCT_LATE", "INSTAL_RECENT_MAX_DAYS_LATE"]
case2 = case2.drop(columns=recent_cols + ["TARGET"])

case2.to_csv(f"{OUT}/case2_early_warning.csv.gz", index=False, compression="gzip")
print(f"  -> {case2.shape}")
print(case2["EARLY_WARNING_SEGMENT"].value_counts())

# ---------------------------------------------------------------------------
# Case 3 - Collections Prioritization
# population: customers who defaulted (TARGET == 1) - the collections pool.
# Features: application + bureau + full-history behavioural severity + exposure.
# Target: RECOVERY_PRIORITY_SCORE (0-100, continuous) and PRIORITY_TIER (4-class),
# built from exposure, behavioural severity, and lateness percentiles.
# ---------------------------------------------------------------------------
print("Building case3_collections.csv ...")
case3 = merge_all(app, bureau_agg, pos_full_agg, cc_full_agg, inst_full_agg)
case3 = case3[case3["TARGET"] == 1].copy()

case3["EXPOSURE"] = case3["AMT_CREDIT"].fillna(0) + case3["BUREAU_AMT_CREDIT_SUM_DEBT_TOTAL"].fillna(0)
case3["DPD_SEVERITY"] = case3[["POS_FULL_MAX_DPD_DEF", "CC_FULL_MAX_DPD_DEF"]].max(axis=1).fillna(0)
case3["LATE_PCT"] = case3["INSTAL_FULL_PCT_LATE"].fillna(0)

exposure_pct = case3["EXPOSURE"].rank(pct=True)
dpd_pct = case3["DPD_SEVERITY"].rank(pct=True)
late_pct_rank = case3["LATE_PCT"].rank(pct=True)

case3["RECOVERY_PRIORITY_SCORE"] = (
    100 * (0.40 * exposure_pct + 0.35 * dpd_pct + 0.25 * late_pct_rank)
).round(1)

case3["PRIORITY_TIER"] = pd.qcut(
    case3["RECOVERY_PRIORITY_SCORE"], 4,
    labels=["Tier 4 - Low", "Tier 3 - Medium", "Tier 2 - High", "Tier 1 - Immediate"],
)

case3 = case3.drop(columns=["TARGET"])
case3.to_csv(f"{OUT}/case3_collections.csv.gz", index=False, compression="gzip")
print(f"  -> {case3.shape}")
print(case3["PRIORITY_TIER"].value_counts())

print("\nDone.")
