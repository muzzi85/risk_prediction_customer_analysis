import pandas as pd
import joblib

# ============================================================
# LOAD MODEL
# ============================================================

xgb_model = joblib.load(

    "../models/final_credit_risk_xgboost_model.pkl"
)

# ============================================================
# LOAD DATASET
# ============================================================

application_train = pd.read_csv(

    "../data/processed/engineered_credit_risk_dataset.csv"
)

# ============================================================
# FEATURE COLUMNS
# ============================================================


feature_columns = [

    # AFFORDABILITY

    "ANNUITY_TO_INCOME",

    "CREDIT_TO_INCOME",

    "FREE_CASH_FLOW",

    "DEBT_TO_INCOME",

    "CREDIT_TO_ANNUITY_RATIO",

    "CREDIT_TO_GOODS_RATIO",

    "DOWN_PAYMENT",

    # EXT_SOURCE

    "EXT_SOURCE_1",

    "EXT_SOURCE_2",

    "EXT_SOURCE_3",

    "EXT_SOURCE_MEAN",

    "EXT_SOURCE_STD",

    "CREDIT_EXT_RATIO",

    # LEVERAGE

    "DEBT_PER_BUREAU_RECORD",

    "OVERDUE_PER_BUREAU_RECORD",

    "ACTIVE_DEBT_RATIO",

    "MEAN_DAYS_CREDIT",

    "LAST_ACTIVE_DAYS_CREDIT",

    # BORROWING

    "APPLICATIONS_PER_INCOME",

    "PREVIOUS_APPLICATION_COUNT",

    "RECENT_APPLICATION_COUNT",

    "BORROWING_ACCELERATION_RATIO",

    # REPAYMENT

    "LATE_PAYMENT_COUNT",

    "MISSED_PAYMENTS_PER_LOAN",

    "AVG_PAYMENT_DELAY",

    "AVG_PAYMENT_DEFICIT",

    "REPAYMENT_INSTABILITY",

    # TEMPORAL

    "LATE_PAYMENTS_LAST_90D",

    "AVG_PAYMENT_DELAY_LAST_90D",

    "RECENT_TO_HISTORICAL_DELAY_RATIO",

    "RECENT_DELAY_TREND",

    "RECENT_PAYMENT_DEFICIT",

    "PAYMENT_DEFICIT_TREND",

    "RECENT_PAYMENT_STABILITY",

    # EMPLOYMENT

    "YEARS_EMPLOYED",

    "EMPLOYMENT_TO_AGE_RATIO",

    "AGE_YEARS",

    # HISTORICAL CAPACITY

    "MAX_INSTALLMENT",

    "ANNUITY_TO_MAX_INSTALLMENT_RATIO",

    # COMPOSITE

    "BEHAVIORAL_RISK_SCORE"
]
