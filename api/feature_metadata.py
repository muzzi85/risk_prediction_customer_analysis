feature_metadata = {

    "DEBT_TO_INCOME": {

        "description":
        "Measures customer leverage relative to income.",

        "formula":
        "AMT_CREDIT / AMT_INCOME_TOTAL"
    },

    "ACTIVE_DEBT_RATIO": {

        "description":
        "Ratio of active debt compared to total bureau debt.",

        "formula":
        "ACTIVE_EXTERNAL_DEBT / TOTAL_EXTERNAL_DEBT"
    },

    "PREVIOUS_APPLICATION_COUNT": {

        "description":
        "Total historical credit applications.",

        "formula":
        "COUNT(SK_ID_PREV)"
    },

    "AVG_PAYMENT_DELAY": {

        "description":
        "Average late payment days across installments.",

        "formula":
        "MEAN(MAX(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0))"
    },

    "LATE_PAYMENT_COUNT": {

        "description":
        "Number of installments paid late.",

        "formula":
        "COUNT(PAYMENT_DELAY > 0)"
    },

    "MISSED_PAYMENTS_PER_LOAN": {

        "description":
        "Average missed or underpaid installments per loan.",

        "formula":
        "TOTAL_MISSED_PAYMENTS / TOTAL_LOANS"
    },

    "BORROWING_ACCELERATION_RATIO": {

        "description":
        "Measures recent borrowing growth intensity.",

        "formula":
        "RECENT_APPLICATION_COUNT / PREVIOUS_APPLICATION_COUNT"
    },

    "EXT_SOURCE_MEAN": {

        "description":
        "Average external bureau risk score.",

        "formula":
        "MEAN(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)"
    },

    "BEHAVIORAL_RISK_SCORE": {

        "description":
        "Composite behavioral delinquency indicator.",

        "formula":
        "Weighted combination of repayment risk metrics"
    }
}