from fastapi import FastAPI

app = FastAPI(

    title="Enterprise Credit Risk API",

    version="1.0"
)

@app.get("/")
def root():

    return {

        "message": "Credit Risk API Running"
    }

from model_utils import *

@app.get("/predict/{customer_id}")

def predict_customer(customer_id: int):

    customer = application_train[

        application_train["SK_ID_CURR"] == customer_id
    ]

    if customer.empty:

        return {

            "error": "Customer not found"
        }

    X = customer[
        feature_columns
    ]

    pd_score = float(

        xgb_model.predict_proba(X)[0][1]
    )

    return {

        "customer_id": customer_id,

        "probability_of_default": round(pd_score, 4)
    }

from explainability import *

@app.get("/explain/{customer_id}")

def explain_customer(customer_id: int):

    shap_df = generate_shap(
        customer_id
    )

    return {

        "customer_id": customer_id,

        "top_shap_features": shap_df.to_dict(
            orient="records"
        )
    }

from tracing import *

@app.get("/trace-events/{customer_id}")

def trace_customer(customer_id: int):

    inst = trace_installments(
        customer_id
    )

    return {

        "customer_id": customer_id,

        "installment_events": inst.to_dict(
            orient="records"
        )
    }

@app.get("/investigate/{customer_id}")

def investigate(customer_id: int):

    customer = application_train[

        application_train["SK_ID_CURR"] == customer_id
    ]

    X = customer[
        feature_columns
    ]

    pd_score = float(

        xgb_model.predict_proba(X)[0][1]
    )

    shap_df = generate_shap(
        customer_id
    )

    events = trace_installments(
        customer_id
    )

    return {

        "customer_id": customer_id,

        "probability_of_default": round(
            pd_score,
            4
        ),

        "top_shap_features": shap_df.head(5).to_dict(
            orient="records"
        ),

        "top_installment_events": events.head(10).to_dict(
            orient="records"
        )
    }
@app.get("/portfolio-summary")

def portfolio_summary():

    # ========================================================
    # BUILD FEATURES
    # ========================================================

    X = application_train[
        feature_columns
    ]

    # ========================================================
    # GENERATE PD
    # ========================================================

    application_train["PD"] = (

        xgb_model.predict_proba(X)[:,1]
    )

    # ========================================================
    # CREATE RISK SEGMENTS
    # ========================================================

    application_train["RISK_SEGMENT"] = pd.cut(

        application_train["PD"],

        bins=[0, 0.10, 0.30, 0.60, 1.0],

        labels=[

            "LOW",
            "MEDIUM",
            "HIGH",
            "VERY_HIGH"
        ]
    )

    # ========================================================
    # DEFAULT RATE
    # ========================================================

    default_rates = application_train.groupby(

        "RISK_SEGMENT"
    )["TARGET"].mean()

    # ========================================================
    # COUNTS
    # ========================================================

    segment_counts = application_train[
        "RISK_SEGMENT"
    ].value_counts()

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "default_rates": default_rates.to_dict(),

        "segment_counts": segment_counts.to_dict()
    }