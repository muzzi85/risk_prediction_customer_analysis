import streamlit as st

st.set_page_config(

    page_title="Enterprise Credit Risk AI",

    layout="wide"
)

st.title(

    "Enterprise Credit Risk Intelligence Platform"
)

st.write(

    "AI-driven credit risk investigation dashboard."
)

import requests

st.sidebar.header(

    "Customer Investigation"
)

customer_id = st.sidebar.number_input(

    "Enter Customer ID",

    value=100020
)

predict_url = (

    f"http://127.0.0.1:8000/predict/{customer_id}"
)

predict_response = requests.get(
    predict_url
)

predict_data = predict_response.json()

st.subheader(

    "Probability of Default"
)

st.metric(

    label="PD",

    value=round(

        predict_data[
            "probability_of_default"
        ],

        4
    )
)

explain_url = (

    f"http://127.0.0.1:8000/explain/{customer_id}"
)

explain_response = requests.get(
    explain_url
)

explain_data = explain_response.json()

import pandas as pd

shap_df = pd.DataFrame(

    explain_data[
        "top_shap_features"
    ]
)

st.subheader(

    "Top SHAP Risk Drivers"
)

st.dataframe(shap_df)

st.bar_chart(

    shap_df.set_index(
        "feature"
    )["shap_value"]
)

events_url = (

    f"http://127.0.0.1:8000/trace-events/{customer_id}"
)

events_response = requests.get(
    events_url
)

events_data = events_response.json()

events_df = pd.DataFrame(

    events_data[
        "installment_events"
    ]
)

st.subheader(

    "Installment Payment Events"
)

st.dataframe(events_df)

portfolio_url = (

    "http://127.0.0.1:8000/portfolio-summary"
)

portfolio_response = requests.get(
    portfolio_url
)

portfolio_data = portfolio_response.json()

st.subheader(

    "Portfolio Risk Segments"
)

segment_df = pd.DataFrame({

    "Segment": list(

        portfolio_data[
            "segment_counts"
        ].keys()
    ),

    "Count": list(

        portfolio_data[
            "segment_counts"
        ].values()
    )
})

st.dataframe(segment_df)

st.bar_chart(

    segment_df.set_index(
        "Segment"
    )["Count"]
)

default_df = pd.DataFrame({

    "Segment": list(

        portfolio_data[
            "default_rates"
        ].keys()
    ),

    "Default_Rate": list(

        portfolio_data[
            "default_rates"
        ].values()
    )
})

st.subheader(

    "Observed Default Rates"
)

st.dataframe(default_df)

st.bar_chart(

    default_df.set_index(
        "Segment"
    )["Default_Rate"]
)

