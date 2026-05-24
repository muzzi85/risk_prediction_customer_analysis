import pandas as pd

def trace_installments(customer_id):

    installments = pd.read_csv(

        "../data/raw/installments_payments.csv"
    )

    inst = installments[

        installments["SK_ID_CURR"] == customer_id
    ].copy()

    inst["PAYMENT_DELAY"] = (

        inst["DAYS_ENTRY_PAYMENT"]
        -
        inst["DAYS_INSTALMENT"]
    )

    inst["PAYMENT_DEFICIT"] = (

        inst["AMT_INSTALMENT"]
        -
        inst["AMT_PAYMENT"]
    )

    inst = inst.sort_values(

        by="PAYMENT_DELAY",

        ascending=False
    )

    return inst.head(20)