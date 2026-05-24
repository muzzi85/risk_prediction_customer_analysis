import shap
import pandas as pd
from feature_metadata import feature_metadata
from model_utils import *

explainer = shap.TreeExplainer(
    xgb_model
)

# ============================================================
# GENERATE SHAP EXPLANATION
# ============================================================

import pandas as pd
import shap

from model_utils import (
    xgb_model,
    application_train,
    feature_columns
)

from feature_metadata import (
    feature_metadata
)

# ============================================================
# SHAP GENERATION FUNCTION
# ============================================================

def generate_shap(customer_id):

    # --------------------------------------------------------
    # CUSTOMER DATA
    # --------------------------------------------------------

    customer = application_train[

        application_train["SK_ID_CURR"]
        ==
        customer_id
    ]

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if customer.empty:

        return pd.DataFrame({

            "error": [
                "Customer not found"
            ]
        })

    # --------------------------------------------------------
    # FEATURE MATRIX
    # --------------------------------------------------------

    X = customer[
        feature_columns
    ]

    # --------------------------------------------------------
    # SHAP VALUES
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        X
    )

    # --------------------------------------------------------
    # BUILD OUTPUT ROWS
    # --------------------------------------------------------

    shap_rows = []

    for i, feature in enumerate(feature_columns):

        # ----------------------------------------------------
        # FEATURE VALUE
        # ----------------------------------------------------

        feature_value = X.iloc[0][feature]

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        metadata = feature_metadata.get(

            feature,

            {

                "description":
                "No description available.",

                "formula":
                "N/A"
            }
        )

        # ----------------------------------------------------
        # SHAP VALUE
        # ----------------------------------------------------

        shap_value = float(
            shap_values[0][i]
        )

        # ----------------------------------------------------
        # APPEND
        # ----------------------------------------------------

        shap_rows.append({

            "feature":
            feature,

            "feature_value":
            round(
                float(feature_value),
                4
            ),

            "description":
            metadata[
                "description"
            ],

            "formula":
            metadata[
                "formula"
            ],

            "shap_value":
            round(
                shap_value,
                4
            ),

            "abs_shap":
            round(
                abs(shap_value),
                4
            )
        })

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    shap_df = pd.DataFrame(
        shap_rows
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    shap_df = shap_df.sort_values(

        by="abs_shap",

        ascending=False
    )

    # --------------------------------------------------------
    # RETURN TOP FEATURES
    # --------------------------------------------------------

    return shap_df.head(15)