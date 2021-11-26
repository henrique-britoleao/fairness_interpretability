# import libraries
import os
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import json

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from xgboost import XGBClassifier
from PIL import Image
from src.global_interpretability import plot_pdp, plot_ale, plot_heatmap
from src.model_evaluation import (
    get_auc_scores,
    plot_decision_tree_surrogate,
    data_prep,
    train_predict_xgb,
    plot_compare_auc,
)
from src.shap_plot import force_plot, summary_plot_numerical, summary_plot_categorical
from src.fpdp import (
    get_fpdp_results,
    compute_vanilla_chi_squared,
    compute_conditional_chi_squared,
)

sys.path.insert(0, "..")

st.set_option("deprecation.showPyplotGlobalUse", False)

PATH = "data/data_project.xlsx"


@st.cache()
def load_data(path=PATH):
    data = pd.read_excel(path)
    with open("config/categories.json", "r") as f:
        dic = f.read()
    # reconstructing the data as a dictionary
    js = json.loads(dic)
    # On remplace les données codées par du texte
    data = data.replace(js)
    return data


########################################Preprocessing#####################################################
# import data
data = load_data()

# set globals
TARGET_COLS = ["y_hat", "CreditRisk (y)"]
CATEGORICAL_COLS = [
    "CreditHistory",
    "EmploymentDuration",
    "Housing",
    "Purpose",
    "Savings",
]
NUMERICAL_COLS = data.loc[
    :, ~data.columns.isin(CATEGORICAL_COLS + TARGET_COLS)
].columns.tolist()

# select full data
data_full = data.copy()
train_data = data_full.loc[data_full.y_hat.isna()]
test_data = data_full.loc[data_full.y_hat.notnull()]

y_train = train_data["CreditRisk (y)"].to_numpy()
y_test = test_data["CreditRisk (y)"].to_numpy()
X_train = train_data.drop(["CreditRisk (y)", "y_hat"], axis=1)
X_test = test_data.drop(["CreditRisk (y)", "y_hat"], axis=1)
X_full = data_full.drop(["CreditRisk (y)", "y_hat"], axis=1)

transformer = make_column_transformer(
    (OneHotEncoder(), CATEGORICAL_COLS), remainder="passthrough"
)

X_train_prep = transformer.fit_transform(X_train)
X_test_prep = transformer.fit_transform(X_test)
X_full_prep = transformer.fit_transform(X_full)

clf = XGBClassifier()
clf.fit(X_train_prep, y_train)

y_pred = clf.predict(X_full_prep)
y_true = data_full.loc[:, "CreditRisk (y)"]

ONE_HOT_COLS = (
    transformer.transformers_[0][1].get_feature_names(CATEGORICAL_COLS).tolist()
)

dataset = pd.DataFrame(data=X_full_prep, columns=ONE_HOT_COLS + NUMERICAL_COLS)

dataset.columns = ONE_HOT_COLS + NUMERICAL_COLS
dataset = dataset.dropna()

model_features = ONE_HOT_COLS + NUMERICAL_COLS
##########################################################################################################


class App:
    """Class to generate a streamlit app combined all required graphs in 4 pages"""

    def __init__(self):
        self.dataset = dataset
        self.option = None

    def configure_page(self):
        """
        Configures app page
        Creates sidebar with selectbox leading to different main pages
        Returns:
            option (str): Name of main page selected by user
        """
        # create sidebar
        st.sidebar.title("Model explainability")
        option = st.sidebar.selectbox(
            "Pick Dashboard:",
            (
                "Model Performance Analysis",
                "Global interpretability",
                "Local interpretability",
                "Fairness assessment",
            ),
        )
        self.option = option

    def create_main_pages(self):
        """
        Creates pages for all options available in sidebar
        Page is loaded according to option picked in sidebar
        """
        # Model Performance Analysis
        if self.option == "Model Performance Analysis":
            st.title("Model Performance Analysis")
            st.header("Interpretation of the given model with a surrogate")
            X_check, y_check = data_prep(data, transformer=transformer)
            # Plot most import features of decision tree
            fig = plot_decision_tree_surrogate(
                X_check, y_check, model_features=model_features
            )
            st.pyplot(fig)

            st.header("Custom model building")
            col1, col2 = st.columns(2)
            col1.image(Image.open("images/xgboost_v2.png"))
            col2.json(
                {
                    "lambda": 0.38873365526971265,
                    "alpha": 0.01058538249593625,
                    "colsample_bytree": 0.5,
                    "subsample": 0.5,
                    "learning_rate": 0.018,
                    "max_depth": 9,
                    "random_state": 48,
                    "min_child_weight": 1,
                    "n_estimators": 1000,
                }
            )

            st.header("Performance comparison with unknown model")
            custom_score, pre_score = get_auc_scores(
                test_data, X_train_prep, y_train, X_test_prep, y_test
            )
            fig2 = plot_compare_auc(custom_score, pre_score)
            st.pyplot(fig2)

            st.header("Interpretation of our own model with a surrogate")
            y_prediction = train_predict_xgb(X_train_prep, y_train, X_test_prep, y_test)
            fig3 = plot_decision_tree_surrogate(
                X_test_prep, y_prediction, model_features=model_features
            )
            st.pyplot(fig3)

        # Global interpretability
        if self.option == "Global interpretability":
            st.title("Global interpretability")
            st.header("Correlation matrix")
            fig_corr = plot_heatmap(data_full[NUMERICAL_COLS])
            st.pyplot(fig_corr)
            column = st.selectbox("Variable:", NUMERICAL_COLS + CATEGORICAL_COLS)

            st.header("PDP plot")
            st.info(
                "💡The Partial Dependence Plot shows the marginal effect one or"
                " two features have on the predicted outcome of a machine"
                " learning model. A partial dependence plot can show whether"
                " the relationship between the target and a feature is linear, "
                "monotonic or more complex."
            )
            ice = st.checkbox("ICE")
            center = st.checkbox("Center")
            fig = plot_pdp(
                model=clf,
                dataset=dataset,
                model_features=model_features,
                column=column,
                ICE=ice,
                center=center,
            )
            st.pyplot(fig)

            st.header("ALE plot")
            st.info(
                "💡Accumulated local effects describe how features influence the"
                " prediction of a machine learning model on average. ALE plots "
                "are a faster and unbiased alternative to partial dependence plots "
            )
            ale_eff = plot_ale(clf, X_train_prep, X_train, transformer, column)
            st.pyplot(ale_eff)

        # Local interpretability
        if self.option == "Local interpretability":
            st.title("Local interpretability")
            st.header("Summary for numerical variables")
            fig2 = summary_plot_numerical(X_test)
            st.pyplot(fig2)

            st.header("Summary for categorical variables")
            fig3 = summary_plot_categorical(X_test)
            st.pyplot(fig3)

            st.header("Force plot")
            st.info(
                "💡One can deep dive to explore for a given account what have "
                "been the main drivers of the prediction"
            )
            index = st.number_input("Client id", min_value=0, max_value=599)
            force_plot(index, clf, X_train_prep, transformer)
            st.pyplot()

        # Model Performance Analysis
        if self.option == "Fairness assessment":
            st.title("Fairness assessment")
            st.header("Statistical parity")
            test_statistic, p_val = compute_vanilla_chi_squared(
                y_pred, data_full["Gender"]
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Chi-squared statistic", round(test_statistic, 2))
            col2.metric("P-value", round(p_val, 4))
            col3.image(traffic_light(p_val), width=150)

            st.header("Conditional statistical parity")
            test_statistic2, p_val2 = (
                test_statistic,
                p_val,
            ) = compute_conditional_chi_squared(
                [0, 1], data_full["Group"], y_pred, data_full["Gender"]
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Chi-squared statistic", round(test_statistic2, 2))
            col2.metric("P-value", round(p_val2, 4))
            col3.image(traffic_light(p_val2), width=150)

            st.header("Equal odds")
            test_statistic3, p_val3 = (
                test_statistic,
                p_val,
            ) = compute_conditional_chi_squared(
                [0, 1], y_true, y_pred, pd.Series(data_full["Gender"])
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Chi-squared statistic", round(test_statistic3, 2))
            col2.metric("P-value", round(p_val3, 4))
            col3.image(traffic_light(p_val3), width=150)

            st.header("FPDP")
            column = st.selectbox("Variable:", NUMERICAL_COLS + CATEGORICAL_COLS)
            title = f"Statistical parity for feature {column}"
            test_1 = get_fpdp_results(
                clf, dataset, model_features, column, title, group_column=y_true
            )
            st.pyplot(test_1)
            title2 = f"Conditional statistical parity for feature {column} depending on the 'group' column"
            test_2 = get_fpdp_results(
                clf,
                dataset,
                model_features,
                column,
                title2,
                group_column=dataset["Group"],
            )
            st.pyplot(test_2)
            title3 = f"Conditional statistical parity for feature {column}"
            test_3 = get_fpdp_results(clf, dataset, model_features, column, title3)
            st.pyplot(test_3)

            st.header("Correction of the model")
            TO_DROP = st.multiselect(
                "Variables to drop:", NUMERICAL_COLS + CATEGORICAL_COLS
            )
            retrain = st.checkbox("Retrain", value=False)
            if retrain:
                TO_DROP = 1


def traffic_light(p_val: float):
    if p_val < 0.05:
        return Image.open("images/feux-feurouge.jpg")
    else:
        return Image.open("images/feux-feuvert.jpg")
