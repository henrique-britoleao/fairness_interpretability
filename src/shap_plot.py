import sys
import pandas as pd
import matplotlib.pyplot as plt
import shap  # package used to calculate Shap values
import pandas as pd

sys.path.insert(0, "..")


def load_shap_data():
    df_categorical = pd.read_csv("data/df_categorical_sum_shap.csv")
    df_shap_numerical = pd.read_csv("data/df_shap_numerical.csv")
    X_test_shap_good_columns = pd.read_csv("data/X_test_shap_good_columns.csv")
    df_shap = pd.read_csv("data/df_shap.csv")
    return df_categorical, df_shap_numerical, X_test_shap_good_columns, df_shap


df_categorical, df_shap_numerical, X_test_shap_good_columns, df_shap = load_shap_data()


def force_plot(customer_index, model, X_train_prep, transformer):
    explainer = shap.TreeExplainer(
        model, pd.DataFrame(X_train_prep, columns=transformer.get_feature_names())
    )
    fig, ax = plt.subplots()
    return shap.force_plot(
        explainer.expected_value,
        df_shap.values[customer_index],
        X_test_shap_good_columns.iloc[customer_index, :],
        link="logit",
        matplotlib=True,
    )


def summary_plot_numerical(X_test):
    fig1, ax1 = plt.subplots()
    shap.summary_plot(df_shap_numerical.values, X_test[df_shap_numerical.columns])
    return fig1


def summary_plot_categorical(X_test):
    fig2, ax2 = plt.subplots()
    shap.summary_plot(
        df_categorical.values,
        X_test[
            [
                "CreditHistory",
                "EmploymentDuration",
                "Housing",
                "Purpose",
                "Savings",
                "Group",
                "Gender",
            ]
        ],
    )
    return fig2
