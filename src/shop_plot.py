import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import shap  # package used to calculate Shap values

# import data
data = pd.read_excel("../data/data_project.xlsx")

# set globals
CATEGORICAL_COLS = [
    "CreditHistory",
    "EmploymentDuration",
    "Housing",
    "Purpose",
    "Savings",
    "Group",
    "Gender",
]
NUMERICAL_COLS = data.loc[:, ~data.columns.isin(CATEGORICAL_COLS)].columns


# select full data
data_full = data.copy()
train_data = data_full.loc[data_full.y_hat.isna()]
test_data = data_full.loc[data_full.y_hat.notnull()]

y_train = train_data["CreditRisk (y)"].to_numpy()
y_test = test_data["CreditRisk (y)"].to_numpy()
X_train = train_data.drop(["CreditRisk (y)", "y_hat"], axis=1)
X_test = test_data.drop(["CreditRisk (y)", "y_hat"], axis=1)

transformer = make_column_transformer(
    (OneHotEncoder(), CATEGORICAL_COLS), remainder="passthrough"
)

X_train_prep = transformer.fit_transform(X_train)

reg = XGBRegressor()
reg.fit(X_train_prep, y_train)


def plotting_shap():

    explainer = shap.TreeExplainer(
        reg, pd.DataFrame(X_train_prep, columns=transformer.get_feature_names())
    )

    # calculate shap values. This is what we will plot.
    # Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
    shap_values = explainer.shap_values(
        pd.DataFrame(
            transformer.fit_transform(X_test), columns=transformer.get_feature_names()
        )
    )

    df_shap = pd.DataFrame(shap_values, columns=transformer.get_feature_names())
    features = df_shap.columns
    df_shap = df_shap[
        [
            "CreditAmount",
            "CreditDuration",
            "Age",
            "onehotencoder__x3_A40",
            "onehotencoder__x0_A34",
            "onehotencoder__x4_A61",
            "InstallmentRate",
            "onehotencoder__x2_A151",
            "onehotencoder__x3_A43",
            "onehotencoder__x2_A152",
        ]
    ].rename(
        columns={
            "onehotencoder__x3_A40": "Purpose buy is a car",
            "onehotencoder__x0_A34": "Other credits existing",
            "onehotencoder__x4_A61": "Savings inferior to 100",
            "onehotencoder__x2_A151": "Home is rented",
            "onehotencoder__x3_A43": "Purpose of buy is radio/television",
            "onehotencoder__x2_A152": "The home is a possession",
        }
    )

    X_test_shap_good_columns = (
        pd.DataFrame(
            transformer.fit_transform(X_test), columns=transformer.get_feature_names()
        )
        .loc[:, features][
            [
                "CreditAmount",
                "CreditDuration",
                "Age",
                "onehotencoder__x3_A40",
                "onehotencoder__x0_A34",
                "onehotencoder__x4_A61",
                "InstallmentRate",
                "onehotencoder__x2_A151",
                "onehotencoder__x3_A43",
                "onehotencoder__x2_A152",
            ]
        ]
        .rename(
            columns={
                "onehotencoder__x3_A40": "For buying a car",
                "onehotencoder__x0_A34": "Other credits existing",
                "onehotencoder__x4_A61": "Savings < 100",
                "onehotencoder__x2_A151": "Home is rented",
                "onehotencoder__x3_A43": "For buying radio/TV",
                "onehotencoder__x2_A152": "Home owner",
            }
        )
    )

    # Make plot. Index of [1] is explained in text below.
    shap.summary_plot(df_shap.values, X_test_shap_good_columns)


def shap_force_plot():

    index = int(input("Enter a number between 0 and 599 (the length of samples)"))

    explainer = shap.TreeExplainer(
        reg, pd.DataFrame(X_train_prep, columns=transformer.get_feature_names())
    )

    # calculate shap values. This is what we will plot.
    # Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
    shap_values = explainer.shap_values(
        pd.DataFrame(
            transformer.fit_transform(X_test), columns=transformer.get_feature_names()
        )
    )

    df_shap = pd.DataFrame(shap_values, columns=transformer.get_feature_names())
    features = df_shap.columns
    df_shap = df_shap[
        [
            "CreditAmount",
            "CreditDuration",
            "Age",
            "onehotencoder__x3_A40",
            "onehotencoder__x0_A34",
            "onehotencoder__x4_A61",
            "InstallmentRate",
            "onehotencoder__x2_A151",
            "onehotencoder__x3_A43",
            "onehotencoder__x2_A152",
        ]
    ].rename(
        columns={
            "onehotencoder__x3_A40": "Purpose buy is a car",
            "onehotencoder__x0_A34": "Other credits existing",
            "onehotencoder__x4_A61": "Savings inferior to 100",
            "onehotencoder__x2_A151": "Home is rented",
            "onehotencoder__x3_A43": "Purpose of buy is radio/television",
            "onehotencoder__x2_A152": "The home is a possession",
        }
    )

    X_test_shap_good_columns = (
        pd.DataFrame(
            transformer.fit_transform(X_test), columns=transformer.get_feature_names()
        )
        .loc[:, features][
            [
                "CreditAmount",
                "CreditDuration",
                "Age",
                "onehotencoder__x3_A40",
                "onehotencoder__x0_A34",
                "onehotencoder__x4_A61",
                "InstallmentRate",
                "onehotencoder__x2_A151",
                "onehotencoder__x3_A43",
                "onehotencoder__x2_A152",
            ]
        ]
        .rename(
            columns={
                "onehotencoder__x3_A40": "For buying a car",
                "onehotencoder__x0_A34": "Other credits existing",
                "onehotencoder__x4_A61": "Savings < 100",
                "onehotencoder__x2_A151": "Home is rented",
                "onehotencoder__x3_A43": "For buying radio/TV",
                "onehotencoder__x2_A152": "Home owner",
            }
        )
    )

    shap_plot = shap.force_plot(
        explainer.expected_value,
        df_shap.values[index],
        features=X_test_shap_good_columns.iloc[-1:],
        feature_names=X_test_shap_good_columns.columns,
        matplotlib=True,
        show=False,
        plot_cmap=["#77dd77", "#f99191"],
    )
