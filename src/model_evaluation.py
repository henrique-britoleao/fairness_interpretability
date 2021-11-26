import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, roc_auc_score
from xgboost import XGBClassifier


def plot_decision_tree_surrogate(X_test_prep, y_pred, model_features):
    # Train and predict with surrogate model (Decision Tree)
    clf = tree.DecisionTreeRegressor(max_depth=50)
    clf.fit(X_test_prep, y_pred)

    # Plot decision tree most important features
    fig, ax = plt.subplots(figsize=(10, 10))
    tree.plot_tree(
        clf,
        ax=ax,
        max_depth=2,
        impurity=False,
        fontsize=12,
        filled=True,
        feature_names=model_features,
    )
    return fig


def data_prep(data, transformer):
    # Get the data with predictions form the unknown model to check
    data_check = data.loc[data["y_hat"].notna()]
    data_check.drop(["CreditRisk (y)"], axis=1, inplace=True)

    y_check = data_check["y_hat"].to_numpy()
    X_check = data_check.drop(["y_hat"], axis=1)

    # Preprocessing data with one hot encoder
    preprocessing = make_pipeline(transformer)
    clf = tree.DecisionTreeRegressor(max_depth=50)

    X_check_prep = preprocessing.fit_transform(X_check)
    return X_check_prep, y_check


def train_predict_xgb(X_train_prep, y_train, X_test_prep, y_test):

    # Training our classifier with the best parameters
    param = {
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

    model = XGBClassifier(**param)

    model.fit(
        X_train_prep,
        y_train,
        eval_set=[(X_test_prep, y_test)],
        early_stopping_rounds=100,
        verbose=False,
    )

    y_pred = model.predict(X_test_prep)

    return y_pred


def get_auc_scores(test_data, X_train_prep, y_train, X_test_prep, y_test):

    # Getting score from unknow model
    y_check_pred = test_data["y_hat"].round().to_numpy()
    pre_score = roc_auc_score(y_check_pred, y_test)

    # Getting score for our model
    y_pred = train_predict_xgb(X_train_prep, y_train, X_test_prep, y_test)
    custom_score = roc_auc_score(y_test, y_pred)

    return custom_score, pre_score


def plot_compare_auc(custom_score, pre_score):

    fig, ax = plt.subplots()

    # create dataset for visualization
    performance = [round(custom_score, 2), round(pre_score, 2)]
    bars = ("Custom XGB", "Unknow Model")
    y_pos = np.arange(len(bars))

    # Create horizontal bars
    plt.barh(y_pos, performance)

    # Create names on the x-axis
    plt.yticks(y_pos, bars)
    ax.set_title("Comparing AUC of XGBoost and Unknown Model")
    ax.set_xlabel("AUC")

    for i, v in enumerate(performance):
        ax.text(v - 0.1, i, str(v), color="white", fontweight="bold")

    return fig
