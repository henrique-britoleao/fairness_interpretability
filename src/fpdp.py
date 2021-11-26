from pdpbox.pdp import PDPIsolate
from pdpbox.pdp_calc_utils import _calc_ice_lines, _prepare_pdp_count_data
from pdpbox.utils import (
    _check_model,
    _check_dataset,
    _check_percentile_range,
    _check_feature,
    _check_grid_type,
    _check_memory_limit,
    _make_list,
    _calc_memory_usage,
    _get_grids,
    _get_string,
)
from scipy.stats import chi2_contingency, chi2

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from joblib import Parallel, delayed

import warnings

warnings.filterwarnings("ignore")


def fpdp_isolate(
    model,
    dataset,
    model_features,
    feature,
    num_grid_points=10,
    grid_type="percentile",
    percentile_range=None,
    grid_range=None,
    cust_grid_points=None,
    memory_limit=0.5,
    n_jobs=1,
    predict_kwds={},
    data_transformer=None,
):
    """Calculate PDP isolation plot
    Parameters
    ----------
    model: a fitted sklearn model
    dataset: pandas DataFrame
        data set on which the model is trained
    model_features: list or 1-d array
        list of model features
    feature: string or list
        feature or feature list to investigate,
        for one-hot encoding features, feature list is required
    num_grid_points: integer, optional, default=10
        number of grid points for numeric feature
    grid_type: string, optional, default='percentile'
        'percentile' or 'equal',
        type of grid points for numeric feature
    percentile_range: tuple or None, optional, default=None
        percentile range to investigate,
        for numeric feature when grid_type='percentile'
    grid_range: tuple or None, optional, default=None
        value range to investigate,
        for numeric feature when grid_type='equal'
    cust_grid_points: Series, 1d-array, list or None, optional, default=None
        customized list of grid points for numeric feature
    memory_limit: float, (0, 1)
        fraction of memory to use
    n_jobs: integer, default=1
        number of jobs to run in parallel.
        make sure n_jobs=1 when you are using XGBoost model.
        check:
        1. https://pythonhosted.org/joblib/parallel.html#bad-interaction-of-multiprocessing-and-third-party-libraries
        2. https://github.com/scikit-learn/scikit-learn/issues/6627
    predict_kwds: dict, optional, default={}
        keywords to be passed to the model's predict function
    data_transformer: function or None, optional, default=None
        function to transform the data set as some features changing values
    Returns
    -------
    pdp_isolate_out: instance of PDPIsolate
    """

    # check function inputs
    n_classes, predict = _check_model(model=model)

    # avoid polluting the original dataset
    # copy training data set and get the model features
    # it's extremely important to keep the original feature order
    _check_dataset(df=dataset)
    _dataset = dataset.copy()

    feature_type = _check_feature(feature=feature, df=_dataset)
    _check_grid_type(grid_type=grid_type)
    _check_percentile_range(percentile_range=percentile_range)
    _check_memory_limit(memory_limit=memory_limit)

    # feature_grids: grid points to calculate on
    # display_columns: xticklabels for grid points
    percentile_info = []
    if feature_type == "binary":
        feature_grids = np.array([0, 1])
        display_columns = ["%s_0" % feature, "%s_1" % feature]
    elif feature_type == "onehot":
        feature_grids = np.array(feature)
        display_columns = feature
    else:
        # calculate grid points for numeric features
        if cust_grid_points is None:
            feature_grids, percentile_info = _get_grids(
                feature_values=_dataset[feature].values,
                num_grid_points=num_grid_points,
                grid_type=grid_type,
                percentile_range=percentile_range,
                grid_range=grid_range,
            )
        else:
            # make sure grid points are unique and in ascending order
            feature_grids = np.array(sorted(np.unique(cust_grid_points)))
        display_columns = [_get_string(v) for v in feature_grids]

    # Parallel calculate ICE lines
    true_n_jobs = _calc_memory_usage(
        df=_dataset,
        total_units=len(feature_grids),
        n_jobs=n_jobs,
        memory_limit=memory_limit,
    )
    grid_results = Parallel(n_jobs=true_n_jobs)(
        delayed(_calc_ice_lines)(
            feature_grid,
            data=_dataset,
            model=model,
            model_features=model_features,
            n_classes=n_classes,
            feature=feature,
            feature_type=feature_type,
            predict_kwds=predict_kwds,
            data_transformer=data_transformer,
        )
        for feature_grid in feature_grids
    )

    if n_classes > 2:
        ice_lines = []
        for n_class in range(n_classes):
            ice_line_n_class = pd.concat(
                [grid_result[n_class] for grid_result in grid_results], axis=1
            )
            ice_lines.append(ice_line_n_class)
    else:
        ice_lines = pd.concat(grid_results, axis=1)

    return ice_lines

    # # calculate the counts
    # count_data = _prepare_pdp_count_data(
    #     feature=feature, feature_type=feature_type, data=_dataset[_make_list(feature)], feature_grids=feature_grids)

    # # prepare histogram information for numeric feature
    # hist_data = None
    # if feature_type == 'numeric':
    #     hist_data = _dataset[feature].values

    # # combine the final results
    # pdp_params = {'n_classes': n_classes, 'feature': feature, 'feature_type': feature_type,
    #               'feature_grids': feature_grids, 'percentile_info': percentile_info,
    #               'display_columns': display_columns, 'count_data': count_data, 'hist_data': hist_data}
    # if n_classes > 2:
    #     pdp_isolate_out = []
    #     for n_class in range(n_classes):
    #         pdp = ice_lines[n_class][feature_grids].mean().values
    #         pdp_isolate_out.append(
    #             PDPIsolate(which_class=n_class, ice_lines=ice_lines[n_class], pdp=pdp, **pdp_params))
    # else:
    #     pdp = ice_lines[feature_grids].mean().values
    #     pdp_isolate_out = PDPIsolate(which_class=None, ice_lines=ice_lines, pdp=pdp, **pdp_params)

    # return pdp_isolate_out


def get_fpdp_results(
    model,
    dataset: pd.DataFrame,
    model_features: list,
    column: str,
    title: str,
    group_column=None,
):
    cols = [feature for feature in dataset.columns if feature.split("_")[0] == column]
    if len(cols) == 1:
        feature = column
    else:
        feature = cols
    pdp_fare = fpdp_isolate(
        model=model, dataset=dataset, model_features=model_features, feature=feature
    )

    results = pd.DataFrame(columns=["p_val"])

    for col in pdp_fare.columns:
        preds = pdp_fare[col].apply(lambda x: 0 if x < 0.5 else 1)

        if group_column is not None:
            results.loc[col, "p_val"] = compute_conditional_chi_squared(
                groups=[0, 1],
                group_column=group_column,
                preds=preds,
                fairness_column=dataset["Gender"],
            )[1]
        else:
            results.loc[col, "p_val"] = compute_vanilla_chi_squared(
                preds=preds, fairness_column=dataset["Gender"]
            )[1]

    fig, ax = plt.subplots(figsize=(8, 5))
    results.plot(title=f"FPDP plot for column {column}", ax=ax)
    ax.axhline(y=0.05, color="r", linestyle="-")
    plt.xticks(range(len(cols)), rotation=45)
    ax.set_title(title)
    ax.set_xticklabels(cols)
    return fig


def compute_vanilla_chi_squared(
    preds: np.ndarray, fairness_column: pd.Series
):
    contingency_table = pd.crosstab(preds, fairness_column)
    test_statistic, p_val, _, _ = chi2_contingency(contingency_table)

    return test_statistic, p_val


def compute_conditional_chi_squared(
    groups: list[int],
    group_column: pd.Series,
    preds: np.ndarray,
    fairness_column: pd.Series,
):
    # initialize test_statistic result
    test_statistic = 0

    for group in groups:
        # select indexes of corresponding group
        group_idx = np.where(group_column == group)[0]
        group_preds = preds[group_idx]

        contingency_table = pd.crosstab(group_preds, fairness_column[group_idx])
        chi_sq_stats, _, _, _ = chi2_contingency(contingency_table)
        test_statistic += chi_sq_stats

    # calculate the p-value
    p_val = chi2.pdf(test_statistic, 2)

    return test_statistic, p_val
