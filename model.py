import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from xgboost import XGBClassifier

# Typing and error catching
from functools import partial
from typing import Tuple, Any, Union

MODEL_TYPE = Union[LogisticRegression, RandomForestClassifier, XGBClassifier]


def features_target_split(
    df: pd.DataFrame, col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split features and target values

    Input:
    data (pd.DataFrame) : input DataFrame with features and target values
    col (str) : name of the target value column

    Output:
    X (pd.DataFrame): DataFrame with features
    y (pd.DataFrame): DataFrame with target values
    """
    X = df.drop(columns=[col])
    y = df[[col]]

    return X, y


def train_test_split(
    df: pd.DataFrame, col: str, ratio: float = 0.75
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Split inputs into 2 sets of data: training (train) and test (test).
    Each set of data is splitted into features (X) and target values (y).

    Input:
    data (pd.DataFrame) : input DataFrame with features and target values
    col (str) : name of the target value column
    ratio (float) : split ratio, between 0 and 1, to split train and validation data

    Output:
    X_train (pd.DataFrame): DataFrame for training with features
    y_train (pd.DataFrame): DataFrame for training with target values
    X_test (pd.DataFrame): DataFrame for testing with features
    y_test (pd.DataFrame): DataFrame for testing with target values
    """
    # spint train and test sets
    df = df.sort_values(by=["DISCOVERY_DATE"])
    index_ratio = int(ratio * df.shape[0])  # find the row number where we want to split
    split_date = df.iloc[
        index_ratio, df.columns.get_loc("DISCOVERY_DATE")
    ]  # find the corresponding data
    data_train = (
        df[df["DISCOVERY_DATE"] < split_date]
        .set_index(["DISCOVERY_DATE", "STATE"])
        .copy()
    )
    data_test = (
        df[df["DISCOVERY_DATE"] >= split_date]
        .set_index(["DISCOVERY_DATE", "STATE"])
        .copy()
    )

    # split between features and target values
    X_train, y_train = features_target_split(data_train, col)
    X_test, y_test = features_target_split(data_test, col)

    return X_train, y_train, X_test, y_test


def model_fit_predict(
    model: MODEL_TYPE, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame
) -> Tuple[pd.DataFrame, Any]:
    """
    Create a model, fit it on X_train and y_train, and predict the target values from X_val

    Input:
    model (MODEL_TYPE) : The model to fit and to then use for predictions on the validation set
    X_train (pd.DataFrame) : input DataFrame with features for training
    y_train (pd.DataFrame) : input DataFrame with target values for training
    X_val (pd.DataFrame) : input DataFrame with features for validation

    Output:
    y_pre (pd.DataFrame): predictions based on the features from X_val
    model : trained model
    """
    if "fit" in dir(model):
        model.fit(X_train, y_train.values.ravel())
    else:
        raise ValueError(
            "Model not supported by this helper function. You have to write your own code"
        )
    if "predict" in dir(model):
        y_pre = model.predict(X_val)
    else:
        raise ValueError(
            "Model not supported by this helper function. You have to write your own code"
        )
    return y_pre, model


def scoring(y_true: pd.DataFrame, y_pre: pd.DataFrame) -> dict:
    """
    Return a dictionary with keys corresponding to score name and values corresponding to the associated score

    Input:
    y_true (pd.DataFrame) : input DataFrame with true labels
    y_pre (pd.DataFrame) : input DataFrame with predicted labels

    Output:
    (dict): output dictionary with scores
    """
    return {
        "f1-micro": f1_score(y_true, y_pre, average="micro"),
        "f1-macro": f1_score(y_true, y_pre, average="macro"),
        "f1-weighted": f1_score(y_true, y_pre, average="weighted"),
        "accuracy": accuracy_score(y_true, y_pre),
    }


def print_scoring(scores: dict) -> None:
    """
    Print scores from a dictionary

    Input:
    scores (dict) : dictionary with keys corresponding to score name and values corresponding to the associated score

    Output:
    None
    """
    for name, score in scores.items():
        print(f"{name}: {score}")


def plot_confusion_matrix(
    y_test: pd.DataFrame, y_pre: pd.DataFrame, nb_values: int
) -> None:
    """
    Plot the confusion matrix based on the provided true and predicted labels

    Input:
    y_true (pd.DataFrame) : input DataFrame with true labels
    y_pre (pd.DataFrame) : input DataFrame with predicted labels
    nb_values (int) : nb of target values

    Output:
    None
    """
    s = sns.heatmap(
        confusion_matrix(y_test, y_pre),
        xticklabels=range(nb_values),
        yticklabels=range(nb_values),
        annot=True,
        cmap="Blues",
        fmt="g",
        cbar=False,
    )
    s.set(xlabel="True label", ylabel="Predicted label")
    plt.show()


def rf_features_importance(model: MODEL_TYPE, cols: list) -> None:
    """
    Plot feature_importance from a random forest model

    Input:
    model (MODEL_TYPE) : Model for which to compute the feature importance
    y_true (pd.DataFrame) : input DataFrame with true labels
    y_pre (pd.DataFrame) : input DataFrame with predicted labels

    Output:
    None
    """
    # get feature importance from model
    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=cols).sort_values(
        ascending=False
    )[:10]
    # plot results
    fig, ax = plt.subplots()
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances")
    fig.tight_layout()


def cross_validation_score(
    model: MODEL_TYPE, X_train_val: pd.DataFrame, y_train_val: pd.DataFrame, gap: int
) -> Tuple[MODEL_TYPE, dict]:
    """
    Cross validation for time series, with 5 splits

    Input:
    X_train_val (pd.DataFrame) : input DataFrame with features for training
    y_train_val (pd.DataFrame) : input DataFrame with target values for training
    gap (int) : number of entries between 2 sets during the cross-validation

    Output:
    model: trained model
    """
    # cross-validation for time series
    tscv = TimeSeriesSplit(n_splits=3, gap=gap)
    y_pre, y_val, trained_model = None, None, None
    scores_history = []
    i = 0
    for train_index, val_index in tscv.split(X_train_val):
        i += 1  # increase iteration
        # get datasets (train and val)
        train_index = list(train_index)
        val_index = list(val_index)
        X_train, X_val = (
            X_train_val.iloc[train_index, :],
            X_train_val.iloc[val_index, :],
        )
        y_train, y_val = (
            y_train_val.iloc[train_index, :],
            y_train_val.iloc[val_index, :],
        )
        # fit model and predict y_val
        y_pre, trained_model = model_fit_predict(model, X_train, y_train, X_val)
        # scoring
        scores = scoring(y_val, y_pre)
        scores_history.append(scores)
        print("")
        print(f"Step {i}")
        print_scoring(scores)

    return trained_model, scores


# to be done only few times (risk of overfitting)
def score_test_set(
    model: Any,
    X_train_val: pd.DataFrame,
    y_train_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> None:
    """
    Fit the model on the entire X_train_val and y_train_val data, and predict values for the test set

    Input:
    model (MODEL_TYPE) : model used for training the the previous section of the notebook
    X_train_val (pd.DataFrame) : input DataFrame with features for training
    y_train_val (pd.DataFrame) : input DataFrame with target values for training
    X_test (pd.DataFrame) : input DataFrame with features for testing
    y_test (pd.DataFrame) : input DataFrame with target values for testing

    Output:
    None
    """
    # fit model on entire X_train_val, y_train_val datasets
    model.fit(X_train_val, y_train_val.values.ravel())
    y_pred = model.predict(X_test)
    # score test set
    scores = scoring(y_test, y_pred)
    print_scoring(scores)
    # plot the last confusion matrix
    plot_confusion_matrix(y_test, y_pred, nb_values=2)
    # plot features importance
    cols = X_train_val.columns  # get columns names
    if "feature_importances_" in dir(model):
        rf_features_importance(model, cols)  # plot feature importance


def train_model_for_feature_engineering(data, target_col, ratio=0.75):
    X_train_val, y_train_val, X_test, y_test = train_test_split(data, target_col, ratio)
    random_forest = RandomForestClassifier()
    max_occ_day = data.groupby("DISCOVERY_DATE").agg({"STATE": "count"}).max().values[0]
    random_forest, random_forest_scores = cross_validation_score(
        random_forest, X_train_val, y_train_val, gap=max_occ_day
    )
    score_test_set(random_forest, X_train_val, y_train_val, X_test, y_test)
