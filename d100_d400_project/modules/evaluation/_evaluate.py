import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss


def lorenz_curve(y_true, y_pred, weights=None):
    """
    Generate Lorenz curve values for Gini coefficient calculation.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    weights : array-like, optional
        Weights for each observation. If None, equal weights are assumed.

    Returns
    -------
    tuple
        A tuple containing:
        - cum_exposure (array-like): Cumulative exposure values.
        - cum_true (array-like): Cumulative true target values.
    """
    order = np.argsort(y_pred)
    y_true, y_pred = y_true[order], y_pred[order]
    if weights is not None:
        weights = weights[order]
    else:
        weights = np.ones_like(y_true)
    cum_true = np.cumsum(y_true * weights) / np.sum(y_true * weights)
    cum_exposure = np.cumsum(weights) / np.sum(weights)
    return cum_exposure, cum_true


def evaluate_predictions(
    df,
    outcome_column,
    *,
    prob_column=None,
    preds_column=None,
    model=None,
    model_name="Model",
    weights_column=None,
):
    """
    Evaluate predictions for multi-class classification models using various metrics.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing actual and predicted values.
    outcome_column : str
        The column name for true target values.
    prob_column : list of str, optional
        List of column names for predicted probabilities. Required if `model` is not provided.
    preds_column : str, optional
        The column name for predicted classes. If not provided, derived from `prob_column`.
    model : sklearn.base.BaseEstimator, optional
        A trained model to generate predictions if `prob_column` is not provided.
    model_name : str, optional
        Name of the model for display purposes (default is "Model").
    weights_column : str, optional
        Column name for weights to be used in calculations. If None, equal weights are assumed.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing various evaluation metrics.

    Notes
    -----
    - Calculates metrics including bias, deviance, log loss, accuracy, F1 score and MAE.
    - If neither `prob_column` nor `model` is provided, raises a ValueError.
    - Outputs a classification report for detailed performance analysis.
    """
    evals = {}
    actuals = df[outcome_column].values  # Actual target values

    # Predicted probabilities
    if prob_column is not None:
        # Correctly extract values as a NumPy array
        probs = df[prob_column].to_numpy()
    elif model is not None:
        # Predict probabilities using the model
        probs = model.predict_proba(df.drop(columns=outcome_column, errors="ignore"))
    else:
        raise ValueError("Provide either prob_column or a fitted model.")

    # Predicted classes
    if preds_column is not None:
        preds = df[preds_column].values
    else:
        preds = np.argmax(probs, axis=1)

    # Weights for bias and deviance calculations
    weights = df[weights_column].values if weights_column else np.ones_like(actuals)

    # 1. Bias
    actual_mean = np.average(actuals, weights=weights)
    pred_mean = np.average(np.argmax(probs, axis=1), weights=weights)
    evals["Bias"] = (pred_mean - actual_mean) / actual_mean

    # 2. Deviance
    evals["Deviance"] = np.average(
        (actuals - np.argmax(probs, axis=1)) ** 2, weights=weights
    )

    # 3. Log Loss, Accuracy, and F1
    evals["log_loss"] = log_loss(actuals, probs, labels=np.unique(actuals))
    evals["accuracy"] = accuracy_score(actuals, preds)
    evals["f1_weighted"] = f1_score(actuals, preds, average="weighted")

    # Print evaluation results
    print(f"\nEvaluation Metrics for {model_name}: ")
    for metric, value in evals.items():
        print(f"{metric.upper()}: {value: .4f}")

    print("\nClassification Report:")
    print(classification_report(actuals, preds))

    return pd.DataFrame(evals, index=[model_name])


def get_lgbm_feature_importance(
    lgbm_model,
    fitted_preprocessor,
    remaining_numericals,
    spline_features,
    categoricals,
    ordinal,
):
    """
    Extract feature importances from an LGBM pipeline with preprocessing.

    Parameters
    ----------
    lgbm_model : lightgbm.LGBMClassifier
        A trained LightGBM classifier.
    fitted_preprocessor : sklearn.compose.ColumnTransformer
        The preprocessor pipeline applied to the dataset.
    remaining_numericals : list of str
        List of remaining numerical features passed to StandardScaler.
    spline_features : list of str
        List of features transformed using splines.
    categoricals : list of str
        List of categorical features passed to OneHotEncoder.
    ordinal : list of str
        List of ordinal features.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing feature names and their corresponding importance scores.

    Notes
    -----
    - Dynamically generates feature names based on the preprocessor's transformers.
    - Raises a ValueError if the number of feature names does not match the feature importances.
    - Outputs a sorted DataFrame by importance scores in descending order.
    """
    # Dynamically get feature names based on transformers
    spline_output_features = (
        fitted_preprocessor.named_transformers_["spline"]
        .named_steps["spline"]
        .get_feature_names_out(input_features=spline_features)
    )

    cat_output_features = fitted_preprocessor.named_transformers_[
        "cat"
    ].get_feature_names_out(input_features=categoricals)

    ord_output_features = ordinal  # CustomOrdinalEncoder doesn't expand features

    # Combine all output feature names
    all_feature_names = (
        list(spline_output_features)
        + list(remaining_numericals)
        + list(cat_output_features)
        + list(ord_output_features)
    )

    # Get feature importances from the LGBMClassifier
    feature_importance = lgbm_model.feature_importances_

    # Verify the lengths match
    if len(all_feature_names) != len(feature_importance):
        raise ValueError("Mismatch in feature names and importances length!")

    # Create and return a DataFrame for feature importances
    lgbm_feature_importance = pd.DataFrame(
        {"Feature": all_feature_names, "Importance": feature_importance}
    ).sort_values(by="Importance", ascending=False)

    return lgbm_feature_importance
