import numpy as np
from scipy.stats import skew


def handle_skewed_columns(df, numerical_features):
    """
    Handle skewed numerical columns in a DataFrame by applying log transformation.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    numerical_features : list of str
        A list of column names corresponding to numerical features in the DataFrame.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with additional log-transformed columns for skewed features.

    Notes:
    ------
    - Only features with an absolute skewness greater than 1 are transformed.
    - Columns where the minimum value is greater than -1 are eligible for log transformation.
    - Log-transformed columns are added with the suffix '_log'.
    """
    numerical_features = df[numerical_features]
    skewness = numerical_features.apply(lambda x: skew(x.dropna()))
    skewed_features = skewness[abs(skewness) > 1]

    for col in skewed_features.index:
        if df[col].min() > -1:
            df[f"{col}_log"] = np.log1p(df[col])
    return df
