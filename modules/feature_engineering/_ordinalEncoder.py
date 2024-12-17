import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    Custom ordinal encoder for transforming categorical data into ordinal values.

    This transformer assigns a unique integer to each category in each column. It ensures
    unknown categories during transformation are assigned a value of -1.

    Attributes
    ----------
    mapping_ : dict
        A dictionary mapping each column to its unique category-to-index mapping.

    Methods
    -------
    fit(X, y=None)
        Fits the encoder to the provided data and creates the category-to-index mapping.
    transform(X)
        Transforms the input data into ordinal-encoded values using the fitted mappings.
    """

    def __init__(self):
        self.mapping_ = {}

    def fit(self, X, y=None):
        """
        Fit the ordinal encoder to the provided data.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Input data with categorical features to fit the encoder.
        y : None, optional
            Ignored. Exists for compatibility with sklearn pipelines.

        Returns
        -------
        CustomOrdinalEncoder
            The fitted encoder instance.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.mapping_ = {
            col: {category: idx for idx, category in enumerate(X[col].unique())}
            for col in X.columns
        }
        return self

    def transform(self, X):
        """
        Transform the input data into ordinal encoded values using the fitted mappings.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Input data with categorical features to transform.

        Returns
        -------
        pandas.DataFrame
            Transformed DataFrame with ordinal-encoded values.

        Raises
        ------
        ValueError
            If a column in X was not present during fitting or contains unmapped categories.
        """
        check_is_fitted(self, "mapping_")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X_transformed = X.copy()
        for col in X.columns:
            if col not in self.mapping_:
                raise ValueError(f"Column {col} was not present during fit.")
            # Map unknown categories to -1
            X_transformed[col] = X[col].map(self.mapping_[col]).fillna(-1).astype(int)
        return X_transformed
