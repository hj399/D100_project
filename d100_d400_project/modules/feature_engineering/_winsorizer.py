import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    A transformer for winsorizing numerical data by capping extreme values
    based on specified quantiles.

    The Winsorizer reduces the impact of outliers by clipping values outside
    a specified range, determined by the lower and upper quantiles.

    Attributes
    ----------
    lower_quantile : float
        The lower quantile for clipping (default is 0.01, i.e., 1st percentile).
    upper_quantile : float
        The upper quantile for clipping (default is 0.99, i.e., 99th percentile).
    lower_bound_ : float
        The computed lower bound based on the lower quantile during fitting.
    upper_bound_ : float
        The computed upper bound based on the upper quantile during fitting.

    Methods
    -------
    fit(X, y=None)
        Computes the lower and upper bounds for clipping based on the specified quantiles.
    transform(X)
        Clips the data to the computed bounds, capping extreme values.
    """

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        """
        Initialize the Winsorizer with specified quantiles.

        Parameters
        ----------
        lower_quantile : float, optional
            The lower quantile for clipping (default is 0.01, i.e., 1st percentile).
        upper_quantile : float, optional
            The upper quantile for clipping (default is 0.99, i.e., 99th percentile).
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        """
        Fit the Winsorizer by calculating the clipping bounds.

        Parameters
        ----------
        X : array-like
            The input data to compute the quantile bounds.
        y : None, optional
            Ignored. Exists for compatibility with sklearn pipelines.

        Returns
        -------
        Winsorizer
            The fitted Winsorizer instance with computed bounds.
        """
        self.lower_bound_ = np.percentile(X, self.lower_quantile * 100)
        self.upper_bound_ = np.percentile(X, self.upper_quantile * 100)
        return self

    def transform(self, X):
        """
        Transform the data by clipping values to the computed bounds.

        Parameters
        ----------
        X : array-like
            The input data to be transformed.

        Returns
        -------
        array-like
            The transformed data with extreme values clipped to the computed bounds.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the Winsorizer is not fitted before calling this method.
        """
        check_is_fitted(self)
        X_clipped = np.clip(X, self.lower_bound_, self.upper_bound_)
        return X_clipped
