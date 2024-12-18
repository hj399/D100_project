import hashlib

import numpy as np


def create_sample_split(df, id_column, training_frac=0.8):
    """
    Create a train-test sample split based on a unique ID column.

    This function deterministically splits the dataset into train and test sets
    using the modulo operation on integer IDs or an MD5 hash of non-integer IDs.
    The split ensures reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    id_column : str
        The name of the column containing unique IDs for each row.
    training_frac : float, optional
        The fraction of the dataset to include in the training set (default is 0.8).

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an additional 'sample' column indicating the split:
        - 'train' for training samples
        - 'test' for testing samples

    Notes
    -----
    - If `id_column` contains integers, the modulo operation is used.
    - If `id_column` contains non-integer values, an MD5 hash is applied for splitting.
    - The function ensures deterministic splitting for reproducibility.
    """
    if df[id_column].dtype == np.int64:
        modulo = df[id_column] % 100
    else:
        modulo = df[id_column].apply(
            lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 100
        )

    df["sample"] = np.where(modulo < training_frac * 100, "train", "test")

    return df
