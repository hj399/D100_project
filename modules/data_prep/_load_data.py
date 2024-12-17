from pathlib import Path
import pandas as pd

def load_data():
    """
    Load and transform the student dropout dataset.

    This function locates the dataset file in the "data" directory relative 
    to the script's parent path and loads it into a pandas DataFrame.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the student dropout dataset.

    Notes:
    ------
    - Assumes the dataset is in CSV format and located in the "data" folder.
    - Uses a comma (',') as the delimiter for loading the file.
    """
    # Locate the dataset using a relative path
    file_path = Path(__file__).resolve().parent.parent.parent / "data" / "dataset.csv"
    
    # Load the dataset
    df = pd.read_csv(file_path, delimiter=',')  # Ensure the delimiter is correct

    return df
