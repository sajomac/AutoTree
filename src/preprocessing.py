
import numpy as np
import pandas as pd
from scipy.stats import skew



def correct_skewness(data, column):
    """
    Correct the skewness of a column by applying a log transformation if necessary.

    For highly positive skew (> 1), a log1p transformation is applied. For highly negative
    skew (< -1), a reverse log1p transformation is used.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): The column to be transformed.

    Returns:
        pd.Series: The transformed column.
    """
    col_skew = skew(data[column].dropna())
    if col_skew > 1:
        # Apply log transformation for positive values
        if (data[column] >= 0).all():
            return np.log1p(data[column])
        else:
            return np.log1p(data[column] + abs(data[column].min()) + 1)
    elif col_skew < -1:
        # Apply reverse-log transformation for negative skew
        if (data[column] <= 0).all():
            return -np.log1p(-data[column])
        else:
            return -np.log1p(abs(data[column] - data[column].max()) + 1)
    return data[column]


def map_non_numeric_columns(data):
    """
    Maps non-numeric columns with 5 or fewer unique values to integers.
    Drops remaining non-numeric columns.
    
    Parameters:
        data (pd.DataFrame): Input dataset.
    
    Returns:
        pd.DataFrame: Processed dataset with mapped categorical columns.
        dict: Dictionary containing the mappings used.
    """
    non_numeric = data.select_dtypes(exclude=['number', 'bool']).columns
    mappings = {}
    
    for col in non_numeric:
        unique_values = data[col].unique()
        if len(unique_values) <= 5:
            mapping = {value: idx for idx, value in enumerate(unique_values)}
            data[col] = data[col].map(mapping)
            mappings[col] = mapping
    
    data.drop(columns=non_numeric, inplace=True, errors='ignore')
    return data, mappings
