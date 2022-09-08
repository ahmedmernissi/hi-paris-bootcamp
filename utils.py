
import pandas as pd

def check_duplicates(df: pd.DataFrame, k: list) -> bool:
    '''
    Checks if a dataframe has duplicate values, based on the provided primary key
    Returns True is there is no duplicate values

    Input:
    df (pd.DataFrame): input DataFrame
    k (list): list of columns, corresponding to the primary key

    Output:
    res (bool): True if there is no duplicate value, False otherwise
    '''
    count = df.shape[0] - df.drop_duplicates(subset=k).shape[0]
    res = True if count == 0 else False
    return res