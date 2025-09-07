# analysis.py
import pandas as pd

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Basic cleaning: drop fully empty cols, strip column names
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Remove duplicated rows
    df = df.drop_duplicates()
    return df

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick numeric summary (count, mean, std, min, 25%, 50%, 75%, max)
    """
    desc = df.describe(include="all", datetime_is_numeric=True).T
    return desc.reset_index().rename(columns={"index": "column"})
