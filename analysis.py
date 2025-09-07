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
    Robust summary that works across pandas versions.
    - Numeric columns: count/mean/std/min/25%/50%/75%/max
    - Datetime columns: count/min/max
    - Categorical/object: count/unique/top/freq
    """
    if df is None or df.empty:
        return pd.DataFrame({"column": [], "summary": []})

    parts = []

    # Numeric
    num_cols = df.select_dtypes(include="number")
    if not num_cols.empty:
        desc_num = num_cols.describe().T
        desc_num["type"] = "numeric"
        parts.append(desc_num)

    # Datetime
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"])
    if not dt_cols.empty:
        # For datetime, describe gives count/unique/top/freq in some versions; compute min/max manually
        dt_desc = pd.DataFrame(index=dt_cols.columns)
        dt_desc["count"] = dt_cols.notna().sum()
        dt_desc["min"] = dt_cols.min(numeric_only=False)
        dt_desc["max"] = dt_cols.max(numeric_only=False)
        dt_desc["type"] = "datetime"
        parts.append(dt_desc)

    # Categorical / object / boolean
    cat_cols = df.select_dtypes(include=["object", "category", "bool"])
    if not cat_cols.empty:
        desc_cat = cat_cols.describe().T  # count, unique, top, freq
        desc_cat["type"] = "categorical"
        parts.append(desc_cat)

    if not parts:
        # Fallback: attempt generic describe without special args
        try:
            generic = df.describe(include="all").T
        except Exception:
            generic = pd.DataFrame(index=df.columns)
        generic["type"] = "unknown"
        parts.append(generic)

    out = pd.concat(parts, axis=0, sort=False)
    out = out.reset_index().rename(columns={"index": "column"})
    return out
