# api_utils.py
import requests
import pandas as pd
from typing import List, Dict, Optional, Tuple

BASE = "https://datos.gob.cl/api/3/action"

def _get(url: str, params: Optional[dict] = None) -> dict:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("success", True) and "result" not in data:
        raise RuntimeError(f"API error: {data}")
    return data

def search_datasets(q: Optional[str] = None, group: str = "clima", rows: int = 25, start: int = 0) -> List[Dict]:
    """
    Search datasets within a CKAN group (default 'clima').
    """
    params = {"fq": f"groups:{group}", "rows": rows, "start": start}
    if q:
        params["q"] = q
    url = f"{BASE}/package_search"
    data = _get(url, params=params)
    return data["result"]["results"]

def dataset_resources(dataset: Dict) -> List[Dict]:
    """
    Return the list of resources for a dataset object from package_search.
    """
    return dataset.get("resources", [])

def fetch_resource_to_df(resource: Dict, limit: int = 10000) -> Tuple[pd.DataFrame, str]:
    """
    Attempt to fetch resource data as a DataFrame.
    - If resource is in Datastore, use datastore_search.
    - Else if resource URL looks like CSV, read via pandas.
    Returns (df, source_description)
    """
    # Try datastore first
    res_id = resource.get("id")
    if res_id:
        try:
            url = f"{BASE}/datastore_search"
            params = {"resource_id": res_id, "limit": limit}
            data = _get(url, params=params)
            records = data["result"]["records"]
            df = pd.DataFrame.from_records(records)
            if not df.empty:
                return df, "API (datastore_search)"
        except Exception:
            pass  # fallback paths below

    # Fallback: try direct CSV
    url = resource.get("url")
    format_hint = (resource.get("format") or "").lower()
    if isinstance(url, str) and (format_hint == "csv" or url.lower().endswith(".csv")):
        df = pd.read_csv(url)
        return df, "HTTP GET (CSV)"
    
    # Could not parse
    raise ValueError("Recurso no compatible automáticamente (no datastore y no CSV).")

def best_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to guess a datetime-like column.
    """
    for col in df.columns:
        # Try common names first
        lowered = col.lower()
        if any(k in lowered for k in ["fecha", "date", "datetime", "hora", "timestamp"]):
            try:
                pd.to_datetime(df[col])
                return col
            except Exception:
                pass
    # Else try to infer any column that parses to datetime with low error
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() >= max(5, int(0.5 * len(df))):
                return col
        except Exception:
            continue
    return None

def numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
