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


# -----------------------
# Curated climate sources
# -----------------------
def fetch_mindicador(indicador: str, year: str = "2024") -> pd.DataFrame:
    """
    Fetch economic/weather-related indicator time series from mindicador.cl
    Example indicador: 'dolar', 'euro', 'uf', etc.
    Returns a DataFrame with columns: fecha (datetime), valor (float)
    """
    url = f"https://mindicador.cl/api/{indicador}/{year}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    serie = data.get("serie", [])
    if not serie:
        return pd.DataFrame()
    df = pd.DataFrame(serie)
    # Normalize
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    if "valor" in df.columns:
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.sort_values("fecha")
    return df[["fecha", "valor"]]

def fetch_sismos_chile() -> pd.DataFrame:
    """
    Últimos sismos desde GAEL Cloud.
    https://api.gael.cloud/general/public/sismos
    """
    url = "https://api.gael.cloud/general/public/sismos"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    # Parse timestamps and numeric magnitudes
    for c in df.columns:
        lc = str(c).lower()
        if "fecha" in lc or "time" in lc or "hora" in lc:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "Magnitud" in df.columns:
        df["Magnitud"] = pd.to_numeric(df["Magnitud"], errors="coerce")
    return df

def fetch_openweather_current(city: str, api_key: str) -> pd.DataFrame:
    """
    Current weather snapshot from OpenWeather.
    Returns a flattened one-row DataFrame for display.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=es"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Flatten some common fields
    out = {
        "ciudad": data.get("name"),
        "pais": (data.get("sys") or {}).get("country"),
        "descripcion": (data.get("weather") or [{}])[0].get("description"),
        "temp_C": (data.get("main") or {}).get("temp"),
        "sensacion_C": (data.get("main") or {}).get("feels_like"),
        "temp_min_C": (data.get("main") or {}).get("temp_min"),
        "temp_max_C": (data.get("main") or {}).get("temp_max"),
        "humedad_%": (data.get("main") or {}).get("humidity"),
        "viento_mps": (data.get("wind") or {}).get("speed"),
        "nubes_%": (data.get("clouds") or {}).get("all"),
        "lat": (data.get("coord") or {}).get("lat"),
        "lon": (data.get("coord") or {}).get("lon"),
        "dt": pd.to_datetime(data.get("dt"), unit="s", errors="coerce")
    }
    return pd.DataFrame([out])
