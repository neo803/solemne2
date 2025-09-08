
from __future__ import annotations
import re, requests
import pandas as pd
import numpy as _np
from typing import Optional
from bs4 import BeautifulSoup
import datetime as _dt

STANDARD_COLS = ['fecha_dt','fecha_local','magnitud','profundidad','latitud','longitud','referencia']

def _ensure_standard(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_COLS)
    for c in STANDARD_COLS:
        if c not in df.columns:
            df[c] = _np.nan
    return df[STANDARD_COLS].sort_values('fecha_dt', ascending=False).reset_index(drop=True)

# Two regex patterns to handle CSN formats
_pat_slash = re.compile(
    r"(?P<local>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"(?P<lugar>[^\n]+?)\s+(?P<utc>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"Latitud\s*/\s*Longitud\s*(?P<lat>-?\d+(?:\.\d+)?)\s*/\s*(?P<lon>-?\d+(?:\.\d+)?).*?"
    r"Profundidad\s*(?P<prof>\d+)\s*km.*?"
    r"Magnitud\s*(?P<mag>[\d\.]+)\s*[A-Za-z]+",
    re.S
)
_pat_multiline = re.compile(
    r"(?P<local>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"(?P<lugar>[^\n]+?)\s+(?P<utc>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"(?P<lat>-?\d+\.\d+)\s+\n\s*(?P<lon>-?\d+\.\d+)\s+"
    r"(?P<prof>\d+)\s+km\s+(?P<mag>[\d\.]+)\s+[A-Za-z]+",
    re.S
)

def _parse_csn_text(text: str) -> pd.DataFrame:
    rows = []
    for pat in (_pat_slash, _pat_multiline):
        for m in pat.finditer(text):
            rows.append({
                'fecha_local_str': m.group('local'),
                'fecha_utc_str':   m.group('utc'),
                'referencia':      m.group('lugar'),
                'latitud': float(m.group('lat')),
                'longitud': float(m.group('lon')),
                'profundidad': float(m.group('prof')),
                'magnitud': float(m.group('mag'))
            })
        if rows: break
    if not rows:
        return pd.DataFrame(columns=STANDARD_COLS)
    df = pd.DataFrame(rows)
    df['fecha_dt'] = pd.to_datetime(df['fecha_utc_str'], utc=True, errors='coerce')
    df['fecha_local'] = pd.to_datetime(df['fecha_local_str'], errors='coerce').dt.tz_localize('America/Santiago', nonexistent='shift_forward', ambiguous='NaT')
    df['dia'] = df['fecha_local'].dt.date
    df['hora'] = df['fecha_local'].dt.strftime('%H:%M')
    return _ensure_standard(df[['fecha_dt','fecha_local','magnitud','profundidad','latitud','longitud','referencia']])

def fetch_from_csn(date: _dt.date, timeout: int = 20) -> pd.DataFrame:
    y = date.strftime('%Y'); m = date.strftime('%m'); d = date.strftime('%Y%m%d')
    url = f"https://www.sismologia.cl/sismicidad/catalogo/{y}/{m}/{d}.html"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'lxml')
    text = soup.get_text('\n', strip=True)
    df = _parse_csn_text(text)
    if not df.empty:
        return df
    # Fallback: try read_html if regex fails
    try:
        tables = pd.read_html(url, flavor='lxml')
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if any("lat" in c for c in cols) and any(("lon" in c or "long" in c or "longitud" in c) for c in cols):
                def pick(opts):
                    for o in opts:
                        for j,c in enumerate(cols):
                            if o in c: return t.columns[j]
                    return None
                col_lat = pick(["latitud","lat"]); col_lon = pick(["longitud","lon","long"])
                col_prof = pick(["profundidad","prof","depth"]); col_mag = pick(["magnitud","mag"])
                col_fecha = pick(["fecha utc","utc","fecha","datetime","hora"])
                df2 = pd.DataFrame()
                if col_fecha is not None: df2['fecha_dt'] = pd.to_datetime(t[col_fecha], utc=True, errors='coerce')
                if col_lat is not None: df2['latitud'] = pd.to_numeric(t[col_lat], errors='coerce')
                if col_lon is not None: df2['longitud'] = pd.to_numeric(t[col_lon], errors='coerce')
                if col_prof is not None: df2['profundidad'] = pd.to_numeric(t[col_prof], errors='coerce')
                if col_mag is not None: df2['magnitud'] = pd.to_numeric(t[col_mag], errors='coerce')
                df2['fecha_local'] = df2.get('fecha_dt', pd.NaT).dt.tz_convert('America/Santiago')
                df2['referencia'] = ""
                return _ensure_standard(df2)
    except Exception:
        pass
    return _ensure_standard(pd.DataFrame())

def fetch_csn_last_days(days: int = 7, timeout: int = 20) -> pd.DataFrame:
    days = max(1, int(days))
    tz = 'America/Santiago'
    today_cl = pd.Timestamp.now(tz=tz).date()
    all_df = []
    for i in range(days):
        d = today_cl - pd.Timedelta(days=i)
        try:
            df = fetch_from_csn(date=d, timeout=timeout)
            if not df.empty:
                all_df.append(df)
        except Exception:
            continue
    if not all_df:
        return _ensure_standard(pd.DataFrame())
    out = pd.concat(all_df, ignore_index=True)
    out = out.drop_duplicates(subset=['fecha_dt','latitud','longitud']).reset_index(drop=True)
    return _ensure_standard(out)

def filter_sismos(df: pd.DataFrame,
                  mag_min: Optional[float] = None,
                  fecha_desde: Optional[pd.Timestamp] = None,
                  fecha_hasta: Optional[pd.Timestamp] = None,
                  region_keyword: str = "") -> pd.DataFrame:
    out = df.copy()
    if mag_min is not None:
        out = out[(out["magnitud"].fillna(-999) >= mag_min)]
    if fecha_desde is not None:
        out = out[out["fecha_dt"] >= fecha_desde]
    if fecha_hasta is not None:
        out = out[out["fecha_dt"] <= fecha_hasta]
    if region_keyword:
        rk = region_keyword.strip().lower()
        out = out[out["referencia"].astype(str).str.lower().str.contains(rk, na=False)]
    return out.reset_index(drop=True)
