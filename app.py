import streamlit as st
import pandas as pd
from api_utils import fetch_sismos_chile

st.set_page_config(page_title="Sismos Chile – Magnitud, Profundidad y Mapa", layout="wide")
st.title("🌎 Sismos en Chile – Magnitud, Profundidad y Mapa")

st.markdown(
    "Datos en tiempo real desde **GAEL Cloud**. Filtra por magnitud y fechas, "
    "y visualiza un mapa con la ubicación de los eventos."
)

@st.cache_data(ttl=300)
def load_sismos():
    df = fetch_sismos_chile()
    # Normalización de nombres para facilitar el manejo
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols_lower:
            return cols_lower[key]
    # búsqueda parcial
    for c in df.columns:
        lc = c.lower()
        if any(key in lc for key in [cand.lower() for cand in candidates]):
            return c
    return None

def coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def coerce_datetime(series):
    return pd.to_datetime(series, errors="coerce")

def infer_coords_from_reference(text):
    """Try to infer (lat, lon) from a reference string if it contains coordinates.
    Only accept values within Chile's rough bounds (lat -56..-17, lon -76..-66).
    Returns tuple (lat, lon) or (None, None)."""
    if not isinstance(text, str):
        return (None, None)
    # Common patterns: "-33.45, -70.66" or "lat -33.45 lon -70.66"
    patterns = [
        r'([-+]?\d{1,2}(?:[\.,]\d+)?)[\s,;]+([-+]?\d{1,3}(?:[\.,]\d+)?)',
        r'lat(?:itud)?\s*[:=]?\s*([-+]?\d{1,2}(?:[\.,]\d+)?)[^\d-]+lon(?:gitud)?\s*[:=]?\s*([-+]?\d{1,3}(?:[\.,]\d+)?)',
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            lat = str(m.group(1)).replace(',', '.')
            lon = str(m.group(2)).replace(',', '.')
            try:
                lat = float(lat); lon = float(lon)
                if -56 <= lat <= -17 and -76 <= lon <= -66:
                    return (lat, lon)
            except Exception:
                pass
    return (None, None)

def extract_region_from_reference(text):
    """Extract region-like tokens from reference text. Returns a normalized name or None."""
    if not isinstance(text, str):
        return None
    # Examples: "Región de Coquimbo", "Región Metropolitana", "Region del Biobío"
    m = re.search(r'(Regi[oó]n\s+(?:de|del)\s+[A-Za-zÁÉÍÓÚÑáéíóú\s]+)', text, flags=re.IGNORECASE)
    if m:
        reg = m.group(1)
        # Normalize spaces/casing
        reg = re.sub(r'\s+', ' ', reg).strip().title()
        # Fix common Spanish accents capitalization
        reg = reg.replace('Biobío', 'Biobío').replace('Metropolitana', 'Metropolitana')
        return reg
    return None

    return pd.to_datetime(series, errors="coerce")

df = load_sismos()
# Try to build region column from reference
if df is not None and not df.empty:
    # detect columns again in case names differ
    def find_col(df, candidates):
        cols_lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            key = cand.lower()
            if key in cols_lower:
                return cols_lower[key]
        for c in df.columns:
            lc = c.lower()
            if any(key in lc for key in [cand.lower() for cand in candidates]):
                return c
        return None

    col_ref = find_col(df, ["Referencia Geografica", "Referencia", "refgeo", "lugar", "place"])
    if col_ref and "region_extraida" not in df.columns:
        df["region_extraida"] = df[col_ref].apply(extract_region_from_reference)

# If dataset lacks lat/lon, try to infer from reference text
if 'lat' not in df.columns and 'Latitud' not in df.columns and 'latitude' not in {c.lower() for c in df.columns}:
    if 'lon' not in df.columns and 'Longitud' not in df.columns and 'longitude' not in {c.lower() for c in df.columns}:
        # Infer only if we have reference
        if 'region_extraida' in df.columns or col_ref:
            col_r = col_ref if col_ref else 'region_extraida'
            # Create inferred columns
            lat_list, lon_list = [], []
            for txt in df[col_ref] if col_ref else df['region_extraida']:
                lat, lon = infer_coords_from_reference(txt)
                lat_list.append(lat); lon_list.append(lon)
            df["lat_inferida"] = pd.Series(lat_list, index=df.index)
            df["lon_inferida"] = pd.Series(lon_list, index=df.index)

if df.empty:
    st.error("No se pudieron cargar los datos de sismos.")
    st.stop()

# Intentar detectar columnas relevantes (caso-insensible y tolerante a variaciones)
col_mag = find_col(df, ["Magnitud", "mag", "magnitude"])
col_prof = find_col(df, ["Profundidad", "depth", "prof"])
col_lat = find_col(df, ["Latitud", "lat", "latitude"])
col_lon = find_col(df, ["Longitud", "lon", "lng", "longitude"])
col_time = find_col(df, ["Fecha", "fecha", "time", "fechaLocal", "Fecha UTC", "TimeStamp"])
col_ref = find_col(df, ["Referencia Geografica", "Referencia", "refgeo", "lugar", "place"])

# Coerciones
if col_mag:
    df[col_mag] = coerce_numeric(df[col_mag])
if col_prof:
    # A veces viene con 'km' como string
    df[col_prof] = df[col_prof].astype(str).str.extract(r"([\d\.,]+)", expand=False).str.replace(",", ".", regex=False)
    df[col_prof] = coerce_numeric(df[col_prof])
if col_lat:
    df[col_lat] = coerce_numeric(df[col_lat])
if col_lon:
    df[col_lon] = coerce_numeric(df[col_lon])
if col_time:
    df[col_time] = coerce_datetime(df[col_time])

# Sidebar de filtros
with st.sidebar:
    st.header("Filtros")
    min_mag = st.slider("Magnitud mínima", 0.0, 10.0, 3.0, 0.1)
    if col_time and pd.api.types.is_datetime64_any_dtype(df[col_time]):
        tmin, tmax = df[col_time].min(), df[col_time].max()
        if pd.isna(tmin) or pd.isna(tmax):
            date_range = None
        else:
            date_range = st.date_input("Rango de fechas", value=(tmin.date(), tmax.date()))
    else:
        date_range = None
    region_opts = sorted([r for r in df.get('region_extraida', []).dropna().unique().tolist()]) if 'region_extraida' in df.columns else []
    region_sel = st.selectbox('Región (opcional)', ['(todas)'] + region_opts if region_opts else ['(no detectadas)'])
    text_ref = st.text_input('Texto a buscar en referencia (opcional)', '')
    apply_btn = st.button("Aplicar filtros")

if apply_btn:
    if col_mag:
        df = df[df[col_mag] >= min_mag]
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2 and col_time:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[(df[col_time] >= start) & (df[col_time] <= end)]
    if 'region_extraida' in df.columns and region_sel not in ['(todas)', '(no detectadas)']:
        df = df[df['region_extraida'] == region_sel]
    if 'text_ref' in locals() and text_ref and (col_ref or 'Referencia' in df.columns):
        ref_c = col_ref if col_ref in df.columns else 'Referencia'
        df = df[df[ref_c].astype(str).str.contains(text_ref, case=False, na=False)]

# Vista general
st.subheader("Tabla de sismos")
cols_show = []
for c in [col_time, col_ref, col_mag, col_prof, col_lat, col_lon]:
    if c and c not in cols_show:
        cols_show.append(c)

if cols_show:
    st.dataframe(df[cols_show].rename(columns={
        col_time or "": "Fecha/Hora",
        col_ref or "": "Referencia",
        col_mag or "": "Magnitud",
        col_prof or "": "Profundidad (km)",
        col_lat or "": "Lat",
        col_lon or "": "Lon",
    }), use_container_width=True)
else:
    st.dataframe(df.head(50), use_container_width=True)

# KPIs rápidos
k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Total de sismos", f"{len(df):,}")
with k2:
    st.metric("Magnitud máx.", f"{df[col_mag].max():.1f}" if col_mag and not df[col_mag].dropna().empty else "N/D")
with k3:
    st.metric("Profundidad media", f"{df[col_prof].mean():.1f} km" if col_prof and not df[col_prof].dropna().empty else "N/D")

# Gráficos
if col_mag:
    st.subheader("Distribución de magnitudes")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(df[col_mag].dropna(), bins=25)
    ax.set_xlabel("Magnitud")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Histograma de magnitudes")
    st.pyplot(fig)

# Mapa
st.subheader("Mapa de sismos")
if (col_lat and col_lon) or ('lat_inferida' in df.columns and 'lon_inferida' in df.columns):
    lat_c = col_lat if col_lat else 'lat_inferida'
    lon_c = col_lon if col_lon else 'lon_inferida'
    map_df = df[[lat_c, lon_c] + ([col_mag] if col_mag else [])].dropna()
    map_df = map_df.rename(columns={lat_c: 'lat', lon_c: 'lon', (col_mag or ''): 'magnitud'})
    if not map_df.empty:
        st.map(map_df[['lat', 'lon']])
    else:
        st.info('No hay coordenadas válidas para mostrar en el mapa luego de los filtros.')
    # Lista destacada
    st.markdown("**Eventos destacados (top 10 por magnitud)**")
    if "magnitud" in map_df.columns:
        top = df.sort_values(col_mag, ascending=False).head(10)
        show_cols = []
        for c in [col_time, col_ref, col_mag, col_prof, col_lat, col_lon]:
            if c and c not in show_cols:
                show_cols.append(c)
        st.dataframe(top[show_cols].rename(columns={
            col_time or "": "Fecha/Hora",
            col_ref or "": "Referencia",
            col_mag or "": "Magnitud",
            col_prof or "": "Profundidad (km)",
            col_lat or "": "Lat",
            col_lon or "": "Lon",
        }), use_container_width=True)
else:
    st.info("Este dataset no incluye columnas de latitud/longitud reconocibles para mapa.")
