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

df = load_sismos()
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
    apply_btn = st.button("Aplicar filtros")

if apply_btn:
    if col_mag:
        df = df[df[col_mag] >= min_mag]
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2 and col_time:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[(df[col_time] >= start) & (df[col_time] <= end)]

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
if col_lat and col_lon:
    map_df = df[[col_lat, col_lon] + ([col_mag] if col_mag else [])].dropna()
    map_df = map_df.rename(columns={col_lat: "lat", col_lon: "lon", col_mag or "": "magnitud"})
    st.map(map_df[["lat", "lon"]])
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
