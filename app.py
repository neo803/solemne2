import streamlit as st
import pandas as pd
import re
import pydeck as pdk
from api_utils import fetch_sismos_chile

st.set_page_config(page_title="Sismos Chile – Magnitud, Profundidad y Mapa", layout="wide")
st.title("🌎 Sismos en Chile – Magnitud, Profundidad y Mapa")
st.markdown('<style>div.block-container{padding-top:1rem;} .stMetric{background: #f8fafc; padding: 12px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);} .stButton>button{border-radius:10px; padding:0.4rem 0.9rem;} .stSelectbox>div, .stTextInput>div, .stSlider>div{border-radius:10px;}</style>', unsafe_allow_html=True)


st.markdown(
    "Datos en tiempo real desde **GAEL Cloud**. Filtra por magnitud, fechas o región, "
    "y visualiza un mapa interactivo con la ubicación de los eventos."
)

# ----------------------
# Helper functions
# ----------------------
def coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def coerce_datetime(series):
    return pd.to_datetime(series, errors="coerce")

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

def infer_coords_from_reference(text):
    if not isinstance(text, str):
        return (None, None)
    patterns = [
        r'([-+]?\\d{1,2}(?:[\\.,]\\d+)?)[\\s,;]+([-+]?\\d{1,3}(?:[\\.,]\\d+)?)',
        r'lat(?:itud)?\\s*[:=]?\\s*([-+]?\\d{1,2}(?:[\\.,]\\d+)?)[^\\d-]+lon(?:gitud)?\\s*[:=]?\\s*([-+]?\\d{1,3}(?:[\\.,]\\d+)?)',
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
    if not isinstance(text, str):
        return None
    m = re.search(r'(Regi[oó]n\\s+(?:de|del)\\s+[A-Za-zÁÉÍÓÚÑáéíóú\\s]+)', text, flags=re.IGNORECASE)
    if m:
        reg = m.group(1)
        reg = re.sub(r'\\s+', ' ', reg).strip().title()
        return reg
    return None

@st.cache_data(ttl=300)
def load_sismos():
    df = fetch_sismos_chile()
    df.columns = [str(c).strip() for c in df.columns]
    return df

df = load_sismos()
if df.empty:
    st.error("No se pudieron cargar los datos de sismos.")
    st.stop()

# Detectar columnas
col_mag = find_col(df, ["Magnitud", "mag", "magnitude"])
col_prof = find_col(df, ["Profundidad", "depth", "prof"])
col_lat = find_col(df, ["Latitud", "lat", "latitude"])
col_lon = find_col(df, ["Longitud", "lon", "lng", "longitude"])
col_time = find_col(df, ["Fecha", "fecha", "time", "fechaLocal", "Fecha UTC", "TimeStamp"])
col_ref = find_col(df, ["Referencia Geografica", "Referencia", "refgeo", "lugar", "place"])

# Parsear
if col_mag: df[col_mag] = coerce_numeric(df[col_mag])
if col_prof:
    df[col_prof] = df[col_prof].astype(str).str.extract(r"([\\d\\.,]+)", expand=False).str.replace(",", ".", regex=False)
    df[col_prof] = coerce_numeric(df[col_prof])
if col_lat: df[col_lat] = coerce_numeric(df[col_lat])
if col_lon: df[col_lon] = coerce_numeric(df[col_lon])
if col_time: df[col_time] = coerce_datetime(df[col_time])

# Extraer región
if col_ref and "region_extraida" not in df.columns:
    df["region_extraida"] = df[col_ref].apply(extract_region_from_reference)

# Inferir coords si no existen
if not col_lat or not col_lon:
    if col_ref:
        lat_list, lon_list = [], []
        for txt in df[col_ref]:
            lat, lon = infer_coords_from_reference(txt)
            lat_list.append(lat); lon_list.append(lon)
        df["lat_inferida"] = pd.Series(lat_list, index=df.index)
        df["lon_inferida"] = pd.Series(lon_list, index=df.index)

# ----------------------
# Sidebar filtros
# ----------------------
with st.sidebar:
    st.header("Filtros")
    min_mag = st.slider("Magnitud mínima", 0.0, 10.0, 3.0, 0.1)
    if col_time and pd.api.types.is_datetime64_any_dtype(df[col_time]):
        tmin, tmax = df[col_time].min(), df[col_time].max()
        date_range = st.date_input("Rango de fechas", value=(tmin.date(), tmax.date()))
    else:
        date_range = None
    region_opts = sorted(df["region_extraida"].dropna().unique().tolist()) if "region_extraida" in df.columns else []
    region_sel = st.selectbox("Región (opcional)", ["(todas)"] + region_opts if region_opts else ["(no detectadas)"])
    text_ref = st.text_input("Texto a buscar en referencia (opcional)", "")
    apply_btn = st.button("Aplicar filtros")

if apply_btn:
    if col_mag:
        df = df[df[col_mag] >= min_mag]
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2 and col_time:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        df = df[(df[col_time] >= start) & (df[col_time] <= end)]
    if "region_extraida" in df.columns and region_sel not in ["(todas)", "(no detectadas)"]:
        df = df[df["region_extraida"] == region_sel]
    if text_ref and col_ref:
        df = df[df[col_ref].astype(str).str.contains(text_ref, case=False, na=False)]

# ----------------------
# Métricas rápidas
# ----------------------
st.subheader("📊 Métricas rápidas")
k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Total de sismos", f"{len(df):,}")
with k2:
    st.metric("Magnitud máx.", f"{df[col_mag].max():.1f}" if col_mag and not df[col_mag].dropna().empty else "N/D")
with k3:
    st.metric("Profundidad media", f"{df[col_prof].mean():.1f} km" if col_prof and not df[col_prof].dropna().empty else "N/D")

# ----------------------
# Tabla de eventos
# ----------------------
st.subheader("📋 Tabla de sismos")
cols_show = []
for c in [col_time, col_ref, col_mag, col_prof, col_lat or "lat_inferida", col_lon or "lon_inferida"]:
    if c and c not in cols_show and c in df.columns:
        cols_show.append(c)

if cols_show:
    st.dataframe(df[cols_show].rename(columns={
        col_time or "": "Fecha/Hora",
        col_ref or "": "Referencia",
        col_mag or "": "Magnitud",
        col_prof or "": "Profundidad (km)",
        col_lat or "lat_inferida": "Lat",
        col_lon or "lon_inferida": "Lon",
    }), use_container_width=True)
else:
    st.dataframe(df.head(50), use_container_width=True)

# ----------------------
# Histograma magnitudes
# ----------------------
if col_mag:
    st.subheader("📈 Distribución de magnitudes")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(df[col_mag].dropna(), bins=25, color="orange", edgecolor="black")
    ax.set_xlabel("Magnitud")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Histograma de magnitudes")
    st.pyplot(fig)

# ----------------------
# Mapa interactivo
# ----------------------
st.subheader("🗺️ Mapa de sismos")
lat_c = col_lat if col_lat in df.columns else ("lat_inferida" if "lat_inferida" in df.columns else None)
lon_c = col_lon if col_lon in df.columns else ("lon_inferida" if "lon_inferida" in df.columns else None)
if lat_c and lon_c:
    map_df = df[[lat_c, lon_c, col_mag or df.columns[0], col_prof or df.columns[0], col_ref or df.columns[0]]].dropna()
    map_df = map_df.rename(columns={lat_c: "lat", lon_c: "lon", col_mag or "": "magnitud", col_prof or "": "profundidad", col_ref or "": "referencia"})
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=-33.45, longitude=-70.66, zoom=4, pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_fill_color="[255, 140, 0, 160]",
                get_radius="magnitud * 10000",
                pickable=True,
            )
        ],
        tooltip={"text": "Magnitud: {magnitud}\nProfundidad: {profundidad} km\n{referencia}"}
    ))
else:
    st.info("No hay coordenadas disponibles para mostrar en el mapa.")
