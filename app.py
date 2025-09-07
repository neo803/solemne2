import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import re, math
import requests
import folium
from streamlit_folium import st_folium
from api_utils import fetch_sismos_chile

# -----------------------------
# Config & styles
# -----------------------------
st.set_page_config(page_title="Sismos Chile – Magnitud, Profundidad y Mapa", layout="wide")
st.title("🌎 Sismos en Chile – Magnitud, Profundidad y Mapa")
st.markdown('<style>div.block-container{padding-top:1rem;} .stMetric{background:#f8fafc;padding:12px;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,0.06);} .stButton>button{border-radius:10px;padding:0.45rem 0.9rem;} .stSelectbox>div, .stTextInput>div, .stSlider>div{border-radius:10px;}</style>', unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=300)
def load_sismos():
    df = fetch_sismos_chile()
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

def normalize_txt(s: str) -> str:
    if not isinstance(s, str):
        return ""
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()

# Full list of Chilean regions (display names)
REGIONES_CL = [
    "Región Arica y Parinacota",
    "Región Tarapacá",
    "Región Antofagasta",
    "Región Atacama",
    "Región Coquimbo",
    "Región Valparaíso",
    "Región Metropolitana",
    "Región O'Higgins",
    "Región Maule",
    "Región Ñuble",
    "Región Biobío",
    "Región La Araucanía",
    "Región Los Ríos",
    "Región Los Lagos",
    "Región Aysén",
    "Región Magallanes",
]

# City/reference anchors: (lat, lon, region)
ANCHORS = {
    # Norte
    "arica": (-18.474, -70.308, "Región Arica y Parinacota"),
    "iquique": (-20.216, -70.142, "Región Tarapacá"),
    "antofagasta": (-23.650, -70.400, "Región Antofagasta"),
    "copiapo": (-27.366, -70.333, "Región Atacama"),
    "la serena": (-29.904, -71.248, "Región Coquimbo"),
    "coquimbo": (-29.953, -71.338, "Región Coquimbo"),
    # Centro
    "valparaiso": (-33.045, -71.620, "Región Valparaíso"),
    "viña del mar": (-33.0246, -71.5518, "Región Valparaíso"),
    "quillota": (-32.880, -71.250, "Región Valparaíso"),
    "san felipe": (-32.750, -70.722, "Región Valparaíso"),
    "los andes": (-32.833, -70.598, "Región Valparaíso"),
    "san antonio": (-33.600, -71.610, "Región Valparaíso"),
    "santiago": (-33.450, -70.660, "Región Metropolitana"),
    "rancagua": (-34.170, -70.740, "Región O'Higgins"),
    "curicó": (-34.985, -71.239, "Región Maule"),
    "curico": (-34.985, -71.239, "Región Maule"),
    "talca": (-35.426, -71.655, "Región Maule"),
    "linares": (-35.846, -71.594, "Región Maule"),
    "chillán": (-36.606, -72.103, "Región Ñuble"),
    "chillan": (-36.606, -72.103, "Región Ñuble"),
    "los angeles": (-37.470, -72.353, "Región Biobío"),
    "concepción": (-36.827, -73.050, "Región Biobío"),
    "concepcion": (-36.827, -73.050, "Región Biobío"),
    # Sur
    "temuco": (-38.735, -72.590, "Región La Araucanía"),
    "valdivia": (-39.819, -73.245, "Región Los Ríos"),
    "osorno": (-40.574, -73.133, "Región Los Lagos"),
    "puerto montt": (-41.469, -72.942, "Región Los Lagos"),
    "coyhaique": (-45.571, -72.068, "Región Aysén"),
    "punta arenas": (-53.163, -70.917, "Región Magallanes"),
}

DIR_BEARINGS = {
    "N": 0, "NORTE": 0,
    "NE": 45, "NORESTE": 45,
    "E": 90, "ESTE": 90,
    "SE": 135, "SURESTE": 135,
    "S": 180, "SUR": 180,
    "SO": 225, "SUROESTE": 225, "SW": 225,
    "O": 270, "OESTE": 270, "W": 270,
    "NO": 315, "NOROESTE": 315, "NW": 315,
}

def parse_directional_reference(text):
    \"\"\"Parse strings like '35 km al NO de Los Andes' -> (35.0, 315, 'los andes').\"\"\"
    if not isinstance(text, str):
        return None
    t = normalize_txt(text)
    # Variantes: 'km al NO de X', 'km al noroeste de X', 'km al oeste de X'
    pat = r\"(\\d+(?:[\\.,]\\d+)?)\\s*km\\s+(?:al\\s+)?(n|ne|no|e|se|s|so|o|w|nw|sw|noreste|noroeste|sureste|suroeste|este|oeste|norte|sur)\\s+de\\s+(.+)$\"
    m = re.search(pat, t, flags=re.IGNORECASE)
    if not m:
        return None
    dist = float(m.group(1).replace(',', '.'))
    dir_token = m.group(2).upper()
    dir_token = {'NORTE':'N','SUR':'S','ESTE':'E','OESTE':'O','NORESTE':'NE','NOROESTE':'NO','SURESTE':'SE','SUROESTE':'SO','W':'O','NW':'NO','SW':'SO'}.get(dir_token, dir_token)
    place = m.group(3).strip()
    # cortar ruido después de coma/paréntesis
    place = re.split(r\"[,\\(]\", place)[0].strip()
    return (dist, DIR_BEARINGS.get(dir_token), place)

def destination_point(lat, lon, distance_km, bearing_deg):
    R = 6371.0
    br = math.radians(bearing_deg)
    d = distance_km / R
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(d) + math.cos(lat1)*math.sin(d)*math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br)*math.sin(d)*math.cos(lat1), math.cos(d) - math.sin(lat1)*math.sin(lat2))
    return (math.degrees(lat2), math.degrees(lon2))

def infer_coords_from_reference(text):
    \"\"\"Return (lat, lon) from reference:
    1) Detect explicit coords.
    2) Parse 'X km al DIR de PLACE' and compute from anchor city.\n    \"\"\"
    # 1) explicit coords
    if isinstance(text, str):
        m = re.search(r'([-+]?\\d{1,2}(?:[\\.,]\\d+)?)[\\s,;]+([-+]?\\d{1,3}(?:[\\.,]\\d+)?)', text)
        if m:
            try:
                la = float(m.group(1).replace(',', '.')); lo = float(m.group(2).replace(',', '.'))
                if -56 <= la <= -17 and -76 <= lo <= -66:
                    return (la, lo)
            except Exception:
                pass
    # 2) directional
    parsed = parse_directional_reference(text)
    if parsed:
        dist, bearing, place = parsed
        place_norm = normalize_txt(place)
        if place_norm in ANCHORS:
            lat0, lon0, _ = ANCHORS[place_norm]
            la, lo = destination_point(lat0, lon0, dist, bearing)
            if -56 <= la <= -17 and -76 <= lo <= -66:
                return (la, lo)
    return (None, None)

# Region extraction from reference
def extract_region_from_reference(text):
    try:
        if not isinstance(text, str) or not text.strip():
            return None
        t = re.sub(r\"\\s+\", \" \", text).strip()
        m = re.search(r\"Regi[oó]n\\s+(?:de|del)?\\s*([A-Za-zÁÉÍÓÚÑáéíóú\\s\\-]+)\", t, flags=re.IGNORECASE)
        if m:
            reg = m.group(0).strip()
            reg = \" \".join(w.capitalize() for w in reg.split())
            reg = reg.replace(\"Ohiggins\",\"O'Higgins\").replace(\"Nuble\",\"Ñuble\").replace(\"Aysen\",\"Aysén\")
            return reg
    except Exception:
        return None
    return None

def detect_region_from_name(text):
    t = normalize_txt(text)
    for r in REGIONES_CL:
        if normalize_txt(r).split(' ',1)[1] in t:  # check region name part
            return r
    return None

def region_from_lat(lat):
    if lat is None or (isinstance(lat, float) and np.isnan(lat)):
        return None
    try:
        lat = float(lat)
    except Exception:
        return None
    if lat >= -20: return \"Región Arica y Parinacota\"
    if lat >= -23: return \"Región Tarapacá\"
    if lat >= -27: return \"Región Antofagasta\"
    if lat >= -30: return \"Región Atacama\"
    if lat >= -32: return \"Región Coquimbo\"
    if lat >= -33: return \"Región Valparaíso\"
    if lat >= -34: return \"Región Metropolitana\"
    if lat >= -34.9: return \"Región O'Higgins\"
    if lat >= -36: return \"Región Maule\"
    if lat >= -36.5: return \"Región Ñuble\"
    if lat >= -38: return \"Región Biobío\"
    if lat >= -39.5: return \"Región La Araucanía\"
    if lat >= -41.5: return \"Región Los Ríos\"
    if lat >= -45: return \"Región Los Lagos\"
    if lat >= -52: return \"Región Aysén\"
    return \"Región Magallanes\"

@st.cache_data(ttl=86400)
def get_chile_regions_geojson():
    urls = [
        \"https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/chile-regions.geojson\",
        \"https://raw.githubusercontent.com/johan/world.geo.json/master/countries/CHL/CHL-ADM1.geo.json\",
    ]
    for u in urls:
        try:
            r = requests.get(u, timeout=30)
            if r.ok:
                return r.json()
        except Exception:
            continue
    return None

def en_to_es_region(name_en: str) -> str | None:
    if not isinstance(name_en, str):
        return None
    t = name_en.lower()
    if \"arica\" in t: return \"Región Arica y Parinacota\"
    if \"tarap\" in t: return \"Región Tarapacá\"
    if \"antof\" in t: return \"Región Antofagasta\"
    if \"atacama\" in t: return \"Región Atacama\"
    if \"coquimbo\" in t: return \"Región Coquimbo\"
    if \"valpar\" in t: return \"Región Valparaíso\"
    if \"metropolitan\" in t or \"santiago\" in t: return \"Región Metropolitana\"
    if \"higgins\" in t: return \"Región O'Higgins\"
    if \"maule\" in t: return \"Región Maule\"
    if \"nuble\" in t or \"ñuble\" in t: return \"Región Ñuble\"
    if \"bio\" in t: return \"Región Biobío\"
    if \"araucan\" in t: return \"Región La Araucanía\"
    if \"los rios\" in t or \"r\\u00edos\" in t: return \"Región Los Ríos\"
    if \"los lagos\" in t: return \"Región Los Lagos\"
    if \"ays\" in t: return \"Región Aysén\"
    if \"magallanes\" in t: return \"Región Magallanes\"
    return None

# -----------------------------
# Data
# -----------------------------
df = load_sismos()
if df.empty:
    st.error(\"No se pudieron cargar los datos de sismos.\")
    st.stop()

# Detect columns
col_mag = find_col(df, [\"Magnitud\", \"mag\", \"magnitude\"])
col_prof = find_col(df, [\"Profundidad\", \"depth\", \"prof\"])
col_lat = find_col(df, [\"Latitud\", \"lat\", \"latitude\"])
col_lon = find_col(df, [\"Longitud\", \"lon\", \"lng\", \"longitude\"])
col_time = find_col(df, [\"Fecha\", \"fecha\", \"time\", \"fechaLocal\", \"Fecha UTC\", \"TimeStamp\"])
col_ref  = find_col(df, [\"Referencia Geografica\", \"Referencia\", \"refgeo\", \"lugar\", \"place\"])

# Coercions
if col_mag: df[col_mag] = coerce_numeric(df[col_mag])
if col_prof:
    df[col_prof] = df[col_prof].astype(str).str.extract(r\"([\\d\\.,]+)\", expand=False).str.replace(\",\", \".\", regex=False)
    df[col_prof] = coerce_numeric(df[col_prof])
if col_lat: df[col_lat] = coerce_numeric(df[col_lat])
if col_lon: df[col_lon] = coerce_numeric(df[col_lon])
if col_time: df[col_time] = coerce_datetime(df[col_time])

# Region from reference or by name
if col_ref and \"region_extraida\" not in df.columns:
    df[\"region_extraida\"] = df[col_ref].apply(extract_region_from_reference)
    mask = df[\"region_extraida\"].isna()
    df.loc[mask, \"region_extraida\"] = df.loc[mask, col_ref].apply(detect_region_from_name)

# Infer lat/lon from reference when missing
if not col_lat and not col_lon and col_ref:
    lats, lons = [], []
    for txt in df[col_ref]:
        la, lo = infer_coords_from_reference(txt)
        lats.append(la); lons.append(lo)
    df[\"lat_inferida\"], df[\"lon_inferida\"] = pd.Series(lats, index=df.index), pd.Series(lons, index=df.index)

# Region calculated from lat fallback
lat_c = col_lat if col_lat else (\"lat_inferida\" if \"lat_inferida\" in df.columns else None)
if lat_c:
    if \"region_extraida\" in df.columns:
        df[\"region_calculada\"] = df[\"region_extraida\"]
        mask = df[\"region_calculada\"].isna()
        df.loc[mask, \"region_calculada\"] = df.loc[mask, lat_c].apply(region_from_lat)
    else:
        df[\"region_calculada\"] = df[lat_c].apply(region_from_lat)

# -----------------------------
# Sidebar filters & options
# -----------------------------
with st.sidebar:
    st.header(\"Filtros\")
    # Full region list
    region_sel = st.selectbox(\"Región\", [\"(todas)\"] + REGIONES_CL, index=0)
    min_mag = st.slider(\"Magnitud mínima\", 0.0, 10.0, 3.0, 0.1)
    if col_time and pd.api.types.is_datetime64_any_dtype(df[col_time]):
        tmin, tmax = df[col_time].min(), df[col_time].max()
        date_range = st.date_input(\"Rango de fechas\", value=(tmin.date(), tmax.date()))
    else:
        date_range = None
    text_ref = st.text_input(\"Texto a buscar en referencia (opcional)\", \"\")
    st.header(\"Opciones de mapa\")
    color_by = st.selectbox(\"Color por\", [\"profundidad\", \"magnitud\"])
    radius_base = st.slider(\"Radio base (px ~ escala)\", 1000, 80000, 15000, 1000)
    st.caption(\"También puedes seleccionar la región clicando en el mapa de polígonos más abajo.\")
    apply_btn = st.button(\"Aplicar filtros\")

# -----------------------------
# Region picker map (folium)
# -----------------------------
st.subheader(\"Selector de región en mapa\")
selected_region_click = None
geo = get_chile_regions_geojson()
if geo:
    m = folium.Map(location=[-33.5, -70.6], zoom_start=4, tiles=\"CartoDB positron\")
    gj = folium.GeoJson(
        geo,
        name=\"Regiones\",
        style_function=lambda f: {\"fillColor\": \"#74add1\", \"color\": \"#2c7fb8\", \"weight\": 1, \"fillOpacity\": 0.2},
        highlight_function=lambda f: {\"weight\": 3, \"fillOpacity\": 0.6},
        tooltip=folium.GeoJsonTooltip(fields=[\"name\"], aliases=[\"Región\"])
    )
    gj.add_to(m)
    out = st_folium(m, height=420, use_container_width=True, returned_objects=[\"last_object_clicked\")
    if out and out.get(\"last_object_clicked\"):
        props = out[\"last_object_clicked\"].get(\"properties\", {})
        name_en = props.get(\"name\") or props.get(\"NAME_1\")
        selected_region_click = en_to_es_region(name_en)
        if selected_region_click:
            st.success(f\"Región seleccionada en el mapa: {selected_region_click}\")
else:
    st.info(\"No fue posible cargar el GeoJSON de regiones. Usa el selector de la barra lateral.\")

# -----------------------------
# Apply filters
# -----------------------------
chosen_region = selected_region_click if selected_region_click else (region_sel if region_sel != \"(todas)\" else None)

if min_mag and col_mag:
    df = df[df[col_mag] >= min_mag]
if date_range and isinstance(date_range, tuple) and len(date_range) == 2 and col_time:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[(df[col_time] >= start) & (df[col_time] <= end)]
if text_ref and col_ref:
    df = df[df[col_ref].astype(str).str.contains(text_ref, case=False, na=False)]
if chosen_region:
    if \"region_calculada\" in df.columns:
        df = df[df[\"region_calculada\"] == chosen_region]
    elif \"region_extraida\" in df.columns:
        df = df[df[\"region_extraida\"] == chosen_region]

# -----------------------------
# Shareable link & CSV
# -----------------------------
params = {\"min_mag\": min_mag, \"region\": chosen_region or \"\", \"text\": text_ref or \"\", \"color\": color_by, \"radius\": radius_base}
try:
    st.experimental_set_query_params(**params)
except Exception:
    pass
c1, c2 = st.columns([1,1])
with c1:
    if st.button(\"Copiar enlace con filtros\"):
        try:
            st.write(\"Parámetros guardados en la URL. Usa copiar del navegador.\")
        except Exception:
            pass
with c2:
    csv = df.to_csv(index=False).encode(\"utf-8\")
    st.download_button(\"Descargar CSV filtrado\", data=csv, file_name=\"sismos_filtrados.csv\", mime=\"text/csv\")

# -----------------------------
# Table
# -----------------------------
st.subheader(\"Tabla de sismos\")
cols_show = []
for c in [col_time, col_ref, col_mag, col_prof, col_lat, col_lon, \"lat_inferida\", \"lon_inferida\"]:
    if c and c in df.columns and c not in cols_show:
        cols_show.append(c)

ren = {}
if col_time: ren[col_time] = \"Fecha/Hora\"
if col_ref: ren[col_ref] = \"Referencia\"
if col_mag: ren[col_mag] = \"Magnitud\"
if col_prof: ren[col_prof] = \"Profundidad (km)\"
if col_lat: ren[col_lat] = \"Lat\"
if col_lon: ren[col_lon] = \"Lon\"
if \"lat_inferida\" in df.columns: ren[\"lat_inferida\"] = \"Lat\"
if \"lon_inferida\" in df.columns: ren[\"lon_inferida\"] = \"Lon\"

st.dataframe(df[cols_show].rename(columns=ren) if cols_show else df.head(50), use_container_width=True)

# -----------------------------
# KPIs
# -----------------------------
k1, k2, k3 = st.columns(3)
with k1:
    st.metric(\"Total de sismos\", f\"{len(df):,}\")
with k2:
    st.metric(\"Magnitud máx.\", f\"{df[col_mag].max():.1f}\" if col_mag and not df[col_mag].dropna().empty else \"N/D\")
with k3:
    st.metric(\"Profundidad media\", f\"{df[col_prof].mean():.1f} km\" if col_prof and not df[col_prof].dropna().empty else \"N/D\")

# -----------------------------
# Histogram
# -----------------------------
if col_mag and not df[col_mag].dropna().empty:
    st.subheader(\"Distribución de magnitudes\")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(df[col_mag].dropna(), bins=25)
    ax.set_xlabel(\"Magnitud\"); ax.set_ylabel(\"Frecuencia\"); ax.set_title(\"Histograma de magnitudes\")
    st.pyplot(fig)

# -----------------------------
# Map of points (pydeck)
# -----------------------------
st.subheader(\"Mapa de sismos (puntos)\")
lat_used = col_lat if col_lat in df.columns else (\"lat_inferida\" if \"lat_inferida\" in df.columns else None)
lon_used = col_lon if col_lon in df.columns else (\"lon_inferida\" if \"lon_inferida\" in df.columns else None)

if lat_used and lon_used:
    map_cols = [lat_used, lon_used]
    if col_mag: map_cols.append(col_mag)
    if col_prof: map_cols.append(col_prof)
    if col_ref: map_cols.append(col_ref)
    if col_time: map_cols.append(col_time)

    map_df = df[map_cols].dropna(subset=[lat_used, lon_used]).copy()
    map_df = map_df.rename(columns={lat_used: \"lat\", lon_used: \"lon\"})
    if col_mag and col_mag in map_df.columns: map_df = map_df.rename(columns={col_mag: \"magnitud\"})
    if col_prof and col_prof in map_df.columns: map_df = map_df.rename(columns={col_prof: \"prof_km\"})
    if col_ref and col_ref in map_df.columns: map_df = map_df.rename(columns={col_ref: \"referencia\"})
    if col_time and col_time in map_df.columns: map_df = map_df.rename(columns={col_time: \"fecha\"})

    if color_by == \"profundidad\" and \"prof_km\" in map_df.columns and not map_df[\"prof_km\"].dropna().empty:
        prof = map_df[\"prof_km\"].fillna(map_df[\"prof_km\"].median())
        norm = (prof - prof.min()) / (prof.max() - prof.min() + 1e-9)
        colors = np.stack([(norm*255).astype(int), ((1-norm)*200+30).astype(int), np.full(len(norm),80), np.full(len(norm),180)], axis=1)
    elif color_by == \"magnitud\" and \"magnitud\" in map_df.columns and not map_df[\"magnitud\"].dropna().empty:
        mag = map_df[\"magnitud\"].fillna(map_df[\"magnitud\"].median())
        norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-9)
        colors = np.stack([(norm*255).astype(int), np.full(len(norm),120), ((1-norm)*255).astype(int), np.full(len(norm),180)], axis=1)
    else:
        colors = np.tile(np.array([30,144,255,180]), (len(map_df),1))

    map_df[\"_color_r\"], map_df[\"_color_g\"], map_df[\"_color_b\"], map_df[\"_color_a\"] = colors[:,0], colors[:,1], colors[:,2], colors[:,3]
    if \"magnitud\" in map_df.columns and not map_df[\"magnitud\"].dropna().empty:
        map_df[\"_radius\"] = (map_df[\"magnitud\"].fillna(3.0) * radius_base).clip(radius_base*0.2, radius_base*4)
    else:
        map_df[\"_radius\"] = np.full(len(map_df), radius_base)

    layer = pdk.Layer(\"ScatterplotLayer\", data=map_df, get_position=\"[lon, lat]\", get_color=\"[_color_r,_color_g,_color_b,_color_a]\", get_radius=\"_radius\", pickable=True, auto_highlight=True)
    view_state = pdk.ViewState(latitude=-33.45, longitude=-70.66, zoom=3.8, pitch=0)
    tooltip = {\"html\": \"<b>Magnitud:</b> {magnitud}<br/><b>Profundidad:</b> {prof_km} km<br/><b>Fecha:</b> {fecha}<br/><b>Ref:</b> {referencia}<br/><b>Lat:</b> {lat} · <b>Lon:</b> {lon}\", \"style\": {\"backgroundColor\": \"rgba(0,0,0,0.72)\", \"color\": \"white\"}}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
else:
    st.info(\"No hay columnas de coordenadas (ni reales ni inferidas) disponibles para dibujar el mapa.\")
