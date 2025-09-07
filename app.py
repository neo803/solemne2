# app.py
import json
import streamlit as st
import pandas as pd
from api_utils import search_datasets, dataset_resources, fetch_resource_to_df, best_time_column, numeric_columns
from analysis import clean_dataframe, summarize
from visuals import plot_timeseries, plot_histogram

st.set_page_config(page_title="Clima Chile - DataViz con APIs (datos.gob.cl)", layout="wide")

st.title("Clima Chile – Análisis y Visualización desde APIs públicas (datos.gob.cl)")
st.markdown("""
Esta aplicación permite **explorar datasets del grupo *Clima* de datos.gob.cl**, seleccionar un recurso y **analizar/visualizar** los datos.
- Fuente: API CKAN `package_search` y `datastore_search` de datos.gob.cl.
- Librerías: `requests`, `json`, `pandas`, `matplotlib`, `streamlit`.
""")

with st.sidebar:
    st.header("Búsqueda")
    q = st.text_input("Filtrar por palabra clave", value="temperatura")
    rows = st.slider("Cantidad de datasets a listar", 5, 50, 15, step=5)
    st.caption("Tip: prueba con términos como *temperatura*, *precipitaciones*, *meteorología*, *clima*.")
    run = st.button("Buscar")

if run or "datasets_cache" not in st.session_state:
    try:
        datasets = search_datasets(q=q, group="clima", rows=rows)
        st.session_state["datasets_cache"] = datasets
    except Exception as e:
        st.error(f"Error al consultar la API: {e}")
        st.stop()

datasets = st.session_state.get("datasets_cache", [])

if not datasets:
    st.warning("No se encontraron datasets en el grupo 'clima' con ese filtro.")
    st.stop()

# Selector de dataset
ds_titles = [f"{i+1}. {d.get('title', 'Sin título')}" for i, d in enumerate(datasets)]
ds_idx = st.selectbox("Selecciona un dataset del grupo *Clima*:", range(len(ds_titles)), format_func=lambda i: ds_titles[i])

ds = datasets[ds_idx]
st.subheader("Dataset seleccionado")
st.write(ds.get("title", "Sin título"))
st.write(ds.get("notes", ""))

res_list = dataset_resources(ds)
if not res_list:
    st.warning("El dataset no tiene recursos listados.")
    st.stop()

# Filtrar recursos CSV o Datastore
filtered = []
for r in res_list:
    fmt = (r.get("format") or "").lower()
    if fmt in ["csv", "json"] or r.get("id"):
        filtered.append(r)

if not filtered:
    st.warning("El dataset no tiene recursos compatibles (CSV/JSON/Datastore).")
    st.stop()

res_labels = [f"{r.get('name') or r.get('id')}  •  formato: {r.get('format')} " for r in filtered]
res_idx = st.selectbox("Recurso:", range(len(filtered)), format_func=lambda i: res_labels[i])

resource = filtered[res_idx]

st.info("⚙️ Intentando cargar datos del recurso vía API… (Datastore primero; si no, CSV)")
try:
    df, source_desc = fetch_resource_to_df(resource, limit=10000)
except Exception as e:
    st.error(f"No fue posible leer el recurso automáticamente: {e}")
    st.stop()

with st.expander("Metadatos del recurso"):
    st.json({k: resource.get(k) for k in ["id", "name", "description", "format", "url", "created", "last_modified"]})

st.success(f"Datos cargados desde: **{source_desc}**. Registros: {len(df):,}  •  Columnas: {len(df.columns)}")

df = clean_dataframe(df)

# Mostrar vista previa
st.subheader("Vista previa de datos")
st.dataframe(df.head(50))

# Detección de columna temporal y columnas numéricas
time_col_default = best_time_column(df)
num_cols = numeric_columns(df)

st.subheader("Exploración y filtros")
left, right = st.columns(2)
with left:
    time_col = st.selectbox("Columna de fecha/hora (opcional)", ["(ninguna)"] + list(df.columns), index=(list(df.columns).index(time_col_default)+1 if time_col_default in df.columns else 0))
with right:
    value_col = st.selectbox("Columna numérica para serie temporal", ["(ninguna)"] + num_cols, index=(num_cols.index(num_cols[0])+1 if num_cols else 0))

if time_col != "(ninguna)":
    # Intentar convertir a datetime
    try:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    except Exception as e:
        st.warning(f"No fue posible convertir {time_col} a fecha/hora automáticamente: {e}")

    # Filtro por rango de fechas si aplica
    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        min_dt = pd.to_datetime(df[time_col]).min()
        max_dt = pd.to_datetime(df[time_col]).max()
        st.caption(f"Rango detectado: {min_dt} — {max_dt}")
        date_range = st.date_input("Rango de fechas", value=(min_dt.date(), max_dt.date()))
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df[(df[time_col] >= start) & (df[time_col] <= end)]

# Resumen
st.subheader("Resumen estadístico")
st.dataframe(summarize(df))

# Gráficos
st.subheader("Visualizaciones")
if time_col != "(ninguna)" and value_col != "(ninguna)" and pd.api.types.is_numeric_dtype(df[value_col]) and pd.api.types.is_datetime64_any_dtype(df[time_col]):
    fig = plot_timeseries(df, time_col, value_col)
    st.pyplot(fig)
else:
    st.caption("Para la serie temporal, selecciona una columna de fecha/hora y una columna numérica.")

# Histogramas de otras columnas numéricas
if len(num_cols) >= 1:
    st.markdown("**Histogramas** de columnas numéricas")
    for col in num_cols[:4]:
        fig = plot_histogram(df, col)
        st.pyplot(fig)

st.markdown("---")
st.caption("Desarrollado para Solemne II – Clima (datos.gob.cl) • by Streamlit + Pandas + Matplotlib")
