import streamlit as st
import pandas as pd
from api_utils import (
    search_datasets, dataset_resources, fetch_resource_to_df,
    best_time_column, numeric_columns,
    fetch_mindicador, fetch_sismos_chile, fetch_openweather_current
)
from analysis import summarize
from visuals import plot_timeseries, plot_histogram

# Configuración de la página
st.set_page_config(page_title="Clima Chile - DataViz", layout="wide")

st.title("Clima Chile – Análisis y Visualización desde APIs públicas")

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Explorador (datos.gob.cl)",
    "📈 Mindicador",
    "🌎 Sismos Chile",
    "☁️ OpenWeather"
])

# -------------------------------
# Tab 1 – Explorador CKAN
# -------------------------------
with tab1:
    st.markdown("Explora datasets por **categoría** (grupo CKAN) y recursos compatibles (Datastore/CSV).")

    with st.sidebar:
        st.header("Búsqueda (CKAN)")
        group = st.selectbox(
            "Categoría (grupo CKAN)",
            [
                "medio_ambiente", "economia", "educacion", "salud",
                "transporte", "agricultura", "ciencia", "cultura",
                "gobierno", "territorio", "energia", "tecnologia", "general"
            ],
            index=0
        )
        q = st.text_input("Filtrar por palabra clave (opcional)", value="")
        rows = st.slider("Cantidad de datasets a listar", 5, 50, 15, step=5)
        run = st.button("Buscar datasets")

    if run or "datasets_cache" not in st.session_state:
        try:
            datasets = search_datasets(q=q or None, group=group, rows=rows)
            st.session_state["datasets_cache"] = datasets
        except Exception as e:
            st.error(f"Error al consultar la API CKAN: {e}")
            datasets = []

    datasets = st.session_state.get("datasets_cache", [])
    if not datasets:
        st.warning("No se encontraron datasets para esos filtros.")
    else:
        ds_titles = [f"{i+1}. {d.get('title', 'Sin título')}" for i, d in enumerate(datasets)]
        ds_idx = st.selectbox("Dataset:", range(len(ds_titles)), format_func=lambda i: ds_titles[i])
        ds = datasets[ds_idx]
        st.subheader(ds.get("title", "Sin título"))
        st.write(ds.get("notes", ""))

        res_list = dataset_resources(ds)
        filtered = []
        for r in res_list:
            fmt = (r.get("format") or "").lower()
            if fmt in ["csv", "json"] or r.get("id"):
                filtered.append(r)

        if not filtered:
            st.info("El dataset no tiene recursos CSV/JSON/Datastore parseables.")
        else:
            res_labels = [f"{r.get('name') or r.get('id')}  •  formato: {r.get('format')}" for r in filtered]
            res_idx = st.selectbox("Recurso:", range(len(filtered)), format_func=lambda i: res_labels[i])
            resource = filtered[res_idx]

            try:
                df, source_desc = fetch_resource_to_df(resource, limit=10000)
                st.success(f"Cargado desde {source_desc}. {len(df):,} filas × {len(df.columns)} cols")
                st.dataframe(df.head(100))

                time_col_default = best_time_column(df)
                num_cols = numeric_columns(df)

                left, right = st.columns(2)
                with left:
                    time_col = st.selectbox(
                        "Columna fecha/hora (opcional)",
                        ["(ninguna)"] + list(df.columns),
                        index=(list(df.columns).index(time_col_default)+1 if time_col_default in df.columns else 0)
                    )
                with right:
                    value_col = st.selectbox("Columna numérica (opcional)", ["(ninguna)"] + num_cols, index=0)

                st.subheader("Resumen estadístico")
                st.dataframe(summarize(df))

                st.subheader("Visualizaciones")
                if time_col != "(ninguna)" and value_col != "(ninguna)":
                    df_plot = df.copy()
                    df_plot[time_col] = pd.to_datetime(df_plot[time_col], errors="coerce")
                    if pd.api.types.is_datetime64_any_dtype(df_plot[time_col]) and pd.api.types.is_numeric_dtype(df_plot[value_col]):
                        fig = plot_timeseries(df_plot.dropna(subset=[time_col, value_col]), time_col, value_col)
                        st.pyplot(fig)

                if num_cols:
                    st.markdown("**Histogramas**")
                    for col in num_cols[:4]:
                        fig = plot_histogram(df, col)
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"No fue posible leer el recurso: {e}")

# -------------------------------
# Tab 2 – Mindicador
# -------------------------------
with tab2:
    st.markdown("Serie histórica de **mindicador.cl** (ej.: `uf`, `dolar`, `euro`, `ipc`).")
    c1, c2 = st.columns(2)
    with c1:
        indicador = st.text_input("Indicador", value="dolar")
    with c2:
        year = st.text_input("Año", value="2024")
    if st.button("Consultar Mindicador"):
        try:
            df = fetch_mindicador(indicador, year)
            if df.empty:
                st.warning("Sin datos para ese indicador/año.")
            else:
                st.dataframe(df.tail(20))
                fig = plot_timeseries(df, "fecha", "valor")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error al consultar mindicador: {e}")

# -------------------------------
# Tab 3 – Sismos Chile
# -------------------------------
with tab3:
    st.markdown("Últimos **sismos en Chile** (GAEL Cloud).")
    if st.button("Cargar sismos"):
        try:
            df = fetch_sismos_chile()
            st.dataframe(df.head(50))
            if "Magnitud" in df.columns:
                fig = plot_histogram(df, "Magnitud")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error al consultar sismos: {e}")

# -------------------------------
# Tab 4 – OpenWeather
# -------------------------------
with tab4:
    st.markdown("Clima **actual** por ciudad desde **OpenWeather** (requiere tu API key).")
    c1, c2 = st.columns(2)
    with c1:
        city = st.text_input("Ciudad", value="Santiago")
    with c2:
        api_key = st.text_input("OpenWeather API key", type="password")
    if st.button("Consultar OpenWeather"):
        if not api_key:
            st.warning("Ingresa tu API key de OpenWeather.")
        else:
            try:
                df = fetch_openweather_current(city, api_key)
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error al consultar OpenWeather: {e}")
