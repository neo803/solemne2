# Solemne II – Proyecto Clima (datos.gob.cl)
**Objetivo:** Aplicación completa en **Python + Streamlit** que consulta **APIs públicas** de **datos.gob.cl** (grupo *Clima*), analiza y visualiza datos.

## Estructura
```
solemne_clima/
├─ app.py              # Aplicación web Streamlit (UI)
├─ api_utils.py        # Lógica de consultas a la API CKAN/recursos
├─ analysis.py         # Limpieza y resúmenes con Pandas
├─ visuals.py          # Gráficos con Matplotlib
├─ requirements.txt    # Dependencias
├─ Informe_Solemne_II.md
└─ Poster_Solemne_II.png
```

## Cómo ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```
La app hace **GET** a:
- `https://datos.gob.cl/api/3/action/package_search` (grupo `clima`)
- `https://datos.gob.cl/api/3/action/datastore_search` (cuando el recurso tiene Datastore)

Si el recurso no está en Datastore, intenta leer vía **CSV** por HTTP GET.

## Despliegue (opcional)
- Streamlit Community Cloud o servicios similares.
- En Streamlit Cloud, sube el repo y define `app.py` como entrypoint.

## Notas de evaluación
- Se cumple con las librerías permitidas (requests, json, pandas, matplotlib, streamlit).
- Se implementa **interactividad** (filtros, selección de dataset/recurso, rango de fechas, gráficos).
- Se incluyen **entregables**: código + informe + póster.
