# Informe – Solemne II (Clima, datos.gob.cl)

**Autor(es):** _Completar_  
**Asignatura:** DataViz Python – Unidad 3, Semana 13  
**Fecha:** 2025-09-07

## 1. Desafío
Desarrollar una aplicación completa en Python que **consuma datos climáticos** desde una **API REST pública** de **datos.gob.cl**, realice **análisis** con Pandas y presente los resultados en una **interfaz interactiva** con Streamlit.

## 2. Fuente de datos
- Portal: https://datos.gob.cl/group/clima  
- API CKAN utilizada: `package_search` (búsqueda de datasets del grupo *Clima*) y `datastore_search` (lectura tabular cuando corresponde).  
- En caso de recursos no alojados en Datastore, se realiza lectura por **HTTP GET (CSV)**.

## 3. Tecnologías
- **Lenguaje:** Python  
- **Librerías:** requests, json, pandas, matplotlib, streamlit

## 4. Proceso de desarrollo
1. **Exploración de datasets** con `package_search` filtrando por grupo `clima` y por palabra clave (ej. “temperatura”).  
2. **Selección de recurso** (CSV/JSON/Datastore).  
3. **Ingesta y limpieza**: carga a DataFrame, eliminación de duplicados y normalización básica.  
4. **Análisis**: resumen estadístico rápido y detección heurística de columna temporal.  
5. **Visualización**:  
   - Serie temporal si existen columnas de tiempo y valores numéricos.  
   - Histogramas básicos de las columnas numéricas detectadas.  
6. **Interactividad** en Streamlit: filtros por dataset/recurso, rango de fechas y selección de columnas.

## 5. Hallazgos y aprendizajes
- La disponibilidad de datos en **Datastore** varía entre datasets; es necesario implementar **rutas de fallback** (CSV).  
- La **calidad de metadatos** (nombres de columnas, tipos) impacta la detección automática de fechas y numéricos.  
- Streamlit permite construir **rápidamente** una UI intuitiva cumpliendo con interactividad y visualización.

## 6. Instrucciones de ejecución
Ver `README.md`. Para generar evidencias del funcionamiento, tomar **capturas** de la app en ejecución con un recurso climático (ej. series de temperatura o precipitaciones) y adjuntarlas.

## 7. Conclusiones
El proyecto cumple con: uso de **API REST**, análisis con **Pandas**, visualización con **Matplotlib**, e **interactividad** vía **Streamlit**. Se adjunta **código fuente**, **informe** (este documento) y **póster**.

---

**Anexos**  
- Código: `app.py`, `api_utils.py`, `analysis.py`, `visuals.py`  
- Póster: `Poster_Solemne_II.png`
