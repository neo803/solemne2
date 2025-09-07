# visuals.py
import io
import pandas as pd
import matplotlib.pyplot as plt

def plot_timeseries(df: pd.DataFrame, time_col: str, value_col: str):
    fig, ax = plt.subplots()
    df_sorted = df.sort_values(time_col)
    ax.plot(df_sorted[time_col], df_sorted[value_col])
    ax.set_xlabel(time_col)
    ax.set_ylabel(value_col)
    ax.set_title(f"Serie temporal: {value_col} por {time_col}")
    fig.tight_layout()
    return fig

def plot_histogram(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots()
    ax.hist(df[col].dropna(), bins=30)
    ax.set_xlabel(col)
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución: {col}")
    fig.tight_layout()
    return fig

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return buf.read()
