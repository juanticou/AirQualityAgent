"""Conversión de PM2.5 diario (µg/m³) a categoría ICA (Res. 2254/2017) y a clase del proyecto.

Decisión: el promedio diario se redondea al entero más cercano (mitades hacia arriba)
antes de clasificar, porque los rangos oficiales están en enteros (12 | 13, 37 | 38, ...).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import CLASSES, ICA_CATEGORIES


def _round_half_up(x: float) -> int:
    """Redondea con mitades hacia arriba (12.5 -> 13), a diferencia de round() de Python."""
    return int(math.floor(x + 0.5))


def pm25_to_ica(pm25: float) -> dict | None:
    """Devuelve {'categoria', 'color', 'rango'} para un PM2.5 diario; None si es NaN o negativo."""
    if pm25 is None or (isinstance(pm25, float) and math.isnan(pm25)) or pm25 < 0:
        return None
    v = _round_half_up(float(pm25))
    for name, lo, hi, color in ICA_CATEGORIES:
        if lo <= v <= hi:
            return {"categoria": name, "color": color, "rango": f"{lo}-{hi} µg/m³"}
    # > 500: se reporta como Peligrosa (fuera de escala)
    name, lo, hi, color = ICA_CATEGORIES[-1]
    return {"categoria": name, "color": color, "rango": f"> {hi} µg/m³ (fuera de escala)"}


def pm25_to_class(pm25: float) -> str | None:
    """Clase del proyecto: normal (≤ 37), alerta (38–55), critico (≥ 56). None si NaN."""
    if pm25 is None or (isinstance(pm25, float) and math.isnan(pm25)) or pm25 < 0:
        return None
    v = _round_half_up(float(pm25))
    if v <= 37:
        return "normal"
    if v <= 55:
        return "alerta"
    return "critico"


def class_series(pm25: pd.Series) -> pd.Series:
    """Versión vectorizada de pm25_to_class para una serie de pandas."""
    v = np.floor(pm25.astype(float) + 0.5)
    out = pd.Series(pd.NA, index=pm25.index, dtype="object")
    out[v <= 37] = "normal"
    out[(v > 37) & (v <= 55)] = "alerta"
    out[v > 55] = "critico"
    out[pm25.isna() | (pm25 < 0)] = pd.NA
    return out


def category_series(pm25: pd.Series) -> pd.Series:
    """Categoría ICA para una serie de pandas."""
    return pm25.map(lambda x: (pm25_to_ica(x) or {}).get("categoria"))


CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
