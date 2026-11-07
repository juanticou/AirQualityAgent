"""Features discretizadas del día t para predecir la clase del día t+1 (por estación).

Sin fuga de información: todo lo observado viene del día t (o t-1). Del día t+1 solo se usa el
calendario (mes y día de la semana), que se conoce de antemano. Los terciles de lluvia y viento
se calculan SOLO con el conjunto de entrenamiento (`Discretizer.fit`).

| feature            | categorías                                                     |
|--------------------|----------------------------------------------------------------|
| nivel_hoy          | PM2.5 de t: 0–12, 13–25, 26–37, 38–55, ≥56                     |
| tendencia          | PM2.5(t) − PM2.5(t−1): baja (< −3), estable, sube (> 3), sin dato |
| lluvia_hoy         | terciles de lluvia diaria en entrenamiento + sin dato          |
| viento_hoy         | terciles de viento medio en entrenamiento + sin dato           |
| mes_manana         | 1..12 del día t+1 (temporadas críticas feb–abr, oct–nov)       |
| dia_semana_manana  | lunes..domingo del día t+1                                     |
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import CLASSES, PM25_DAILY_CSV
from src.data.ica import CLASS_TO_IDX

NIVEL_EDGES = [12.5, 25.5, 37.5, 55.5]
NIVEL_LABELS = ["0-12", "13-25", "26-37", "38-55", ">=56"]
TREND_DELTA = 3.0
FEATURES = ["nivel_hoy", "tendencia", "lluvia_hoy", "viento_hoy", "mes_manana", "dia_semana_manana"]
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def load_daily() -> pd.DataFrame:
    df = pd.read_csv(PM25_DAILY_CSV, dtype={"estacion_id": str}, parse_dates=["fecha"])
    return df.sort_values(["estacion_id", "fecha"]).reset_index(drop=True)


def make_pairs(daily: pd.DataFrame) -> pd.DataFrame:
    """Une cada día t con t-1 y t+1 de la misma estación (exige que t+1 exista)."""
    d = daily[["estacion_id", "fecha", "pm25", "clase"] +
              [c for c in ("lluvia_mm", "viento_medio_kmh") if c in daily.columns]].copy()
    prev = d[["estacion_id", "fecha", "pm25"]].copy()
    prev["fecha"] = prev["fecha"] + pd.Timedelta(days=1)
    nxt = d[["estacion_id", "fecha", "pm25", "clase"]].copy()
    nxt["fecha"] = nxt["fecha"] - pd.Timedelta(days=1)
    out = (d.merge(prev.rename(columns={"pm25": "pm25_ayer"}), on=["estacion_id", "fecha"], how="left")
             .merge(nxt.rename(columns={"pm25": "pm25_manana", "clase": "clase_manana"}),
                    on=["estacion_id", "fecha"], how="inner"))
    out["fecha_manana"] = out["fecha"] + pd.Timedelta(days=1)
    return out.reset_index(drop=True)


@dataclass
class Discretizer:
    """Convierte variables continuas en categorías enteras; los cortes se aprenden en train."""
    lluvia_edges: list[float] = field(default_factory=list)
    viento_edges: list[float] = field(default_factory=list)

    def fit(self, pairs: pd.DataFrame) -> "Discretizer":
        if "lluvia_mm" in pairs:
            self.lluvia_edges = _tercile_edges(pairs["lluvia_mm"])
        if "viento_medio_kmh" in pairs:
            self.viento_edges = _tercile_edges(pairs["viento_medio_kmh"])
        return self

    @property
    def n_categories(self) -> list[int]:
        return [len(NIVEL_EDGES) + 1, 4, 4, 4, 12, 7]

    def transform(self, pairs: pd.DataFrame) -> np.ndarray:
        """Matriz (n, 6) de enteros en el orden de FEATURES."""
        nivel = np.digitize(pairs["pm25"].to_numpy(), NIVEL_EDGES)
        diff = (pairs["pm25"] - pairs["pm25_ayer"]).to_numpy()
        trend = np.where(diff < -TREND_DELTA, 0, np.where(diff > TREND_DELTA, 2, 1))
        trend[np.isnan(diff)] = 3
        lluvia = _bin_or_missing(pairs.get("lluvia_mm"), self.lluvia_edges, len(pairs))
        viento = _bin_or_missing(pairs.get("viento_medio_kmh"), self.viento_edges, len(pairs))
        mes = pairs["fecha_manana"].dt.month.to_numpy() - 1
        dow = pairs["fecha_manana"].dt.dayofweek.to_numpy()
        return np.column_stack([nivel, trend, lluvia, viento, mes, dow]).astype(int)

    def describe_row(self, x: np.ndarray) -> dict:
        """Traducción legible de un vector de features (para el agente)."""
        tend = ["baja", "estable", "sube", "sin dato"]
        terc = ["bajo", "medio", "alto", "sin dato"]
        meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
        return {"nivel_hoy": NIVEL_LABELS[x[0]] + " µg/m³", "tendencia": tend[x[1]],
                "lluvia_hoy": terc[x[2]], "viento_hoy": terc[x[3]],
                "mes_manana": meses[x[4]], "dia_semana_manana": DIAS[x[5]]}

    def to_dict(self) -> dict:
        return {"lluvia_edges": self.lluvia_edges, "viento_edges": self.viento_edges}

    @classmethod
    def from_dict(cls, d: dict) -> "Discretizer":
        return cls(lluvia_edges=d["lluvia_edges"], viento_edges=d["viento_edges"])


def _tercile_edges(s: pd.Series) -> list[float]:
    v = s.dropna().to_numpy()
    return [float(np.quantile(v, 1 / 3)), float(np.quantile(v, 2 / 3))] if len(v) else []


def _bin_or_missing(s: pd.Series | None, edges: list[float], n: int) -> np.ndarray:
    """Terciles 0..2; 3 = sin dato."""
    if s is None or not edges:
        return np.full(n, 3)
    v = s.to_numpy(dtype=float)
    out = np.digitize(v, edges, right=True)
    out[np.isnan(v)] = 3
    return out


def encode_target(classes: pd.Series) -> np.ndarray:
    return classes.map(CLASS_TO_IDX).to_numpy(dtype=int)


def temporal_split(pairs: pd.DataFrame, train_end: int, val_year: int) -> dict[str, pd.DataFrame]:
    """Partición por el año del día objetivo (t+1): train ≤ train_end, val = val_year, test > val_year."""
    y = pairs["fecha_manana"].dt.year
    return {"train": pairs[y <= train_end], "val": pairs[y == val_year], "test": pairs[y > val_year]}


__all__ = ["FEATURES", "CLASSES", "Discretizer", "make_pairs", "temporal_split", "load_daily"]
