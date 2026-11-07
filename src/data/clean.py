"""Limpieza: PM2.5 horario (SISAIRE + SIATA) -> promedio diario de 24 h por estación + meteorología.

Reglas (CLAUDE.md §4):
- Valores horarios válidos: 0 <= PM2.5 <= 500 µg/m³.
- Meses con escala inconsistente: en SISAIRE varias estaciones reportan en 2020-2023 valores
  varias veces mayores que los de las estaciones vecinas (medianas de 60-117 µg/m³ frente a ~15).
  Esos estación-mes se descartan completos (ver `flag_bad_months`) y se reporta cuántos.
- SIATA (ventana jul-2023 a jul-2024, bandera de calidad 1.0 = válido) solo rellena horas
  que falten en SISAIRE; las estaciones se emparejan por coordenadas (< 500 m).
- Promedio diario solo si hay >= 75 % de horas válidas (18 de 24); el resto se descarta.
- Se conservan estaciones fijas con >= 365 días válidos.

Uso: python -m src.data.clean
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import (MIN_VALID_HOURS_FRAC, PM25_DAILY_CSV, PROCESSED_DIR, RAW_DIR,
                        STATIONS_CSV)
from src.data.ica import category_series, class_series

MAX_MONTHLY_MEDIAN = 100.0
MAX_RATIO_TO_REGION = 3.0
OWN_REF_FROM_YEAR = 2024
MIN_DAYS_STATION = 365


def load_sisaire() -> pd.DataFrame:
    """Lee los CSV anuales de SISAIRE (solo estaciones fijas) en formato largo horario."""
    files = sorted(RAW_DIR.glob("sisaire_amva_pm25_*.csv"))
    if not files:
        raise FileNotFoundError("No hay data/raw/sisaire_amva_pm25_*.csv; corre `python -m src.data.download`.")
    df = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    df = df[df["tipo_estacion"] == "Fija"].copy()
    df["fecha_hora"] = pd.to_datetime(df["med_fecha_inicio"], errors="coerce")
    df["pm25"] = pd.to_numeric(df["med_concentracion_estandar"], errors="coerce")
    df["lat"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitud"], errors="coerce")
    return df.rename(columns={"nombre_est": "nombre"})[
        ["estacion_id", "nombre", "municipio", "lat", "lon", "fecha_hora", "pm25"]]


def load_raw_stations() -> pd.DataFrame:
    """Estaciones presentes en los CSV crudos de SISAIRE (id, nombre, municipio, lat, lon)."""
    df = load_sisaire()
    return (df.groupby("estacion_id", as_index=False)
              .agg(nombre=("nombre", "first"), municipio=("municipio", "first"),
                   lat=("lat", "first"), lon=("lon", "first")))


def load_siata(stations: pd.DataFrame, max_dist_m: float = 500.0) -> pd.DataFrame:
    """Lee el snapshot de SIATA y lo mapea a los estacion_id de SISAIRE por cercanía."""
    path = RAW_DIR / "siata_pm25_snapshot.json"
    if not path.exists():
        return pd.DataFrame(columns=["estacion_id", "fecha_hora", "pm25"])
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for s in data:
        # distancia aproximada (equirectangular) a cada estación SISAIRE
        dlat = (stations["lat"] - s["latitud"]) * 111_320
        dlon = (stations["lon"] - s["longitud"]) * 111_320 * np.cos(np.radians(s["latitud"]))
        dist = np.hypot(dlat, dlon)
        if dist.min() > max_dist_m:
            continue
        sid = stations.loc[dist.idxmin(), "estacion_id"]
        for d in s["datos"]:
            if d["calidad"] == "1.0" and d["valor"] != -9999:
                rows.append((sid, d["fecha"], d["valor"]))
    out = pd.DataFrame(rows, columns=["estacion_id", "fecha_hora", "pm25"])
    out["fecha_hora"] = pd.to_datetime(out["fecha_hora"])
    return out


def flag_bad_months(h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descarta estación-mes con escala inconsistente.

    Criterio (cualquiera): mediana horaria del mes > MAX_MONTHLY_MEDIAN; o > MAX_RATIO_TO_REGION
    veces el percentil 25 de las medianas de todas las estaciones ese mes; o > MAX_RATIO_TO_REGION
    veces la mediana mensual típica de la propia estación en 2024-2025 (periodo sin inconsistencias).
    El PM2.5 del valle está muy correlacionado entre estaciones; en datos limpios ninguna de las dos
    razones pasa de ~2. Se usa el P25 y no la mediana porque en 2020 casi la mitad de las
    estaciones están corruptas.
    """
    h = h.copy()
    h["mes"] = h["fecha_hora"].dt.to_period("M")
    med = h.groupby(["estacion_id", "mes"])["pm25"].median()
    ref = med.groupby(level="mes").quantile(0.25)
    ratio = med / ref.reindex(med.index.get_level_values("mes")).to_numpy()
    # referencia propia: mediana de la estación en el periodo sin inconsistencias (2024-2025)
    meses = med.index.get_level_values("mes")
    own = med[meses.year >= OWN_REF_FROM_YEAR].groupby(level="estacion_id").median()
    own_ratio = med / own.reindex(med.index.get_level_values("estacion_id")).to_numpy()
    bad_mask = (med > MAX_MONTHLY_MEDIAN) | (ratio > MAX_RATIO_TO_REGION) | (own_ratio > MAX_RATIO_TO_REGION)
    bad = med[bad_mask.fillna(False)].reset_index()
    key = h.set_index(["estacion_id", "mes"]).index
    mask = key.isin(pd.MultiIndex.from_frame(bad[["estacion_id", "mes"]]))
    return h.loc[~mask].drop(columns="mes"), bad


def to_daily(h: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Promedio de 24 h; descarta días con < 75 % de horas válidas. Devuelve (diario, n_descartados)."""
    h = h.copy()
    h["fecha"] = h["fecha_hora"].dt.normalize()
    g = h.groupby(["estacion_id", "fecha"])["pm25"].agg(["mean", "count"]).reset_index()
    ok = g["count"] >= int(np.ceil(24 * MIN_VALID_HOURS_FRAC))
    daily = g[ok].rename(columns={"mean": "pm25", "count": "horas_validas"})
    return daily, int((~ok).sum())


def load_meteo() -> pd.DataFrame | None:
    """Meteorología diaria de Open-Meteo (si se descargó)."""
    path = RAW_DIR / "open_meteo_diario.csv"
    if not path.exists():
        return None
    m = pd.read_csv(path, dtype={"estacion_id": str})
    m["fecha"] = pd.to_datetime(m["fecha"])
    return m.rename(columns={"precipitation_sum": "lluvia_mm", "wind_speed_10m_mean": "viento_medio_kmh",
                             "wind_speed_10m_max": "viento_max_kmh"})


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    sis = load_sisaire()
    stations = load_raw_stations()
    report["horas_sisaire"] = int(len(sis))

    # 1) meses con escala inconsistente (antes del filtro de rango, para no sesgar la mediana)
    sis = sis.loc[sis["pm25"].notna() & (sis["pm25"] >= 0), ["estacion_id", "fecha_hora", "pm25"]]
    n_before = len(sis)
    sis, bad = flag_bad_months(sis)
    report["meses_estacion_escala_inconsistente"] = int(len(bad))
    report["horas_en_meses_inconsistentes"] = int(n_before - len(sis))

    # 2) rango físico
    valid = sis["pm25"].between(0, 500)
    report["horas_fuera_de_rango"] = int((~valid).sum())
    sis = sis.loc[valid]

    # 3) relleno con SIATA solo donde falta SISAIRE
    siata = load_siata(stations)
    siata = siata[siata["pm25"].between(0, 500)]
    merged = pd.concat([sis.assign(fuente="SISAIRE"), siata.assign(fuente="SIATA")])
    merged = merged.drop_duplicates(["estacion_id", "fecha_hora"], keep="first")
    report["horas_rellenadas_siata"] = int((merged["fuente"] == "SIATA").sum())

    # 4) promedio diario con criterio de 75 %
    daily, n_drop = to_daily(merged)
    report["dias_descartados_menos_75pct"] = n_drop

    # 5) estaciones con suficiente historia
    counts = daily.groupby("estacion_id").size()
    keep = counts[counts >= MIN_DAYS_STATION].index
    report["estaciones_descartadas_poca_historia"] = sorted(set(counts.index) - set(keep))
    daily = daily[daily["estacion_id"].isin(keep)].copy()

    # 6) meteorología, ICA y clase
    meteo = load_meteo()
    if meteo is not None:
        daily = daily.merge(meteo, on=["estacion_id", "fecha"], how="left")
    daily["pm25"] = daily["pm25"].round(2)
    daily["categoria_ica"] = category_series(daily["pm25"])
    daily["clase"] = class_series(daily["pm25"])
    daily = daily.sort_values(["estacion_id", "fecha"])
    daily["fecha"] = daily["fecha"].dt.date
    daily.to_csv(PM25_DAILY_CSV, index=False)

    st = stations[stations["estacion_id"].isin(keep)].copy()
    agg = daily.groupby("estacion_id").agg(n_dias=("pm25", "size"), desde=("fecha", "min"),
                                           hasta=("fecha", "max"), pm25_medio=("pm25", "mean"))
    st = st.merge(agg.reset_index(), on="estacion_id")
    st["pm25_medio"] = st["pm25_medio"].round(1)
    st.to_csv(STATIONS_CSV, index=False)

    report["estaciones_finales"] = int(len(st))
    report["dias_estacion_finales"] = int(len(daily))
    report["distribucion_clases"] = daily["clase"].value_counts().to_dict()
    (PROCESSED_DIR / "clean_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
