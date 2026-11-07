"""Descarga de datos de PM2.5 y meteorología para el Valle de Aburrá.

Orden de fuentes (CLAUDE.md §4), con el resultado verificado el 2026-11-06:
1. SIATA (EntregaData1/Datos_SIATA_Aire_pm25.json): responde, pero solo trae una ventana
   móvil de ~1 año (jul-2023 a jul-2024). Se guarda como referencia, no alcanza para entrenar.
2. Portal de datos abiertos del AMVA (CKAN): devolvió 502 Bad Gateway en todos los intentos.
3. datos.gov.co, dataset IDEAM-SISAIRE "Calidad del Aire en Colombia" (g4t8-zkc3):
   PM2.5 horario medido por las estaciones del AMVA, 2020-2025. **Fuente principal.**
Meteorología: Open-Meteo Historical Weather API (reanálisis ERA5, *modelado*, no medido)
en las coordenadas de cada estación.

Uso:  python -m src.data.download            (descarga lo que falte en data/raw/)
      python -m src.data.download --force    (vuelve a descargar todo)
"""
from __future__ import annotations

import argparse
import io
import json
import time
from datetime import date

import pandas as pd
import requests

from src.config import PROCESSED_DIR, RAW_DIR

SIATA_PM25_URL = "https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json"
AMVA_CKAN_URL = (
    "https://datosabiertos.metropol.gov.co/api/3/action/package_show"
    "?id=mediciones-estaciones-calidad-del-aire"
)
SISAIRE_URL = "https://www.datos.gov.co/resource/g4t8-zkc3.csv"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
YEARS = range(2020, 2026)
PAGE = 50_000
HEADERS = {"User-Agent": "calidad-aire-aburra/0.1 (proyecto academico EAFIT)"}


def probe(url: str, timeout: int = 30) -> dict:
    """Consulta una URL y reporta si respondió (código HTTP, tipo de contenido)."""
    try:
        r = requests.get(url, timeout=timeout, headers=HEADERS, stream=True)
        r.close()
        return {"url": url, "ok": r.status_code == 200, "status": r.status_code,
                "content_type": r.headers.get("content-type")}
    except requests.RequestException as e:
        return {"url": url, "ok": False, "error": str(e)}


def download_siata_snapshot(force: bool = False) -> dict:
    """Guarda el JSON de SIATA (ventana reciente) en data/raw/ y resume su cobertura."""
    out = RAW_DIR / "siata_pm25_snapshot.json"
    if force or not out.exists():
        r = requests.get(SIATA_PM25_URL, timeout=180, headers=HEADERS)
        r.raise_for_status()
        out.write_bytes(r.content)
    data = json.loads(out.read_text(encoding="utf-8"))
    fechas = [d["fecha"] for s in data for d in s["datos"]]
    return {"fuente": "SIATA", "ok": True, "estaciones": len(data),
            "desde": min(fechas), "hasta": max(fechas), "archivo": out.name}


def _sisaire_year(year: int) -> pd.DataFrame:
    """Descarga (paginando) el PM2.5 horario de estaciones AMVA para un año desde SISAIRE."""
    where = (f"nombre_fgda='AMVA' AND msfl_code='PM2.5' AND med_fecha_inicio >= '{year}-01-01' "
             f"AND med_fecha_inicio < '{year + 1}-01-01'")
    cols = ("estacion_id,nombre_est,municipio,latitud,longitud,tipo_estacion,"
            "med_fecha_inicio,med_concentracion_estandar")
    frames, offset = [], 0
    while True:
        params = {"$select": cols, "$where": where, "$order": "estacion_id,med_fecha_inicio",
                  "$limit": PAGE, "$offset": offset}
        for intento in range(3):
            try:
                r = requests.get(SISAIRE_URL, params=params, timeout=300, headers=HEADERS)
                r.raise_for_status()
                break
            except requests.RequestException:
                if intento == 2:
                    raise
                time.sleep(5)
        df = pd.read_csv(io.StringIO(r.text), dtype=str)
        frames.append(df)
        if len(df) < PAGE:
            break
        offset += PAGE
    return pd.concat(frames, ignore_index=True)


def download_sisaire(force: bool = False) -> dict:
    """Descarga 2020-2025 por año en data/raw/sisaire_amva_pm25_<año>.csv."""
    total = 0
    for y in YEARS:
        out = RAW_DIR / f"sisaire_amva_pm25_{y}.csv"
        if force or not out.exists():
            df = _sisaire_year(y)
            df.to_csv(out, index=False)
            print(f"  SISAIRE {y}: {len(df):,} filas")
        total += sum(1 for _ in open(out, encoding="utf-8")) - 1
    return {"fuente": "IDEAM-SISAIRE (datos.gov.co g4t8-zkc3)", "ok": True, "filas": total}


def _open_meteo_point(lat: float, lon: float) -> pd.DataFrame:
    """Serie diaria 2020-2025 para un punto; caché en data/raw/open_meteo_cache/ y reintentos ante 429."""
    cache = RAW_DIR / "open_meteo_cache" / f"{lat:.2f}_{lon:.2f}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    params = {"latitude": lat, "longitude": lon, "start_date": f"{YEARS[0]}-01-01",
              "end_date": f"{YEARS[-1]}-12-31", "timezone": "America/Bogota",
              "daily": "precipitation_sum,wind_speed_10m_mean,wind_speed_10m_max"}
    for intento in range(4):
        r = requests.get(OPEN_METEO_URL, params=params, timeout=180, headers=HEADERS)
        if r.status_code == 429 and intento < 3:   # límite de la API gratuita: esperar y reintentar
            time.sleep(65)
            continue
        r.raise_for_status()
        break
    df = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "fecha"})
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    time.sleep(2.0)
    return df


def download_meteo(stations: pd.DataFrame, force: bool = False) -> dict:
    """Lluvia y viento diarios (Open-Meteo, reanálisis ERA5) para cada estación.

    Las estaciones se agrupan en celdas de 0.1° (resolución de ERA5-Land): estaciones a pocos km
    comparten la misma serie, y así se hacen menos llamadas a la API.
    """
    out = RAW_DIR / "open_meteo_diario.csv"
    if out.exists() and not force:
        return {"fuente": "Open-Meteo (ERA5)", "ok": True, "archivo": out.name}
    st = stations.copy()
    st["glat"], st["glon"] = (st["lat"] / 0.1).round() * 0.1, (st["lon"] / 0.1).round() * 0.1
    frames = []
    cells = st.groupby(["glat", "glon"])
    for (glat, glon), group in cells:
        df = _open_meteo_point(round(glat, 2), round(glon, 2))
        for sid in group["estacion_id"]:
            frames.append(df.assign(estacion_id=sid))
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    return {"fuente": "Open-Meteo (ERA5)", "ok": True, "estaciones": int(len(st)),
            "celdas_consultadas": int(cells.ngroups), "archivo": out.name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    report: list[dict] = []

    print("1) SIATA ...")
    try:
        report.append(download_siata_snapshot(args.force))
    except Exception as e:  # noqa: BLE001 - se reporta y se sigue
        report.append({"fuente": "SIATA", "ok": False, "error": str(e)})
    print("2) Portal AMVA (CKAN) ...")
    report.append({"fuente": "AMVA datos abiertos", **probe(AMVA_CKAN_URL)})
    print("3) datos.gov.co / IDEAM-SISAIRE ...")
    try:
        report.append(download_sisaire(args.force))
    except Exception as e:  # noqa: BLE001
        report.append({"fuente": "SISAIRE", "ok": False, "error": str(e)})

    print("4) Meteorología Open-Meteo ...")
    from src.data.clean import load_raw_stations  # import diferido: evita ciclo

    try:
        report.append(download_meteo(load_raw_stations(), args.force))
    except Exception as e:  # noqa: BLE001
        report.append({"fuente": "Open-Meteo", "ok": False, "error": str(e)})

    payload = {"fecha_descarga": date.today().isoformat(), "fuentes": report}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for report_path in (RAW_DIR / "download_report.json", PROCESSED_DIR / "download_report.json"):
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in report:
        print(" -", json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
