"""Herramientas del agente: get_pm25, predict_tomorrow, search_docs, list_stations.

Cada herramienta devuelve un dict serializable a JSON. Los errores esperables (estación o fecha
inexistente) se devuelven como {"error": "...", ...} para que el LLM los explique; `execute_tool`
además atrapa cualquier excepción y la convierte en observación, así el loop nunca se rompe.
"""
from __future__ import annotations

import json
import time
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import CLASSES, MODELS_DIR, PM25_DAILY_CSV, STATIONS_CSV
from src.data.features import Discretizer
from src.data.ica import pm25_to_ica
from src.models.hmm import GaussianHMM
from src.models.naive_bayes import CategoricalNaiveBayes

DATA_SOURCE = "IDEAM-SISAIRE (estaciones AMVA) + SIATA; promedios diarios de 24 h"


# --- carga en caché --------------------------------------------------------------
@lru_cache(maxsize=1)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(PM25_DAILY_CSV, dtype={"estacion_id": str}, parse_dates=["fecha"])
    stations = pd.read_csv(STATIONS_CSV, dtype={"estacion_id": str})
    return daily, stations


@lru_cache(maxsize=1)
def load_models() -> dict:
    nb, nb_extra = CategoricalNaiveBayes.load(MODELS_DIR / "nb.json")
    out = {"nb": nb, "nb_extra": nb_extra, "disc": Discretizer.from_dict(nb_extra["discretizer"])}
    if (MODELS_DIR / "hmm.json").exists():
        out["hmm"], out["hmm_extra"] = GaussianHMM.load(MODELS_DIR / "hmm.json")
    return out


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(c for c in s if not unicodedata.combining(c)).strip()


def resolve_station(estacion: str) -> tuple[pd.Series | None, dict | None]:
    """Busca por id exacto, o por nombre/municipio (sin tildes). Devuelve (fila, error)."""
    _, st = load_data()
    q = _norm(estacion)
    if not q:
        return None, {"error": "Debes indicar una estación."}
    exact = st[st["estacion_id"] == q]
    if len(exact):
        return exact.iloc[0], None
    hay = (st["nombre"].map(_norm) + " | " + st["municipio"].map(_norm))
    match = st[hay.str.contains(q, regex=False)]
    if len(match) == 0:  # coincidencia por palabras
        words = [w for w in q.split() if len(w) > 2]
        match = st[hay.apply(lambda h: bool(words) and all(w in h for w in words))]
    if len(match) == 1:
        return match.iloc[0], None
    if len(match) > 1:
        return None, {"error": f"'{estacion}' coincide con varias estaciones; elige una.",
                      "opciones": _station_list(match)}
    return None, {"error": f"No existe la estación '{estacion}' en los datos.",
                  "estaciones_disponibles": _station_list(st)}


def _station_list(st: pd.DataFrame) -> list[dict]:
    return [{"id": r.estacion_id, "nombre": r.nombre, "municipio": r.municipio} for r in st.itertuples()]


def _parse_date(fecha: str | None) -> tuple[pd.Timestamp | None, dict | None]:
    if fecha in (None, "", "hoy", "null"):
        return None, None
    try:
        return pd.Timestamp(fecha).normalize(), None
    except (ValueError, TypeError):
        return None, {"error": f"Fecha '{fecha}' inválida; usa el formato AAAA-MM-DD."}


# --- herramientas ----------------------------------------------------------------
def list_stations(municipio: str | None = None) -> dict:
    """Estaciones disponibles con municipio, coordenadas y rango de datos."""
    _, st = load_data()
    if municipio:
        st = st[st["municipio"].map(_norm).str.contains(_norm(municipio), regex=False)]
        if st.empty:
            return {"error": f"No hay estaciones en '{municipio}'.", "municipios": sorted(load_data()[1]["municipio"].unique())}
    return {"estaciones": [{"id": r.estacion_id, "nombre": r.nombre, "municipio": r.municipio,
                            "lat": round(r.lat, 5), "lon": round(r.lon, 5), "datos_desde": r.desde,
                            "datos_hasta": r.hasta} for r in st.itertuples()]}


def get_pm25(estacion: str, fecha: str | None = None) -> dict:
    """PM2.5 diario, categoría ICA, color y últimos 7 días para una estación y fecha."""
    row, err = resolve_station(estacion)
    if err:
        return err
    day, err = _parse_date(fecha)
    if err:
        return err
    daily, _ = load_data()
    d = daily[daily["estacion_id"] == row.estacion_id].set_index("fecha").sort_index()
    if day is None:
        day = d.index.max()
    if day not in d.index:
        near = d.index[(d.index >= day - pd.Timedelta(days=10)) & (d.index <= day + pd.Timedelta(days=10))]
        return {"error": f"No hay dato válido para {row.nombre} el {day.date()} "
                         f"(día sin dato o con < 75 % de horas válidas).",
                "rango_disponible": f"{d.index.min().date()} a {d.index.max().date()}",
                "fechas_cercanas_con_dato": [str(x.date()) for x in near[:6]]}
    pm = float(d.loc[day, "pm25"])
    ica = pm25_to_ica(pm)
    last7 = d.loc[day - pd.Timedelta(days=6): day]
    return {
        "estacion": {"id": row.estacion_id, "nombre": row.nombre, "municipio": row.municipio},
        "fecha": str(day.date()), "pm25_ug_m3": round(pm, 1), "categoria_ica": ica["categoria"],
        "color": ica["color"], "rango_categoria": ica["rango"], "clase_proyecto": d.loc[day, "clase"],
        "supera_limite_diario_37": bool(round(pm) > 37),
        "ultimos_7_dias": [{"fecha": str(i.date()), "pm25": round(float(v), 1)}
                           for i, v in last7["pm25"].items()],
        "fuente_datos": DATA_SOURCE,
    }


def predict_tomorrow(estacion: str, fecha: str | None = None) -> dict:
    """Probabilidad de normal/alerta/crítico para el día siguiente a `fecha` (por defecto el último dato)."""
    row, err = resolve_station(estacion)
    if err:
        return err
    day, err = _parse_date(fecha)
    if err:
        return err
    daily, _ = load_data()
    m = load_models()
    d = daily[daily["estacion_id"] == row.estacion_id].set_index("fecha").sort_index()
    if day is None:
        day = d.index.max()
    if day not in d.index:
        return {"error": f"No hay dato de PM2.5 para {row.nombre} el {day.date()}; "
                         "la predicción necesita el dato de ese día.",
                "rango_disponible": f"{d.index.min().date()} a {d.index.max().date()}"}
    prev = day - pd.Timedelta(days=1)
    feat_row = pd.DataFrame([{
        "pm25": d.loc[day, "pm25"], "pm25_ayer": d.loc[prev, "pm25"] if prev in d.index else np.nan,
        "lluvia_mm": d.loc[day].get("lluvia_mm", np.nan),
        "viento_medio_kmh": d.loc[day].get("viento_medio_kmh", np.nan),
        "fecha_manana": day + pd.Timedelta(days=1)}])
    x = m["disc"].transform(feat_row)
    p_nb = m["nb"].predict_proba(x)[0]
    out = {
        "estacion": {"id": row.estacion_id, "nombre": row.nombre, "municipio": row.municipio},
        "fecha_base": str(day.date()), "fecha_prediccion": str((day + pd.Timedelta(days=1)).date()),
        "naive_bayes": {"probabilidades": {c: round(float(p), 3) for c, p in zip(CLASSES, p_nb)},
                        "clase_mas_probable": CLASSES[int(np.argmax(p_nb))]},
        "features_usadas": m["disc"].describe_row(x[0]),
        "desempeno_en_prueba": m["nb_extra"].get("metricas_test", {}),
        "nota": "Clases del proyecto (normal ≤ 37, alerta 38–55, crítico ≥ 56 µg/m³), no niveles oficiales POECA.",
    }
    if "hmm" in m:
        hmm = m["hmm"]
        s = d["pm25"].reindex(pd.date_range(d.index.min(), day, freq="D"))
        xs = np.log(s.clip(lower=1.0)).to_numpy()
        alpha, _ = hmm.forward(xs)
        nxt = alpha[-1] @ hmm.A
        pc = hmm.class_probs_from_states(nxt[None], m["hmm_extra"]["class_edges"])[0]
        est = m["hmm_extra"].get("estados", ["bajo", "medio", "alto"])
        med = m["hmm_extra"].get("pm25_mediana_estado", [])
        out["hmm"] = {"estado_latente_hoy": {f"{e} (~{v} µg/m³)" if med else e: round(float(p), 3)
                                             for e, v, p in zip(est, med or est, alpha[-1])},
                      "probabilidades": {c: round(float(p), 3) for c, p in zip(CLASSES, pc)},
                      "clase_mas_probable": CLASSES[int(np.argmax(pc))]}
    return out


def search_docs(query: str, k: int = 4) -> dict:
    """Fragmentos del corpus normativo/salud con fuente y página."""
    from src.rag.retriever import get_retriever

    res = get_retriever().search(query, k)
    for r in res["resultados"]:
        r["texto"] = r["texto"][:900]
        r["cita"] = f"[{r['fuente']}, p. {r['pagina']}]"   # el LLM la copia literal
    if not res["encontrado"]:
        res["mensaje"] = "Ningún fragmento supera el umbral de relevancia: no hay respaldo documental."
    return res


# --- esquemas JSON para tool calling ---------------------------------------------
TOOL_FUNCTIONS = {"get_pm25": get_pm25, "predict_tomorrow": predict_tomorrow,
                  "search_docs": search_docs, "list_stations": list_stations}

_EST = {"type": "string", "description": "Id numérico de la estación, o parte de su nombre o municipio (p. ej. 'Itagüí', 'Altavista')."}
_FECHA = {"type": "string", "description": "Fecha AAAA-MM-DD. Omitir para usar el dato más reciente."}

TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "get_pm25",
        "description": "PM2.5 promedio diario (24 h) de una estación del Valle de Aburrá en una fecha: valor, categoría ICA, color y últimos 7 días.",
        "parameters": {"type": "object", "properties": {"estacion": _EST, "fecha": _FECHA}, "required": ["estacion"]}}},
    {"type": "function", "function": {
        "name": "predict_tomorrow",
        "description": "Probabilidad de que el día siguiente sea normal / alerta / crítico en una estación (Naive Bayes y HMM propios), con las features usadas y el desempeño del modelo en prueba.",
        "parameters": {"type": "object", "properties": {"estacion": _EST, "fecha": {**_FECHA, "description": "Día base 'hoy' (AAAA-MM-DD); se predice el día siguiente. Omitir para el último dato."}}, "required": ["estacion"]}}},
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Busca en la normativa y guías oficiales (Resolución 2254 de 2017, POECA del AMVA, guías OMS 2021) fragmentos con fuente y página para citar.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Consulta en español, específica (p. ej. 'recomendaciones para niños cuando el ICA es dañino')."},
            "k": {"type": "integer", "description": "Número de fragmentos (1-8).", "default": 4}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "list_stations",
        "description": "Lista las estaciones de PM2.5 disponibles (id, nombre, municipio, coordenadas). Opcionalmente filtra por municipio.",
        "parameters": {"type": "object", "properties": {"municipio": {"type": "string", "description": "Municipio, p. ej. 'Medellín', 'Bello'."}}, "required": []}}},
]


def execute_tool(name: str, arguments: str | dict) -> tuple[dict, float]:
    """Ejecuta una herramienta; cualquier falla se devuelve como {'error': ...}. Retorna (resultado, segundos)."""
    t0 = time.perf_counter()
    try:
        args = json.loads(arguments or "{}") if isinstance(arguments, str) else dict(arguments or {})
        fn = TOOL_FUNCTIONS.get(name)
        if fn is None:
            result = {"error": f"Herramienta desconocida '{name}'.", "disponibles": list(TOOL_FUNCTIONS)}
        else:
            result = fn(**{k: v for k, v in args.items() if v is not None})
    except TypeError as e:
        result = {"error": f"Argumentos inválidos para {name}: {e}"}
    except json.JSONDecodeError:
        result = {"error": f"Los argumentos de {name} no son JSON válido."}
    except Exception as e:  # noqa: BLE001 - nunca romper el loop
        result = {"error": f"Falla interna en {name}: {type(e).__name__}: {e}"}
    return result, time.perf_counter() - t0
