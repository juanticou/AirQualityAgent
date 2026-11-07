"""Tests de las herramientas del agente sobre los datos procesados versionados."""
import json

import pytest

from src.agent.tools import TOOLS_SCHEMA, execute_tool, get_pm25, predict_tomorrow, resolve_station
from src.config import PM25_DAILY_CSV

pytestmark = pytest.mark.skipif(not PM25_DAILY_CSV.exists(), reason="faltan datos procesados")


def test_esquemas_tienen_nombres_esperados():
    names = {t["function"]["name"] for t in TOOLS_SCHEMA}
    assert names == {"get_pm25", "predict_tomorrow", "search_docs", "list_stations"}


def test_resolver_estacion_sin_tildes():
    row, err = resolve_station("altavista")
    assert err is None and row.estacion_id == "31371"
    row, err = resolve_station("itagui casa")
    assert err is None and row.estacion_id == "29493"


def test_get_pm25_estructura():
    r = get_pm25("29493", "2024-03-06")
    assert {"pm25_ug_m3", "categoria_ica", "color", "ultimos_7_dias"} <= set(r)
    assert len(r["ultimos_7_dias"]) <= 7


def test_errores_amigables():
    assert "error" in get_pm25("Estación inexistente")
    assert "opciones" in get_pm25("Medellín")              # ambigua
    assert "error" in get_pm25("29493", "1999-01-01")      # fecha sin datos
    assert "error" in get_pm25("29493", "no-es-fecha")


def test_prediccion_probabilidades_suman_uno():
    r = predict_tomorrow("31371", "2024-03-06")
    assert abs(sum(r["naive_bayes"]["probabilidades"].values()) - 1) < 1e-2
    assert r["fecha_prediccion"] == "2024-03-07"


def test_execute_tool_atrapa_todo():
    r, _ = execute_tool("get_pm25", '{"estacion": ')
    assert "error" in r
    r, _ = execute_tool("get_pm25", json.dumps({"otro": 1}))
    assert "error" in r
