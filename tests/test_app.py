"""Smoke tests del dashboard con streamlit.testing (sin navegador ni red)."""
from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

import src.config as cfg
from src.config import PM25_DAILY_CSV, ROOT

pytestmark = pytest.mark.skipif(not PM25_DAILY_CSV.exists(), reason="faltan datos procesados")
APP = str(ROOT / "app.py")


def test_app_arranca_sin_api_key(monkeypatch):
    monkeypatch.setattr(cfg, "LLM_API_KEY", None)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any("Chat desactivado" in i.value for i in at.info)
    assert len(at.get("plotly_chart")) >= 3          # mapa, serie, barras de predicción


def test_app_cambia_estacion_y_fecha(monkeypatch):
    monkeypatch.setattr(cfg, "LLM_API_KEY", None)
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox[0].set_value("31371").run()
    at.date_input[0].set_value(date(2024, 3, 6)).run()
    assert not at.exception
    assert any("Predicción para 2024-03-07" in m.value for m in at.markdown)
