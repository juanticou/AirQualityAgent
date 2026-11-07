"""Tests del loop agéntico con un LLM simulado (no requieren API key ni internet)."""
from types import SimpleNamespace as NS

import pytest

import src.agent.loop as loop
from src.agent.llm import LLMUnavailable


def _tool_call(i, name, args):
    return NS(id=f"call_{i}", function=NS(name=name, arguments=args))


def _resp(content=None, tool_calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))])


class FakeLLM:
    """Devuelve respuestas guionizadas y guarda los mensajes que recibió."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, messages, tools=None, model=None, client=None, temperature=0.1):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.script.pop(0)


@pytest.fixture
def patch_llm(monkeypatch):
    def _apply(script):
        fake = FakeLLM(script)
        monkeypatch.setattr(loop, "get_client", lambda: object())
        monkeypatch.setattr(loop, "chat", fake)
        return fake
    return _apply


def test_flujo_con_herramienta_y_cita(patch_llm, monkeypatch):
    monkeypatch.setattr(loop, "execute_tool", lambda n, a: ({"pm25_ug_m3": 20.0}, 0.01))
    fake = patch_llm([
        _resp(tool_calls=[_tool_call(1, "get_pm25", '{"estacion": "Itagüí"}')]),
        _resp(content="Hoy hay 20 µg/m³ [Res. 2254/2017 MinAmbiente, p. 9]."),
    ])
    r = loop.run_agent("¿cómo está el aire?", estacion="Itagüí", fecha="2025-12-31")
    assert r.tools_called() == ["get_pm25"]
    assert r.citations == [("Res. 2254/2017 MinAmbiente", 9)]
    assert r.iterations == 2 and not r.hit_t_max
    # el resultado de la herramienta vuelve al LLM como mensaje 'tool'
    assert any(m["role"] == "tool" for m in fake.calls[1]["messages"])
    # el contexto del dashboard viaja como mensaje de sistema
    assert "Itagüí" in fake.calls[0]["messages"][1]["content"]


def test_error_de_herramienta_es_observacion(patch_llm):
    fake = patch_llm([
        _resp(tool_calls=[_tool_call(1, "get_pm25", '{"estacion": "Estación Marte"}')]),
        _resp(content="No existe esa estación."),
    ])
    r = loop.run_agent("PM2.5 en Marte")
    step = [s for s in r.trace if s["tipo"] == "tool"][0]
    assert step["error"] is True
    tool_msg = [m for m in fake.calls[1]["messages"] if m["role"] == "tool"][0]
    assert '"error"' in tool_msg["content"]
    assert r.error is None


def test_t_max_fuerza_respuesta_sin_herramientas(patch_llm, monkeypatch):
    monkeypatch.setattr(loop, "execute_tool", lambda n, a: ({"ok": True}, 0.0))
    script = [_resp(tool_calls=[_tool_call(i, "list_stations", "{}")]) for i in range(3)]
    script.append(_resp(content="Respuesta parcial."))
    fake = patch_llm(script)
    r = loop.run_agent("pregunta en bucle", t_max=4)
    assert r.hit_t_max and r.iterations == 4
    assert fake.calls[-1]["tools"] is None          # última llamada sin herramientas
    assert "límite de pasos" in r.answer


def test_sin_api_key_no_rompe(monkeypatch):
    def _raise():
        raise LLMUnavailable("Falta GROQ_API_KEY")
    monkeypatch.setattr(loop, "get_client", _raise)
    r = loop.run_agent("hola")
    assert r.error and "GROQ_API_KEY" in r.answer


def test_falla_del_llm_a_mitad_de_camino(patch_llm, monkeypatch):
    monkeypatch.setattr(loop, "execute_tool", lambda n, a: ({"ok": True}, 0.0))
    fake = patch_llm([_resp(tool_calls=[_tool_call(1, "list_stations", "{}")])])

    def chat_then_fail(*a, **k):
        if fake.script:
            return fake(*a, **k)
        raise LLMUnavailable("rate limit")
    monkeypatch.setattr(loop, "chat", chat_then_fail)
    r = loop.run_agent("x")
    assert r.error == "rate limit" and "No pude completar" in r.answer
