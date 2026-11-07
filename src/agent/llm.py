"""Cliente LLM compatible con OpenAI (Groq por defecto), con timeout y un reintento.

Uso: python -m src.agent.llm   -> lista los modelos disponibles en el endpoint configurado.
"""
from __future__ import annotations

import time

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_S


class LLMUnavailable(RuntimeError):
    """No hay API key o el endpoint no respondió tras el reintento."""


def get_client():
    if not LLM_API_KEY:
        raise LLMUnavailable("Falta GROQ_API_KEY en .env: el chat está desactivado.")
    from openai import OpenAI

    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT_S, max_retries=0)


def chat(messages: list[dict], tools: list[dict] | None = None, model: str | None = None,
         client=None, temperature: float = 0.1):
    """Una llamada al LLM. Reintenta una vez ante error de red / rate limit / 5xx."""
    from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

    client = client or get_client()
    model = model or LLM_MODEL
    if not model:
        raise LLMUnavailable("Falta LLM_MODEL en .env (ver `python -m src.agent.llm`).")
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if tools:
        kwargs.update(tools=tools, tool_choice="auto")
    for intento in range(2):
        try:
            return client.chat.completions.create(**kwargs)
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            if intento == 1:
                raise LLMUnavailable(f"El LLM no respondió tras reintentar: {type(e).__name__}") from e
            time.sleep(3 if isinstance(e, RateLimitError) else 1)
        except APIStatusError as e:
            if e.status_code >= 500 and intento == 0:
                time.sleep(1)
                continue
            raise LLMUnavailable(f"Error del LLM ({e.status_code}): {e.message}") from e
    raise LLMUnavailable("El LLM no respondió.")


def list_models() -> list[str]:
    return sorted(m.id for m in get_client().models.list().data)


if __name__ == "__main__":
    for m in list_models():
        print(m)
