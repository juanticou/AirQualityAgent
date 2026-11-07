"""Loop agéntico manual con tool calling (sin frameworks).

1. mensajes = system + contexto del dashboard + historial + pregunta
2. llamar al LLM con las herramientas; si pide tool calls -> ejecutarlas -> agregar mensajes 'tool'
3. repetir hasta respuesta final o T_MAX iteraciones; en T_MAX se fuerza una respuesta sin herramientas
4. errores de herramientas = observaciones; errores del LLM = mensaje claro al usuario
5. traza de cada paso (herramienta, argumentos, resumen, latencia) para el dashboard / evaluación

Uso: python -m src.agent.loop "¿Cómo está el aire hoy en Itagüí?"
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from src.agent.llm import LLMUnavailable, chat, get_client
from src.agent.prompts import SYSTEM_PROMPT, context_message
from src.agent.tools import TOOLS_SCHEMA, execute_tool
from src.config import T_MAX

# acepta [..] y los corchetes 【..】 que usan algunos modelos (p. ej. gpt-oss)
CITATION_RE = re.compile(r"[\[【](?:fuente:\s*)?([^\[\]【】]+?),\s*(?:p\.|pág\.|página)\s*(\d+)[\]】]",
                         re.IGNORECASE)


def extract_citations(text: str) -> list[tuple[str, int]]:
    """Citas [fuente, p. N] de la respuesta (tolera 'página', 'fuente:' y espacios no estándar)."""
    text = text.replace(" ", " ").replace(" ", " ")
    return sorted(set((f.strip(), int(p)) for f, p in CITATION_RE.findall(text)))


@dataclass
class AgentResult:
    answer: str
    trace: list[dict] = field(default_factory=list)
    citations: list[tuple[str, int]] = field(default_factory=list)
    unverified_citations: list[tuple[str, int]] = field(default_factory=list)  # no recuperadas
    iterations: int = 0
    hit_t_max: bool = False
    error: str | None = None

    def tools_called(self) -> list[str]:
        return [s["herramienta"] for s in self.trace if s["tipo"] == "tool"]


def _summarize(result: dict, limit: int = 300) -> str:
    if "error" in result:
        return f"ERROR: {result['error']}"
    if "resultados" in result:
        return f"{len(result['resultados'])} fragmentos: " + "; ".join(
            f"{r['fuente']} p.{r['pagina']}" for r in result["resultados"])
    s = json.dumps(result, ensure_ascii=False)
    return s[:limit] + ("…" if len(s) > limit else "")


def run_agent(question: str, history: list[dict] | None = None, estacion: str | None = None,
              fecha: str | None = None, model: str | None = None, t_max: int = T_MAX) -> AgentResult:
    """Responde `question` usando herramientas. Nunca lanza excepción: los fallos van en AgentResult.error."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    ctx = context_message(estacion, fecha)
    if ctx:
        messages.append({"role": "system", "content": ctx})
    messages += list(history or [])
    messages.append({"role": "user", "content": question})
    trace: list[dict] = []
    retrieved: set[tuple[str, int]] = set()   # (fuente, pagina) devueltos por search_docs
    try:
        client = get_client()
    except LLMUnavailable as e:
        return AgentResult(answer=str(e), error=str(e))

    for it in range(1, t_max + 1):
        last = it == t_max
        t0 = time.perf_counter()
        try:
            # en la última iteración se quitan las herramientas para obligar a responder
            resp = chat(messages, tools=None if last else TOOLS_SCHEMA, model=model, client=client)
        except LLMUnavailable as e:
            msg = f"No pude completar la respuesta: {e}. Intenta de nuevo en un momento."
            return AgentResult(answer=msg, trace=trace, iterations=it, error=str(e))
        msg = resp.choices[0].message
        trace.append({"tipo": "llm", "iteracion": it, "latencia_s": round(time.perf_counter() - t0, 2),
                      "tool_calls": len(msg.tool_calls or [])})
        if not msg.tool_calls:
            answer = msg.content or ""
            if last:
                answer = ("(Alcancé el límite de pasos de razonamiento; respondo con lo que tengo.)\n\n"
                          + answer)
            cits = extract_citations(answer)
            return AgentResult(answer=answer, trace=trace, iterations=it, hit_t_max=last,
                               citations=cits,
                               unverified_citations=[c for c in cits if c not in retrieved])
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments or "{}"}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            result, dt = execute_tool(tc.function.name, tc.function.arguments)
            for r in result.get("resultados", []) if tc.function.name == "search_docs" else []:
                retrieved.add((r["fuente"], int(r["pagina"])))
            trace.append({"tipo": "tool", "iteracion": it, "herramienta": tc.function.name,
                          "argumentos": tc.function.arguments, "resultado": _summarize(result),
                          "error": "error" in result, "latencia_s": round(dt, 2)})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, ensure_ascii=False, default=str)})
    return AgentResult(answer="No logré responder dentro del límite de pasos.", trace=trace,
                       iterations=t_max, hit_t_max=True)


def print_result(q: str, r: AgentResult) -> None:
    print(f"\n{'=' * 80}\nPREGUNTA: {q}")
    for s in r.trace:
        if s["tipo"] == "tool":
            print(f"  [it {s['iteracion']}] {s['herramienta']}({s['argumentos']}) "
                  f"{s['latencia_s']}s -> {s['resultado'][:220]}")
        else:
            print(f"  [it {s['iteracion']}] LLM {s['latencia_s']}s, tool_calls={s['tool_calls']}")
    print(f"--- RESPUESTA ({r.iterations} iteraciones, T_MAX alcanzado={r.hit_t_max}, citas={r.citations}, "
          f"citas_no_verificadas={r.unverified_citations})")
    print(r.answer)


if __name__ == "__main__":
    import sys

    for q in sys.argv[1:] or ["¿Cómo está el aire hoy en Itagüí, Casa de Justicia?"]:
        print_result(q, run_agent(q))
