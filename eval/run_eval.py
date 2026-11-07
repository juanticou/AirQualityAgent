"""Evaluación automática del agente sobre eval/questions.yaml.

Criterios por pregunta (todos deben cumplirse para "correcta"):
- herramientas: cada herramienta esperada aparece en la traza ("a|b" = cualquiera);
- contenido: la respuesta contiene cada elemento de `debe_contener` (sin tildes ni mayúsculas);
- cita (si `requiere_cita`): >= 1 cita y ninguna fuera de los fragmentos recuperados;
- sin error del LLM.
Salidas: eval/results.md (tabla + resumen) y eval/traces.json (trazas completas).

Uso: python eval/run_eval.py [--pausa 20] [--ids 1 5 10]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.loop import run_agent  # noqa: E402
from src.config import EVAL_DIR, LLM_MODEL  # noqa: E402


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def check(q: dict, r) -> dict:
    called = r.tools_called()
    tools_ok = all(any(alt in called for alt in exp.split("|")) for exp in q["herramienta_esperada"])
    ans = norm(r.answer)
    missing = [item for item in q.get("debe_contener", [])
               if not any(norm(alt) in ans for alt in item.split("|"))]
    cites = bool(r.citations)
    cite_ok = (cites and not r.unverified_citations) if q["requiere_cita"] else True
    correct = tools_ok and not missing and cite_ok and r.error is None
    obs = []
    if r.error:
        obs.append(f"error LLM: {r.error}")
    if not tools_ok:
        obs.append(f"faltó herramienta (esperada {q['herramienta_esperada']})")
    if missing:
        obs.append(f"no menciona {missing}")
    if q["requiere_cita"] and not cites:
        obs.append("sin cita")
    if r.unverified_citations:
        obs.append(f"citas no verificadas {r.unverified_citations}")
    if r.hit_t_max:
        obs.append("alcanzó T_MAX")
    return {"correcta": correct, "herramientas_ok": tools_ok, "cita_ok": cite_ok, "cito": cites,
            "observacion": "; ".join(obs) or "ok"}


def run(ids: list[int] | None, pause: float) -> list[dict]:
    questions = yaml.safe_load((EVAL_DIR / "questions.yaml").read_text(encoding="utf-8"))
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    rows = []
    for i, q in enumerate(questions):
        if i:
            time.sleep(pause)               # límite de tokens por minuto del plan gratuito
        ctx = q.get("contexto") or {}
        t0 = time.perf_counter()
        r = run_agent(q["pregunta"], estacion=ctx.get("estacion"), fecha=ctx.get("fecha"))
        if r.error and "RateLimit" in r.error:   # un segundo intento tras esperar
            time.sleep(60)
            r = run_agent(q["pregunta"], estacion=ctx.get("estacion"), fecha=ctx.get("fecha"))
        res = check(q, r)
        rows.append({**q, **res, "herramientas_llamadas": r.tools_called(), "citas": r.citations,
                     "citas_no_verificadas": r.unverified_citations, "iteraciones": r.iterations,
                     "latencia_s": round(time.perf_counter() - t0, 1), "respuesta": r.answer,
                     "traza": r.trace})
        print(f"[{q['id']:>2}] {'OK ' if res['correcta'] else 'NO '} {q['categoria']}: {res['observacion']}")
    return rows


def write_report(rows: list[dict]) -> None:
    n = len(rows)
    ok = sum(r["correcta"] for r in rows)
    need_cite = [r for r in rows if r["requiere_cita"]]
    lines = [
        "# Evaluación del agente", "",
        f"- Fecha: {datetime.now():%Y-%m-%d %H:%M} · modelo: `{LLM_MODEL}` · preguntas: {n}",
        f"- **Correctas: {ok}/{n} ({ok / n:.0%})**",
        f"- Herramientas esperadas usadas: {sum(r['herramientas_ok'] for r in rows)}/{n}",
        f"- Citas correctas cuando se requieren: {sum(r['cita_ok'] for r in need_cite)}/{len(need_cite)}",
        f"- Iteraciones promedio: {sum(r['iteraciones'] for r in rows) / n:.1f} · "
        f"latencia promedio: {sum(r['latencia_s'] for r in rows) / n:.1f} s", "",
        "Criterios automáticos: herramientas esperadas en la traza, contenido mínimo en la respuesta, "
        "citas presentes y verificadas contra los fragmentos recuperados (si se requieren) y sin error del LLM. "
        "Ver `eval/run_eval.py`; respuestas y trazas completas en `eval/traces.json`.", "",
        "| # | Categoría | Pregunta | Herramientas llamadas | ¿Correcta? | ¿Citó fuente? | Observación |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cite = ("sí" if r["cito"] else "no") + ("" if r["requiere_cita"] else " (no requerida)")
        lines.append(f"| {r['id']} | {r['categoria']} | {r['pregunta']} | "
                     f"{', '.join(r['herramientas_llamadas']) or '—'} | {'✅' if r['correcta'] else '❌'} | "
                     f"{cite} | {r['observacion']} |")
    (EVAL_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (EVAL_DIR / "traces.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str),
                                          encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pausa", type=float, default=20.0, help="segundos entre preguntas")
    ap.add_argument("--ids", type=int, nargs="*")
    args = ap.parse_args()
    rows = run(args.ids, args.pausa)
    write_report(rows)
    ok = sum(r["correcta"] for r in rows)
    print(f"\nCorrectas: {ok}/{len(rows)} -> eval/results.md")


if __name__ == "__main__":
    main()
