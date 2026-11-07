# Declaración de uso de IA generativa

Proyecto final SI3003 – Inteligencia Artificial (EAFIT, 2026-1). Grupo: _[completar]_.

Usamos IA generativa como asistente de programación. Todo el código lo revisó el equipo, que
debe poder explicarlo y defenderlo; las decisiones de diseño y la interpretación de resultados
son nuestras. Cada sesión queda registrada abajo.

## Registro por sesión

### 2026-11-06 — Claude Code (Claude Opus 5.5)
- **Qué se pidió:** leer `CLAUDE.md`, proponer un plan e implementar en una sola sesión hasta el
  agente con RAG funcional: andamiaje, descarga y limpieza de datos, Naive Bayes y HMM propios,
  líneas base y evaluación, corpus e índice RAG, herramientas y loop del agente.
- **Qué generó la herramienta:** la estructura del repositorio; `src/config.py`; `src/data/`
  (descarga desde SIATA, AMVA, datos.gov.co/IDEAM-SISAIRE y Open-Meteo; limpieza; ICA; features);
  `src/models/` (NB categórico y HMM gaussiano en NumPy, líneas base, evaluación, entrenamiento);
  `src/rag/` (ingesta e índice FAISS/BM25); `src/agent/` (cliente LLM, herramientas, loop, prompt);
  tests en `tests/`. También buscó y descargó el corpus y verificó los rangos ICA contra la Res. 2254.
- **Qué revisó o modificó el estudiante:** _[completar: qué se leyó, qué se cambió, qué se verificó a mano]_.

### 2026-11-06 (cont.) — Claude Code (Claude Opus 5.5): primera prueba con Groq
- **Qué se pidió:** correr el agente con la clave de Groq (`openai/gpt-oss-120b`) sobre 4 preguntas de prueba.
- **Qué se encontró y corrigió:** límite de solicitudes por minuto (el reintento ahora respeta `retry-after`);
  recomendaciones sin respaldo documental (prompt más estricto); una cita atribuida al documento equivocado
  (campo `cita` listo para copiar y verificación de citas contra los fragmentos recuperados); formatos de cita
  alternativos (`página`, `【…】`).
- **Qué revisó o modificó el estudiante:** _[completar]_.
