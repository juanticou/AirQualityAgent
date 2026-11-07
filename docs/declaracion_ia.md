# Declaración de uso de IA generativa

Proyecto final SI3003 – Inteligencia Artificial (EAFIT, 2026-1) - Juan Tomás Macías Yepes y Carlos Mario Arroyave Aguirre. Grupo: Jueves

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
- **Qué revisó o modificó el estudiante:** El programa fue revisado por Juan Tomás y Carlos Mario, se hizo una consulta previa acerca de datos y documentos pertinentes para el índice RAG. Se crearon las preguntas y se supervisaron las respuestas cuidadosamente, teniendo en cuenta de donde cita, y si la información es correcta y completa.

### 2026-11-06 (cont.) — Claude Code (Claude Opus 5.5): primera prueba con Groq
- **Qué se pidió:** correr el agente con la clave de Groq (`openai/gpt-oss-120b`) sobre 4 preguntas de prueba.
- **Qué se encontró y corrigió:** límite de solicitudes por minuto (el reintento ahora respeta `retry-after`);
  recomendaciones sin respaldo documental (prompt más estricto); una cita atribuida al documento equivocado
  (campo `cita` listo para copiar y verificación de citas contra los fragmentos recuperados); formatos de cita
  alternativos (`página`, `【…】`).
- **Qué revisó o modificó el estudiante:** El programa fue revisado por Carlos Mario, se desarrollaron correcciones respecto a las respuesta que tenía el agente, dado que citaba el documento que no era.

### 2026-11-06 (cierre) — Claude Code (Claude Opus 5.5)
- **Qué se pidió:** terminar el proyecto: dashboard, evaluación del agente, README final, informe y prueba de instalación limpia.
- **Qué generó la herramienta:** `app.py` (mapa, serie con bandas ICA y estado HMM, predicción, chat con traza) y sus
  smoke tests; `eval/questions.yaml` (14 preguntas), `eval/run_eval.py` y `eval/results.md`; diagrama de arquitectura y
  secciones finales del README; `docs/informe.md`. Corrió la evaluación dos veces (12/14 → 13/14), analizó los fallos
  y ajustó el prompt con reglas generales, sin ajustar preguntas. Verificó la instalación en un clon y un venv nuevos (52 tests en verde).
- **Qué revisó o modificó el estudiante:** decidió no incluir un video demo; _[completar]_.
