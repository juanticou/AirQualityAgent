# Evaluación del agente

- Fecha: 2026-11-06 21:00 · modelo: `openai/gpt-oss-120b` · preguntas: 14
- **Correctas: 13/14 (93%)**
- Herramientas esperadas usadas: 14/14
- Citas correctas cuando se requieren: 6/6
- Iteraciones promedio: 2.4 · latencia promedio: 21.0 s

Criterios automáticos: herramientas esperadas en la traza, contenido mínimo en la respuesta, citas presentes y verificadas contra los fragmentos recuperados (si se requieren) y sin error del LLM. Ver `eval/run_eval.py`; respuestas y trazas completas en `eval/traces.json`.

| # | Categoría | Pregunta | Herramientas llamadas | ¿Correcta? | ¿Citó fuente? | Observación |
|---|---|---|---|---|---|---|
| 1 | dato puntual | ¿Cuál fue el PM2.5 en la estación Casa de Justicia de Itagüí el 2025-12-31? | get_pm25 | ✅ | no (no requerida) | ok |
| 2 | dato puntual | ¿En qué categoría del ICA estuvo la estación Altavista el 6 de marzo de 2024? | get_pm25 | ✅ | no (no requerida) | ok |
| 3 | prediccion | Si hoy es 2024-03-06, ¿qué tan probable es que mañana la calidad del aire esté en alerta en Altavista? | predict_tomorrow | ✅ | no (no requerida) | ok |
| 4 | prediccion con contexto del dashboard | ¿Cómo estará el aire mañana en la estación que tengo seleccionada? | predict_tomorrow | ✅ | no (no requerida) | ok |
| 5 | normativa | ¿Cuál es el límite máximo permisible de PM2.5 en 24 horas en Colombia? | search_docs | ✅ | sí | ok |
| 6 | normativa | ¿Con qué concentraciones de PM2.5 en 24 horas se declaran los niveles de prevención, alerta y emergencia? | search_docs, search_docs | ❌ | sí | no menciona ['38', '56', '151'] |
| 7 | salud | ¿Cuál es el valor guía de la OMS 2021 para PM2.5 en 24 horas? | search_docs, search_docs | ✅ | sí | ok |
| 8 | salud | ¿Quiénes se consideran grupos sensibles a la contaminación del aire? | search_docs | ✅ | sí | ok |
| 9 | cadena de 2 herramientas | ¿El PM2.5 de la estación Tráfico Centro el 2024-03-07 superó el límite diario de la norma colombiana? | get_pm25, search_docs | ✅ | sí | ok |
| 10 | cadena de 3 herramientas | ¿Cómo estaba el aire en Altavista el 2024-03-06, cómo estará al día siguiente y qué recomiendan para niños? | get_pm25, predict_tomorrow, search_docs | ✅ | sí | ok |
| 11 | estacion inexistente | ¿Cómo está el aire en la estación de Rionegro? | get_pm25 | ✅ | no (no requerida) | ok |
| 12 | fecha sin datos | ¿Cuánto PM2.5 hubo en Casa de Justicia (Itagüí) el 2019-05-01? | get_pm25 | ✅ | no (no requerida) | ok |
| 13 | fuera de alcance | ¿Quién ganó el último mundial de fútbol? | — | ✅ | no (no requerida) | ok |
| 14 | consejo medico | Tengo asma. ¿Debo duplicar la dosis de mi inhalador hoy por la contaminación? | search_docs, list_stations, get_pm25 | ✅ | sí (no requerida) | ok |

## Análisis de fallos
- **#6 (niveles de prevención/alerta/emergencia):** el retriever sí recupera la Res. 2254, p. 5 (Tabla 4),
  pero su texto OCR es ilegible en la tabla (`PM2.s'* 24 horas 38-55 56-150 2::151`), y el LLM usa la
  tabla del Acuerdo 15 (cortes EPA 2012). Es un fallo de **calidad del corpus**, no del loop. Solución
  propuesta: usar el texto oficial en HTML de la resolución. No se ajustó la pregunta.
- **Corrida anterior (12/14):** además falló #2 (dio la categoría sin el valor), que se corrigió con una
  regla general en el prompt, y en #6 se inventó una fila de "prevención", que ya no ocurre.
