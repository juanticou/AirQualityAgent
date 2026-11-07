# Prompts para Claude Code — Agente de calidad del aire del Valle de Aburrá

Uso: crea una carpeta vacía `calidad-aire-aburra/`, pon dentro `CLAUDE.md`, abre Claude Code ahí
(`claude`) y pega el **Prompt 0**. Los demás prompts van uno por fase, en sesiones o en momentos distintos.
Revisa y entiende cada entrega antes de pasar a la siguiente: lo que no puedas explicar, no lo entregues.

---

## Prompt 0 — Arranque (pegar primero)

```
Lee CLAUDE.md completo. Es el contexto del proyecto final de SI3003 (EAFIT): un agente de calidad del
aire del Valle de Aburrá con un modelo clásico implementado por nosotros (Naive Bayes + HMM), un agente
LLM con RAG y tool calling, y un dashboard en Streamlit. La entrega es el viernes 13 de noviembre a las
6:00 p. m. y hoy es viernes 6 de noviembre, así que el alcance es un MVP que corra de punta a punta.

Antes de escribir código:
1. Resume en 5 líneas lo que entendiste del proyecto y de la rúbrica.
2. Dime qué riesgos ves para terminar a tiempo y qué recortarías primero si hace falta.
3. Propón un plan de trabajo por fases alineado con el cronograma de CLAUDE.md, con la lista de tareas.
4. Hazme las preguntas que necesites (por ejemplo: nombre del grupo, si ya tengo API key de Groq,
   si trabajo en Lightning AI o en local, si ya descargué algún dato).

No empieces a implementar hasta que confirme el plan.
```

---

## Prompt 1 — Andamiaje del repositorio

```
Crea la estructura del repositorio exactamente como en la sección 9 de CLAUDE.md: carpetas, __init__.py,
src/config.py (rutas relativas, constantes ICA, clases normal/alerta/critico, T_MAX=6, lectura de .env),
.gitignore, .env.example, requirements.txt con versiones fijadas y compatibles con Python 3.11 en CPU,
un README esqueleto y un Makefile con objetivos data, train, index, eval, app y test.
Incluye src/data/ica.py con la función que convierte PM2.5 diario a categoría ICA y a la clase del proyecto,
con tests en tests/test_ica.py para los bordes de cada rango. Corre pytest y muéstrame el resultado.
Inicializa git y haz el primer commit.
```

---

## Prompt 2 — Datos reales y plan B (prioridad máxima)

```
Objetivo: tener datos diarios de PM2.5 por estación del Valle de Aburrá y meteorología (lluvia, viento),
limpios y guardados en data/processed/, con el acceso verificado y el plan B listo.

1. Escribe src/data/download.py que intente, en este orden: SIATA, el portal de datos abiertos del AMVA
   (dataset mediciones-estaciones-calidad-del-aire), datos.gov.co. Para cada fuente reporta si respondió,
   qué formato trae, qué estaciones y qué rango de fechas. Si una falla, sigue con la siguiente sin romper.
2. Si ninguna fuente automática funciona, dime exactamente qué archivo debo descargar a mano y desde dónde,
   y deja el código listo para leerlo desde data/raw/.
3. Meteorología: usa estaciones SIATA si están disponibles; si no, propón Open-Meteo histórico por
   coordenadas de cada estación y documenta que es una fuente distinta.
4. src/data/clean.py: agregación a promedio diario de 24 h, descarte de días con menos de 75 % de horas
   válidas (reporta cuántos), unión con meteorología, columna de categoría ICA y clase del proyecto.
   Guarda data/processed/pm25_diario.csv y data/processed/estaciones.csv (id, nombre, municipio, lat, lon).
5. Agrega las fuentes con URL y fecha de descarga a la sección de datos del README.
6. Crea notebooks/01_eda.ipynb que use src/ (sin duplicar lógica): cobertura por estación, serie de tiempo,
   distribución de clases por mes (deben verse las temporadas feb-abr y oct-nov), relación lluvia/viento vs PM2.5.

Dime al final cuántas estaciones y cuántos días útiles quedaron, y qué tan desbalanceadas están las clases.
```

---

## Prompt 3 — Parte 1: Naive Bayes, líneas base y evaluación

```
Implementa la sección 5.1 y 5.3 de CLAUDE.md.

1. src/data/features.py: features discretizadas del día t para predecir la clase de t+1 por estación
   (categoría de hoy, tendencia vs t-1, lluvia, viento en terciles calculados SOLO con entrenamiento,
   temporada, día de la semana). Cuidado con la fuga de información: nada del día t+1 en las features.
2. src/models/naive_bayes.py: Naive Bayes categórico en NumPy con suavizado de Laplace y log-probabilidades.
   Comenta las ecuaciones en el código. Métodos fit, predict_proba, predict, save, load.
3. tests/test_naive_bayes.py: compara predict_proba con sklearn CategoricalNB en datos sintéticos (tol 1e-6).
4. src/models/baselines.py: persistencia y climatología mensual.
5. src/models/evaluate.py: partición temporal (propón los cortes según el rango real de los datos),
   accuracy, F1 macro, recall de alerta+critico, matriz de confusión. Guarda results/metrics_part1.json
   y figuras en results/figures/.
6. Script de entrenamiento (make train) que deja el modelo en models/.
7. notebooks/02_parte1_modelos.ipynb con la comparación NB vs líneas base y una interpretación honesta.

Al terminar, muéstrame la tabla de métricas y explícame en pocas líneas si el NB le gana a la persistencia
y por qué (o por qué no). Si no le gana, propón 1-2 ajustes de features razonables, no más.
```

---

## Prompt 4 — Parte 1: HMM (solo si el Prompt 3 está cerrado)

```
Implementa la sección 5.2 de CLAUDE.md: HMM de 3 estados con emisiones gaussianas sobre log(PM2.5 diario),
en NumPy, con forward escalado, Viterbi y Baum-Welch. Comenta las ecuaciones.
Tests: log-verosimilitud no decreciente en Baum-Welch y recuperación aproximada de parámetros en datos simulados.
Etiqueta los estados por media de emisión. Agrega la predicción a un paso P(z_{t+1}|x_{1:t}) = A^T α_t
como segundo predictor de mañana y evalúalo con la misma partición y métricas que el NB.
Actualiza metrics_part1.json, el notebook 02 y guarda el modelo en models/.
Si ves que esto pone en riesgo el checkpoint del lunes, dímelo antes y lo dejamos para después.
```

---

## Prompt 5 — Checkpoint del lunes: agente mínimo + dashboard esqueleto

```
Objetivo del checkpoint: agente mínimo con UNA herramienta y dashboard esqueleto, ambos corriendo.

1. src/agent/llm.py: cliente openai con base_url, api_key y modelo desde .env. Lista los modelos disponibles
   con client.models.list() y dime cuáles soportan tool calling para elegir LLM_MODEL. Timeout y un reintento.
2. src/agent/tools.py: get_pm25(estacion, fecha=None) leyendo data/processed/, con JSON Schema y errores
   amigables (estación inexistente, fecha sin datos) devueltos como {"error": ...}.
3. src/agent/loop.py: loop manual con T_MAX, ejecución de tool calls, errores como observación y traza de pasos.
4. src/agent/prompts.py: system prompt según la sección 6 de CLAUDE.md.
5. app.py: sidebar (estación, rango de fechas), mapa de estaciones coloreado por ICA, serie de tiempo con bandas
   ICA y línea de 37 µg/m³, y chat que usa el agente. Sin API key la app debe arrancar y avisar.
6. Pruébalo: corre la app en modo headless para verificar que arranca sin errores y haz 3 preguntas al agente
   desde un script. Muéstrame las trazas.
```

---

## Prompt 6 — RAG y herramientas completas

```
1. Ya puse los documentos del corpus en data/corpus/ y sus URLs en data/corpus/SOURCES.md.
   [Si aún no: dime qué documentos descargar (Resolución 2254 de 2017, POECA del AMVA, material SIATA/AMVA
   sobre ICA y recomendaciones de salud, guías OMS 2021) y espera.]
2. src/rag/ingest.py: texto por página con pypdf, chunks con solapamiento, metadatos fuente y pagina,
   índice FAISS con un modelo multilingüe pequeño en CPU. Fallback a BM25 si falla la descarga del modelo.
   Comando: make index.
3. src/rag/retriever.py con top-k y umbral. Pruébalo con 5 consultas y muéstrame qué recupera.
4. Herramientas search_docs(query, k) y predict_tomorrow(estacion) (usa el NB y el HMM si existe, devuelve
   probabilidades, clase más probable y desempeño del modelo en test). Opcional: list_stations().
5. Verifica los rangos ICA de CLAUDE.md contra el anexo de la Resolución 2254 ya indexada; corrige si difieren.
6. Ajusta el system prompt para exigir citas [fuente, p. N] y el aviso de no consejo médico.
7. Prueba 5 preguntas que obliguen a encadenar herramientas, p. ej.
   "¿Cómo está el aire hoy en la estación X, cómo estará mañana y qué recomienda la norma para niños?"
```

---

## Prompt 7 — Pulido del dashboard e interacción

```
Mejora app.py según la sección 7 de CLAUDE.md:
- Tarjeta de predicción de mañana con barras de probabilidad por clase y desempeño del modelo.
- Si existe HMM, franja de estado latente sobre la serie de tiempo.
- En el chat: expander "¿Cómo llegó a esta respuesta?" con herramientas, argumentos y citas.
- La estación y fecha seleccionadas en la sidebar se pasan como contexto al agente.
- Aviso visible de herramienta académica, no consejo médico ni declaratoria oficial.
- Caché de datos, modelos e índice; la app no reentrena ni reindexa al arrancar.
Que se vea limpio y legible, sin recargarlo. Prueba en headless.
```

---

## Prompt 8 — Evaluación del agente

```
Implementa la sección 8 de CLAUDE.md. Escribe eval/questions.yaml con 13 preguntas que cubran todos los
casos listados (incluye una estación inexistente, una fuera de alcance y una que pida consejo médico),
con herramienta_esperada y requiere_cita. eval/run_eval.py corre todo, guarda las trazas y genera
eval/results.md con la tabla y el resumen. Córrelo, muéstrame los resultados y analiza los fallos:
¿son del prompt, de las herramientas o del retriever? Propón correcciones puntuales y vuelve a correr.
No ajustes el sistema para que pase preguntas específicas de forma artificial; reporta los fallos que queden.
```

---

## Prompt 9 — Congelamiento (jueves 12)

```
Vamos a congelar el código.
1. Simula una instalación limpia: entorno virtual nuevo, pip install -r requirements.txt, seguir el README
   paso a paso sin atajos. Corrige todo lo que falle. Corre pytest y make eval.
2. Completa el README: descripción, arquitectura (diagrama en Mermaid), instalación, datos y fuentes citadas,
   cómo correr cada parte, resultados principales, limitaciones y aviso de no consejo médico.
3. Redacta docs/informe.md (máximo 4 páginas): problema, datos, método de la Parte 1, diseño del agente y RAG,
   resultados (métricas Parte 1 y evaluación del agente) y limitaciones. Usa solo números que estén en results/ y eval/.
4. Completa docs/declaracion_ia.md con lo registrado en cada sesión.
5. Dame un guion de 3-4 minutos para el video demo de respaldo y una lista de verificación final.
6. Revisa que no haya claves, .env ni archivos pesados innecesarios en git. Commit final.
```

---

## Prompts útiles en cualquier momento

- **Para entender el código:** `Explícame src/models/hmm.py como si me fueran a preguntar en un examen: ecuaciones, cada función y qué fallaría si quito el escalado.`
- **Para recortar alcance:** `Estamos atrasados. Con lo que hay, ¿qué es lo mínimo que cumple todos los requisitos obligatorios? Propón el recorte y espera confirmación.`
- **Al cerrar cada sesión:** `Agrega la entrada de hoy a docs/declaracion_ia.md, actualiza el README si cambió algo y haz commit.`
