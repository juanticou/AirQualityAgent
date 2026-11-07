# CLAUDE.md — Agente de calidad del aire del Valle de Aburrá

> Contexto persistente para Claude Code. Léelo completo al inicio de cada sesión.
> Proyecto final de **SI3003 – Inteligencia Artificial (EAFIT, 2026-1)**, idea 3.1 del enunciado.
> **Entrega: viernes 13 de noviembre de 2026, 6:00 p. m. (hora Colombia).** Código congelado el jueves 12.

---

## 1. Qué estamos construyendo

Un **MVP funcional** que le explica a un ciudadano qué significa el PM2.5 del día en el Valle de Aburrá,
qué tan probable es que mañana empeore y qué hacer al respecto, citando normativa y protocolos oficiales.

Tres piezas obligatorias, combinadas de forma **no trivial**:

| Pieza | Qué es | Cómo se conecta |
|---|---|---|
| **Parte 1 (IA clásica)** | Modelos implementados por nosotros en NumPy: Naive Bayes categórico para la categoría de mañana + HMM para el estado latente (normal / alerta / crítico) | Sus salidas son lo que devuelve la herramienta `predict_tomorrow()` del agente |
| **Parte 2 (agente LLM)** | Loop agéntico manual con tool calling + RAG sobre normativa, POECA y recomendaciones de salud | El agente decide cuándo consultar datos, predecir o buscar documentos, y cita fuentes |
| **Dashboard** | Streamlit: mapa de estaciones, serie de tiempo con bandas ICA, predicción y chat | El usuario elige estación y fechas; el chat usa la misma selección como contexto |

**Principio rector: alcance pequeño que corre de punta a punta > alcance ambicioso que falla en la demo.**
Una demo que falla se califica como no funcional. Ante la duda, recorta funcionalidad, no robustez.

---

## 2. Rúbrica (guía todas las decisiones)

| Criterio | Peso | Qué implica para el código |
|---|---|---|
| Técnica Parte 1: implementación propia, corrección, métrica vs. línea base | 20 % | NB y HMM escritos a mano en NumPy; tests que comparen con implementación de referencia; métricas vs. persistencia y climatología |
| Agente LLM: loop, RAG, ≥ 2 herramientas, manejo de errores, citas | 20 % | `T_MAX` en el loop, errores de herramientas devueltos como observación, citas `[doc, p. N]` |
| Dashboard: visualización clara | 10 % | ≥ 2 visualizaciones significativas (mapa + serie de tiempo como mínimo) |
| Interacción y funcionalidad (sin fallas en la demo) | 15 % | Selección de estación/fecha + chat; la app arranca aunque falte API key o internet |
| Rigor experimental | 15 % | Partición temporal, métricas con matriz de confusión, evaluación del agente con 10–15 preguntas fijas, limitaciones escritas |
| Reproducibilidad | 10 % | README que funciona copiando y pegando, `requirements.txt` con versiones, datos procesados versionados |
| Presentación | 5 % | — |
| Declaración y uso ético de IA generativa | 5 % | `docs/declaracion_ia.md` actualizado en cada sesión |

**No hay sustentación**: el repositorio debe explicarse solo.

---

## 3. Stack y entorno

- **Python 3.11**. Debe correr en **CPU** (Lightning AI Studio o laptop). Nada de GPU.
- **LLM**: cliente `openai` apuntando a un endpoint compatible gratuito. Por defecto **Groq**
  (`base_url=https://api.groq.com/openai/v1`). Alternativas configurables: NVIDIA NIM, Ollama local.
  - El modelo se lee de `LLM_MODEL` en `.env`. **No asumas un nombre de modelo**: verifica con
    `client.models.list()` cuáles están disponibles y elige uno que soporte tool calling.
- **RAG**: `sentence-transformers` con un modelo multilingüe pequeño
  (p. ej. `paraphrase-multilingual-MiniLM-L12-v2`) + **FAISS** (`faiss-cpu`). Fallback a BM25 (`rank-bm25`)
  si no se puede descargar el modelo de embeddings.
- **Dashboard**: Streamlit + Plotly (mapa con `scatter_mapbox`/`scatter_map` y estilo de mapa sin token).
- **Datos**: pandas. **Modelos Parte 1**: solo NumPy (scikit-learn se permite únicamente para métricas,
  particiones y como referencia en tests).
- Secretos en `.env` (nunca en el repo). Mantener `.env.example` actualizado.

---

## 4. Datos

### Fuentes (en orden de preferencia)
1. **SIATA** (siata.gov.co): estaciones de calidad del aire y meteorología del Valle de Aburrá.
2. **Portal de datos abiertos del AMVA**: dataset *mediciones-estaciones-calidad-del-aire*
   (`https://datosabiertos.metropol.gov.co/dataset/mediciones-estaciones-calidad-del-aire`).
3. **datos.gov.co** (buscar PM2.5 Valle de Aburrá / Medellín).
4. **Plan B obligatorio**: CSV históricos descargados y versionados en `data/processed/`.
   Si ninguna fuente oficial es accesible, documentar el problema y usar una fuente abierta verificable
   (p. ej. OpenAQ o la API de calidad del aire de Open-Meteo, que es **modelada, no medida** — decirlo explícitamente).

Meteorología (lluvia, viento) para las features: estaciones SIATA si están; si no, Open-Meteo histórico
para las coordenadas de cada estación (documentar).

### Reglas
- **Toda fuente se cita** en el README y en el informe: URL, fecha de descarga, licencia si existe.
- Guardar crudos en `data/raw/` (gitignored si pesan > 50 MB) y procesados en `data/processed/` (versionados).
- La app **nunca** depende de una API en vivo para arrancar: lee `data/processed/`. La consulta en vivo es opcional
  y con fallback silencioso al CSV.
- Agregar a **promedio diario (24 h)** por estación; descartar días con < 75 % de horas válidas
  (criterio usado en los informes de SIATA). Registrar cuántos días se descartan.

### Categorías ICA para PM2.5 (24 h, Resolución 2254 de 2017 de MinAmbiente)
| Categoría | µg/m³ | Color |
|---|---|---|
| Buena | 0–12 | Verde |
| Aceptable | 13–37 | Amarillo |
| Dañina a grupos sensibles | 38–55 | Naranja |
| Dañina a la salud | 56–150 | Rojo |
| Muy dañina | 151–250 | Morado |
| Peligrosa | 251–500 | Marrón |

Límite máximo permisible diario PM2.5: **37 µg/m³**. **Verificado el 2026-11-06** contra el corpus:
la Tabla 1 (p. 2) fija 50 µg/m³ en 24 h desde enero de 2018 y el parágrafo 1 lo baja a 37 µg/m³ desde el 1 de julio de 2018;
los rangos ICA de esta tabla coinciden con la Tabla 6 (p. 9). Según la Tabla 4 (p. 5), los niveles de declaratoria para
PM2.5 (24 h) son Prevención 38–55, Alerta 56–150 y Emergencia ≥ 151 µg/m³.

**Clases del proyecto (3)**, derivadas del ICA:
`normal` (≤ 37) · `alerta` (38–55) · `critico` (≥ 56).
Son una **simplificación nuestra**: los niveles oficiales del POECA (normal, prevención, alerta, emergencia)
los declara el AMVA con criterios adicionales. Decirlo en la interfaz y en el informe. Por rangos, nuestra clase `alerta`
coincide con el nivel oficial de *Prevención* y `critico` con *Alerta* o *Emergencia* (Res. 2254, Tabla 4).

---

## 5. Parte 1 — Modelos clásicos (implementación propia)

### 5.1 Naive Bayes categórico — predicción de la categoría de mañana (núcleo, obligatorio)
- Target: clase de `t+1` (`normal` / `alerta` / `critico`) por estación.
- Features discretizadas del día `t`: categoría PM2.5 de hoy, tendencia (sube/baja vs. `t-1`), lluvia (sí/no o terciles),
  viento (terciles), mes o temporada (feb–abr y oct–nov son temporadas críticas en el valle), día de la semana.
- Implementación en `src/models/naive_bayes.py`: `fit`, `predict_proba`, `predict`, suavizado de Laplace,
  trabajo en log-probabilidades.
- Test: en un dataset sintético, `predict_proba` coincide con `sklearn.naive_bayes.CategoricalNB` (tolerancia 1e-6).

### 5.2 HMM — estado latente (segundo hito, solo si 5.1 está listo y evaluado)
- 3 estados ocultos, emisiones gaussianas sobre `log(PM2.5 diario)` (opcional: + lluvia).
- Implementar en `src/models/hmm.py`: forward (filtrado, escalado para estabilidad), Viterbi, Baum-Welch.
- Etiquetar estados ordenándolos por media de emisión.
- Predicción a un paso: `P(z_{t+1} | x_{1:t}) = A^T α_t` → sirve como segundo predictor de mañana.
- Test: log-verosimilitud no decreciente en Baum-Welch; en datos simulados recupera parámetros razonablemente.

### 5.3 Líneas base y evaluación (`src/models/baselines.py`, `src/models/evaluate.py`)
- **Persistencia**: mañana = hoy. **Climatología**: clase más frecuente por mes en entrenamiento.
- **Partición temporal** (nunca aleatoria): p. ej. entrenar hasta 2023, validar 2024, probar 2025+. Ajustar a los datos reales.
- Métricas: accuracy, **F1 macro** (clases desbalanceadas), recall de `alerta`+`critico`, matriz de confusión.
- Resultados guardados en `results/metrics_part1.json` y figuras en `results/figures/`.
- Los modelos entrenados se serializan en `models/` para que la app no reentrene al arrancar.

---

## 6. Parte 2 — Agente LLM con RAG

### Loop (`src/agent/loop.py`) — manual, sin LangGraph salvo que sobre tiempo
1. Mensajes = system prompt + historial + pregunta (+ estación/fecha seleccionadas en el dashboard como contexto).
2. Llamar al LLM con `tools`. Si pide herramientas → ejecutarlas → agregar resultados como mensajes `tool`.
3. Repetir hasta respuesta final o **`T_MAX = 6`** iteraciones; al llegar a `T_MAX`, responder con lo disponible y decirlo.
4. **Errores de herramientas se devuelven como observación** (`{"error": "..."}`), nunca rompen el loop.
5. Registrar cada paso (herramienta, argumentos, resultado resumido, latencia) en una traza que el dashboard puede mostrar.
6. Timeout y reintento único ante errores de red/rate limit del LLM; mensaje claro si persiste.

### Herramientas (`src/agent/tools.py`), con JSON Schema y docstrings claros
| Herramienta | Devuelve |
|---|---|
| `get_pm25(estacion, fecha=None)` | PM2.5 diario, categoría ICA, color y últimos 7 días; error amigable si la estación o fecha no existen |
| `predict_tomorrow(estacion)` | Probabilidades por clase (NB y, si existe, HMM), clase más probable, features usadas, desempeño del modelo en test |
| `search_docs(query, k=4)` | Fragmentos del corpus con `fuente`, `pagina`, `texto` |
| `list_stations()` *(opcional)* | Estaciones disponibles con municipio y coordenadas |

### RAG (`src/rag/`)
- Corpus en `data/corpus/` (PDF/HTML descargados por el equipo, con `data/corpus/SOURCES.md` listando URL y fecha):
  Resolución 2254 de 2017; POECA del AMVA (Acuerdo Metropolitano 04 de 2018 y su modificación de 2021);
  material del SIATA/AMVA sobre ICA y recomendaciones; guías de calidad del aire de la OMS (2021).
- `ingest.py`: extraer texto por página (`pypdf`), chunks de ~500–800 tokens con solapamiento, metadatos `fuente` + `pagina`,
  construir índice FAISS en `data/index/` (versionado si pesa poco, si no, se reconstruye con un comando del README).
- `retriever.py`: búsqueda top-k con umbral de similitud; si nada supera el umbral, devolverlo explícitamente.

### System prompt (`src/agent/prompts.py`) — requisitos
- Responde en español claro para ciudadanos.
- Usa herramientas para cualquier dato numérico; **nunca inventa concentraciones ni predicciones**.
- Toda afirmación normativa o de salud lleva cita `[fuente, p. N]` tomada de `search_docs`. Si no hay documento que la respalde, lo dice.
- Incluye siempre que no es consejo médico ni declaratoria oficial; remite a SIATA/AMVA para avisos oficiales.
- Explica la incertidumbre de la predicción (probabilidades, desempeño del modelo).

---

## 7. Dashboard (`app.py`)

- **Sidebar**: estación, rango de fechas, (opcional) proveedor/modelo LLM, botón "limpiar chat".
- **Visualizaciones mínimas**: (1) mapa de estaciones coloreado por categoría ICA del día seleccionado;
  (2) serie de tiempo de PM2.5 con bandas ICA y línea de 37 µg/m³; (3) tarjeta de predicción de mañana con barras de probabilidad;
  (4) *(si hay HMM)* franja de estado latente sobre la serie.
- **Chat** con `st.chat_message`, mostrando herramientas usadas y citas (expander "¿Cómo llegó a esta respuesta?").
- **Robustez**: sin `GROQ_API_KEY` la app arranca, muestra visualizaciones y avisa que el chat está desactivado.
  Usar `st.cache_data`/`st.cache_resource` para datos, modelos e índice.
- Aviso visible: herramienta académica, no consejo médico ni oficial.

---

## 8. Evaluación del agente (`eval/`)

- `eval/questions.yaml`: **12–15 preguntas fijas** con `herramienta_esperada` y `requiere_cita` (sí/no). Cubrir:
  dato puntual, predicción, normativa, recomendación de salud, preguntas que encadenan 2–3 herramientas,
  estación inexistente (manejo de error), pregunta fuera de alcance, intento de pedir consejo médico.
- `eval/run_eval.py`: corre todas, guarda trazas y produce `eval/results.md` con tabla:
  pregunta · herramientas llamadas · ¿correcta? · ¿citó fuente? · observación. Más un resumen (% aciertos).
- Correrla por comando, no a mano.

---

## 9. Estructura del repositorio

```
calidad-aire-aburra/
├── CLAUDE.md
├── README.md                 # instalación, datos, cómo correr app/eval, fuentes, limitaciones
├── requirements.txt          # versiones fijadas
├── .env.example              # GROQ_API_KEY=, LLM_BASE_URL=, LLM_MODEL=
├── Makefile                  # data, train, index, eval, app, test (opcional pero útil)
├── app.py
├── src/
│   ├── config.py             # rutas, constantes ICA, T_MAX, lectura de .env
│   ├── data/                 # download.py, clean.py, ica.py, features.py
│   ├── models/               # naive_bayes.py, hmm.py, baselines.py, evaluate.py
│   ├── rag/                  # ingest.py, retriever.py
│   └── agent/                # llm.py, tools.py, loop.py, prompts.py
├── notebooks/                # 01_eda, 02_parte1_modelos, 03_eval_agente (consumen src/, no duplican lógica)
├── data/{raw,processed,corpus,index}/
├── models/                   # modelos entrenados serializados
├── results/                  # métricas y figuras
├── eval/                     # questions.yaml, run_eval.py, results.md
├── tests/                    # pytest
└── docs/                     # informe.md (≤ 4 págs.), declaracion_ia.md
```

---

## 10. Cómo trabajar en este repo

- **Antes de codificar una fase, propón un plan corto y espera confirmación.** Luego implementa en pasos pequeños.
- Después de cada módulo: correr `pytest` y un *smoke test* (`python -m src.<modulo>` o `streamlit run app.py` en headless).
- **Explica cada módulo nuevo** en 3–5 líneas al terminarlo (qué hace, decisiones, cómo probarlo). El estudiante
  debe entender y poder defender todo el código; la falta de comprensión es falta a la integridad académica.
  Para los modelos de la Parte 1, comenta las ecuaciones en el código.
- Funciones pequeñas, type hints, docstrings en español. Nombres de herramientas exactamente como en la sección 6.
- Nada de rutas absolutas; todo relativo a la raíz vía `src/config.py`. Semillas fijas.
- **No** añadir dependencias sin justificarlo; actualizar `requirements.txt` en el mismo cambio.
- **No** commitear `.env`, claves ni datos individuales sensibles.
- Commits pequeños y descriptivos en español (`feat: ...`, `fix: ...`, `docs: ...`).
- Al final de cada sesión, agregar una entrada a `docs/declaracion_ia.md`: fecha, herramienta (Claude Code),
  qué se pidió, qué se generó, qué revisó/modificó el estudiante.
- Si algo del plan no es viable a tiempo, **dilo y propone el recorte**, no lo escondas.

---

## 11. Cronograma (hoy es la referencia; ajusta según avance)

| Fecha | Hito |
|---|---|
| Vie 6 – Sáb 7 nov | Acceso a datos verificado (plan B listo), datos limpios, EDA, NB + líneas base con métricas |
| Dom 8 nov | Evaluación Parte 1 cerrada; HMM si hay tiempo; esqueleto de app |
| **Lun 9 nov** | **Checkpoint**: agente mínimo con 1 herramienta + dashboard esqueleto |
| Lun 9 – Mié 11 nov | RAG, 3 herramientas, visualizaciones, interacción, evaluación del agente |
| Jue 12 nov | **Congelar código.** README probado en entorno limpio, informe, declaración de IA, video plan B |
| Vie 13 nov, 6:00 p. m. | Entrega en GitHub + correo con asunto `si3003-AI-grupo-nombre` (cada integrante) |

## 12. Definición de "terminado"

- [ ] Clonar → `pip install -r requirements.txt` → `streamlit run app.py` funciona en un entorno limpio, siguiendo solo el README.
- [ ] La app funciona sin internet para datos y sin API key (chat desactivado con aviso).
- [ ] NB (y HMM si existe) implementados a mano, con tests y métricas vs. persistencia y climatología.
- [ ] Agente con `T_MAX`, ≥ 3 herramientas, errores como observación, citas.
- [ ] `eval/results.md` con 12–15 preguntas.
- [ ] README con fuentes citadas, limitaciones y aviso de no consejo médico.
- [ ] `docs/informe.md` ≤ 4 páginas y `docs/declaracion_ia.md` completos.
