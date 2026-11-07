# Atajos (en Windows sin make, usa los comandos equivalentes del README).
PY ?= python

.PHONY: data train index eval app test

data:      ## descarga crudos y genera data/processed/
	$(PY) -m src.data.download
	$(PY) -m src.data.clean

train:     ## entrena NB + HMM y guarda métricas/figuras
	$(PY) -m src.models.train

index:     ## construye el índice RAG en data/index/
	$(PY) -m src.rag.ingest

eval:      ## evaluación del agente (requiere GROQ_API_KEY)
	$(PY) eval/run_eval.py

app:
	streamlit run app.py

test:
	$(PY) -m pytest -q
