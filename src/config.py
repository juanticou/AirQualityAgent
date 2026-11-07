"""Configuración central del proyecto: rutas, constantes ICA, clases y variables de entorno.

Todas las rutas son relativas a la raíz del repositorio (nunca absolutas).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Rutas -------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CORPUS_DIR = DATA_DIR / "corpus"
INDEX_DIR = DATA_DIR / "index"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
EVAL_DIR = ROOT / "eval"

PM25_DAILY_CSV = PROCESSED_DIR / "pm25_diario.csv"
STATIONS_CSV = PROCESSED_DIR / "estaciones.csv"
METRICS_PART1_JSON = RESULTS_DIR / "metrics_part1.json"

# --- Reproducibilidad --------------------------------------------------------
SEED = 42

# --- Categorías ICA para PM2.5 (24 h), Resolución 2254 de 2017 ---------------
# (nombre, límite inferior, límite superior, color). Rangos en µg/m³ enteros:
# el promedio diario se redondea al entero más cercano antes de clasificar.
ICA_CATEGORIES: list[tuple[str, int, int, str]] = [
    ("Buena", 0, 12, "#00e400"),
    ("Aceptable", 13, 37, "#ffff00"),
    ("Dañina a grupos sensibles", 38, 55, "#ff7e00"),
    ("Dañina a la salud", 56, 150, "#ff0000"),
    ("Muy dañina", 151, 250, "#8f3f97"),
    ("Peligrosa", 251, 500, "#7e0023"),
]
PM25_DAILY_LIMIT = 37  # µg/m³, límite máximo permisible diario

# --- Clases del proyecto (simplificación nuestra, NO son niveles oficiales POECA)
CLASSES: list[str] = ["normal", "alerta", "critico"]
CLASS_UPPER: dict[str, float] = {"normal": 37, "alerta": 55, "critico": float("inf")}

# Temporadas críticas en el Valle de Aburrá (transición a lluvias)
CRITICAL_MONTHS = {2, 3, 4, 10, 11}

# --- Calidad de datos --------------------------------------------------------
MIN_VALID_HOURS_FRAC = 0.75  # días con < 75 % de horas válidas se descartan

# --- Agente ------------------------------------------------------------------
T_MAX = 6
LLM_TIMEOUT_S = 60

load_dotenv(ROOT / ".env")
LLM_API_KEY: str | None = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or None
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL: str = os.getenv("LLM_MODEL", "")

# --- RAG ---------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
RAG_MIN_SCORE = 0.40  # umbral de similitud coseno (FAISS); BM25 usa su propio umbral
