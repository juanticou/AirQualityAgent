"""Líneas base para la clase de mañana.

- Persistencia: clase(t+1) = clase(t).
- Climatología: clase más frecuente del mes de t+1 en entrenamiento (y sus frecuencias como probabilidades).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CLASSES
from src.data.ica import CLASS_TO_IDX


def persistence(pairs: pd.DataFrame) -> np.ndarray:
    return pairs["clase"].map(CLASS_TO_IDX).to_numpy(dtype=int)


class Climatology:
    """Frecuencias de clase por mes (con suavizado de Laplace) aprendidas en entrenamiento."""

    def fit(self, pairs: pd.DataFrame, alpha: float = 1.0) -> "Climatology":
        m = pairs["fecha_manana"].dt.month
        y = pairs["clase_manana"].map(CLASS_TO_IDX)
        counts = np.zeros((12, len(CLASSES)))
        np.add.at(counts, (m.to_numpy() - 1, y.to_numpy(dtype=int)), 1.0)
        self.probs_ = (counts + alpha) / (counts + alpha).sum(axis=1, keepdims=True)
        return self

    def predict_proba(self, pairs: pd.DataFrame) -> np.ndarray:
        return self.probs_[pairs["fecha_manana"].dt.month.to_numpy() - 1]

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(pairs), axis=1)
