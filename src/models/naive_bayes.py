"""Naive Bayes categórico implementado en NumPy (Parte 1).

Modelo
------
Para una clase y in {0..K-1} y features categóricas x = (x_1, ..., x_d), Naive Bayes supone
independencia condicional de las features dada la clase:

    P(y = k | x)  ∝  P(y = k) · Π_j P(x_j | y = k)

Estimación con suavizado de Laplace (aditivo, parámetro α > 0):

    P(y = k)          = N_k / N                                   (o uniforme si prior="uniform")
    P(x_j = v | y = k) = (N_{k,j,v} + α) / (N_k + α · V_j)

donde N_k es el número de ejemplos de la clase k, N_{k,j,v} cuántos de ellos tienen x_j = v
y V_j el número de categorías de la feature j. Así ninguna probabilidad es 0.

Para evitar underflow se trabaja en log-probabilidades:

    log P(y = k | x) = log P(y = k) + Σ_j log P(x_j | y = k) − log Z
    log Z = logsumexp_k( log P(y = k) + Σ_j log P(x_j | y = k) )

Una categoría no vista en entrenamiento (v >= V_j) no aporta evidencia: se omite ese término.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    """log(Σ exp(a)) estable: se resta el máximo antes de exponenciar."""
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


class CategoricalNaiveBayes:
    """Naive Bayes para features enteras 0..V_j-1, con suavizado de Laplace."""

    def __init__(self, alpha: float = 1.0, prior: str = "empirical", prior_alpha: float = 0.0) -> None:
        """prior_alpha > 0 suaviza el prior empírico: P(y=k) = (N_k + prior_alpha) / (N + K·prior_alpha).
        Con 0 (por defecto, igual que sklearn) una clase sin ejemplos tendría probabilidad 0 para siempre."""
        if alpha <= 0:
            raise ValueError("alpha debe ser > 0")
        if prior not in {"empirical", "uniform"}:
            raise ValueError("prior debe ser 'empirical' o 'uniform'")
        self.alpha = alpha
        self.prior = prior
        self.prior_alpha = prior_alpha
        self.class_log_prior_: np.ndarray | None = None
        self.feature_log_prob_: list[np.ndarray] = []  # por feature: (K, V_j)
        self.n_categories_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, n_categories: list[int] | None = None,
            n_classes: int | None = None) -> "CategoricalNaiveBayes":
        """Cuenta frecuencias y calcula log P(y) y log P(x_j | y)."""
        X = np.asarray(X, dtype=int)
        y = np.asarray(y, dtype=int)
        K = int(n_classes if n_classes is not None else y.max() + 1)
        V = np.asarray(n_categories if n_categories is not None else X.max(axis=0) + 1, dtype=int)
        self.classes_ = np.arange(K)
        self.n_categories_ = V

        class_count = np.bincount(y, minlength=K).astype(float)          # N_k
        if self.prior == "empirical":
            smoothed = class_count + self.prior_alpha
            with np.errstate(divide="ignore"):   # log(0) = -inf es correcto si prior_alpha = 0
                self.class_log_prior_ = np.log(smoothed) - np.log(smoothed.sum())
        else:
            self.class_log_prior_ = np.full(K, -np.log(K))

        self.feature_log_prob_ = []
        for j, Vj in enumerate(V):
            counts = np.zeros((K, Vj))                                      # N_{k,j,v}
            np.add.at(counts, (y, X[:, j]), 1.0)
            num = counts + self.alpha
            den = class_count[:, None] + self.alpha * Vj
            self.feature_log_prob_.append(np.log(num) - np.log(den))
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """log P(y=k) + Σ_j log P(x_j | y=k), forma (n, K)."""
        if self.class_log_prior_ is None:
            raise RuntimeError("El modelo no está entrenado; llama fit() primero.")
        X = np.asarray(X, dtype=int)
        jll = np.tile(self.class_log_prior_, (X.shape[0], 1))
        for j, flp in enumerate(self.feature_log_prob_):
            xj = X[:, j]
            known = (xj >= 0) & (xj < flp.shape[1])                          # categorías vistas
            jll[known] += flp[:, xj[known]].T
        return jll

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        return jll - _logsumexp(jll, axis=1)[:, None]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self._joint_log_likelihood(X), axis=1)

    @staticmethod
    def decide(proba: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Regla de decisión con umbral para episodios (clases 1..K-1 = alerta, crítico).

        Si P(alerta) + P(crítico) >= threshold se predice la más probable entre alerta y crítico;
        si no, normal. Con threshold=None equivale a argmax. Bajar el umbral cambia precisión por
        recall de episodios, que es el error más costoso para un ciudadano.
        """
        if threshold is None:
            return np.argmax(proba, axis=1)
        episode = proba[:, 1:].sum(axis=1) >= threshold
        return np.where(episode, 1 + np.argmax(proba[:, 1:], axis=1), 0)

    # --- persistencia (JSON legible, sin pickle) -------------------------------
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        payload = {
            "alpha": self.alpha, "prior": self.prior, "prior_alpha": self.prior_alpha,
            "class_log_prior": self.class_log_prior_.tolist(),
            "feature_log_prob": [f.tolist() for f in self.feature_log_prob_],
            "n_categories": self.n_categories_.tolist(),
            "extra": extra or {},
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> tuple["CategoricalNaiveBayes", dict]:
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(alpha=p["alpha"], prior=p["prior"], prior_alpha=p.get("prior_alpha", 0.0))
        m.class_log_prior_ = np.array(p["class_log_prior"])
        m.feature_log_prob_ = [np.array(f) for f in p["feature_log_prob"]]
        m.n_categories_ = np.array(p["n_categories"])
        m.classes_ = np.arange(len(m.class_log_prior_))
        return m, p.get("extra", {})
