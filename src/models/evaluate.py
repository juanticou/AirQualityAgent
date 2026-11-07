"""Métricas y figuras de evaluación de la Parte 1 (scikit-learn solo para métricas)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             log_loss, recall_score)

from src.config import CLASSES  # noqa: E402

LABELS = list(range(len(CLASSES)))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           proba: np.ndarray | None = None) -> dict:
    """accuracy, F1 macro, recall por clase, recall de episodios (alerta o crítico), matriz de confusión."""
    episode_true = y_true >= 1
    episode_pred = y_pred >= 1
    rec = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    # F1 macro sobre las clases con soporte en y_true (una clase ausente no se puede evaluar)
    present = sorted(set(np.asarray(y_true).tolist()))
    out = {
        "n": int(len(y_true)),
        "soporte": {c: int((y_true == i).sum()) for i, c in enumerate(CLASSES)},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "f1_macro_clases": [CLASSES[i] for i in present],
        "recall_por_clase": {c: float(r) for c, r in zip(CLASSES, rec)},
        "recall_alerta_o_critico": float(episode_pred[episode_true].mean()) if episode_true.any() else None,
        "precision_alerta_o_critico": (float(episode_true[episode_pred].mean())
                                       if episode_pred.any() else None),
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }
    if proba is not None:
        p = np.clip(proba, 1e-12, 1)
        out["log_loss"] = float(log_loss(y_true, p / p.sum(1, keepdims=True), labels=LABELS))
        onehot = np.eye(len(CLASSES))[y_true]
        out["brier"] = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    return out


def plot_confusion_matrices(results: dict[str, dict], path: Path, title: str) -> None:
    """Una matriz de confusión (normalizada por fila) por modelo."""
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.6))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        cm = np.array(results[name]["matriz_confusion"], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.0%}", ha="center", va="center",
                        fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(LABELS, CLASSES, fontsize=8)
        ax.set_yticks(LABELS, CLASSES, fontsize=8)
        ax.set_xlabel("predicho")
        ax.set_title(f"{name}\nF1m={results[name]['f1_macro']:.2f}", fontsize=9)
    axes[0].set_ylabel("real")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_metric_bars(results: dict[str, dict], path: Path, title: str) -> None:
    names = list(results)
    metrics = ["accuracy", "f1_macro", "recall_alerta_o_critico"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, m in enumerate(metrics):
        vals = [results[n][m] or 0 for n in names]
        ax.bar(x + (i - 1) * 0.27, vals, width=0.27, label=m)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_loglik(history: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(history, marker=".")
    ax.set_xlabel("iteración Baum-Welch")
    ax.set_ylabel("log-verosimilitud")
    ax.set_title("HMM: convergencia de EM", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
