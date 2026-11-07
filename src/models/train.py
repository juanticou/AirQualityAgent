"""Entrena y evalúa la Parte 1: Naive Bayes, HMM y líneas base, con partición temporal.

Protocolo
---------
1. Partición por año del día objetivo (t+1): entrenamiento 2020-2023, prueba 2024-2025.
   (2025 solo tiene 6 días de alerta; 2024 incluye el episodio de feb-abr, por eso la prueba
   abarca ambos años.)
2. Selección de hiperparámetros del NB (α, prior) con validación temporal de ventana expandible
   dentro de entrenamiento: se valida en 2021, 2022 y 2023 entrenando con los años anteriores.
   Criterio: F1 macro de las predicciones agregadas de los 3 pliegues.
3. Re-entrenamiento con 2020-2023 y evaluación única en prueba.
4. HMM: Baum-Welch con las series de log(PM2.5) hasta 2023-12-31; la predicción para t+1
   usa solo el filtrado hasta t (sin mirar el futuro).
Salidas: models/nb.json, models/hmm.json, results/metrics_part1.json, results/figures/*.png

Uso: python -m src.models.train
"""
from __future__ import annotations

import json
from itertools import product

import numpy as np
import pandas as pd

from src.config import CLASSES, FIGURES_DIR, METRICS_PART1_JSON, MODELS_DIR, SEED
from src.data.features import FEATURES, Discretizer, encode_target, load_daily, make_pairs
from src.models.baselines import Climatology, persistence
from src.models.evaluate import (classification_metrics, plot_confusion_matrices, plot_loglik,
                                 plot_metric_bars)
from src.models.hmm import GaussianHMM
from src.models.naive_bayes import CategoricalNaiveBayes

TRAIN_END = 2023
CV_YEARS = [2021, 2022, 2023]
ALPHAS = [0.5, 1.0, 2.0, 5.0]
PRIORS = ["empirical", "uniform"]
THRESHOLDS = [None, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]  # None = argmax
PRIOR_ALPHA = 1.0  # suavizado del prior: en 2020-2023 no hay días "critico"
HMM_STATES = ["bajo", "medio", "alto"]  # estados ordenados por media; NO son las clases del proyecto
CLASS_EDGES = [37.5, 55.5]  # µg/m³: normal | alerta | crítico (tras redondeo)


def station_series(daily: pd.DataFrame, end: pd.Timestamp | None = None) -> dict[str, pd.Series]:
    """Serie diaria continua (calendario completo, NaN = sin dato) de log(PM2.5) por estación."""
    out = {}
    for sid, g in daily.groupby("estacion_id"):
        s = g.set_index("fecha")["pm25"]
        idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        s = np.log(s.reindex(idx).clip(lower=1.0))
        out[sid] = s if end is None else s[s.index <= end]
    return out


def fit_hmm(daily: pd.DataFrame, end: pd.Timestamp) -> GaussianHMM:
    seqs = [s.to_numpy() for s in station_series(daily, end).values() if s.notna().sum() > 30]
    return GaussianHMM(n_states=3, n_iter=150, tol=1e-3, seed=SEED).fit(seqs)


def hmm_predictions(model: GaussianHMM, daily: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """P(clase_{t+1} | x_{1:t}) para cada fila de pairs, filtrando la serie completa de su estación."""
    rows = []
    for sid, s in station_series(daily).items():
        state_next = model.predict_next_state(s.to_numpy())
        pc = model.class_probs_from_states(state_next, CLASS_EDGES)
        rows.append(pd.DataFrame(pc, columns=CLASSES).assign(estacion_id=sid, fecha=s.index))
    table = pd.concat(rows)
    merged = pairs[["estacion_id", "fecha"]].merge(table, on=["estacion_id", "fecha"], how="left")
    return merged[CLASSES].to_numpy()


def fit_nb(pairs: pd.DataFrame, alpha: float, prior: str) -> tuple[CategoricalNaiveBayes, Discretizer]:
    disc = Discretizer().fit(pairs)
    X, y = disc.transform(pairs), encode_target(pairs["clase_manana"])
    nb = CategoricalNaiveBayes(alpha=alpha, prior=prior, prior_alpha=PRIOR_ALPHA).fit(
        X, y, disc.n_categories, len(CLASSES))
    return nb, disc


def select_nb(train: pd.DataFrame) -> tuple[list[dict], dict]:
    """Validación temporal de ventana expandible: entrena con años < v, valida en el año v."""
    year = train["fecha_manana"].dt.year
    grid = []
    for a, pr in product(ALPHAS, PRIORS):
        ys, ps = [], []
        for v in CV_YEARS:
            nb, disc = fit_nb(train[year < v], a, pr)
            fold = train[year == v]
            ys.append(encode_target(fold["clase_manana"]))
            ps.append(nb.predict_proba(disc.transform(fold)))
        y, p = np.concatenate(ys), np.concatenate(ps)
        for thr in THRESHOLDS:
            met = classification_metrics(y, CategoricalNaiveBayes.decide(p, thr), p)
            grid.append({"alpha": a, "prior": pr, "umbral": thr, "f1_macro": round(met["f1_macro"], 4),
                         "accuracy": round(met["accuracy"], 4), "log_loss": round(met["log_loss"], 4),
                         "recall_alerta_o_critico": met["recall_alerta_o_critico"]})
    best = max(grid, key=lambda g: (g["f1_macro"], -g["log_loss"]))
    return grid, best


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    daily = load_daily()
    pairs = make_pairs(daily)
    year = pairs["fecha_manana"].dt.year
    train, test = pairs[year <= TRAIN_END], pairs[year > TRAIN_END]
    print(f"train={len(train)}  test={len(test)}")

    # 1) selección del NB
    grid, best = select_nb(train)
    print("NB elegido (CV temporal):", best)

    # 2) entrenamiento final y prueba
    nb, disc = fit_nb(train, best["alpha"], best["prior"])
    clim = Climatology().fit(train)
    hmm = fit_hmm(daily, pd.Timestamp(f"{TRAIN_END}-12-31"))
    yte = encode_target(test["clase_manana"])
    p_nb = nb.predict_proba(disc.transform(test))
    thr = best["umbral"]
    p_hmm = hmm_predictions(hmm, daily, test)
    test_res = {
        "persistencia": classification_metrics(yte, persistence(test)),
        "climatologia": classification_metrics(yte, clim.predict(test), clim.predict_proba(test)),
        "naive_bayes": classification_metrics(yte, CategoricalNaiveBayes.decide(p_nb, thr), p_nb),
        "naive_bayes_argmax": classification_metrics(yte, p_nb.argmax(1), p_nb),
        "hmm": classification_metrics(yte, p_hmm.argmax(1), p_hmm),
    }

    # 3) guardar
    keys = ("accuracy", "f1_macro", "recall_alerta_o_critico", "precision_alerta_o_critico")
    summary = {k: {m: round(v[m], 4) for m in keys if v.get(m) is not None} for k, v in test_res.items()}
    nb.save(MODELS_DIR / "nb.json", extra={
        "features": FEATURES, "discretizer": disc.to_dict(), "classes": CLASSES, "umbral": thr,
        "entrenado_con": f"{train['fecha_manana'].min().date()} a {TRAIN_END}-12-31",
        "metricas_test": {"periodo": "2024-2025", **summary["naive_bayes"],
                          "persistencia_f1_macro": summary["persistencia"]["f1_macro"]}})
    hmm.save(MODELS_DIR / "hmm.json", extra={
        "estados": HMM_STATES, "pm25_mediana_estado": np.exp(hmm.mu).round(1).tolist(),
        "variable": "log(PM2.5 diario)", "class_edges": CLASS_EDGES,
        "metricas_test": {"periodo": "2024-2025", **summary["hmm"]}})

    split_info = {k: {"n": int(len(v)), "desde": str(v["fecha_manana"].min().date()),
                      "hasta": str(v["fecha_manana"].max().date()),
                      "clases": v["clase_manana"].value_counts().to_dict()}
                  for k, v in {"train": train, "test": test}.items()}
    metrics = {
        "particion": split_info, "cv_years": CV_YEARS,
        "nb_grid_cv": grid, "nb_elegido": best, "test": test_res, "resumen_test": summary,
        "hmm_parametros": {"mu_log": hmm.mu.round(3).tolist(),
                           "pm25_mediana_estado": np.exp(hmm.mu).round(1).tolist(),
                           "sd_log": np.sqrt(hmm.var).round(3).tolist(),
                           "A": hmm.A.round(3).tolist(), "iteraciones": len(hmm.loglik_history_)},
    }
    METRICS_PART1_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    core = {k: test_res[k] for k in ("persistencia", "climatologia", "naive_bayes", "hmm")}
    plot_confusion_matrices(core, FIGURES_DIR / "confusion_test.png", "Prueba 2024-2025 (entrenado 2020-2023)")
    plot_metric_bars(core, FIGURES_DIR / "metricas_test.png", "Prueba 2024-2025")
    plot_loglik(hmm.loglik_history_, FIGURES_DIR / "hmm_loglik.png")

    print("\nPrueba 2024-2025:")
    print(pd.DataFrame(summary).T.to_string())
    print("HMM medianas por estado (µg/m³):", metrics["hmm_parametros"]["pm25_mediana_estado"])


if __name__ == "__main__":
    main()
