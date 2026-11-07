"""HMM de 3 estados con emisiones gaussianas sobre x_t = log(PM2.5 diario), en NumPy (Parte 1).

Modelo
------
    z_1 ~ Cat(π),   z_t | z_{t-1} ~ Cat(A[z_{t-1}, :]),   x_t | z_t = s ~ N(μ_s, σ_s²)

Días faltantes (x_t = NaN) se tratan como observación ausente: b_s(x_t) = 1 para todo s,
así el filtro solo propaga la dinámica (α_t ∝ A^T α_{t-1}) sin romper la secuencia.

Forward escalado (filtrado). Con b_t(s) = N(x_t; μ_s, σ_s²):
    α̃_1(s) = π_s b_1(s);              c_1 = Σ_s α̃_1(s);   α_1 = α̃_1 / c_1
    α̃_t(s) = b_t(s) Σ_r α_{t-1}(r) A[r,s];  c_t = Σ_s α̃_t(s); α_t = α̃_t / c_t
  α_t(s) = P(z_t = s | x_{1:t})   y   log P(x_{1:T}) = Σ_t log c_t.
  Sin escalar, α_t es un producto de T densidades y hace underflow para T ~ cientos.

Backward escalado:  β_T = 1;  β_t(r) = Σ_s A[r,s] b_{t+1}(s) β_{t+1}(s) / c_{t+1}
Posteriores:  γ_t(s) = α_t(s) β_t(s);   ξ_t(r,s) = α_t(r) A[r,s] b_{t+1}(s) β_{t+1}(s) / c_{t+1}

Baum-Welch (EM), sumando sobre todas las secuencias (una por estación):
    π_s = mean γ_1(s);   A[r,s] = Σ_t ξ_t(r,s) / Σ_t γ_t(r)   (t = 1..T-1)
    μ_s = Σ_t γ_t(s) x_t / Σ_t γ_t(s);   σ_s² = Σ_t γ_t(s)(x_t − μ_s)² / Σ_t γ_t(s)   (solo x_t observados)

Predicción a un paso:  P(z_{t+1} | x_{1:t}) = A^T α_t.
Para la clase de mañana se integra la emisión sobre cada rango de PM2.5:
    P(clase_{t+1} = c) = Σ_s P(z_{t+1}=s | x_{1:t}) · P(log lo_c ≤ x ≤ log hi_c | s).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

LOG_2PI = math.log(2 * math.pi)
_erf = np.vectorize(math.erf)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(np.asarray(z) / math.sqrt(2.0)))


class GaussianHMM:
    """HMM con emisiones gaussianas univariadas; estados ordenados por media creciente."""

    def __init__(self, n_states: int = 3, n_iter: int = 100, tol: float = 1e-4,
                 min_var: float = 1e-3, seed: int = 42) -> None:
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.min_var = min_var
        self.seed = seed
        self.pi: np.ndarray | None = None
        self.A: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.loglik_history_: list[float] = []

    # --- emisiones --------------------------------------------------------------
    def _emission(self, x: np.ndarray) -> np.ndarray:
        """b_t(s) = N(x_t; μ_s, σ_s²), forma (T, K); filas NaN -> 1 (observación ausente)."""
        x = np.asarray(x, dtype=float)[:, None]
        logb = -0.5 * (LOG_2PI + np.log(self.var) + (x - self.mu) ** 2 / self.var)
        b = np.exp(logb)
        b[np.isnan(x[:, 0])] = 1.0
        return np.maximum(b, 1e-300)

    # --- forward / backward ----------------------------------------------------
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (α normalizado (T, K), c (T,)). log P(x) = Σ log c."""
        b = self._emission(x)
        T = len(b)
        alpha = np.zeros((T, self.K))
        c = np.zeros(T)
        a = self.pi * b[0]
        for t in range(T):
            if t > 0:
                a = (alpha[t - 1] @ self.A) * b[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        return alpha, c

    def backward(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        b = self._emission(x)
        T = len(b)
        beta = np.ones((T, self.K))
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (b[t + 1] * beta[t + 1])) / c[t + 1]
        return beta

    def score(self, sequences: list[np.ndarray]) -> float:
        """Log-verosimilitud total Σ_secuencias log P(x)."""
        return float(sum(np.log(self.forward(x)[1]).sum() for x in sequences))

    # --- entrenamiento ---------------------------------------------------------
    def _init_params(self, sequences: list[np.ndarray]) -> None:
        allx = np.concatenate([x[~np.isnan(x)] for x in sequences])
        qs = np.linspace(0, 1, self.K + 2)[1:-1]                 # cuantiles equiespaciados
        self.mu = np.quantile(allx, qs)
        self.var = np.full(self.K, max(allx.var() / self.K, self.min_var))
        self.pi = np.full(self.K, 1.0 / self.K)
        self.A = np.full((self.K, self.K), 0.1 / (self.K - 1))
        np.fill_diagonal(self.A, 0.9)                             # estados persistentes

    def fit(self, sequences: list[np.ndarray]) -> "GaussianHMM":
        """Baum-Welch sobre varias secuencias (NaN = día sin dato)."""
        sequences = [np.asarray(s, dtype=float) for s in sequences if np.isfinite(s).any()]
        self._init_params(sequences)
        self.loglik_history_ = []
        for _ in range(self.n_iter):
            pi_acc = np.zeros(self.K)
            A_num = np.zeros((self.K, self.K))
            A_den = np.zeros(self.K)
            g_sum = np.zeros(self.K)
            gx_sum = np.zeros(self.K)
            gxx_sum = np.zeros(self.K)
            ll = 0.0
            for x in sequences:
                alpha, c = self.forward(x)
                beta = self.backward(x, c)
                b = self._emission(x)
                ll += np.log(c).sum()
                gamma = alpha * beta                                       # γ_t(s)
                gamma /= gamma.sum(axis=1, keepdims=True)
                # ξ_t(r,s) sumado sobre t
                xi = (alpha[:-1, :, None] * self.A[None] *
                      (b[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
                pi_acc += gamma[0]
                A_num += xi.sum(axis=0)
                A_den += gamma[:-1].sum(axis=0)
                obs = ~np.isnan(x)
                go, xo = gamma[obs], x[obs]
                g_sum += go.sum(axis=0)
                gx_sum += (go * xo[:, None]).sum(axis=0)
                gxx_sum += (go * xo[:, None] ** 2).sum(axis=0)
            self.loglik_history_.append(float(ll))
            # M-step
            self.pi = pi_acc / pi_acc.sum()
            self.A = A_num / A_den[:, None]
            self.A /= self.A.sum(axis=1, keepdims=True)
            self.mu = gx_sum / g_sum
            self.var = np.maximum(gxx_sum / g_sum - self.mu ** 2, self.min_var)
            if len(self.loglik_history_) > 1 and \
                    abs(self.loglik_history_[-1] - self.loglik_history_[-2]) < self.tol:
                break
        self._sort_states()
        return self

    def _sort_states(self) -> None:
        """Reordena estados por μ creciente (0 = nivel bajo, 1 = medio, 2 = alto)."""
        o = np.argsort(self.mu)
        self.mu, self.var, self.pi = self.mu[o], self.var[o], self.pi[o]
        self.A = self.A[np.ix_(o, o)]

    # --- inferencia ------------------------------------------------------------
    def viterbi(self, x: np.ndarray) -> np.ndarray:
        """Secuencia de estados más probable (en log, sin escalado necesario)."""
        logb = np.log(self._emission(x))
        logA = np.log(self.A)
        T = len(logb)
        delta = np.log(self.pi) + logb[0]
        psi = np.zeros((T, self.K), dtype=int)
        for t in range(1, T):
            cand = delta[:, None] + logA                 # (r, s)
            psi[t] = np.argmax(cand, axis=0)
            delta = cand[psi[t], np.arange(self.K)] + logb[t]
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta))
        for t in range(T - 1, 0, -1):
            path[t - 1] = psi[t, path[t]]
        return path

    def predict_next_state(self, x: np.ndarray) -> np.ndarray:
        """P(z_{t+1} | x_{1:t}) = A^T α_t para cada t, forma (T, K)."""
        alpha, _ = self.forward(x)
        return alpha @ self.A

    def class_probs_from_states(self, state_probs: np.ndarray,
                                edges: list[float]) -> np.ndarray:
        """P(clase) = Σ_s P(s) P(x en [log lo_c, log hi_c] | s); edges en µg/m³, p.ej. [37.5, 55.5]."""
        cuts = np.concatenate([[-np.inf], np.log(edges), [np.inf]])
        sd = np.sqrt(self.var)
        cdf = _norm_cdf((cuts[:, None] - self.mu[None]) / sd[None])   # (n_cuts, K)
        p_class_given_state = np.diff(cdf, axis=0).T                     # (K, C)
        return state_probs @ p_class_given_state

    # --- persistencia ----------------------------------------------------------
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        payload = {"pi": self.pi.tolist(), "A": self.A.tolist(), "mu": self.mu.tolist(),
                   "var": self.var.tolist(), "loglik_history": self.loglik_history_,
                   "extra": extra or {}}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> tuple["GaussianHMM", dict]:
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(n_states=len(p["pi"]))
        m.pi, m.A = np.array(p["pi"]), np.array(p["A"])
        m.mu, m.var = np.array(p["mu"]), np.array(p["var"])
        m.loglik_history_ = p["loglik_history"]
        return m, p.get("extra", {})
