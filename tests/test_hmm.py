import numpy as np
import pytest

from src.models.hmm import GaussianHMM


def simulate(T: int, seed: int):
    rng = np.random.default_rng(seed)
    pi = np.array([0.6, 0.3, 0.1])
    A = np.array([[0.90, 0.08, 0.02], [0.15, 0.75, 0.10], [0.05, 0.25, 0.70]])
    mu, sd = np.array([2.5, 3.5, 4.3]), np.array([0.25, 0.2, 0.2])
    z = np.zeros(T, dtype=int)
    z[0] = rng.choice(3, p=pi)
    for t in range(1, T):
        z[t] = rng.choice(3, p=A[z[t - 1]])
    x = rng.normal(mu[z], sd[z])
    return x, z, A, mu, sd


@pytest.fixture(scope="module")
def fitted():
    seqs, states = [], []
    for s in range(4):
        x, z, A, mu, sd = simulate(600, seed=s)
        x[rng_idx(s)] = np.nan  # algunos faltantes
        seqs.append(x)
        states.append(z)
    m = GaussianHMM(n_iter=200, tol=1e-6).fit(seqs)
    return m, seqs, states, A, mu, sd


def rng_idx(s):
    return np.random.default_rng(100 + s).choice(600, 30, replace=False)


def test_loglik_no_decreciente(fitted):
    m = fitted[0]
    ll = np.array(m.loglik_history_)
    assert np.all(np.diff(ll) >= -1e-6), "Baum-Welch debe ser monótono"


def test_recupera_parametros(fitted):
    m, _, _, A, mu, sd = fitted
    np.testing.assert_allclose(m.mu, mu, atol=0.1)
    np.testing.assert_allclose(np.sqrt(m.var), sd, atol=0.06)
    np.testing.assert_allclose(m.A, A, atol=0.08)


def test_viterbi_y_prediccion(fitted):
    m, seqs, states, *_ = fitted
    path = m.viterbi(seqs[0])
    assert (path == states[0]).mean() > 0.85
    p = m.predict_next_state(seqs[0])
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    pc = m.class_probs_from_states(p, [np.exp(3.0), np.exp(4.0)])
    np.testing.assert_allclose(pc.sum(axis=1), 1.0, atol=1e-9)


def test_forward_estable_en_secuencia_larga():
    x, *_ = simulate(5000, seed=9)
    m = GaussianHMM(n_iter=5).fit([x])
    alpha, c = m.forward(x)
    assert np.isfinite(np.log(c).sum())
    np.testing.assert_allclose(alpha.sum(axis=1), 1.0)
