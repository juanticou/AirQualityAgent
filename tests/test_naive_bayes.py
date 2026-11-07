import numpy as np
import pytest
from sklearn.naive_bayes import CategoricalNB

from src.models.naive_bayes import CategoricalNaiveBayes


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    n, V = 600, [3, 5, 2, 7]
    y = rng.integers(0, 3, n)
    # features dependientes de la clase para que el problema no sea trivial
    X = np.column_stack([(rng.integers(0, v, n) + y) % v for v in V])
    return X, y, V


@pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0])
def test_predict_proba_igual_a_sklearn(synthetic, alpha):
    X, y, V = synthetic
    ours = CategoricalNaiveBayes(alpha=alpha).fit(X[:500], y[:500], n_categories=V)
    ref = CategoricalNB(alpha=alpha, min_categories=V).fit(X[:500], y[:500])
    np.testing.assert_allclose(ours.predict_proba(X[500:]), ref.predict_proba(X[500:]), atol=1e-6)
    np.testing.assert_array_equal(ours.predict(X[500:]), ref.predict(X[500:]))


def test_prior_uniforme_y_probabilidades_suman_uno(synthetic):
    X, y, V = synthetic
    m = CategoricalNaiveBayes(prior="uniform").fit(X, y, n_categories=V)
    np.testing.assert_allclose(np.exp(m.class_log_prior_), 1 / 3)
    np.testing.assert_allclose(m.predict_proba(X).sum(axis=1), 1.0)


def test_categoria_no_vista_se_ignora(synthetic):
    X, y, V = synthetic
    m = CategoricalNaiveBayes().fit(X, y, n_categories=V)
    x = X[:1].copy()
    x_unseen = x.copy()
    x_unseen[0, 0] = 99
    # con la feature 0 desconocida, el resultado equivale a no usarla
    m0 = CategoricalNaiveBayes().fit(X[:, 1:], y, n_categories=V[1:])
    np.testing.assert_allclose(m.predict_proba(x_unseen), m0.predict_proba(x[:, 1:]), atol=1e-12)


def test_guardar_y_cargar(tmp_path, synthetic):
    X, y, V = synthetic
    m = CategoricalNaiveBayes(alpha=0.7).fit(X, y, n_categories=V)
    m.save(tmp_path / "nb.json", extra={"features": ["a", "b", "c", "d"]})
    m2, extra = CategoricalNaiveBayes.load(tmp_path / "nb.json")
    np.testing.assert_allclose(m.predict_proba(X), m2.predict_proba(X))
    assert extra["features"][0] == "a"
