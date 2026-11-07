import math

import pandas as pd
import pytest

from src.data.ica import category_series, class_series, pm25_to_class, pm25_to_ica


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "Buena"), (12, "Buena"), (12.49, "Buena"), (12.5, "Aceptable"), (13, "Aceptable"),
        (37, "Aceptable"), (37.4, "Aceptable"), (38, "Dañina a grupos sensibles"),
        (55, "Dañina a grupos sensibles"), (56, "Dañina a la salud"), (150, "Dañina a la salud"),
        (151, "Muy dañina"), (250, "Muy dañina"), (251, "Peligrosa"), (500, "Peligrosa"),
        (800, "Peligrosa"),
    ],
)
def test_bordes_ica(value, expected):
    assert pm25_to_ica(value)["categoria"] == expected


@pytest.mark.parametrize(
    "value,expected",
    [(0, "normal"), (37, "normal"), (37.49, "normal"), (37.5, "alerta"), (38, "alerta"),
     (55, "alerta"), (55.5, "critico"), (56, "critico"), (300, "critico")],
)
def test_bordes_clase(value, expected):
    assert pm25_to_class(value) == expected


def test_nan_y_negativos():
    assert pm25_to_ica(math.nan) is None
    assert pm25_to_class(-1) is None


def test_vectorizado_coincide():
    vals = [0, 12.5, 37, 37.5, 55, 56, 200, float("nan")]
    s = pd.Series(vals)
    cls = class_series(s)
    for v, c in zip(vals, cls):
        expected = pm25_to_class(v)
        assert (c is pd.NA and expected is None) or c == expected
    assert category_series(pd.Series([10]))[0] == "Buena"
