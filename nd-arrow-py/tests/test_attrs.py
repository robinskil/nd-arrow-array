import numpy as np
import pytest

from nd_arrow_array import _core


def test_scalars():
    attrs = {"s": "K", "i": 3, "f": 0.5}
    back = _core._roundtrip_attrs(attrs)
    assert back == attrs
    assert type(back["i"]) is int
    assert type(back["f"]) is float


def test_bool_stays_bool():
    back = _core._roundtrip_attrs({"flag": True})
    assert back["flag"] is True


def test_list_of_str():
    back = _core._roundtrip_attrs({"names": ["a", "b"]})
    assert list(back["names"]) == ["a", "b"]


def test_numeric_list_keeps_dtype():
    back = _core._roundtrip_attrs({"valid_range": np.array([0, 40], dtype="float32")})
    assert isinstance(back["valid_range"], np.ndarray)
    assert back["valid_range"].dtype == np.float32
    np.testing.assert_array_equal(back["valid_range"], [0.0, 40.0])


def test_python_number_list():
    back = _core._roundtrip_attrs({"a": [1, 2], "b": (0.5, 1.5)})
    assert back["a"].dtype == np.int64
    assert back["b"].dtype == np.float64


def test_numpy_scalar_keeps_dtype():
    back = _core._roundtrip_attrs({"x": np.int16(4)})
    assert isinstance(back["x"], np.int16)
    assert back["x"] == 4


def test_order_is_kept():
    back = _core._roundtrip_attrs({"b": 1, "a": 2})
    assert list(back) == ["b", "a"]


def test_unsupported_value_raises():
    with pytest.raises(TypeError):
        _core._roundtrip_attrs({"x": object()})


def test_non_str_key_raises():
    with pytest.raises(TypeError):
        _core._roundtrip_attrs({1: "a"})


def test_null_attribute_value_raises():
    with pytest.raises(ValueError):
        _core._roundtrip_attrs({"d": np.datetime64("NaT", "s")})
