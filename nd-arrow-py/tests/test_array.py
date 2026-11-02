import numpy as np
import pytest

import nd_arrow as nda


def make():
    return nda.NdArray.from_numpy(
        np.arange(6.0).reshape(2, 3), dims=["time", "lat"], attrs={"units": "K"}
    )


def test_from_numpy_metadata():
    a = make()
    assert a.dims == ("time", "lat")
    assert a.shape == (2, 3)
    assert a.ndim == 2
    assert a.sizes == {"time": 2, "lat": 3}
    assert a.dtype == "double"
    assert a.attrs == {"units": "K"}
    np.testing.assert_array_equal(a.to_numpy(), np.arange(6.0).reshape(2, 3))


def test_scalar_from_zero_d():
    a = nda.NdArray.from_numpy(np.array(273.15), dims=[])
    assert a.dims == () and a.shape == ()
    assert a.to_numpy() == 273.15


def test_dims_length_mismatch_raises():
    with pytest.raises(ValueError):
        nda.NdArray.from_numpy(np.zeros((2, 3)), dims=["x"])


def test_duplicate_dim_raises():
    with pytest.raises(ValueError):
        nda.NdArray.from_numpy(np.zeros((2, 3)), dims=["x", "x"])


def test_attrs_getter_returns_copy():
    a = make()
    a.attrs["x"] = 1
    assert a.attrs == {"units": "K"}


def test_attrs_setter_replaces():
    a = make()
    a.attrs = {"long_name": "temperature"}
    assert a.attrs == {"long_name": "temperature"}


def test_set_and_del_attr():
    a = make()
    a.set_attr("scale_factor", 0.5)
    assert a.attrs == {"units": "K", "scale_factor": 0.5}
    a.del_attr("units")
    assert a.attrs == {"scale_factor": 0.5}
    with pytest.raises(KeyError):
        a.del_attr("missing")


def test_with_attrs_keeps_original():
    a = make()
    b = a.with_attrs({"units": "degC"})
    assert a.attrs == {"units": "K"}
    assert b.attrs == {"units": "degC"}


def test_reshape():
    a = make()
    b = a.reshape({"x": 6})
    assert b.shape == (6,)
    assert b.attrs == {"units": "K"}
    with pytest.raises(ValueError):
        a.reshape([("x", 4)])


def test_eq_and_repr():
    assert make() == make()
    assert make() != make().with_attrs({})
    assert "time" in repr(make())
