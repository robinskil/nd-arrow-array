import numpy as np
import pytest

import nd_arrow_array as nda

xr = pytest.importorskip("xarray")


def sst():
    return nda.NdArray.from_numpy(
        np.arange(6.0).reshape(2, 3), dims=["time", "lat"], attrs={"units": "K"}
    )


def test_array_to_xarray():
    da = sst().to_xarray(name="sst")
    assert isinstance(da, xr.DataArray)
    assert da.name == "sst"
    assert da.dims == ("time", "lat")
    assert da.attrs == {"units": "K"}
    np.testing.assert_array_equal(da.values, np.arange(6.0).reshape(2, 3))


def test_array_round_trip():
    a = sst()
    assert nda.NdArray.from_xarray(a.to_xarray()) == a


def test_array_from_xarray_drops_coords():
    da = xr.DataArray([1.0, 2.0], dims=["x"], coords={"x": [10, 20]}, attrs={"a": 1})
    a = nda.NdArray.from_xarray(da)
    assert a.dims == ("x",)
    assert a.attrs == {"a": 1}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.ma.MaskedArray([1.0, 2.0], [False, True]), np.array([1.0, np.nan])),
        (np.ma.MaskedArray([1, 2], [True, False]), np.array([np.nan, 2.0])),
        (
            np.ma.MaskedArray(np.array(["2020-01-01", "2020-01-02"], "datetime64[ns]"), [False, True]),
            np.array(["2020-01-01", "NaT"], "datetime64[ns]"),
        ),
    ],
)
def test_nulls_become_missing_values(values, expected):
    da = nda.NdArray.from_numpy(values, dims=["x"]).to_xarray()
    np.testing.assert_array_equal(da.values, expected)


def test_missing_strings_become_nan():
    values = nda.NdArray.from_numpy(np.ma.MaskedArray(["a", "b"], [True, False]), dims=["x"])
    out = values.to_xarray().values
    assert out.dtype == object
    assert np.isnan(out[0]) and out[1] == "b"


def test_object_strings_with_none_become_nulls():
    da = xr.DataArray(np.array(["a", None], dtype=object), dims=["x"])
    a = nda.NdArray.from_xarray(da)
    assert a.to_numpy().mask.tolist() == [False, True]


def test_object_array_of_other_types_raises():
    with pytest.raises(TypeError):
        nda.NdArray.from_xarray(xr.DataArray(np.array([1, "a"], dtype=object), dims=["x"]))


def test_non_str_dimension_raises():
    with pytest.raises(TypeError):
        nda.NdArray.from_xarray(xr.DataArray([1.0], dims=[0]))


def test_from_xarray_rejects_other_types():
    with pytest.raises(TypeError):
        nda.NdArray.from_xarray(np.zeros(2))
    with pytest.raises(TypeError):
        nda.NdRecordBatch.from_xarray(np.zeros(2))


def dataset():
    return xr.Dataset(
        {
            "sst": (("time", "lat", "lon"), np.arange(12.0).reshape(2, 3, 2), {"units": "K"}),
            "flag": (("time",), np.array([1, 0], dtype="int8")),
        },
        coords={
            "time": np.array(["2020-01-01", "2020-01-02"], "datetime64[ns]"),
            "lat": ("lat", [-10.0, 0.0, 10.0], {"units": "degrees_north"}),
            "lon": [0.0, 90.0],
            "area": (("lat", "lon"), np.ones((3, 2))),
            "label": ("time", ["a", "b"]),
        },
        attrs={"title": "demo"},
    )


def test_batch_from_xarray():
    batch = nda.NdRecordBatch.from_xarray(dataset())
    assert set(batch.keys()) == {"sst", "flag", "time", "lat", "lon", "area", "label"}
    assert batch.target == {"time": 2, "lat": 3, "lon": 2}
    assert batch.attrs == {"title": "demo"}
    assert batch["sst"].attrs == {"units": "K", "coordinates": "area label"}
    assert batch["flag"].attrs == {"coordinates": "label"}
    assert batch["lat"].attrs == {"units": "degrees_north"}
    assert batch["time"].dtype == "timestamp[ns]"


def test_batch_round_trip():
    ds = dataset()
    back = nda.NdRecordBatch.from_xarray(ds).to_xarray()
    xr.testing.assert_identical(back, ds)


def test_batch_round_trip_via_arrow():
    pytest.importorskip("pyarrow")
    ds = dataset()
    (batch,) = nda.NdRecordBatch.from_arrow(nda.NdRecordBatch.from_xarray(ds).to_arrow())
    xr.testing.assert_identical(batch.to_xarray(), ds)


def test_unclaimed_coordinate_goes_to_global_attrs():
    ds = xr.Dataset({"a": ("x", [1.0])}, coords={"station": ("y", ["s1", "s2"])})
    batch = nda.NdRecordBatch.from_xarray(ds)
    assert batch.attrs == {"coordinates": "station"}
    xr.testing.assert_identical(batch.to_xarray(), ds)


def test_batch_from_data_array_keeps_coords():
    da = xr.DataArray([1.0, 2.0], dims=["x"], coords={"x": [10, 20]}, name="v")
    batch = nda.NdRecordBatch.from_xarray(da)
    assert set(batch.keys()) == {"v", "x"}
    xr.testing.assert_identical(batch.to_xarray()["v"], da)


def test_batch_to_xarray_from_nd_batch():
    lat = nda.NdArray.from_numpy(np.array([-10.0, 0.0, 10.0]), dims=["lat"])
    ds = nda.NdRecordBatch({"sst": sst(), "lat": lat}, attrs={"title": "t"}).to_xarray()
    assert list(ds.data_vars) == ["sst"]
    assert list(ds.coords) == ["lat"]
    assert ds.attrs == {"title": "t"}


def test_existing_coordinates_attribute_raises():
    ds = xr.Dataset({"a": ("x", [1.0], {"coordinates": "b"})})
    with pytest.raises(ValueError):
        nda.NdRecordBatch.from_xarray(ds)


def test_unknown_coordinate_name_raises():
    a = nda.NdArray.from_numpy(np.zeros(2), dims=["x"], attrs={"coordinates": "missing"})
    with pytest.raises(ValueError):
        nda.NdRecordBatch({"a": a}).to_xarray()


cftime = pytest.importorskip("cftime")


def cftime_array(dates, calendar):
    return xr.DataArray(
        np.array([cftime.datetime(*d, calendar=calendar) if d else None for d in dates], dtype=object),
        dims=["time"],
    )


@pytest.mark.parametrize("calendar", ["standard", "proleptic_gregorian", "noleap", "360_day", "all_leap"])
def test_cftime_dates_become_timestamps(calendar):
    a = nda.NdArray.from_xarray(cftime_array([(2000, 1, 15, 12, 30), (2001, 2, 28)], calendar))
    assert a.dtype == "timestamp[us]"
    expected = np.array(["2000-01-15T12:30", "2001-02-28"], dtype="datetime64[us]")
    np.testing.assert_array_equal(a.to_numpy(), expected)


def test_cftime_julian_keeps_the_instant():
    # 1 January 1500 in the Julian calendar is 10 January 1500 in the proleptic Gregorian.
    a = nda.NdArray.from_xarray(cftime_array([(1500, 1, 1)], "julian"))
    assert a.to_numpy()[0] == np.datetime64("1500-01-10", "us")


def test_cftime_standard_before_1582_keeps_the_instant():
    a = nda.NdArray.from_xarray(cftime_array([(1500, 1, 1)], "standard"))
    assert a.to_numpy()[0] == np.datetime64("1500-01-10", "us")


def test_cftime_outside_nanosecond_range():
    a = nda.NdArray.from_xarray(cftime_array([(1, 1, 1), (5000, 6, 1)], "noleap"))
    np.testing.assert_array_equal(
        a.to_numpy(), np.array(["0001-01-01", "5000-06-01"], dtype="datetime64[us]")
    )


def test_cftime_missing_dates_become_nulls():
    a = nda.NdArray.from_xarray(cftime_array([(2000, 1, 1), None], "noleap"))
    assert a.to_numpy().mask.tolist() == [False, True]
    assert np.isnat(a.to_xarray().values[1])


def test_cftime_date_missing_in_standard_calendar_raises():
    with pytest.raises(ValueError, match="convert_calendar"):
        nda.NdArray.from_xarray(cftime_array([(2001, 2, 30)], "360_day"))


def test_cftime_mixed_calendars_raise():
    da = xr.DataArray(
        np.array(
            [cftime.datetime(2000, 1, 1, calendar="noleap"), cftime.datetime(2000, 1, 1, calendar="360_day")],
            dtype=object,
        ),
        dims=["time"],
    )
    with pytest.raises(ValueError, match="one calendar"):
        nda.NdArray.from_xarray(da)


def test_cftime_time_coordinate_in_dataset():
    time = xr.date_range("2000-01-01", periods=3, freq="D", calendar="noleap", use_cftime=True)
    ds = xr.Dataset({"a": ("time", [1.0, 2.0, 3.0])}, coords={"time": time})
    batch = nda.NdRecordBatch.from_xarray(ds)
    assert batch["time"].dtype == "timestamp[us]"
    back = batch.to_xarray()
    assert back.indexes["time"].dtype.kind == "M"
    np.testing.assert_array_equal(
        back["time"].values.astype("M8[us]"),
        np.array(["2000-01-01", "2000-01-02", "2000-01-03"], dtype="datetime64[us]"),
    )
