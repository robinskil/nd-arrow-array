"""Conversion between nd arrays and xarray objects.

The methods ``to_xarray`` and ``from_xarray`` of ``NdArray`` and
``NdRecordBatch`` call the functions in this module.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Hashable, Iterable, Mapping

import numpy as np

from ._core import NdArray, NdRecordBatch

# CF attribute that names the non-index coordinates of a variable.
COORDINATES = "coordinates"


def _xr(feature: str) -> ModuleType:
    try:
        import xarray
    except ImportError as error:
        raise ImportError(
            f"xarray is required for {feature}: pip install nd-arrow-array[xarray]"
        ) from error
    return xarray


def _name(key: Hashable, kind: str) -> str:
    if not isinstance(key, str):
        raise TypeError(f"{kind} names must be str, got {key!r}")
    return key


def _decode(values: Any) -> np.ndarray[Any, Any]:
    """Nulls become NaN or NaT, as in xarray."""
    if not isinstance(values, np.ma.MaskedArray):
        return np.asarray(values)
    if not values.mask.any():
        return values.data
    kind = values.dtype.kind
    if kind in "fc":
        return values.filled(np.nan)
    if kind in "biu":
        # Like xarray CF decoding: integers with missing values become float64.
        return np.where(values.mask, np.nan, values.data.astype(np.float64))
    if kind in "mM":
        return values.filled(values.dtype.type("NaT", np.datetime_data(values.dtype)[0]))
    out = values.data.astype(object)
    out[values.mask] = np.nan
    return out


def _missing(item: Any) -> bool:
    return item is None or (isinstance(item, float) and item != item)


# Calendars where a cftime date is a real instant. Others are model calendars.
REAL_CALENDARS = {"standard", "gregorian", "proleptic_gregorian", "julian"}
MODEL_CALENDARS = {"noleap", "365_day", "all_leap", "366_day", "360_day"}
US_PER_SECOND = 1_000_000


def _cftime_type() -> type | None:
    try:
        import cftime  # type: ignore[import-untyped, unused-ignore]
    except ImportError:
        return None
    return cftime.datetime  # type: ignore[no-any-return, unused-ignore]


def _elapsed_us(date: Any, calendar: str) -> int:
    """Microseconds since the Unix epoch of the instant `date`."""
    import cftime  # type: ignore[import-untyped, unused-ignore]

    # The Unix epoch, 1970-01-01 Gregorian, is 1969-12-19 in the Julian calendar.
    ymd = (1969, 12, 19) if calendar == "julian" else (1970, 1, 1)
    epoch = cftime.datetime(*ymd, calendar=calendar, has_year_zero=date.has_year_zero)
    delta = date - epoch
    return int((delta.days * 86_400 + delta.seconds) * US_PER_SECOND + delta.microseconds)


def _fields_us(date: Any, calendar: str) -> int:
    """Microseconds since 1970-01-01 of the same date and time in the standard calendar."""
    month = np.datetime64(date.year - 1970, "Y").astype("M8[M]") + np.timedelta64(date.month - 1, "M")
    first = month.astype("M8[D]")
    days = int(((month + np.timedelta64(1, "M")).astype("M8[D]") - first).astype(np.int64))
    if not 1 <= date.day <= days:
        raise ValueError(
            f"{date} of calendar '{calendar}' does not exist in the standard calendar; "
            "use xarray convert_calendar(..., align_on='date') first"
        )
    day_us = int((first + np.timedelta64(date.day - 1, "D")).astype("M8[us]").astype(np.int64))
    seconds = (date.hour * 60 + date.minute) * 60 + date.second
    return int(day_us + seconds * US_PER_SECOND + date.microsecond)


def _cftime_to_timestamps(flat: np.ndarray[Any, Any], missing: np.ndarray[Any, Any]) -> Any:
    """cftime dates become datetime64[us] in the standard calendar."""
    calendars = {item.calendar for item, gap in zip(flat, missing) if not gap}
    if len(calendars) != 1:
        raise ValueError(f"cftime dates must have one calendar, got {sorted(calendars)}")
    (calendar,) = calendars
    if calendar in REAL_CALENDARS:
        convert = _elapsed_us
    elif calendar in MODEL_CALENDARS:
        convert = _fields_us
    else:
        raise ValueError(f"cftime calendar '{calendar}' is not supported")
    raw = [0 if gap else convert(item, calendar) for item, gap in zip(flat, missing)]
    return np.array(raw, dtype=np.int64).view("M8[us]")


def _encode(values: Any) -> Any:
    """Object arrays of str or cftime dates become typed arrays. None and NaN become nulls."""
    values = np.asarray(values)
    if values.dtype != object:
        return values
    flat = values.ravel()
    missing = np.array([_missing(item) for item in flat], dtype=bool)
    present = [item for item, gap in zip(flat, missing) if not gap]
    cftime_type = _cftime_type()
    if cftime_type is not None and present and all(isinstance(i, cftime_type) for i in present):
        out = _cftime_to_timestamps(flat, missing)
    elif all(isinstance(item, str) for item in present):
        out = np.where(missing, "", flat).astype(str)
    else:
        raise TypeError("object arrays must hold only str or cftime dates, with None and NaN")
    out = out.reshape(values.shape)
    missing = missing.reshape(values.shape)
    return np.ma.MaskedArray(out, missing) if missing.any() else out


def _attrs(attrs: Mapping[Any, Any]) -> dict[str, Any]:
    return {_name(key, "attribute"): value for key, value in attrs.items()}


def _array(dims: Iterable[Hashable], values: Any, attrs: Mapping[Any, Any]) -> NdArray:
    names = [_name(dim, "dimension") for dim in dims]
    return NdArray.from_numpy(_encode(values), dims=names, attrs=_attrs(attrs))


def array_to_xarray(array: NdArray, name: str | None = None) -> Any:
    xr = _xr("NdArray.to_xarray")
    return xr.DataArray(_decode(array.to_numpy()), dims=array.dims, name=name, attrs=array.attrs)


def array_from_xarray(data_array: Any) -> NdArray:
    xr = _xr("NdArray.from_xarray")
    if not isinstance(data_array, xr.DataArray):
        raise TypeError(f"expected an xarray.DataArray, got {type(data_array).__name__}")
    return _array(data_array.dims, data_array.values, data_array.attrs)


def _split(names: Any) -> list[str]:
    if not isinstance(names, str):
        raise TypeError(f"the '{COORDINATES}' attribute must be a str, got {names!r}")
    return names.split()


def batch_to_xarray(batch: NdRecordBatch) -> Any:
    xr = _xr("NdRecordBatch.to_xarray")
    attrs = batch.attrs
    coords = set(_split(attrs.pop(COORDINATES, "")))
    variables = {}
    for name in batch.keys():
        array = batch[name]
        var_attrs = array.attrs
        coords.update(_split(var_attrs.pop(COORDINATES, "")))
        if array.dims == (name,):
            coords.add(name)
        variables[name] = xr.Variable(array.dims, _decode(array.to_numpy()), attrs=var_attrs)
    unknown = coords - variables.keys()
    if unknown:
        raise ValueError(f"'{COORDINATES}' names columns that do not exist: {sorted(unknown)}")
    data_vars = {k: v for k, v in variables.items() if k not in coords}
    coord_vars = {k: v for k, v in variables.items() if k in coords}
    return xr.Dataset(data_vars, coords=coord_vars, attrs=attrs)


def _check_free(attrs: Mapping[str, Any], owner: str) -> None:
    if COORDINATES in attrs:
        raise ValueError(
            f"{owner} has a '{COORDINATES}' attribute; move it to the xarray coordinates"
        )


def batch_from_xarray(dataset: Any) -> NdRecordBatch:
    xr = _xr("NdRecordBatch.from_xarray")
    if isinstance(dataset, xr.DataArray):
        dataset = dataset.to_dataset(name=dataset.name if dataset.name is not None else "data")
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(f"expected an xarray.Dataset, got {type(dataset).__name__}")

    # Non-index coordinates go to CF 'coordinates' attributes, as in xarray encoding.
    extra = [
        _name(name, "coordinate")
        for name, coord in dataset.coords.items()
        if coord.dims != (name,)
    ]
    unclaimed = set(extra)
    columns = {}
    for key, var in dataset.variables.items():
        name = _name(key, "variable")
        attrs = _attrs(var.attrs)
        _check_free(attrs, f"variable '{name}'")
        if name in dataset.data_vars:
            linked = [c for c in extra if set(dataset[c].dims) <= set(var.dims)]
            if linked:
                attrs[COORDINATES] = " ".join(linked)
                unclaimed.difference_update(linked)
        columns[name] = _array(var.dims, var.values, attrs)

    attrs = _attrs(dataset.attrs)
    _check_free(attrs, "the dataset")
    if unclaimed:
        attrs[COORDINATES] = " ".join(c for c in extra if c in unclaimed)
    target = {_name(dim, "dimension"): size for dim, size in dataset.sizes.items()}
    return NdRecordBatch(columns, attrs=attrs, target=target)
