"""Generate numpy.json: NumPy reference results for tests/numpy.rs.

Run from the crate root: python3 tests/fixtures/gen_numpy.py
NaN in NumPy maps to null in Arrow. Reductions use the NumPy nan* functions.
"""

import json
import pathlib
import warnings

import numpy as np

rng = np.random.default_rng(42)
DIMS = ["a", "b", "c"]


def encode(array, dims):
    flat = [None if np.isnan(v) else float(v) for v in np.asarray(array, dtype=float).ravel()]
    return {"dims": [[d, int(s)] for d, s in zip(dims, np.shape(array))], "data": flat}


def case(name, op, x, dims, expected, out_dims, **params):
    return {
        "name": name,
        "op": op,
        "input": encode(x, dims),
        "params": params,
        "expected": encode(expected, out_dims),
    }


cases = []

x = rng.normal(size=(3, 4, 5)).round(3)
x[0, 1, 2] = np.nan
x[2, 3, :] = np.nan
x[1, 0, 4] = np.nan

reductions = {
    "sum": np.nansum,
    "mean": np.nanmean,
    "min": np.nanmin,
    "max": np.nanmax,
    "median": np.nanmedian,
    "count": lambda v, axis: np.sum(~np.isnan(v), axis=axis),
    "var0": lambda v, axis: np.nanvar(v, axis=axis, ddof=0),
    "var1": lambda v, axis: np.nanvar(v, axis=axis, ddof=1),
    "std1": lambda v, axis: np.nanstd(v, axis=axis, ddof=1),
    "quantile30": lambda v, axis: np.nanquantile(v, 0.3, axis=axis),
}
axes_options = [None, (0,), (1,), (2,), (0, 2), (0, 1)]

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    for op, fn in reductions.items():
        for axes in axes_options:
            expected = fn(x, axis=axes)
            kept = [d for i, d in enumerate(DIMS) if axes is not None and i not in axes]
            names = None if axes is None else [DIMS[i] for i in axes]
            cases.append(case(f"{op}_{axes}", op, x, DIMS, expected, kept, axes=names))

# Arg reductions on data without all-NaN groups.
y = rng.normal(size=(3, 4, 5)).round(3)
y[0, 0, 0] = np.nan
for axis in range(3):
    for op, fn in [("argmin", np.nanargmin), ("argmax", np.nanargmax)]:
        kept = [d for i, d in enumerate(DIMS) if i != axis]
        cases.append(case(f"{op}_{axis}", op, y, DIMS, fn(y, axis=axis), kept, axes=[DIMS[axis]]))

for axis in range(3):
    expected = np.nancumsum(x, axis=axis)
    expected[np.isnan(x)] = np.nan
    cases.append(case(f"cumsum_{axis}", "cumsum", x, DIMS, expected, DIMS, dim=DIMS[axis]))
    for n in (1, 2):
        expected = np.diff(x, n=n, axis=axis)
        cases.append(case(f"diff_{axis}_{n}", "diff", x, DIMS, expected, DIMS, dim=DIMS[axis], n=n))

for perm in [(2, 0, 1), (1, 2, 0), (2, 1, 0)]:
    cases.append(
        case(f"transpose_{perm}", "transpose", x, DIMS, np.transpose(x, perm), [DIMS[i] for i in perm],
             order=[DIMS[i] for i in perm])
    )

z = rng.permutation(60).reshape(3, 4, 5).astype(float)
z[1, 2, 3] = np.nan
for axis in range(3):
    cases.append(case(f"sort_{axis}", "sort", z, DIMS, np.sort(z, axis=axis), DIMS, dim=DIMS[axis]))
    cases.append(
        case(f"argsort_{axis}", "argsort", z, DIMS, np.argsort(z, axis=axis, kind="stable"), DIMS,
             dim=DIMS[axis])
    )

sorted_values = np.sort(rng.integers(0, 20, size=15)).astype(float)
needles = rng.uniform(-2, 22, size=10).round(1)
for side in ("left", "right"):
    cases.append({
        "name": f"searchsorted_{side}",
        "op": "searchsorted",
        "input": encode(sorted_values, ["x"]),
        "params": {"side": side, "values": encode(needles, ["q"])},
        "expected": encode(np.searchsorted(sorted_values, needles, side=side), ["q"]),
    })

xp = np.array([0.0, 1.0, 2.5, 4.0, 7.0])
fp = rng.normal(size=(2, 5)).round(3)
targets = np.array([0.0, 0.5, 2.0, 3.9, 6.5, 7.0])
cases.append({
    "name": "interp",
    "op": "interp",
    "input": encode(fp, ["y", "x"]),
    "params": {"x": encode(xp, ["x"]), "new_x": encode(targets, ["x"])},
    "expected": encode(np.stack([np.interp(targets, xp, row) for row in fp]), ["y", "x"]),
})

a = rng.normal(size=(3, 4)).round(3)
b = rng.normal(size=(4,)).round(3)
for op, fn in [("add", np.add), ("sub", np.subtract), ("mul", np.multiply), ("div", np.divide)]:
    cases.append({
        "name": f"{op}_broadcast",
        "op": op,
        "input": encode(a, ["x", "y"]),
        "params": {"other": encode(b, ["y"])},
        "expected": encode(fn(a, b), ["x", "y"]),
    })

out = pathlib.Path(__file__).with_name("numpy.json")
out.write_text(json.dumps({"numpy": np.__version__, "cases": cases}, indent=1))
print(f"wrote {len(cases)} cases to {out}")
