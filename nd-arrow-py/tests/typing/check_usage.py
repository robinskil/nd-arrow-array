"""Static type checks for the stubs. Run: mypy --strict tests/typing."""

import sys
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

import nd_arrow as nda

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

a = nda.NdArray.from_numpy(np.zeros((2, 3)), dims=["time", "lat"], attrs={"units": "K"})
assert_type(a, nda.NdArray)
assert_type(a.dims, tuple[str, ...])
assert_type(a.shape, tuple[int, ...])
assert_type(a.sizes, dict[str, int])
assert_type(a.attrs, dict[str, Any])
assert_type(a.to_numpy(), npt.NDArray[Any])
assert_type(a.reshape({"x": 6}), nda.NdArray)
a.attrs = {"valid_range": np.array([0.0, 1.0]), "flag": True, "names": ["a", "b"]}
a.set_attr("scale_factor", np.float32(0.5))

col = nda.NdColumn.from_arrays([a, None])
assert_type(col[0], Optional[nda.NdArray])
for row in col:
    assert_type(row, Optional[nda.NdArray])

ds = nda.NdRecordBatch({"sst": a}, attrs={"title": "demo"}, target=[("time", 2), ("lat", 3)])
assert_type(ds["sst"], nda.NdArray)
assert_type(ds.target, dict[str, int])
assert_type(nda.NdRecordBatch.from_arrow(col), list[nda.NdRecordBatch])

# Each line below must be a type error; warn_unused_ignores catches a pass.
nda.NdArray.from_numpy(np.zeros(2), dims=["x"], attrs={"u": object()})  # type: ignore[dict-item]
nda.NdColumn.from_arrays([a, 1])  # type: ignore[list-item]
nda.NdRecordBatch({"sst": np.zeros(2)})  # type: ignore[dict-item]
ds[0]  # type: ignore[index]
