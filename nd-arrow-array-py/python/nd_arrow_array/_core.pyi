from typing import Any, Iterator, Mapping, Optional, Protocol, Sequence, Tuple, Union, final

import numpy as np
import numpy.typing as npt

__all__ = [
    "NdArray",
    "NdColumn",
    "NdRecordBatch",
    "_ArrowExport",
    "encode_batches",
    "NdFormatError",
    "_roundtrip_numpy",
    "_roundtrip_attrs",
]

class _ArrowArrayExportable(Protocol):
    def __arrow_c_array__(self, requested_schema: Any = None) -> Tuple[Any, Any]: ...

class _ArrowScalar(Protocol):
    # A pyarrow.Scalar, without a dependency on pyarrow.
    def as_py(self) -> Any: ...

class _ArrowStreamExportable(Protocol):
    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any: ...

# Dimensions: {"time": 2, "lat": 3} or [("time", 2), ("lat", 3)].
_Dims = Union[Mapping[str, int], Sequence[Tuple[str, int]]]
_AttrValue = Union[
    str,
    bool,
    int,
    float,
    np.generic,
    npt.NDArray[Any],
    Sequence[str],
    Sequence[int],
    Sequence[float],
    _ArrowScalar,
]
_Attrs = Optional[Mapping[str, _AttrValue]]

class NdFormatError(ValueError):
    """Data does not follow the nd_arrow.array format."""

@final
class NdArray:
    """One N-dimensional array with named dimensions and attributes."""

    def __new__(
        cls, values: _ArrowArrayExportable, dims: _Dims, attrs: _Attrs = None
    ) -> NdArray: ...
    @staticmethod
    def from_numpy(
        arr: npt.ArrayLike, dims: Sequence[str], attrs: _Attrs = None
    ) -> NdArray: ...
    @property
    def dims(self) -> Tuple[str, ...]: ...
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def sizes(self) -> dict[str, int]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    @property
    def attrs(self) -> dict[str, Any]: ...
    @attrs.setter
    def attrs(self, value: _Attrs) -> None: ...
    @property
    def values(self) -> Any:
        """The flat values as a pyarrow.Array. Needs pyarrow."""
    def set_attr(self, name: str, value: _AttrValue) -> None: ...
    def del_attr(self, name: str) -> None: ...
    def with_attrs(self, attrs: _Attrs) -> NdArray: ...
    def reshape(self, dims: _Dims) -> NdArray: ...
    def to_numpy(self) -> npt.NDArray[Any]:
        """A numpy.ndarray, or a numpy.ma.MaskedArray when nulls exist."""
    def __arrow_c_array__(self, requested_schema: Any = None) -> Tuple[Any, Any]: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...

@final
class NdColumn:
    """An Arrow column of the nd_arrow.array extension type."""

    @staticmethod
    def from_arrays(items: Sequence[Optional[NdArray]]) -> NdColumn: ...
    @staticmethod
    def from_arrow(obj: _ArrowArrayExportable) -> NdColumn: ...
    def to_arrow(self) -> Any:
        """The column as a pyarrow.Array. Needs pyarrow."""
    @property
    def null_count(self) -> int: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int, /) -> Optional[NdArray]: ...
    def __iter__(self) -> Iterator[Optional[NdArray]]: ...
    def __arrow_c_array__(self, requested_schema: Any = None) -> Tuple[Any, Any]: ...
    def __arrow_c_schema__(self) -> Any: ...

@final
class NdRecordBatch:
    """Named nd arrays on one shared grid, like an xarray Dataset."""

    def __new__(
        cls,
        columns: Mapping[str, NdArray],
        attrs: _Attrs = None,
        target: Optional[_Dims] = None,
    ) -> NdRecordBatch: ...
    @staticmethod
    def from_arrow(
        obj: Union[_ArrowArrayExportable, _ArrowStreamExportable],
    ) -> list[NdRecordBatch]: ...
    @property
    def target(self) -> dict[str, int]: ...
    @property
    def attrs(self) -> dict[str, Any]: ...
    @attrs.setter
    def attrs(self, value: _Attrs) -> None: ...
    def keys(self) -> list[str]: ...
    def __getitem__(self, name: str, /) -> NdArray: ...
    def __len__(self) -> int: ...
    def to_arrow(self) -> Any:
        """The batch as a one-row pyarrow.RecordBatch. Needs pyarrow."""
    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any: ...
    def __arrow_c_schema__(self) -> Any: ...

def encode_batches(batches: Sequence[NdRecordBatch]) -> Any:
    """Several batches as one pyarrow.RecordBatch, one row each. Needs pyarrow."""

@final
class _ArrowExport:
    def __arrow_c_array__(self, requested_schema: Any = None) -> Tuple[Any, Any]: ...

def _roundtrip_numpy(arr: npt.ArrayLike) -> npt.NDArray[Any]: ...
def _roundtrip_attrs(dict: Mapping[str, _AttrValue]) -> dict[str, Any]: ...
