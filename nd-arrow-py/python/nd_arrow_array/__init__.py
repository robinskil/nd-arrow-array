"""N-dimensional Arrow arrays."""

from importlib.metadata import version as _version

from ._core import NdArray, NdColumn, NdFormatError, NdRecordBatch, encode_batches

__all__ = [
    "NdArray",
    "NdColumn",
    "NdFormatError",
    "NdRecordBatch",
    "encode_batches",
]
__version__ = _version("nd-arrow-array")
