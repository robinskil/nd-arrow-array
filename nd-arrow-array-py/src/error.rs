//! Map nd errors to Python exceptions.

use nd_arrow_array::NdError;
use pyo3::PyErr;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};

create_exception!(
    _core,
    NdFormatError,
    PyValueError,
    "Data does not follow the nd_arrow.array format."
);

/// The Python exception for an nd error.
pub fn to_py_err(error: NdError) -> PyErr {
    let message = error.to_string();
    match error {
        NdError::InvalidDimensions(_)
        | NdError::ShapeMismatch(_)
        | NdError::InvalidAttribute(_)
        | NdError::InvalidArgument(_) => PyValueError::new_err(message),
        NdError::SchemaMismatch(_) | NdError::InvalidEncoding(_) => {
            NdFormatError::new_err(message)
        }
        NdError::NotSupported(_) => PyTypeError::new_err(message),
        NdError::Arrow(_) => PyRuntimeError::new_err(message),
    }
}

/// Convert a Rust result into a Python result.
pub trait IntoPyResult<T> {
    /// The result with the error mapped by [`to_py_err`].
    fn py(self) -> pyo3::PyResult<T>;
}

impl<T> IntoPyResult<T> for nd_arrow_array::Result<T> {
    fn py(self) -> pyo3::PyResult<T> {
        self.map_err(to_py_err)
    }
}
