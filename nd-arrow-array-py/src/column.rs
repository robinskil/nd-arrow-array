//! The `NdColumn` Python class.

use nd_arrow_array::{NdColumn, NdColumnBuilder};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::array::PyNdArray;
use crate::error::IntoPyResult;

/// An Arrow column of the `nd_arrow.array` extension type. Each row is one array.
#[pyclass(name = "NdColumn", module = "nd_arrow_array")]
#[derive(Clone)]
pub struct PyNdColumn(pub NdColumn);

#[pymethods]
impl PyNdColumn {
    /// A column with one row per item. `None` gives a null row.
    #[staticmethod]
    fn from_arrays(items: Vec<Option<PyRef<'_, PyNdArray>>>) -> PyResult<Self> {
        let mut builder = NdColumnBuilder::new();
        for item in &items {
            builder.append_option(item.as_ref().map(|a| &a.0)).py()?;
        }
        Ok(Self(builder.finish().py()?))
    }

    /// The number of null rows.
    #[getter]
    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __getitem__(&self, index: isize) -> PyResult<Option<PyNdArray>> {
        let len = self.0.len() as isize;
        let row = if index < 0 { index + len } else { index };
        if row < 0 || row >= len {
            return Err(PyIndexError::new_err("NdColumn index out of range"));
        }
        Ok(self.0.get(row as usize).map(PyNdArray))
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rows: Vec<Option<PyNdArray>> = self.0.iter().map(|row| row.map(PyNdArray)).collect();
        PyList::new(py, rows)?.try_iter().map(|it| it.into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "NdColumn(len={}, value_type={})",
            self.0.len(),
            crate::array::type_name(self.0.value_type())
        )
    }
}
