//! The `NdArray` Python class.

use arrow::datatypes::{DataType, Field, TimeUnit};
use nd_arrow_array::{Dimension, Dimensions, NdArrowArray, NdColumn};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::arrow_conv::{export_array, import_array, to_pyarrow};
use crate::attrs::{attributes_to_dict, dict_to_attributes, value_to_arrow};
use crate::error::IntoPyResult;
use crate::numpy_conv::{arrow_to_numpy, numpy_to_arrow};

/// One N-dimensional array with named dimensions and attributes.
#[pyclass(name = "NdArray", module = "nd_arrow_array", eq, skip_from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PyNdArray(pub NdArrowArray);

#[pymethods]
impl PyNdArray {
    /// An array from flat Arrow values in C order and named dimensions.
    #[new]
    #[pyo3(signature = (values, dims, attrs=None))]
    fn new(
        values: &Bound<'_, PyAny>,
        dims: &Bound<'_, PyAny>,
        attrs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let (_, values) = import_array(values)?;
        let array = NdArrowArray::try_new(values, parse_dims(dims)?).py()?;
        Ok(Self(array.with_attributes(dict_to_attributes(attrs)?)))
    }

    /// An array from a NumPy array, with one dimension name per axis.
    #[staticmethod]
    #[pyo3(signature = (arr, dims, attrs=None))]
    fn from_numpy(
        arr: &Bound<'_, PyAny>,
        dims: Vec<String>,
        attrs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let (values, shape) = numpy_to_arrow(arr)?;
        if dims.len() != shape.len() {
            return Err(PyValueError::new_err(format!(
                "got {} dimension names for an array with {} axes",
                dims.len(),
                shape.len()
            )));
        }
        let dims = Dimensions::try_new(dims.into_iter().zip(shape)).py()?;
        let array = NdArrowArray::try_new(values, dims).py()?;
        Ok(Self(array.with_attributes(dict_to_attributes(attrs)?)))
    }

    /// The dimension names.
    #[getter]
    fn dims<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.dims().names())
    }

    /// The size of each axis.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.shape())
    }

    /// The dimension sizes by name.
    #[getter]
    fn sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let sizes = PyDict::new(py);
        for dim in self.0.dims().iter() {
            sizes.set_item(dim.name(), dim.size())?;
        }
        Ok(sizes)
    }

    /// The number of axes.
    #[getter]
    fn ndim(&self) -> usize {
        self.0.rank()
    }

    /// The value type, with the pyarrow type name.
    #[getter]
    fn dtype(&self) -> String {
        type_name(self.0.data_type())
    }

    /// A copy of the attributes.
    #[getter]
    fn get_attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        attributes_to_dict(py, self.0.attributes())
    }

    /// Replace all attributes.
    #[setter]
    fn set_attrs(&mut self, attrs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        *self.0.attributes_mut() = dict_to_attributes(attrs)?;
        Ok(())
    }

    /// Insert or replace one attribute.
    fn set_attr(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = value_to_arrow(value)?;
        self.0.attributes_mut().try_insert_array(name, value).py()
    }

    /// Remove one attribute. Raises `KeyError` when it is absent.
    fn del_attr(&mut self, name: &str) -> PyResult<()> {
        match self.0.attributes_mut().remove(name) {
            Some(_) => Ok(()),
            None => Err(PyKeyError::new_err(name.to_owned())),
        }
    }

    /// A copy with other attributes.
    fn with_attrs(&self, attrs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self(
            self.0.clone().with_attributes(dict_to_attributes(attrs)?),
        ))
    }

    /// A copy with other dimensions and the same number of elements.
    fn reshape(&self, dims: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self(self.0.clone().reshape(parse_dims(dims)?).py()?))
    }

    /// The values as a NumPy array, masked when nulls exist.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        arrow_to_numpy(py, self.0.values(), &self.0.shape())
    }

    /// The flat values in C order, as a pyarrow array.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let values = self.0.values().clone();
        let field = Field::new("values", values.data_type().clone(), true);
        to_pyarrow(py, field, values, "NdArray.values")
    }

    /// Export as a one-row nd column.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let _ = requested_schema;
        let column = NdColumn::try_from_arrays([&self.0]).py()?;
        let field = column.to_field("nd");
        export_array(py, &field, &column.into_array_ref())
    }

    fn __repr__(&self) -> String {
        self.0.to_string()
    }
}

/// Dimensions from a dict or a sequence of `(name, size)` pairs.
pub fn parse_dims(dims: &Bound<'_, PyAny>) -> PyResult<Dimensions> {
    let pairs: Vec<(String, usize)> = match dims.cast::<PyDict>() {
        Ok(dict) => dict.items().extract()?,
        Err(_) => dims.extract().map_err(|_| {
            PyTypeError::new_err("dims must be a dict or a sequence of (name, size) pairs")
        })?,
    };
    Dimensions::try_new(
        pairs
            .into_iter()
            .map(|(name, size)| Dimension::new(name, size)),
    )
    .py()
}

/// The pyarrow name of an Arrow type.
pub fn type_name(data_type: &DataType) -> String {
    let unit = |unit: &TimeUnit| match unit {
        TimeUnit::Second => "s",
        TimeUnit::Millisecond => "ms",
        TimeUnit::Microsecond => "us",
        TimeUnit::Nanosecond => "ns",
    };
    match data_type {
        DataType::Boolean => "bool".into(),
        DataType::Int8 => "int8".into(),
        DataType::Int16 => "int16".into(),
        DataType::Int32 => "int32".into(),
        DataType::Int64 => "int64".into(),
        DataType::UInt8 => "uint8".into(),
        DataType::UInt16 => "uint16".into(),
        DataType::UInt32 => "uint32".into(),
        DataType::UInt64 => "uint64".into(),
        DataType::Float16 => "halffloat".into(),
        DataType::Float32 => "float".into(),
        DataType::Float64 => "double".into(),
        DataType::Utf8 => "string".into(),
        DataType::LargeUtf8 => "large_string".into(),
        DataType::Timestamp(u, None) => format!("timestamp[{}]", unit(u)),
        DataType::Duration(u) => format!("duration[{}]", unit(u)),
        other => other.to_string(),
    }
}
