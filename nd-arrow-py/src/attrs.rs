//! Conversion between Python dicts and nd attributes.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Float64Array, Int64Array, ListArray, StringArray,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Float64Type, Int64Type};
use nd_arrow_array::Attributes;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use crate::arrow_conv::{import_array, to_pyarrow};
use crate::error::IntoPyResult;
use crate::numpy_conv::{arrow_to_numpy, numpy_to_arrow};

/// Attributes from a Python dict with `str` keys. `None` gives no attributes.
pub fn dict_to_attributes(dict: Option<&Bound<'_, PyDict>>) -> PyResult<Attributes> {
    let mut attrs = Attributes::new();
    let Some(dict) = dict else {
        return Ok(attrs);
    };
    for (key, value) in dict.iter() {
        if !key.is_instance_of::<PyString>() {
            return Err(PyTypeError::new_err("attribute names must be str"));
        }
        let key: String = key.extract()?;
        let value = value_to_arrow(&value).map_err(|e| prefix_error(e, &key, value.py()))?;
        attrs.try_insert_array(key, value).py()?;
    }
    Ok(attrs)
}

/// A new Python dict with the attributes, in insertion order.
pub fn attributes_to_dict<'py>(
    py: Python<'py>,
    attrs: &Attributes,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in attrs.iter() {
        dict.set_item(key, value_to_python(py, value)?)?;
    }
    Ok(dict)
}

/// One Python attribute value as a length-1 Arrow array. A null value is an
/// error, because a null attribute means an absent key in the format.
pub fn value_to_arrow(value: &Bound<'_, PyAny>) -> PyResult<ArrayRef> {
    let array = convert_value(value)?;
    if array.is_null(0) {
        return Err(PyValueError::new_err(
            "attribute value is null; leave the key out instead",
        ));
    }
    Ok(array)
}

fn convert_value(value: &Bound<'_, PyAny>) -> PyResult<ArrayRef> {
    let py = value.py();
    let np = py.import("numpy")?;
    // NumPy scalars come first: np.float64 is also a Python float.
    if value.is_instance(&np.getattr("generic")?)? {
        let (values, _) = numpy_to_arrow(&np.call_method1("asarray", (value,))?)?;
        return Ok(values);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Arc::new(BooleanArray::from(vec![value.extract::<bool>()?])));
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(Arc::new(Int64Array::from(vec![value.extract::<i64>()?])));
    }
    if value.is_instance_of::<PyFloat>() {
        return Ok(Arc::new(Float64Array::from(vec![value.extract::<f64>()?])));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(Arc::new(StringArray::from(vec![
            value.extract::<String>()?,
        ])));
    }
    if value.is_instance(&np.getattr("ndarray")?)? {
        let ndim: usize = value.getattr("ndim")?.extract()?;
        if ndim != 1 {
            return Err(PyValueError::new_err(format!(
                "array attributes must be 1-D, got {ndim}-D"
            )));
        }
        let (values, _) = numpy_to_arrow(value)?;
        return Ok(list_of(values));
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        let items: Vec<Bound<'_, PyAny>> = value.try_iter()?.collect::<PyResult<_>>()?;
        if !items.is_empty() && items.iter().all(|i| i.is_instance_of::<PyString>()) {
            let strings: Vec<String> =
                items.iter().map(|i| i.extract()).collect::<PyResult<_>>()?;
            return Ok(list_of(Arc::new(StringArray::from(strings))));
        }
        let arr = np.call_method1("asarray", (value,))?;
        let kind: char = arr.getattr("dtype")?.getattr("kind")?.extract()?;
        if !matches!(kind, 'b' | 'i' | 'u' | 'f') {
            return Err(PyTypeError::new_err(
                "list attributes must hold only str or only numbers",
            ));
        }
        let (values, _) = numpy_to_arrow(&arr)?;
        return Ok(list_of(values));
    }
    if let Ok(pa) = py.import("pyarrow")
        && value.is_instance(&pa.getattr("Scalar")?)?
    {
        let array = pa.call_method1("array", (vec![value.clone()], value.getattr("type")?))?;
        return Ok(import_array(&array)?.1);
    }
    Err(PyTypeError::new_err(format!(
        "unsupported attribute type {}",
        value.get_type().name()?
    )))
}

/// One length-1 Arrow attribute value as a Python object.
pub fn value_to_python<'py>(py: Python<'py>, value: &ArrayRef) -> PyResult<Bound<'py, PyAny>> {
    if value.is_null(0) {
        return Ok(py.None().into_bound(py));
    }
    match value.data_type() {
        DataType::Boolean => Ok(PyBool::new(py, value.as_boolean().value(0))
            .to_owned()
            .into_any()),
        DataType::Int64 => Ok(value
            .as_primitive::<Int64Type>()
            .value(0)
            .into_pyobject(py)?
            .into_any()),
        DataType::Float64 => Ok(value
            .as_primitive::<Float64Type>()
            .value(0)
            .into_pyobject(py)?
            .into_any()),
        DataType::Utf8 => Ok(PyString::new(py, value.as_string::<i32>().value(0)).into_any()),
        DataType::List(_) => {
            let items = value.as_list::<i32>().value(0);
            arrow_to_numpy(py, &items, &[items.len()])
        }
        // Other types become a NumPy scalar, so the dtype stays.
        _ => match arrow_to_numpy(py, value, &[]) {
            Ok(scalar) => scalar.get_item(()),
            // Types without a NumPy dtype use the pyarrow Python value.
            Err(e) if e.is_instance_of::<PyTypeError>(py) => {
                let field = Field::new("value", value.data_type().clone(), true);
                to_pyarrow(py, field, value.clone(), "this attribute type")?
                    .get_item(0)?
                    .call_method0("as_py")
            }
            Err(e) => Err(e),
        },
    }
}

fn list_of(values: ArrayRef) -> ArrayRef {
    let field = Arc::new(Field::new("item", values.data_type().clone(), true));
    Arc::new(ListArray::new(
        field,
        OffsetBuffer::from_lengths([values.len()]),
        values,
        None,
    ))
}

fn prefix_error(error: PyErr, key: &str, py: Python<'_>) -> PyErr {
    let message = format!("attribute '{key}': {}", error.value(py));
    if error.is_instance_of::<PyValueError>(py) {
        PyValueError::new_err(message)
    } else {
        PyTypeError::new_err(message)
    }
}

/// Test hook: dict to attributes and back.
#[pyfunction]
pub fn _roundtrip_attrs<'py>(dict: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let attrs = dict_to_attributes(Some(dict))?;
    attributes_to_dict(dict.py(), &attrs)
}
