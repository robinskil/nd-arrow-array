//! Arrow C data interface: import and export through PyCapsules.

use std::ffi::CString;

use arrow::array::{ArrayRef, make_array};
use arrow::datatypes::Field;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi, to_ffi};
use nd_arrow_array::NdColumn;
use nd_arrow_array::tensor::column_from_field;
use pyo3::exceptions::{PyImportError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};

use crate::error::IntoPyResult;

const SCHEMA_CAPSULE: &str = "arrow_schema";
const ARRAY_CAPSULE: &str = "arrow_array";

/// The field and the array of an object with `__arrow_c_array__`.
pub fn import_array(obj: &Bound<'_, PyAny>) -> PyResult<(Field, ArrayRef)> {
    if !obj.hasattr("__arrow_c_array__")? {
        return Err(PyTypeError::new_err(format!(
            "expected an Arrow array (an object with __arrow_c_array__), got {}",
            obj.get_type().name()?
        )));
    }
    let tuple = obj.call_method0("__arrow_c_array__")?;
    let (schema, array): (Bound<'_, PyCapsule>, Bound<'_, PyCapsule>) = tuple
        .extract()
        .map_err(|_| PyTypeError::new_err("__arrow_c_array__ must return two capsules"))?;
    let schema_name = CString::new(SCHEMA_CAPSULE).expect("no nul");
    let array_name = CString::new(ARRAY_CAPSULE).expect("no nul");
    let schema_ptr = schema
        .pointer_checked(Some(&schema_name))?
        .cast::<FFI_ArrowSchema>();
    let array_ptr = array
        .pointer_checked(Some(&array_name))?
        .cast::<FFI_ArrowArray>();
    // SAFETY: the capsule names guarantee the pointer types. from_raw moves
    // the array out and marks the capsule content as released.
    let (field, data) = unsafe {
        let schema = schema_ptr.as_ref();
        let field = Field::try_from(schema).map_err(arrow_err)?;
        let ffi_array = FFI_ArrowArray::from_raw(array_ptr.as_ptr());
        (field, from_ffi(ffi_array, schema).map_err(arrow_err)?)
    };
    Ok((field, make_array(data)))
}

/// The `(schema, array)` capsule pair for `__arrow_c_array__`.
pub fn export_array<'py>(
    py: Python<'py>,
    field: &Field,
    array: &ArrayRef,
) -> PyResult<Bound<'py, PyTuple>> {
    let (ffi_array, _) = to_ffi(&array.to_data()).map_err(arrow_err)?;
    let schema = export_schema(py, field)?;
    let array = PyCapsule::new(
        py,
        ffi_array,
        Some(CString::new(ARRAY_CAPSULE).expect("no nul")),
    )?;
    PyTuple::new(py, [schema.into_any(), array.into_any()])
}

/// The schema capsule for `__arrow_c_schema__`.
pub fn export_schema<'py>(py: Python<'py>, field: &Field) -> PyResult<Bound<'py, PyCapsule>> {
    let schema = FFI_ArrowSchema::try_from(field).map_err(arrow_err)?;
    PyCapsule::new(
        py,
        schema,
        Some(CString::new(SCHEMA_CAPSULE).expect("no nul")),
    )
}

/// The nd column in an imported Arrow array.
///
/// pyarrow drops unknown extension types on import, so a plain struct with
/// the nd storage layout is accepted too.
pub fn import_column(obj: &Bound<'_, PyAny>) -> PyResult<NdColumn> {
    let (field, array) = import_array(obj)?;
    match field.extension_type_name() {
        Some(_) => column_from_field(&field, array.as_ref()).py(),
        None => NdColumn::try_from_array(array.as_ref()).py(),
    }
}

/// The `pyarrow` module, or an `ImportError` that names the feature.
pub fn pyarrow<'py>(py: Python<'py>, feature: &str) -> PyResult<Bound<'py, PyModule>> {
    py.import("pyarrow").map_err(|_| {
        PyImportError::new_err(format!(
            "pyarrow is required for {feature}: pip install nd-arrow-array[arrow]"
        ))
    })
}

/// A pyarrow array from a field and an Arrow array.
pub fn to_pyarrow<'py>(
    py: Python<'py>,
    field: Field,
    array: ArrayRef,
    feature: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let pa = pyarrow(py, feature)?;
    pa.call_method1("array", (ArrowExport { field, array },))
}

/// A field and an array that export through `__arrow_c_array__`.
#[pyclass(name = "_ArrowExport", module = "nd_arrow_array")]
pub struct ArrowExport {
    field: Field,
    array: ArrayRef,
}

#[pymethods]
impl ArrowExport {
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        // The requested schema is a hint; the producer can ignore it.
        let _ = requested_schema;
        export_array(py, &self.field, &self.array)
    }
}

fn arrow_err(error: arrow::error::ArrowError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
