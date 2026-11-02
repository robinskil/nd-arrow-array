//! The `NdRecordBatch` Python class.

use std::ffi::CString;
use std::sync::Arc;

use arrow::array::{Array, RecordBatch, RecordBatchIterator, StructArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ffi::FFI_ArrowSchema;
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use nd_arrow::{NdArrowArray, NdRecordBatch, encode_batches};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};

use crate::array::{PyNdArray, parse_dims};
use crate::arrow_conv::{import_array, to_pyarrow};
use crate::attrs::{attributes_to_dict, dict_to_attributes};
use crate::error::IntoPyResult;

const STREAM_CAPSULE: &str = "arrow_array_stream";

/// Named nd arrays on one shared grid, like an xarray `Dataset`.
#[pyclass(name = "NdRecordBatch", module = "nd_arrow", skip_from_py_object)]
#[derive(Clone)]
pub struct PyNdRecordBatch(pub NdRecordBatch);

#[pymethods]
impl PyNdRecordBatch {
    /// A batch from a dict of arrays. `target=None` infers the grid.
    #[new]
    #[pyo3(signature = (columns, attrs=None, target=None))]
    fn new(
        columns: &Bound<'_, PyDict>,
        attrs: Option<&Bound<'_, PyDict>>,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut fields = Vec::with_capacity(columns.len());
        let mut arrays: Vec<NdArrowArray> = Vec::with_capacity(columns.len());
        for (name, array) in columns.iter() {
            let name: String = name
                .extract()
                .map_err(|_| PyTypeError::new_err("column names must be str"))?;
            let array = array
                .extract::<PyRef<'_, PyNdArray>>()
                .map_err(|_| PyTypeError::new_err(format!("column '{name}' must be an NdArray")))?;
            fields.push(Field::new(name, array.0.data_type().clone(), true));
            arrays.push(array.0.clone());
        }
        let schema = Arc::new(Schema::new(fields));
        let batch = match target {
            Some(target) => NdRecordBatch::try_new(schema, arrays, parse_dims(target)?),
            None => NdRecordBatch::try_new_inferred(schema, arrays),
        }
        .py()?;
        Ok(Self(batch.with_attributes(dict_to_attributes(attrs)?)))
    }

    /// Decode every row of an nd-encoded record batch, table or stream.
    #[staticmethod]
    fn from_arrow(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Self>> {
        let batches = import_record_batches(obj)?;
        let mut out = Vec::new();
        for batch in &batches {
            let decoded = NdRecordBatch::try_from_record_batch(batch).py()?;
            out.extend(decoded.into_iter().map(Self));
        }
        Ok(out)
    }

    /// The shared grid: dimension sizes by name.
    #[getter]
    fn target<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let target = PyDict::new(py);
        for dim in self.0.target().iter() {
            target.set_item(dim.name(), dim.size())?;
        }
        Ok(target)
    }

    /// A copy of the global attributes.
    #[getter]
    fn get_attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        attributes_to_dict(py, self.0.attributes())
    }

    /// Replace all global attributes.
    #[setter]
    fn set_attrs(&mut self, attrs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        *self.0.attributes_mut() = dict_to_attributes(attrs)?;
        Ok(())
    }

    /// The column names, in schema order.
    fn keys(&self) -> Vec<String> {
        self.0
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }

    fn __getitem__(&self, name: &str) -> PyResult<PyNdArray> {
        self.0
            .column_by_name(name)
            .map(|a| PyNdArray(a.clone()))
            .ok_or_else(|| PyKeyError::new_err(name.to_owned()))
    }

    fn __len__(&self) -> usize {
        self.0.num_columns()
    }

    /// Encode as a one-row pyarrow record batch.
    fn to_arrow<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        record_batch_to_pyarrow(py, self.0.to_record_batch().py()?, "NdRecordBatch.to_arrow")
    }

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = requested_schema;
        let batch = self.0.to_record_batch().py()?;
        let schema = batch.schema();
        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        let stream = FFI_ArrowArrayStream::new(Box::new(reader));
        PyCapsule::new(
            py,
            stream,
            Some(CString::new(STREAM_CAPSULE).expect("no nul")),
        )
    }

    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        let batch = self.0.to_record_batch().py()?;
        let schema = FFI_ArrowSchema::try_from(batch.schema().as_ref())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        PyCapsule::new(
            py,
            schema,
            Some(CString::new("arrow_schema").expect("no nul")),
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "NdRecordBatch(columns={:?}, target={})",
            self.keys(),
            self.0.target()
        )
    }
}

/// Encode several batches with one schema as a pyarrow record batch, one row each.
#[pyfunction(name = "encode_batches")]
pub fn py_encode_batches<'py>(
    py: Python<'py>,
    batches: Vec<PyRef<'py, PyNdRecordBatch>>,
) -> PyResult<Bound<'py, PyAny>> {
    let batches: Vec<NdRecordBatch> = batches.iter().map(|b| b.0.clone()).collect();
    record_batch_to_pyarrow(py, encode_batches(&batches).py()?, "encode_batches")
}

fn record_batch_to_pyarrow<'py>(
    py: Python<'py>,
    batch: RecordBatch,
    feature: &str,
) -> PyResult<Bound<'py, PyAny>> {
    // A record batch travels as a struct array; the struct field carries the schema metadata.
    let schema = batch.schema();
    let field = Field::new("", DataType::Struct(schema.fields().clone()), false)
        .with_metadata(schema.metadata().clone());
    let array = Arc::new(StructArray::from(batch));
    let struct_array = to_pyarrow(py, field, array, feature)?;
    let pa = py.import("pyarrow")?;
    let batch = pa
        .getattr("RecordBatch")?
        .call_method1("from_struct_array", (struct_array,))?;
    batch.call_method1("replace_schema_metadata", (schema_metadata(py, &schema)?,))
}

fn schema_metadata<'py>(py: Python<'py>, schema: &Schema) -> PyResult<Bound<'py, PyDict>> {
    let metadata = PyDict::new(py);
    for (key, value) in schema.metadata() {
        metadata.set_item(key, value)?;
    }
    Ok(metadata)
}

/// The record batches of an object with `__arrow_c_array__` or `__arrow_c_stream__`.
fn import_record_batches(obj: &Bound<'_, PyAny>) -> PyResult<Vec<RecordBatch>> {
    if obj.hasattr("__arrow_c_array__")? {
        let (field, array) = import_array(obj)?;
        let structs = array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| PyTypeError::new_err("expected a record batch or a struct array"))?;
        if structs.null_count() > 0 {
            return Err(PyTypeError::new_err(
                "a record batch cannot come from a struct array with null rows",
            ));
        }
        let batch = RecordBatch::from(structs.clone());
        let schema = batch
            .schema()
            .as_ref()
            .clone()
            .with_metadata(field.metadata().clone());
        return Ok(vec![
            batch.with_schema(Arc::new(schema)).map_err(arrow_err)?,
        ]);
    }
    if obj.hasattr("__arrow_c_stream__")? {
        let capsule = obj.call_method0("__arrow_c_stream__")?;
        let capsule = capsule.cast::<PyCapsule>()?;
        let name = CString::new(STREAM_CAPSULE).expect("no nul");
        let ptr = capsule
            .pointer_checked(Some(&name))?
            .cast::<FFI_ArrowArrayStream>();
        // SAFETY: the capsule name guarantees the pointer type. from_raw moves
        // the stream out and marks the capsule content as released.
        let stream = unsafe { FFI_ArrowArrayStream::from_raw(ptr.as_ptr()) };
        let reader = ArrowArrayStreamReader::try_new(stream).map_err(arrow_err)?;
        return reader.collect::<Result<Vec<_>, _>>().map_err(arrow_err);
    }
    Err(PyTypeError::new_err(format!(
        "expected an Arrow record batch or stream, got {}",
        obj.get_type().name()?
    )))
}

fn arrow_err(error: arrow::error::ArrowError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
