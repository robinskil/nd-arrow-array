//! Conversion between NumPy arrays and flat Arrow values in C order.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, PrimitiveArray, StringArray};
use arrow::buffer::{NullBuffer, ScalarBuffer};
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, DurationMicrosecondType, DurationMillisecondType,
    DurationNanosecondType, DurationSecondType, Float16Type, Float32Type, Float64Type, Int8Type,
    Int16Type, Int32Type, Int64Type, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use half::f16;
use numpy::{Element, PyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyList;

/// The flat C-order values and the shape of a NumPy array.
///
/// A `numpy.ma.MaskedArray` mask and `NaT` become nulls. `NaN` stays a value.
pub fn numpy_to_arrow(arr: &Bound<'_, PyAny>) -> PyResult<(ArrayRef, Vec<usize>)> {
    let py = arr.py();
    let np = py.import("numpy")?;
    let ma = np.getattr("ma")?;
    let mask = ma.call_method1("getmaskarray", (arr,))?;
    let data = ma.call_method1("getdata", (arr,))?;
    let dtype = data.getattr("dtype")?;
    let kind: char = dtype.getattr("kind")?.extract()?;
    let shape: Vec<usize> = data.getattr("shape")?.extract()?;

    let mut nulls = mask_to_nulls(&mask)?;
    let values: ArrayRef = match kind {
        'b' => {
            let data = contiguous(&np, &data, "bool")?;
            let view = data.extract::<PyReadonlyArrayDyn<'_, bool>>()?;
            Arc::new(BooleanArray::from(view.as_slice()?.to_vec()))
        }
        'i' | 'u' | 'f' => {
            let name: String = dtype.getattr("name")?.extract()?;
            numeric_to_arrow(&np, &data, &name)?
        }
        'M' | 'm' => {
            let unit = time_unit(&np, &dtype)?;
            let raw = contiguous(&np, &data, "int64")?;
            let raw = raw.extract::<PyReadonlyArrayDyn<'_, i64>>()?;
            let raw = raw.as_slice()?;
            // NaT is i64::MIN in NumPy and has no Arrow value: make it null.
            if raw.contains(&i64::MIN) {
                let valid = raw.iter().map(|v| *v != i64::MIN);
                nulls = merge_nulls(nulls, NullBuffer::from_iter(valid));
            }
            time_array(kind, unit, raw.to_vec())
        }
        'U' => {
            let flat = data
                .call_method1("reshape", (-1,))?
                .call_method0("tolist")?;
            let strings: Vec<String> = flat.extract()?;
            Arc::new(StringArray::from(strings))
        }
        _ => {
            return Err(PyTypeError::new_err(format!(
                "unsupported NumPy dtype {}",
                dtype.str()?
            )));
        }
    };
    let values = match nulls {
        Some(nulls) => with_nulls(values, nulls)?,
        None => values,
    };
    Ok((values, shape))
}

/// A NumPy array of `shape` from flat C-order values.
///
/// Returns a plain `ndarray` without nulls, else a `numpy.ma.MaskedArray`.
pub fn arrow_to_numpy<'py>(
    py: Python<'py>,
    values: &ArrayRef,
    shape: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let flat: Bound<'py, PyAny> = match values.data_type() {
        DataType::Boolean => {
            let bools = values.as_boolean();
            let vec: Vec<bool> = (0..bools.len()).map(|i| bools.value(i)).collect();
            PyArray1::from_vec(py, vec).into_any()
        }
        DataType::Int8 => primitive_to_numpy::<Int8Type>(py, values),
        DataType::Int16 => primitive_to_numpy::<Int16Type>(py, values),
        DataType::Int32 => primitive_to_numpy::<Int32Type>(py, values),
        DataType::Int64 => primitive_to_numpy::<Int64Type>(py, values),
        DataType::UInt8 => primitive_to_numpy::<UInt8Type>(py, values),
        DataType::UInt16 => primitive_to_numpy::<UInt16Type>(py, values),
        DataType::UInt32 => primitive_to_numpy::<UInt32Type>(py, values),
        DataType::UInt64 => primitive_to_numpy::<UInt64Type>(py, values),
        DataType::Float16 => {
            let bits: Vec<u16> = values
                .as_primitive::<Float16Type>()
                .values()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            PyArray1::from_vec(py, bits).call_method1("view", ("float16",))?
        }
        DataType::Float32 => primitive_to_numpy::<Float32Type>(py, values),
        DataType::Float64 => primitive_to_numpy::<Float64Type>(py, values),
        DataType::Timestamp(unit, None) => {
            let raw = time_values(values, unit, true);
            let dtype = format!("datetime64[{}]", unit_code(unit));
            PyArray1::from_vec(py, raw).call_method1("view", (dtype,))?
        }
        DataType::Duration(unit) => {
            let raw = time_values(values, unit, false);
            let dtype = format!("timedelta64[{}]", unit_code(unit));
            PyArray1::from_vec(py, raw).call_method1("view", (dtype,))?
        }
        DataType::Utf8 | DataType::LargeUtf8 => {
            let strings: Vec<&str> = match values.data_type() {
                DataType::Utf8 => string_values(values.as_string::<i32>()),
                _ => string_values(values.as_string::<i64>()),
            };
            let list = PyList::new(py, strings)?;
            if list.is_empty() {
                np.call_method1("array", (list, "U1"))?
            } else {
                np.call_method1("array", (list, "str"))?
            }
        }
        other => {
            return Err(PyTypeError::new_err(format!(
                "no NumPy dtype for Arrow type {other}"
            )));
        }
    };
    let shaped = flat.call_method1("reshape", (shape.to_vec(),))?;
    match values.logical_nulls() {
        Some(nulls) if nulls.null_count() > 0 => {
            let mask: Vec<bool> = nulls.iter().map(|valid| !valid).collect();
            let mask = PyArray1::from_vec(py, mask).call_method1("reshape", (shape.to_vec(),))?;
            np.getattr("ma")?
                .call_method1("MaskedArray", (shaped, mask))
        }
        _ => Ok(shaped),
    }
}

/// The values in C order with native byte order, as dtype `dtype`.
fn contiguous<'py>(
    np: &Bound<'py, PyModule>,
    data: &Bound<'py, PyAny>,
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let flat = np.call_method1("ascontiguousarray", (data,))?;
    // A view keeps the bits: datetime64 and float16 are read as integers.
    let native = flat
        .getattr("dtype")?
        .call_method1("newbyteorder", ("=",))?;
    let flat = flat.call_method1("astype", (native,))?;
    flat.call_method1("view", (dtype,))?
        .call_method1("reshape", (-1,))
}

fn numeric_to_arrow(
    np: &Bound<'_, PyModule>,
    data: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<ArrayRef> {
    Ok(match name {
        "int8" => primitive_from_numpy::<Int8Type>(np, data, name)?,
        "int16" => primitive_from_numpy::<Int16Type>(np, data, name)?,
        "int32" => primitive_from_numpy::<Int32Type>(np, data, name)?,
        "int64" => primitive_from_numpy::<Int64Type>(np, data, name)?,
        "uint8" => primitive_from_numpy::<UInt8Type>(np, data, name)?,
        "uint16" => primitive_from_numpy::<UInt16Type>(np, data, name)?,
        "uint32" => primitive_from_numpy::<UInt32Type>(np, data, name)?,
        "uint64" => primitive_from_numpy::<UInt64Type>(np, data, name)?,
        "float16" => {
            let bits = contiguous(np, data, "uint16")?;
            let bits = bits.extract::<PyReadonlyArrayDyn<'_, u16>>()?;
            let values: Vec<f16> = bits
                .as_slice()?
                .iter()
                .map(|b| f16::from_bits(*b))
                .collect();
            Arc::new(PrimitiveArray::<Float16Type>::from(values))
        }
        "float32" => primitive_from_numpy::<Float32Type>(np, data, name)?,
        "float64" => primitive_from_numpy::<Float64Type>(np, data, name)?,
        _ => {
            return Err(PyTypeError::new_err(format!(
                "unsupported NumPy dtype {name}"
            )));
        }
    })
}

fn primitive_from_numpy<T>(
    np: &Bound<'_, PyModule>,
    data: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<ArrayRef>
where
    T: ArrowPrimitiveType,
    T::Native: Element,
{
    let flat = contiguous(np, data, name)?;
    let view = flat.extract::<PyReadonlyArrayDyn<'_, T::Native>>()?;
    let buffer = ScalarBuffer::from(view.as_slice()?.to_vec());
    Ok(Arc::new(PrimitiveArray::<T>::new(buffer, None)))
}

fn primitive_to_numpy<'py, T>(py: Python<'py>, values: &ArrayRef) -> Bound<'py, PyAny>
where
    T: ArrowPrimitiveType,
    T::Native: Element,
{
    let vec = values.as_primitive::<T>().values().to_vec();
    PyArray1::from_vec(py, vec).into_any()
}

fn time_unit(np: &Bound<'_, PyModule>, dtype: &Bound<'_, PyAny>) -> PyResult<TimeUnit> {
    let (unit, count): (String, i64) = np.call_method1("datetime_data", (dtype,))?.extract()?;
    match (unit.as_str(), count) {
        ("s", 1) => Ok(TimeUnit::Second),
        ("ms", 1) => Ok(TimeUnit::Millisecond),
        ("us", 1) => Ok(TimeUnit::Microsecond),
        ("ns", 1) => Ok(TimeUnit::Nanosecond),
        _ => Err(PyTypeError::new_err(format!(
            "unsupported NumPy time unit {}",
            dtype.str()?
        ))),
    }
}

fn unit_code(unit: &TimeUnit) -> &'static str {
    match unit {
        TimeUnit::Second => "s",
        TimeUnit::Millisecond => "ms",
        TimeUnit::Microsecond => "us",
        TimeUnit::Nanosecond => "ns",
    }
}

fn time_array(kind: char, unit: TimeUnit, raw: Vec<i64>) -> ArrayRef {
    let buffer = ScalarBuffer::from(raw);
    match (kind, unit) {
        ('M', TimeUnit::Second) => {
            Arc::new(PrimitiveArray::<TimestampSecondType>::new(buffer, None))
        }
        ('M', TimeUnit::Millisecond) => Arc::new(PrimitiveArray::<TimestampMillisecondType>::new(
            buffer, None,
        )),
        ('M', TimeUnit::Microsecond) => Arc::new(PrimitiveArray::<TimestampMicrosecondType>::new(
            buffer, None,
        )),
        ('M', TimeUnit::Nanosecond) => {
            Arc::new(PrimitiveArray::<TimestampNanosecondType>::new(buffer, None))
        }
        (_, TimeUnit::Second) => Arc::new(PrimitiveArray::<DurationSecondType>::new(buffer, None)),
        (_, TimeUnit::Millisecond) => {
            Arc::new(PrimitiveArray::<DurationMillisecondType>::new(buffer, None))
        }
        (_, TimeUnit::Microsecond) => {
            Arc::new(PrimitiveArray::<DurationMicrosecondType>::new(buffer, None))
        }
        (_, TimeUnit::Nanosecond) => {
            Arc::new(PrimitiveArray::<DurationNanosecondType>::new(buffer, None))
        }
    }
}

fn time_values(values: &ArrayRef, unit: &TimeUnit, timestamp: bool) -> Vec<i64> {
    macro_rules! raw {
        ($t:ty) => {
            values.as_primitive::<$t>().values().to_vec()
        };
    }
    match (timestamp, unit) {
        (true, TimeUnit::Second) => raw!(TimestampSecondType),
        (true, TimeUnit::Millisecond) => raw!(TimestampMillisecondType),
        (true, TimeUnit::Microsecond) => raw!(TimestampMicrosecondType),
        (true, TimeUnit::Nanosecond) => raw!(TimestampNanosecondType),
        (false, TimeUnit::Second) => raw!(DurationSecondType),
        (false, TimeUnit::Millisecond) => raw!(DurationMillisecondType),
        (false, TimeUnit::Microsecond) => raw!(DurationMicrosecondType),
        (false, TimeUnit::Nanosecond) => raw!(DurationNanosecondType),
    }
}

fn string_values<O: arrow::array::OffsetSizeTrait>(
    strings: &arrow::array::GenericStringArray<O>,
) -> Vec<&str> {
    // A null slot can hold any bytes, so it becomes an empty string.
    (0..strings.len())
        .map(|i| {
            if strings.is_valid(i) {
                strings.value(i)
            } else {
                ""
            }
        })
        .collect()
}

fn mask_to_nulls(mask: &Bound<'_, PyAny>) -> PyResult<Option<NullBuffer>> {
    let flat = mask.call_method1("reshape", (-1,))?;
    let flat = flat.extract::<PyReadonlyArrayDyn<'_, bool>>()?;
    let masked = flat.as_slice()?;
    if !masked.contains(&true) {
        return Ok(None);
    }
    Ok(Some(NullBuffer::from_iter(masked.iter().map(|m| !m))))
}

fn merge_nulls(a: Option<NullBuffer>, b: NullBuffer) -> Option<NullBuffer> {
    NullBuffer::union(a.as_ref(), Some(&b))
}

fn with_nulls(values: ArrayRef, nulls: NullBuffer) -> PyResult<ArrayRef> {
    let data = values
        .into_data()
        .into_builder()
        .nulls(Some(nulls))
        .build()
        .map_err(|e| PyTypeError::new_err(e.to_string()))?;
    Ok(arrow::array::make_array(data))
}

/// Test hook: NumPy to Arrow and back.
#[pyfunction]
pub fn _roundtrip_numpy<'py>(arr: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let (values, shape) = numpy_to_arrow(arr)?;
    arrow_to_numpy(arr.py(), &values, &shape)
}
