//! CF conventions: mask fill values and unpack `scale_factor` / `add_offset`.
//!
//! The behavior follows netCDF4-python and xarray `decode_cf`.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, Scalar};
use arrow::compute::kernels::{boolean, cmp};
use arrow::compute::{cast as arrow_cast, nullif};
use arrow::datatypes::DataType;

use crate::compute::elementwise::{add, cast, div, fill_null, mul, round, sub};
use crate::error::{Result, nd_err};
use crate::{Attributes, NdArrowArray};

pub const FILL_VALUE: &str = "_FillValue";
pub const MISSING_VALUE: &str = "missing_value";
pub const VALID_MIN: &str = "valid_min";
pub const VALID_MAX: &str = "valid_max";
pub const VALID_RANGE: &str = "valid_range";
pub const SCALE_FACTOR: &str = "scale_factor";
pub const ADD_OFFSET: &str = "add_offset";

const CF_KEYS: [&str; 7] = [
    FILL_VALUE,
    MISSING_VALUE,
    VALID_MIN,
    VALID_MAX,
    VALID_RANGE,
    SCALE_FACTOR,
    ADD_OFFSET,
];

/// True for the attribute keys that [`decode_cf`] uses.
pub fn is_cf_key(key: &str) -> bool {
    CF_KEYS.contains(&key)
}

/// The CF keys that [`decode_cf`] removed, plus the packed value type.
/// [`encode_cf`] uses it to restore the packed form.
#[derive(Debug, Clone, PartialEq)]
pub struct CfEncoding {
    pub attributes: Attributes,
    pub packed_type: DataType,
}

/// The values of an attribute: a scalar, or the items of a length-1 list.
fn attribute_values(value: &ArrayRef) -> ArrayRef {
    match value.data_type() {
        DataType::List(_) => value.as_list::<i32>().value(0),
        DataType::LargeList(_) => value.as_list::<i64>().value(0),
        DataType::FixedSizeList(_, _) => value.as_fixed_size_list().value(0),
        _ => value.clone(),
    }
}

fn scalar_at(values: &ArrayRef, index: usize, to: &DataType) -> Result<Option<Scalar<ArrayRef>>> {
    if index >= values.len() || values.is_null(index) {
        return Ok(None);
    }
    Ok(Some(Scalar::new(arrow_cast(
        values.slice(index, 1).as_ref(),
        to,
    )?)))
}

fn or(mask: Option<BooleanArray>, other: BooleanArray) -> Result<BooleanArray> {
    Ok(match mask {
        None => other,
        Some(mask) => boolean::or(&mask, &other)?,
    })
}

/// Mask fill and out-of-range values as nulls, then unpack scale and offset.
///
/// - `_FillValue` and every `missing_value` become null.
/// - Values outside `valid_min`, `valid_max` or `valid_range` become null.
/// - With `scale_factor` or `add_offset`, the result is `value * scale + offset`
///   in the type of those attributes (a float type), else in `Float64`.
///
/// The CF keys move from the attributes to the returned [`CfEncoding`].
pub fn decode_cf(a: &NdArrowArray) -> Result<(NdArrowArray, CfEncoding)> {
    let attrs = a.attributes();
    let values = a.values();
    let data_type = a.data_type().clone();
    let mut invalid: Option<BooleanArray> = None;

    for key in [FILL_VALUE, MISSING_VALUE] {
        if let Some(value) = attrs.get(key) {
            let items = attribute_values(value);
            for i in 0..items.len() {
                if let Some(fill) = scalar_at(&items, i, &data_type)? {
                    invalid = Some(or(invalid, cmp::eq(values, &fill)?)?);
                }
            }
        }
    }

    let range = attrs.get(VALID_RANGE).map(attribute_values);
    let low = match attrs.get(VALID_MIN) {
        Some(v) => scalar_at(&attribute_values(v), 0, &data_type)?,
        None => match &range {
            Some(r) => scalar_at(r, 0, &data_type)?,
            None => None,
        },
    };
    let high = match attrs.get(VALID_MAX) {
        Some(v) => scalar_at(&attribute_values(v), 0, &data_type)?,
        None => match &range {
            Some(r) => scalar_at(r, 1, &data_type)?,
            None => None,
        },
    };
    if let Some(low) = low {
        invalid = Some(or(invalid, cmp::lt(values, &low)?)?);
    }
    if let Some(high) = high {
        invalid = Some(or(invalid, cmp::gt(values, &high)?)?);
    }

    let masked: ArrayRef = match &invalid {
        Some(mask) => nullif(values.as_ref(), mask)?,
        None => values.clone(),
    };
    let mut out = NdArrowArray::try_new(masked, a.dims().clone())?;

    let scale = attrs.get(SCALE_FACTOR).map(attribute_values);
    let offset = attrs.get(ADD_OFFSET).map(attribute_values);
    if scale.is_some() || offset.is_some() {
        let unpacked_type = scale
            .iter()
            .chain(offset.iter())
            .map(|v| v.data_type().clone())
            .find(|t| matches!(t, DataType::Float32 | DataType::Float64))
            .unwrap_or(DataType::Float64);
        out = cast(&out, &unpacked_type)?;
        if let Some(scale) = scale {
            out = mul(&out, &scalar_array(&scale, &unpacked_type)?)?;
        }
        if let Some(offset) = offset {
            out = add(&out, &scalar_array(&offset, &unpacked_type)?)?;
        }
    }

    let mut kept = Attributes::new();
    let mut moved = Attributes::new();
    for (key, value) in attrs.iter() {
        let target = if CF_KEYS.contains(&key) {
            &mut moved
        } else {
            &mut kept
        };
        target.try_insert_array(key, value.clone())?;
    }
    Ok((
        out.with_attributes(kept),
        CfEncoding {
            attributes: moved,
            packed_type: data_type,
        },
    ))
}

fn scalar_array(values: &ArrayRef, to: &DataType) -> Result<NdArrowArray> {
    if values.is_empty() {
        return Err(nd_err!(InvalidAttribute, "empty CF scale or offset"));
    }
    NdArrowArray::try_new_scalar(arrow_cast(values.slice(0, 1).as_ref(), to)?)
}

/// Pack decoded values back with `encoding`, the inverse of [`decode_cf`].
///
/// Computes `(value - offset) / scale`, rounds half to even for integer
/// packed types, casts to the packed type, fills nulls with `_FillValue` (or the
/// first `missing_value`), and restores the CF attributes.
pub fn encode_cf(a: &NdArrowArray, encoding: &CfEncoding) -> Result<NdArrowArray> {
    let attrs = &encoding.attributes;
    let mut out = a.clone();
    let scale = attrs.get(SCALE_FACTOR).map(attribute_values);
    let offset = attrs.get(ADD_OFFSET).map(attribute_values);
    if let Some(offset) = &offset {
        out = sub(&out, &scalar_array(offset, out.data_type())?)?;
    }
    if let Some(scale) = &scale {
        out = div(&out, &scalar_array(scale, out.data_type())?)?;
    }
    let packed_is_integer = encoding.packed_type.is_integer();
    if (scale.is_some() || offset.is_some()) && packed_is_integer {
        out = round(&out)?;
    }
    out = cast(&out, &encoding.packed_type)?;

    let fill = attrs
        .get(FILL_VALUE)
        .or_else(|| attrs.get(MISSING_VALUE))
        .map(attribute_values)
        .filter(|v| !v.is_empty());
    if let Some(fill) = fill {
        out = fill_null(&out, &Arc::new(fill.slice(0, 1)) as &ArrayRef)?;
    }

    let mut merged = a.attributes().clone();
    for (key, value) in attrs.iter() {
        merged.try_insert_array(key, value.clone())?;
    }
    Ok(out.with_attributes(merged))
}

#[cfg(test)]
mod tests {
    use arrow::array::{Float32Array, Int16Array};
    use arrow::datatypes::{Float32Type, Int16Type};

    use super::*;
    use crate::Dimensions;

    fn packed() -> NdArrowArray {
        NdArrowArray::try_new(
            Arc::new(Int16Array::from(vec![-32767, 0, 100, 200, 5000])),
            Dimensions::try_new([("x", 5)]).unwrap(),
        )
        .unwrap()
        .with_attributes(
            Attributes::new()
                .with(FILL_VALUE, -32767i16)
                .with(VALID_RANGE, vec![0i16, 1000])
                .with(SCALE_FACTOR, 0.5f32)
                .with(ADD_OFFSET, 10.0f32)
                .with("units", "degC"),
        )
    }

    #[test]
    fn decode_masks_and_unpacks() {
        let (decoded, encoding) = decode_cf(&packed()).unwrap();
        assert_eq!(decoded.data_type(), &DataType::Float32);
        let values: Vec<Option<f32>> = decoded
            .values()
            .as_primitive::<Float32Type>()
            .iter()
            .collect();
        assert_eq!(
            values,
            vec![None, Some(10.0), Some(60.0), Some(110.0), None]
        );
        assert_eq!(decoded.attributes().get_str("units"), Some("degC"));
        assert!(!decoded.attributes().contains(SCALE_FACTOR));
        assert_eq!(encoding.attributes.len(), 4);
        assert_eq!(encoding.packed_type, DataType::Int16);
    }

    #[test]
    fn encode_inverts_decode() {
        let (decoded, encoding) = decode_cf(&packed()).unwrap();
        let encoded = encode_cf(&decoded, &encoding).unwrap();
        let values: Vec<i16> = encoded
            .values()
            .as_primitive::<Int16Type>()
            .values()
            .to_vec();
        assert_eq!(values, vec![-32767, 0, 100, 200, -32767]);
        assert!(encoded.attributes().contains(SCALE_FACTOR));
        assert_eq!(encoded.attributes().get_str("units"), Some("degC"));
    }

    #[test]
    fn mask_only_keeps_the_type() {
        let a = NdArrowArray::try_new(
            Arc::new(Float32Array::from(vec![1.0, -999.0])),
            Dimensions::try_new([("x", 2)]).unwrap(),
        )
        .unwrap()
        .with_attributes(Attributes::new().with(MISSING_VALUE, vec![-999.0f32, -998.0]));
        let (decoded, _) = decode_cf(&a).unwrap();
        assert_eq!(decoded.data_type(), &DataType::Float32);
        assert_eq!(decoded.values().null_count(), 1);
    }
}
