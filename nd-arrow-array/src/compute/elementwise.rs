//! Elementwise kernels with name-based broadcasting and NumPy type promotion.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, Datum, Scalar};
use arrow::compute::kernels::zip::zip;
use arrow::compute::kernels::{boolean, cmp, numeric};
use arrow::compute::{cast as arrow_cast, is_not_null, is_null};
use arrow::datatypes::{
    DataType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
};

use crate::NdArrowArray;
use crate::compute::shape::{broadcast_arrays, broadcast_to};
use crate::dimensions::Dimensions;
use crate::error::{Result, nd_err};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Bool,
    Int(u8),
    UInt(u8),
    Float(u8),
}

fn kind(data_type: &DataType) -> Option<Kind> {
    Some(match data_type {
        DataType::Boolean => Kind::Bool,
        DataType::Int8 => Kind::Int(8),
        DataType::Int16 => Kind::Int(16),
        DataType::Int32 => Kind::Int(32),
        DataType::Int64 => Kind::Int(64),
        DataType::UInt8 => Kind::UInt(8),
        DataType::UInt16 => Kind::UInt(16),
        DataType::UInt32 => Kind::UInt(32),
        DataType::UInt64 => Kind::UInt(64),
        DataType::Float16 => Kind::Float(16),
        DataType::Float32 => Kind::Float(32),
        DataType::Float64 => Kind::Float(64),
        _ => return None,
    })
}

fn data_type(kind: Kind) -> DataType {
    match kind {
        Kind::Bool => DataType::Boolean,
        Kind::Int(8) => DataType::Int8,
        Kind::Int(16) => DataType::Int16,
        Kind::Int(32) => DataType::Int32,
        Kind::Int(_) => DataType::Int64,
        Kind::UInt(8) => DataType::UInt8,
        Kind::UInt(16) => DataType::UInt16,
        Kind::UInt(32) => DataType::UInt32,
        Kind::UInt(_) => DataType::UInt64,
        Kind::Float(16) => DataType::Float16,
        Kind::Float(32) => DataType::Float32,
        Kind::Float(_) => DataType::Float64,
    }
}

/// The NumPy result type of two numeric types, like `numpy.result_type`.
///
/// Returns `None` for pairs that are not both numeric or boolean. Kernels pass
/// those types to Arrow unchanged, so temporal arithmetic keeps working.
pub fn result_type(a: &DataType, b: &DataType) -> Option<DataType> {
    if a == b {
        return Some(a.clone());
    }
    let float_bits_for = |bits: u8| match bits {
        8 => 16,
        16 => 32,
        _ => 64,
    };
    let kind = match (kind(a)?, kind(b)?) {
        (Kind::Bool, other) | (other, Kind::Bool) => other,
        (Kind::Int(x), Kind::Int(y)) => Kind::Int(x.max(y)),
        (Kind::UInt(x), Kind::UInt(y)) => Kind::UInt(x.max(y)),
        (Kind::Int(s), Kind::UInt(u)) | (Kind::UInt(u), Kind::Int(s)) => {
            if s > u {
                Kind::Int(s)
            } else if u < 64 {
                Kind::Int(u * 2)
            } else {
                Kind::Float(64)
            }
        }
        (Kind::Float(x), Kind::Float(y)) => Kind::Float(x.max(y)),
        (Kind::Float(f), Kind::Int(n) | Kind::UInt(n))
        | (Kind::Int(n) | Kind::UInt(n), Kind::Float(f)) => Kind::Float(f.max(float_bits_for(n))),
    };
    Some(data_type(kind))
}

/// The floating point type that math functions compute in, like NumPy.
fn float_type(data_type: &DataType) -> Result<DataType> {
    match kind(data_type) {
        Some(Kind::Float(64)) => Ok(DataType::Float64),
        Some(Kind::Float(_)) => Ok(DataType::Float32),
        Some(_) => Ok(DataType::Float64),
        None => Err(nd_err!(
            NotSupported,
            "math function on non-numeric type {data_type}"
        )),
    }
}

fn cast_values(values: &ArrayRef, to: &DataType) -> Result<ArrayRef> {
    if values.data_type() == to {
        Ok(values.clone())
    } else {
        Ok(arrow_cast(values.as_ref(), to)?)
    }
}

enum Operand {
    Array(ArrayRef),
    Scalar(Scalar<ArrayRef>),
}

impl Operand {
    fn datum(&self) -> &dyn Datum {
        match self {
            Operand::Array(array) => array,
            Operand::Scalar(scalar) => scalar,
        }
    }
}

/// Cast both sides to `to` and align them. Rank-0 sides stay Arrow scalars.
fn prepare(
    a: &NdArrowArray,
    b: &NdArrowArray,
    to: Option<&DataType>,
) -> Result<(Operand, Operand, Dimensions)> {
    let cast = |x: &NdArrowArray| match to {
        Some(t) => cast_values(x.values(), t),
        None => Ok(x.values().clone()),
    };
    if a.dims() == b.dims() {
        return Ok((
            Operand::Array(cast(a)?),
            Operand::Array(cast(b)?),
            a.dims().clone(),
        ));
    }
    if b.is_scalar() {
        return Ok((
            Operand::Array(cast(a)?),
            Operand::Scalar(Scalar::new(cast(b)?)),
            a.dims().clone(),
        ));
    }
    if a.is_scalar() {
        return Ok((
            Operand::Scalar(Scalar::new(cast(a)?)),
            Operand::Array(cast(b)?),
            b.dims().clone(),
        ));
    }
    let aligned = broadcast_arrays(&[a, b])?;
    Ok((
        Operand::Array(cast(&aligned[0])?),
        Operand::Array(cast(&aligned[1])?),
        aligned[0].dims().clone(),
    ))
}

fn wrap(values: ArrayRef, dims: Dimensions) -> Result<NdArrowArray> {
    NdArrowArray::try_new(values, dims)
}

fn arithmetic(
    a: &NdArrowArray,
    b: &NdArrowArray,
    to: Option<DataType>,
    op: impl Fn(&dyn Datum, &dyn Datum) -> std::result::Result<ArrayRef, arrow::error::ArrowError>,
) -> Result<NdArrowArray> {
    let (x, y, dims) = prepare(a, b, to.as_ref())?;
    wrap(op(x.datum(), y.datum())?, dims)
}

/// `a + b`. Integer overflow wraps, like NumPy.
pub fn add(a: &NdArrowArray, b: &NdArrowArray) -> Result<NdArrowArray> {
    arithmetic(
        a,
        b,
        result_type(a.data_type(), b.data_type()),
        numeric::add_wrapping,
    )
}

/// `a - b`. Integer overflow wraps, like NumPy.
pub fn sub(a: &NdArrowArray, b: &NdArrowArray) -> Result<NdArrowArray> {
    arithmetic(
        a,
        b,
        result_type(a.data_type(), b.data_type()),
        numeric::sub_wrapping,
    )
}

/// `a * b`. Integer overflow wraps, like NumPy.
pub fn mul(a: &NdArrowArray, b: &NdArrowArray) -> Result<NdArrowArray> {
    arithmetic(
        a,
        b,
        result_type(a.data_type(), b.data_type()),
        numeric::mul_wrapping,
    )
}

/// True division `a / b`, like `numpy.true_divide`: integers divide as `Float64`.
pub fn div(a: &NdArrowArray, b: &NdArrowArray) -> Result<NdArrowArray> {
    let to = match result_type(a.data_type(), b.data_type()) {
        Some(t) if matches!(kind(&t), Some(Kind::Float(_))) => Some(t),
        Some(_) => Some(DataType::Float64),
        None => None,
    };
    arithmetic(a, b, to, numeric::div)
}

/// Remainder with the sign of `a`, like `numpy.fmod`. Integer division by zero is an error.
pub fn rem(a: &NdArrowArray, b: &NdArrowArray) -> Result<NdArrowArray> {
    arithmetic(
        a,
        b,
        result_type(a.data_type(), b.data_type()),
        numeric::rem,
    )
}

/// `a ** b` in floating point, like `numpy.float_power`.
pub fn pow(a: &NdArrowArray, b: &NdArrowArray) -> Result<NdArrowArray> {
    let to = result_type(a.data_type(), b.data_type()).ok_or_else(|| {
        nd_err!(
            NotSupported,
            "pow on {} and {}",
            a.data_type(),
            b.data_type()
        )
    })?;
    let to = float_type(&to)?;
    let aligned = broadcast_arrays(&[a, b])?;
    let x = cast_values(aligned[0].values(), &to)?;
    let y = cast_values(aligned[1].values(), &to)?;
    let values: ArrayRef = match to {
        DataType::Float32 => Arc::new(arrow::compute::binary::<_, _, _, Float32Type>(
            x.as_primitive::<Float32Type>(),
            y.as_primitive::<Float32Type>(),
            f32::powf,
        )?),
        _ => Arc::new(arrow::compute::binary::<_, _, _, Float64Type>(
            x.as_primitive::<Float64Type>(),
            y.as_primitive::<Float64Type>(),
            f64::powf,
        )?),
    };
    wrap(values, aligned[0].dims().clone())
}

fn comparison(
    a: &NdArrowArray,
    b: &NdArrowArray,
    op: impl Fn(&dyn Datum, &dyn Datum) -> std::result::Result<BooleanArray, arrow::error::ArrowError>,
) -> Result<NdArrowArray> {
    let to = result_type(a.data_type(), b.data_type());
    let (x, y, dims) = prepare(a, b, to.as_ref())?;
    wrap(Arc::new(op(x.datum(), y.datum())?), dims)
}

/// `a == b`.
pub fn eq(a: &NdArrowArray, b: &NdArrowArray) -> Result<NdArrowArray> {
    comparison(a, b, cmp::eq)
}

/// `a != b`.
pub fn ne(a: &NdArrowArray, b: &NdArrowArray) -> Result<NdArrowArray> {
    comparison(a, b, cmp::neq)
}

/// `a < b`.
pub fn lt(a: &NdArrowArray, b: &NdArrowArray) -> Result<NdArrowArray> {
    comparison(a, b, cmp::lt)
}

/// `a <= b`.
pub fn le(a: &NdArrowArray, b: &NdArrowArray) -> Result<NdArrowArray> {
    comparison(a, b, cmp::lt_eq)
}

/// `a > b`.
pub fn gt(a: &NdArrowArray, b: &NdArrowArray) -> Result<NdArrowArray> {
    comparison(a, b, cmp::gt)
}

/// `a >= b`.
pub fn ge(a: &NdArrowArray, b: &NdArrowArray) -> Result<NdArrowArray> {
    comparison(a, b, cmp::gt_eq)
}

fn boolean_pair(
    a: &NdArrowArray,
    b: &NdArrowArray,
    op: impl Fn(
        &BooleanArray,
        &BooleanArray,
    ) -> std::result::Result<BooleanArray, arrow::error::ArrowError>,
) -> Result<NdArrowArray> {
    for x in [a, b] {
        if x.data_type() != &DataType::Boolean {
            return Err(nd_err!(
                InvalidArgument,
                "logical operation needs Boolean values, got {}",
                x.data_type()
            ));
        }
    }
    let aligned = broadcast_arrays(&[a, b])?;
    let values = op(
        aligned[0].values().as_boolean(),
        aligned[1].values().as_boolean(),
    )?;
    wrap(Arc::new(values), aligned[0].dims().clone())
}

/// Logical AND with Kleene null logic (`false AND null` is `false`).
pub fn logical_and(a: &NdArrowArray, b: &NdArrowArray) -> Result<NdArrowArray> {
    boolean_pair(a, b, boolean::and_kleene)
}

/// Logical OR with Kleene null logic (`true OR null` is `true`).
pub fn logical_or(a: &NdArrowArray, b: &NdArrowArray) -> Result<NdArrowArray> {
    boolean_pair(a, b, boolean::or_kleene)
}

/// Logical XOR. A null input gives a null.
pub fn logical_xor(a: &NdArrowArray, b: &NdArrowArray) -> Result<NdArrowArray> {
    boolean_pair(a, b, |x, y| cmp::neq(x, y))
}

/// Logical NOT.
pub fn logical_not(a: &NdArrowArray) -> Result<NdArrowArray> {
    let values = a
        .values()
        .as_boolean_opt()
        .ok_or_else(|| nd_err!(InvalidArgument, "logical_not needs Boolean values"))?;
    wrap(Arc::new(boolean::not(values)?), a.dims().clone())
}

/// `true` where the value is null.
pub fn is_null_mask(a: &NdArrowArray) -> Result<NdArrowArray> {
    wrap(Arc::new(is_null(a.values().as_ref())?), a.dims().clone())
}

/// `true` where the value is not null.
pub fn is_not_null_mask(a: &NdArrowArray) -> Result<NdArrowArray> {
    wrap(
        Arc::new(is_not_null(a.values().as_ref())?),
        a.dims().clone(),
    )
}

/// `-a`. Integer overflow wraps, like NumPy.
pub fn neg(a: &NdArrowArray) -> Result<NdArrowArray> {
    wrap(
        numeric::neg_wrapping(a.values().as_ref())?,
        a.dims().clone(),
    )
}

/// `|a|`, in the input type. `abs(i8::MIN)` wraps to `i8::MIN`, like NumPy.
pub fn abs(a: &NdArrowArray) -> Result<NdArrowArray> {
    let v = a.values();
    let values: ArrayRef = match v.data_type() {
        DataType::Int8 => Arc::new(
            v.as_primitive::<Int8Type>()
                .unary::<_, Int8Type>(i8::wrapping_abs),
        ),
        DataType::Int16 => Arc::new(
            v.as_primitive::<Int16Type>()
                .unary::<_, Int16Type>(i16::wrapping_abs),
        ),
        DataType::Int32 => Arc::new(
            v.as_primitive::<Int32Type>()
                .unary::<_, Int32Type>(i32::wrapping_abs),
        ),
        DataType::Int64 => Arc::new(
            v.as_primitive::<Int64Type>()
                .unary::<_, Int64Type>(i64::wrapping_abs),
        ),
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => v.clone(),
        DataType::Float32 => Arc::new(
            v.as_primitive::<Float32Type>()
                .unary::<_, Float32Type>(f32::abs),
        ),
        DataType::Float64 => Arc::new(
            v.as_primitive::<Float64Type>()
                .unary::<_, Float64Type>(f64::abs),
        ),
        other => return Err(nd_err!(NotSupported, "abs on {other}")),
    };
    wrap(values, a.dims().clone())
}

/// Apply a floating point function. Integers compute as `Float64`, like NumPy.
pub fn map_float(
    a: &NdArrowArray,
    f32_op: impl Fn(f32) -> f32,
    f64_op: impl Fn(f64) -> f64,
) -> Result<NdArrowArray> {
    let to = float_type(a.data_type())?;
    let values = cast_values(a.values(), &to)?;
    let out: ArrayRef = match to {
        DataType::Float32 => Arc::new(
            values
                .as_primitive::<Float32Type>()
                .unary::<_, Float32Type>(f32_op),
        ),
        _ => Arc::new(
            values
                .as_primitive::<Float64Type>()
                .unary::<_, Float64Type>(f64_op),
        ),
    };
    wrap(out, a.dims().clone())
}

fn float_predicate(
    a: &NdArrowArray,
    f32_op: impl Fn(f32) -> bool,
    f64_op: impl Fn(f64) -> bool,
) -> Result<NdArrowArray> {
    let to = float_type(a.data_type())?;
    let values = cast_values(a.values(), &to)?;
    let out = match to {
        DataType::Float32 => BooleanArray::from_unary(values.as_primitive::<Float32Type>(), f32_op),
        _ => BooleanArray::from_unary(values.as_primitive::<Float64Type>(), f64_op),
    };
    wrap(Arc::new(out), a.dims().clone())
}

macro_rules! float_functions {
    ($($(#[$doc:meta])* $name:ident => $method:ident),* $(,)?) => {$(
        $(#[$doc])*
        pub fn $name(a: &NdArrowArray) -> Result<NdArrowArray> {
            map_float(a, f32::$method, f64::$method)
        }
    )*};
}

float_functions!(
    /// Square root, like `numpy.sqrt`.
    sqrt => sqrt,
    /// Cube root, like `numpy.cbrt`.
    cbrt => cbrt,
    /// `e^a`, like `numpy.exp`.
    exp => exp,
    /// Natural logarithm, like `numpy.log`.
    log => ln,
    /// Base-2 logarithm, like `numpy.log2`.
    log2 => log2,
    /// Base-10 logarithm, like `numpy.log10`.
    log10 => log10,
    /// `ln(1 + a)`, like `numpy.log1p`.
    log1p => ln_1p,
    /// `e^a - 1`, like `numpy.expm1`.
    expm1 => exp_m1,
    /// Sine, like `numpy.sin`.
    sin => sin,
    /// Cosine, like `numpy.cos`.
    cos => cos,
    /// Tangent, like `numpy.tan`.
    tan => tan,
    /// Inverse sine, like `numpy.arcsin`.
    arcsin => asin,
    /// Inverse cosine, like `numpy.arccos`.
    arccos => acos,
    /// Inverse tangent, like `numpy.arctan`.
    arctan => atan,
    /// Hyperbolic sine, like `numpy.sinh`.
    sinh => sinh,
    /// Hyperbolic cosine, like `numpy.cosh`.
    cosh => cosh,
    /// Hyperbolic tangent, like `numpy.tanh`.
    tanh => tanh,
    /// Round down, like `numpy.floor`.
    floor => floor,
    /// Round up, like `numpy.ceil`.
    ceil => ceil,
    /// Round toward zero, like `numpy.trunc`.
    trunc => trunc,
    /// Round half to even, like `numpy.round` with zero decimals.
    round => round_ties_even,
);

/// Degrees to radians, like `numpy.deg2rad`.
pub fn deg2rad(a: &NdArrowArray) -> Result<NdArrowArray> {
    map_float(a, f32::to_radians, f64::to_radians)
}

/// Radians to degrees, like `numpy.rad2deg`.
pub fn rad2deg(a: &NdArrowArray) -> Result<NdArrowArray> {
    map_float(a, f32::to_degrees, f64::to_degrees)
}

/// `true` where the value is NaN, like `numpy.isnan`. Nulls stay null.
pub fn isnan(a: &NdArrowArray) -> Result<NdArrowArray> {
    float_predicate(a, f32::is_nan, f64::is_nan)
}

/// `true` where the value is infinite, like `numpy.isinf`.
pub fn isinf(a: &NdArrowArray) -> Result<NdArrowArray> {
    float_predicate(a, f32::is_infinite, f64::is_infinite)
}

/// `true` where the value is finite, like `numpy.isfinite`.
pub fn isfinite(a: &NdArrowArray) -> Result<NdArrowArray> {
    float_predicate(a, f32::is_finite, f64::is_finite)
}

/// Replace NaN with null, so that NaN counts as missing in all kernels.
pub fn nan_to_null(a: &NdArrowArray) -> Result<NdArrowArray> {
    if !matches!(kind(a.data_type()), Some(Kind::Float(_))) {
        return Ok(a.clone());
    }
    let mask = isnan(a)?;
    let values = arrow::compute::nullif(a.values().as_ref(), mask.values().as_boolean())?;
    Ok(wrap(values, a.dims().clone())?.with_attributes(a.attributes().clone()))
}

/// Cast the values, like `ndarray.astype`. Keeps the attributes.
pub fn cast(a: &NdArrowArray, to: &DataType) -> Result<NdArrowArray> {
    Ok(wrap(cast_values(a.values(), to)?, a.dims().clone())?
        .with_attributes(a.attributes().clone()))
}

fn scalar_operand(x: &NdArrowArray, target: &Dimensions, to: &DataType) -> Result<Operand> {
    if x.is_scalar() {
        return Ok(Operand::Scalar(Scalar::new(cast_values(x.values(), to)?)));
    }
    let aligned = broadcast_to(x, target)?;
    Ok(Operand::Array(cast_values(aligned.values(), to)?))
}

/// `cond ? x : y`, like `numpy.where`. A null condition picks `y`.
/// Keeps the attributes of `x`.
pub fn where_(cond: &NdArrowArray, x: &NdArrowArray, y: &NdArrowArray) -> Result<NdArrowArray> {
    if cond.data_type() != &DataType::Boolean {
        return Err(nd_err!(
            InvalidArgument,
            "where condition must be Boolean, got {}",
            cond.data_type()
        ));
    }
    let target = Dimensions::broadcast(&[cond.dims(), x.dims(), y.dims()])?;
    let to = result_type(x.data_type(), y.data_type()).unwrap_or_else(|| x.data_type().clone());
    let mask = broadcast_to(cond, &target)?;
    let truthy = scalar_operand(x, &target, &to)?;
    let falsy = scalar_operand(y, &target, &to)?;
    let values = if target.is_scalar() {
        let pick = if mask.values().as_boolean().is_valid(0) && mask.values().as_boolean().value(0)
        {
            &truthy
        } else {
            &falsy
        };
        match pick {
            Operand::Array(a) => a.clone(),
            Operand::Scalar(s) => s.clone().into_inner(),
        }
    } else {
        zip(mask.values().as_boolean(), truthy.datum(), falsy.datum())?
    };
    Ok(wrap(values, target)?.with_attributes(x.attributes().clone()))
}

/// Replace nulls with `value`, a length-1 array. Keeps the attributes.
pub fn fill_null(a: &NdArrowArray, value: &ArrayRef) -> Result<NdArrowArray> {
    if value.len() != 1 {
        return Err(nd_err!(InvalidArgument, "fill value must have length 1"));
    }
    if a.values().null_count() == 0 {
        return Ok(a.clone());
    }
    let fill = Scalar::new(cast_values(value, a.data_type())?);
    let valid = is_not_null(a.values().as_ref())?;
    Ok(wrap(zip(&valid, a.values(), &fill)?, a.dims().clone())?
        .with_attributes(a.attributes().clone()))
}

/// Limit values to `[min, max]`, like `numpy.clip`. Keeps the attributes.
pub fn clip(
    a: &NdArrowArray,
    min: Option<&ArrayRef>,
    max: Option<&ArrayRef>,
) -> Result<NdArrowArray> {
    let mut values = a.values().clone();
    for (bound, below) in [(min, true), (max, false)] {
        let Some(bound) = bound else { continue };
        let bound = Scalar::new(cast_values(bound, a.data_type())?);
        let outside = if below {
            cmp::lt(&values, &bound)?
        } else {
            cmp::gt(&values, &bound)?
        };
        values = zip(&outside, &bound, &values)?;
    }
    Ok(wrap(values, a.dims().clone())?.with_attributes(a.attributes().clone()))
}

#[cfg(test)]
mod tests {
    use arrow::array::{Float32Array, Float64Array, Int8Array, Int32Array, UInt8Array};
    use arrow::datatypes::{Float64Type, Int32Type};

    use super::*;
    use crate::{Attributes, IntoScalar};

    fn nd(values: ArrayRef, dims: &[(&str, usize)]) -> NdArrowArray {
        NdArrowArray::try_new(values, Dimensions::try_new(dims.iter().copied()).unwrap()).unwrap()
    }

    fn scalar(value: impl IntoScalar) -> NdArrowArray {
        NdArrowArray::try_new_scalar(value.into_scalar()).unwrap()
    }

    fn f64s(a: &NdArrowArray) -> Vec<Option<f64>> {
        a.values().as_primitive::<Float64Type>().iter().collect()
    }

    #[test]
    fn result_type_follows_numpy() {
        use DataType::*;
        let cases = [
            (Int8, Int16, Int16),
            (Int8, UInt8, Int16),
            (Int64, UInt64, Float64),
            (UInt8, Int32, Int32),
            (Float32, Int16, Float32),
            (Float32, Int32, Float64),
            (Float16, Int8, Float16),
            (Boolean, Int8, Int8),
            (Float32, Float64, Float64),
        ];
        for (a, b, expected) in cases {
            assert_eq!(result_type(&a, &b), Some(expected.clone()), "{a} + {b}");
            assert_eq!(result_type(&b, &a), Some(expected), "{b} + {a}");
        }
        assert_eq!(result_type(&Utf8, &Int8), None);
    }

    #[test]
    fn add_broadcasts_by_name() {
        let a = nd(Arc::new(Int32Array::from(vec![1, 2, 3])), &[("lat", 3)]);
        let b = nd(Arc::new(Int32Array::from(vec![10, 20])), &[("time", 2)]);
        let c = add(&a, &b).unwrap();
        assert_eq!(c.dims().names(), vec!["lat", "time"]);
        assert_eq!(
            c.values().as_primitive::<Int32Type>().values(),
            &[11, 21, 12, 22, 13, 23]
        );
    }

    #[test]
    fn scalar_operands_and_promotion() {
        let a = nd(Arc::new(Int8Array::from(vec![1, 2])), &[("x", 2)]);
        let c = mul(&a, &scalar(1.5f32)).unwrap();
        assert_eq!(c.data_type(), &DataType::Float32);
        let d = div(&a, &scalar(2i8)).unwrap();
        assert_eq!(f64s(&d), vec![Some(0.5), Some(1.0)]);
        let wrapped = add(&nd(Arc::new(Int8Array::from(vec![127])), &[]), &scalar(1i8)).unwrap();
        assert_eq!(wrapped.values().as_primitive::<Int8Type>().value(0), -128);
        let u = sub(&nd(Arc::new(UInt8Array::from(vec![1])), &[]), &scalar(2i8)).unwrap();
        assert_eq!(u.data_type(), &DataType::Int16);
    }

    #[test]
    fn nulls_propagate_and_fill() {
        let a = nd(
            Arc::new(Float64Array::from(vec![Some(1.0), None])),
            &[("x", 2)],
        );
        let b = add(&a, &scalar(1.0f64)).unwrap();
        assert_eq!(f64s(&b), vec![Some(2.0), None]);
        let filled = fill_null(&b, &0.0f64.into_scalar()).unwrap();
        assert_eq!(f64s(&filled), vec![Some(2.0), Some(0.0)]);
    }

    #[test]
    fn comparison_and_logic() {
        let a = nd(Arc::new(Int32Array::from(vec![1, 5, 3])), &[("x", 3)]);
        let gt3 = gt(&a, &scalar(2i32)).unwrap();
        let lt5 = lt(&a, &scalar(5i64)).unwrap();
        let both = logical_and(&gt3, &lt5).unwrap();
        let v: Vec<bool> = both
            .values()
            .as_boolean()
            .iter()
            .map(Option::unwrap)
            .collect();
        assert_eq!(v, vec![false, false, true]);
        let not = logical_not(&both).unwrap();
        assert!(not.values().as_boolean().value(0));
        assert!(logical_and(&a, &a).is_err());
    }

    #[test]
    fn math_functions_promote_to_float() {
        let a = nd(Arc::new(Int32Array::from(vec![1, 4, 9])), &[("x", 3)]);
        assert_eq!(
            f64s(&sqrt(&a).unwrap()),
            vec![Some(1.0), Some(2.0), Some(3.0)]
        );
        let f = nd(
            Arc::new(Float32Array::from(vec![2.5, -0.5, 1.5])),
            &[("x", 3)],
        );
        let r = round(&f).unwrap();
        assert_eq!(
            r.values().as_primitive::<Float32Type>().values(),
            &[2.0, -0.0, 2.0]
        );
        let ab = abs(&nd(Arc::new(Int32Array::from(vec![-3, 2])), &[("x", 2)])).unwrap();
        assert_eq!(ab.values().as_primitive::<Int32Type>().values(), &[3, 2]);
    }

    #[test]
    fn where_clip_and_nan() {
        let a = nd(
            Arc::new(Float64Array::from(vec![1.0, f64::NAN, 7.0])),
            &[("x", 3)],
        );
        let a = a.with_attributes(Attributes::new().with("units", "K"));
        let cond = gt(&a, &scalar(2.0f64)).unwrap();
        let w = where_(&cond, &a, &scalar(0.0f64)).unwrap();
        assert_eq!(w.attributes().get_str("units"), Some("K"));
        let v = f64s(&w);
        assert_eq!(v[0], Some(0.0));
        assert_eq!(v[2], Some(7.0));

        let c = clip(&a, Some(&2.0f64.into_scalar()), Some(&5.0f64.into_scalar())).unwrap();
        assert_eq!(f64s(&c)[0], Some(2.0));
        assert_eq!(f64s(&c)[2], Some(5.0));

        let n = nan_to_null(&a).unwrap();
        assert_eq!(n.values().null_count(), 1);
    }

    #[test]
    fn pow_broadcasts() {
        let a = nd(Arc::new(Int32Array::from(vec![2, 3])), &[("x", 2)]);
        let p = pow(&a, &scalar(2i32)).unwrap();
        assert_eq!(f64s(&p), vec![Some(4.0), Some(9.0)]);
    }
}
