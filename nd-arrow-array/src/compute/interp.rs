//! Linear interpolation along one dimension.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Float64Array};
use arrow::compute::cast as arrow_cast;
use arrow::datatypes::{DataType, Float64Type};

use crate::NdArrowArray;
use crate::compute::search::partition_point;
use crate::dimensions::{Dimension, Dimensions};
use crate::error::{Result, nd_err};

fn to_f64(values: &ArrayRef) -> Result<Float64Array> {
    Ok(arrow_cast(values.as_ref(), &DataType::Float64)?
        .as_primitive::<Float64Type>()
        .clone())
}

/// Interpolate `a` linearly along `dim` from coordinates `x` to `new_x`,
/// like `xarray.DataArray.interp` with `method="linear"`.
///
/// `x` is 1-D, strictly increasing and as long as `dim`. `new_x` is 1-D. The
/// result keeps the axis order and has `new_x.len()` elements along `dim`.
/// Points outside `[x[0], x[-1]]`, null points and points next to a null value
/// give null. The result is `Float64` (`Float32` for `Float32` input).
pub fn interp(
    a: &NdArrowArray,
    dim: &str,
    x: &NdArrowArray,
    new_x: &NdArrowArray,
) -> Result<NdArrowArray> {
    let axis = a.dims().axis(dim)?;
    let n = a.dims().as_slice()[axis].size();
    if x.rank() != 1 || x.len() != n {
        return Err(nd_err!(
            InvalidArgument,
            "interp coordinates must be 1-D with {n} values along '{dim}'"
        ));
    }
    if new_x.rank() != 1 {
        return Err(nd_err!(InvalidArgument, "interp targets must be 1-D"));
    }
    let xs = to_f64(x.values())?;
    if xs.null_count() > 0 || xs.values().windows(2).any(|w| w[1] <= w[0]) {
        return Err(nd_err!(
            InvalidArgument,
            "interp coordinates must be non-null and strictly increasing"
        ));
    }
    let xs = xs.values();
    let targets = to_f64(new_x.values())?;

    // Per target: (lower index, weight of the upper neighbor), or None.
    let plan: Vec<Option<(usize, f64)>> = targets
        .iter()
        .map(|t| {
            let t = t?;
            if n == 0 || t < xs[0] || t > xs[n - 1] {
                return None;
            }
            let upper = partition_point(n, |i| xs[i] < t);
            if xs[upper] == t {
                return Some((upper, 0.0));
            }
            let lower = upper - 1;
            Some((lower, (t - xs[lower]) / (xs[upper] - xs[lower])))
        })
        .collect();

    let values = to_f64(a.values())?;
    let shape = a.shape();
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let m = plan.len();
    let mut out: Vec<Option<f64>> = Vec::with_capacity(outer * m * inner);
    let get = |p: usize| values.is_valid(p).then(|| values.value(p));
    for o in 0..outer {
        for step in &plan {
            for j in 0..inner {
                out.push(step.and_then(|(lower, w)| {
                    let low = get((o * n + lower) * inner + j)?;
                    if w == 0.0 {
                        return Some(low);
                    }
                    let high = get((o * n + lower + 1) * inner + j)?;
                    Some(low + (high - low) * w)
                }));
            }
        }
    }

    let mut result: ArrayRef = Arc::new(Float64Array::from(out));
    if matches!(a.data_type(), DataType::Float32 | DataType::Float16) {
        result = arrow_cast(result.as_ref(), &DataType::Float32)?;
    }
    let dims = a.dims().iter().map(|d| {
        if d.name() == dim {
            Dimension::new(dim, m)
        } else {
            d.clone()
        }
    });
    Ok(NdArrowArray::try_new(result, Dimensions::try_new(dims)?)?
        .with_attributes(a.attributes().clone()))
}

#[cfg(test)]
mod tests {
    use arrow::array::Int32Array;

    use super::*;

    fn nd(values: ArrayRef, dims: &[(&str, usize)]) -> NdArrowArray {
        NdArrowArray::try_new(values, Dimensions::try_new(dims.iter().copied()).unwrap()).unwrap()
    }

    #[test]
    fn interpolates_each_lane() {
        let a = nd(
            Arc::new(Int32Array::from(vec![0, 10, 20, 100, 110, 120])),
            &[("y", 2), ("x", 3)],
        );
        let x = nd(
            Arc::new(Float64Array::from(vec![0.0, 1.0, 3.0])),
            &[("x", 3)],
        );
        let new_x = nd(
            Arc::new(Float64Array::from(vec![0.5, 2.0, 3.0, 4.0])),
            &[("x", 4)],
        );
        let out = interp(&a, "x", &x, &new_x).unwrap();
        assert_eq!(out.shape(), vec![2, 4]);
        let v: Vec<Option<f64>> = out.values().as_primitive::<Float64Type>().iter().collect();
        assert_eq!(
            v,
            vec![
                Some(5.0),
                Some(15.0),
                Some(20.0),
                None,
                Some(105.0),
                Some(115.0),
                Some(120.0),
                None
            ]
        );
    }

    #[test]
    fn rejects_unsorted_coordinates() {
        let a = nd(Arc::new(Int32Array::from(vec![1, 2])), &[("x", 2)]);
        let x = nd(Arc::new(Float64Array::from(vec![1.0, 0.0])), &[("x", 2)]);
        assert!(interp(&a, "x", &x, &x).is_err());
    }
}
