//! Block and weighted aggregation: `coarsen`, `weighted_sum`, `weighted_mean`, `dot`.

use crate::compute::elementwise::{div, is_not_null_mask, mul, where_};
use crate::compute::reduce::{Reduction, reduce, sum};
use crate::compute::shape::{Selector, Slice, isel};
use crate::dimensions::{Dimension, Dimensions};
use crate::error::{Result, nd_err};
use crate::{IntoScalar, NdArrowArray};

/// What [`coarsen`] does when a dimension is not a multiple of its window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoarsenBoundary {
    /// Fail, like xarray `boundary="exact"`.
    #[default]
    Exact,
    /// Drop the last positions, like xarray `boundary="trim"`.
    Trim,
}

/// Reduce non-overlapping blocks, like `xarray.DataArray.coarsen`.
///
/// Each `(dim, window)` pair splits `dim` into blocks of `window` positions.
/// The split itself is a reshape without a copy.
pub fn coarsen(
    a: &NdArrowArray,
    windows: &[(&str, usize)],
    op: Reduction,
    boundary: CoarsenBoundary,
) -> Result<NdArrowArray> {
    let mut trimmed = a.clone();
    for &(name, window) in windows {
        if window == 0 {
            return Err(nd_err!(
                InvalidArgument,
                "coarsen window must be at least 1"
            ));
        }
        let size = a
            .dims()
            .size_of(name)
            .ok_or_else(|| nd_err!(InvalidArgument, "dimension '{name}' is not in {}", a.dims()))?;
        if size % window != 0 {
            if boundary == CoarsenBoundary::Exact {
                return Err(nd_err!(
                    ShapeMismatch,
                    "dimension '{name}' of size {size} is not a multiple of {window}"
                ));
            }
            let keep = (size - size % window) as isize;
            trimmed = isel(&trimmed, &[(name, Selector::Slice(Slice::range(0, keep)))])?;
        }
    }

    let mut dims: Vec<Dimension> = Vec::new();
    let mut window_dims: Vec<String> = Vec::new();
    for dim in trimmed.dims().iter() {
        match windows.iter().find(|(name, _)| *name == dim.name()) {
            Some(&(name, window)) => {
                let inner = format!("{name}__window");
                if trimmed.dims().contains(&inner) {
                    return Err(nd_err!(
                        InvalidArgument,
                        "dimension '{inner}' already exists"
                    ));
                }
                dims.push(Dimension::new(name, dim.size() / window));
                dims.push(Dimension::new(inner.as_str(), window));
                window_dims.push(inner);
            }
            None => dims.push(dim.clone()),
        }
    }
    let attrs = trimmed.attributes().clone();
    let reshaped = trimmed.reshape(Dimensions::try_new(dims)?)?;
    let names: Vec<&str> = window_dims.iter().map(String::as_str).collect();
    Ok(reduce(&reshaped, Some(&names), op)?.with_attributes(attrs))
}

/// Weighted sum over `axes`, like `xarray.DataArray.weighted(w).sum()`.
/// Null values of `a` count as zero. Result type follows `a * weights`.
pub fn weighted_sum(
    a: &NdArrowArray,
    weights: &NdArrowArray,
    axes: Option<&[&str]>,
) -> Result<NdArrowArray> {
    sum(&mul(a, weights)?, axes)
}

/// Weighted mean over `axes`, like `xarray.DataArray.weighted(w).mean()`.
/// Weights of null values do not count. The result is floating point.
pub fn weighted_mean(
    a: &NdArrowArray,
    weights: &NdArrowArray,
    axes: Option<&[&str]>,
) -> Result<NdArrowArray> {
    let numerator = weighted_sum(a, weights, axes)?;
    let present = is_not_null_mask(a)?;
    let zero = NdArrowArray::try_new_scalar(0.0f64.into_scalar())?;
    let used_weights = where_(&present, weights, &zero)?;
    let denominator = sum(&used_weights, axes)?;
    let nonzero = crate::compute::ne(&denominator, &zero)?;
    let null = NdArrowArray::new_null_scalar(&arrow::datatypes::DataType::Float64);
    where_(&nonzero, &div(&numerator, &denominator)?, &null)
}

/// Sum of products over the named shared dimensions, like `xarray.dot`.
///
/// The product broadcasts first, so memory use follows the broadcast shape.
/// Nulls count as zero.
pub fn dot(a: &NdArrowArray, b: &NdArrowArray, dims: &[&str]) -> Result<NdArrowArray> {
    for name in dims {
        if !(a.dims().contains(name) && b.dims().contains(name)) {
            return Err(nd_err!(
                InvalidArgument,
                "dot dimension '{name}' must be in both arrays"
            ));
        }
    }
    sum(&mul(a, b)?, Some(dims))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{AsArray, Float64Array, Int32Array};
    use arrow::datatypes::{Float64Type, Int64Type};

    use super::*;

    fn nd(values: Vec<f64>, dims: &[(&str, usize)]) -> NdArrowArray {
        NdArrowArray::try_new(
            Arc::new(Float64Array::from(values)),
            Dimensions::try_new(dims.iter().copied()).unwrap(),
        )
        .unwrap()
    }

    fn f64s(a: &NdArrowArray) -> Vec<Option<f64>> {
        a.values().as_primitive::<Float64Type>().iter().collect()
    }

    #[test]
    fn coarsen_blocks() {
        let a = NdArrowArray::try_new(
            Arc::new(Int32Array::from_iter_values(0..16)),
            Dimensions::try_new([("y", 4), ("x", 4)]).unwrap(),
        )
        .unwrap();
        let c = coarsen(
            &a,
            &[("y", 2), ("x", 2)],
            Reduction::Sum,
            CoarsenBoundary::Exact,
        )
        .unwrap();
        assert_eq!(
            c.dims(),
            &Dimensions::try_new([("y", 2), ("x", 2)]).unwrap()
        );
        let v: Vec<i64> = c.values().as_primitive::<Int64Type>().values().to_vec();
        assert_eq!(v, vec![10, 18, 42, 50]);

        assert!(coarsen(&a, &[("x", 3)], Reduction::Sum, CoarsenBoundary::Exact).is_err());
        let t = coarsen(&a, &[("x", 3)], Reduction::Max, CoarsenBoundary::Trim).unwrap();
        assert_eq!(t.shape(), vec![4, 1]);
    }

    #[test]
    fn weighted_mean_ignores_null_weights() {
        let a = NdArrowArray::try_new(
            Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)])),
            Dimensions::try_new([("lat", 3)]).unwrap(),
        )
        .unwrap();
        let w = nd(vec![1.0, 5.0, 3.0], &[("lat", 3)]);
        assert_eq!(f64s(&weighted_sum(&a, &w, None).unwrap()), vec![Some(10.0)]);
        assert_eq!(f64s(&weighted_mean(&a, &w, None).unwrap()), vec![Some(2.5)]);
    }

    #[test]
    fn dot_matches_matrix_product() {
        let a = nd(vec![1.0, 2.0, 3.0, 4.0], &[("i", 2), ("k", 2)]);
        let b = nd(vec![5.0, 6.0, 7.0, 8.0], &[("k", 2), ("j", 2)]);
        let c = dot(&a, &b, &["k"]).unwrap();
        assert_eq!(c.dims().names(), vec!["i", "j"]);
        assert_eq!(
            f64s(&c),
            vec![Some(19.0), Some(22.0), Some(43.0), Some(50.0)]
        );
        assert!(dot(&a, &b, &["i"]).is_err());
    }
}
