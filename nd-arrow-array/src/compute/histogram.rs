//! Counting kernels: `histogram` and `bincount`.

use std::sync::Arc;

use arrow::array::{Array, AsArray, Float64Array, Int64Array};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type, Int64Type};

use crate::NdArrowArray;
use crate::dimensions::Dimensions;
use crate::error::{Result, nd_err};

/// Count all elements of `a` per bin, like `numpy.histogram` with explicit edges.
///
/// `edges` is 1-D and increasing. Bins are half open, `[edges[i], edges[i+1])`,
/// except the last bin, which includes its right edge. Nulls, NaN and values
/// outside the edges are not counted. The result is `Int64` along `dim`.
pub fn histogram(a: &NdArrowArray, edges: &NdArrowArray, dim: &str) -> Result<NdArrowArray> {
    let edges = cast(edges.values().as_ref(), &DataType::Float64)?;
    let edges = edges.as_primitive::<Float64Type>();
    if edges.len() < 2 || edges.null_count() > 0 || edges.values().windows(2).any(|w| w[1] <= w[0])
    {
        return Err(nd_err!(
            InvalidArgument,
            "histogram edges must be at least two increasing, non-null values"
        ));
    }
    let edges = edges.values();
    let bins = edges.len() - 1;
    let mut counts = vec![0i64; bins];
    let values = cast(a.values().as_ref(), &DataType::Float64)?;
    for v in values.as_primitive::<Float64Type>().iter().flatten() {
        if v.is_nan() || v < edges[0] || v > edges[bins] {
            continue;
        }
        let bin = edges
            .partition_point(|&e| e <= v)
            .saturating_sub(1)
            .min(bins - 1);
        counts[bin] += 1;
    }
    NdArrowArray::try_new(
        Arc::new(Int64Array::from(counts)),
        Dimensions::try_new([(dim, bins)])?,
    )
}

/// Count each non-negative integer value, like `numpy.bincount`.
///
/// With `weights` (same dimensions as `a`), sums the weights instead and
/// returns `Float64`. The result has at least `min_length` positions along
/// `dim`. Null values are not counted. Negative values are an error.
pub fn bincount(
    a: &NdArrowArray,
    weights: Option<&NdArrowArray>,
    min_length: usize,
    dim: &str,
) -> Result<NdArrowArray> {
    if !a.data_type().is_integer() {
        return Err(nd_err!(
            InvalidArgument,
            "bincount needs integers, got {}",
            a.data_type()
        ));
    }
    let values = cast(a.values().as_ref(), &DataType::Int64)?;
    let values = values.as_primitive::<Int64Type>();
    if values.iter().flatten().any(|v| v < 0) {
        return Err(nd_err!(
            InvalidArgument,
            "bincount values must be non-negative"
        ));
    }
    let len = values
        .iter()
        .flatten()
        .max()
        .map_or(0, |m| m as usize + 1)
        .max(min_length);
    let dims = Dimensions::try_new([(dim, len)])?;
    match weights {
        None => {
            let mut counts = vec![0i64; len];
            for v in values.iter().flatten() {
                counts[v as usize] += 1;
            }
            NdArrowArray::try_new(Arc::new(Int64Array::from(counts)), dims)
        }
        Some(w) => {
            if w.dims() != a.dims() {
                return Err(nd_err!(
                    ShapeMismatch,
                    "bincount weights {} do not match values {}",
                    w.dims(),
                    a.dims()
                ));
            }
            let w = cast(w.values().as_ref(), &DataType::Float64)?;
            let w = w.as_primitive::<Float64Type>();
            let mut sums = vec![0.0f64; len];
            for (v, weight) in values.iter().zip(w.iter()) {
                if let (Some(v), Some(weight)) = (v, weight) {
                    sums[v as usize] += weight;
                }
            }
            NdArrowArray::try_new(Arc::new(Float64Array::from(sums)), dims)
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::Int32Array;

    use super::*;

    fn nd(values: Vec<f64>, dims: &[(&str, usize)]) -> NdArrowArray {
        NdArrowArray::try_new(
            Arc::new(Float64Array::from(values)),
            Dimensions::try_new(dims.iter().copied()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn histogram_matches_numpy() {
        let a = nd(
            vec![0.5, 1.0, 1.5, 2.0, 2.0, 3.0, f64::NAN, -1.0],
            &[("x", 8)],
        );
        let edges = nd(vec![0.0, 1.0, 2.0], &[("e", 3)]);
        let h = histogram(&a, &edges, "bin").unwrap();
        assert_eq!(h.values().as_primitive::<Int64Type>().values(), &[1, 4]);
        assert!(histogram(&a, &nd(vec![1.0], &[("e", 1)]), "bin").is_err());
    }

    #[test]
    fn bincount_with_and_without_weights() {
        let a = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![Some(0), Some(2), Some(2), None])),
            Dimensions::try_new([("x", 4)]).unwrap(),
        )
        .unwrap();
        let c = bincount(&a, None, 5, "value").unwrap();
        assert_eq!(
            c.values().as_primitive::<Int64Type>().values(),
            &[1, 0, 2, 0, 0]
        );
        let w = nd(vec![0.5, 1.0, 2.0, 9.0], &[("x", 4)]);
        let s = bincount(&a, Some(&w), 0, "value").unwrap();
        assert_eq!(
            s.values().as_primitive::<Float64Type>().values(),
            &[0.5, 0.0, 3.0]
        );
    }
}
