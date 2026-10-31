//! Moving windows along one dimension, like `xarray.DataArray.rolling`.

use std::sync::Arc;

use arrow::array::{Array, AsArray, Float64Array};
use arrow::compute::cast as arrow_cast;
use arrow::datatypes::{DataType, Float64Type};

use crate::NdArrowArray;
use crate::error::{Result, nd_err};

/// The statistic of each window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RollingOp {
    Sum,
    Mean,
    Min,
    Max,
    Count,
}

/// Apply `op` over a moving window of `window` elements along `dim`.
///
/// Like xarray, the window ends at each element, or is centered on it when
/// `center` is true. A window with fewer than `min_periods` non-null values
/// gives null (`min_periods` of `None` means `window`). The result has the
/// shape of `a` and type `Float64` (`Int64` for `Count`).
pub fn rolling(
    a: &NdArrowArray,
    dim: &str,
    window: usize,
    op: RollingOp,
    min_periods: Option<usize>,
    center: bool,
) -> Result<NdArrowArray> {
    if window == 0 {
        return Err(nd_err!(
            InvalidArgument,
            "rolling window must be at least 1"
        ));
    }
    let min_periods = min_periods.unwrap_or(window).max(1);
    let axis = a.dims().axis(dim)?;
    let shape = a.shape();
    let n = shape[axis];
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let values = arrow_cast(a.values().as_ref(), &DataType::Float64)?;
    let values = values.as_primitive::<Float64Type>();
    let get = |p: usize| values.is_valid(p).then(|| values.value(p));

    // Window of element i covers [i + 1 - window + shift, i + shift].
    let shift = if center { window / 2 } else { 0 };
    let mut out = vec![None; a.len()];
    let mut lane = Vec::with_capacity(n);
    for o in 0..outer {
        for j in 0..inner {
            lane.clear();
            lane.extend((0..n).map(|i| get((o * n + i) * inner + j)));
            for i in 0..n {
                let end = (i + shift).min(n.saturating_sub(1));
                let start = (i + shift + 1).saturating_sub(window);
                let items = lane[start..=end].iter().flatten();
                let valid = items.clone().count();
                let p = (o * n + i) * inner + j;
                out[p] = match op {
                    RollingOp::Count => Some(valid as f64),
                    _ if valid < min_periods => None,
                    RollingOp::Sum => Some(items.sum()),
                    RollingOp::Mean => Some(items.sum::<f64>() / valid as f64),
                    RollingOp::Min => items.copied().reduce(f64::min),
                    RollingOp::Max => items.copied().reduce(f64::max),
                };
            }
        }
    }
    let result: Arc<dyn Array> = if op == RollingOp::Count {
        arrow_cast(&Float64Array::from(out), &DataType::Int64)?
    } else {
        Arc::new(Float64Array::from(out))
    };
    NdArrowArray::try_new(result, a.dims().clone())
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, Int32Array};

    use super::*;
    use crate::Dimensions;

    fn lane(values: Vec<Option<i32>>) -> NdArrowArray {
        let n = values.len();
        NdArrowArray::try_new(
            Arc::new(Int32Array::from(values)) as ArrayRef,
            Dimensions::try_new([("t", n)]).unwrap(),
        )
        .unwrap()
    }

    fn f64s(a: &NdArrowArray) -> Vec<Option<f64>> {
        a.values().as_primitive::<Float64Type>().iter().collect()
    }

    #[test]
    fn trailing_mean_matches_xarray() {
        let a = lane(vec![Some(1), Some(2), Some(3), Some(4)]);
        let m = rolling(&a, "t", 2, RollingOp::Mean, None, false).unwrap();
        assert_eq!(f64s(&m), vec![None, Some(1.5), Some(2.5), Some(3.5)]);
    }

    #[test]
    fn centered_sum_with_min_periods() {
        let a = lane(vec![Some(1), None, Some(3), Some(4)]);
        let s = rolling(&a, "t", 3, RollingOp::Sum, Some(1), true).unwrap();
        assert_eq!(f64s(&s), vec![Some(1.0), Some(4.0), Some(7.0), Some(7.0)]);
        let c = rolling(&a, "t", 3, RollingOp::Count, None, true).unwrap();
        assert_eq!(c.data_type(), &DataType::Int64);
    }
}
