//! Group-by reductions along one dimension, like `xarray.DataArray.groupby`.
//!
//! For regular time bins, see [`resample`](fn@crate::compute::resample).

use std::cmp::Ordering;

use arrow::array::make_comparator;
use arrow::array::{Array, UInt32Array};
use arrow::compute::{SortOptions, sort_to_indices};

use crate::NdArrowArray;
use crate::compute::reduce::{Reduction, reduce};
use crate::compute::search::unique;
use crate::compute::shape::{concat, expand_dims, take};
use crate::error::{Result, nd_err};

/// Reduce `a` per group of equal `labels` along `dim`.
///
/// `labels` is 1-D with one label per position along `dim`. Null labels are
/// left out. The result replaces `dim` with one position per unique label,
/// in sorted order, under the same dimension name. The second value holds
/// the unique labels along `dim`.
pub fn groupby_reduce(
    a: &NdArrowArray,
    dim: &str,
    labels: &NdArrowArray,
    op: Reduction,
) -> Result<(NdArrowArray, NdArrowArray)> {
    let axis = a.dims().axis(dim)?;
    let n = a.dims().as_slice()[axis].size();
    if labels.rank() != 1 || labels.len() != n {
        return Err(nd_err!(
            InvalidArgument,
            "groupby labels must be 1-D with {n} values along '{dim}'"
        ));
    }
    let groups = unique(labels, dim)?;
    if groups.is_empty() {
        return Err(nd_err!(InvalidArgument, "groupby labels are all null"));
    }

    let options = SortOptions {
        descending: false,
        nulls_first: false,
    };
    let order: UInt32Array = sort_to_indices(labels.values().as_ref(), Some(options), None)?;
    let compare = make_comparator(groups.values().as_ref(), labels.values().as_ref(), options)?;

    let mut parts = Vec::with_capacity(groups.len());
    let mut cursor = 0usize;
    for g in 0..groups.len() {
        let mut members: Vec<isize> = Vec::new();
        while cursor < order.len() {
            let position = order.value(cursor) as usize;
            if labels.values().is_null(position) || compare(g, position) != Ordering::Equal {
                break;
            }
            members.push(position as isize);
            cursor += 1;
        }
        members.sort_unstable();
        let group = take(a, dim, &members)?;
        let reduced = reduce(&group, Some(&[dim]), op)?;
        parts.push(expand_dims(&reduced, dim, axis)?);
    }
    let result = concat(&parts.iter().collect::<Vec<_>>(), dim)?;
    Ok((result, groups))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{AsArray, Float64Array, StringArray};
    use arrow::datatypes::Float64Type;

    use super::*;
    use crate::Dimensions;

    #[test]
    fn mean_per_label() {
        let a = NdArrowArray::try_new(
            Arc::new(Float64Array::from(vec![
                1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0,
            ])),
            Dimensions::try_new([("y", 2), ("t", 4)]).unwrap(),
        )
        .unwrap();
        let labels = NdArrowArray::try_new(
            Arc::new(StringArray::from(vec![
                Some("b"),
                Some("a"),
                Some("b"),
                None,
            ])),
            Dimensions::try_new([("t", 4)]).unwrap(),
        )
        .unwrap();
        let (out, groups) = groupby_reduce(&a, "t", &labels, Reduction::Mean).unwrap();
        assert_eq!(out.dims().names(), vec!["y", "t"]);
        assert_eq!(out.shape(), vec![2, 2]);
        let v: Vec<f64> = out.values().as_primitive::<Float64Type>().values().to_vec();
        assert_eq!(v, vec![2.0, 2.0, 20.0, 20.0]);
        let g: Vec<&str> = groups
            .values()
            .as_string::<i32>()
            .iter()
            .map(Option::unwrap)
            .collect();
        assert_eq!(g, vec!["a", "b"]);
    }
}
