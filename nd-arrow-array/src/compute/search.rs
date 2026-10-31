//! Sorting and searching: `searchsorted`, `sort`, `argsort`, `unique`.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow::array::make_comparator;
use arrow::array::{Array, ArrayRef, UInt32Array, UInt64Array};
use arrow::compute::{SortOptions, cast as arrow_cast, sort_to_indices, take as arrow_take};

use crate::NdArrowArray;
use crate::compute::shape::move_dim;
use crate::dimensions::Dimensions;
use crate::error::{Result, nd_err};

/// Which insertion point [`searchsorted`] returns for equal values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchSide {
    /// The first suitable position, like `side="left"`.
    #[default]
    Left,
    /// The last suitable position, like `side="right"`.
    Right,
}

const ASCENDING: SortOptions = SortOptions {
    descending: false,
    nulls_first: false,
};

/// Insertion points of `values` into the ascending 1-D array `sorted`, like
/// `numpy.searchsorted`. The result has the dimensions of `values` and type
/// `UInt64`. A null value gives a null.
pub fn searchsorted(
    sorted: &NdArrowArray,
    values: &NdArrowArray,
    side: SearchSide,
) -> Result<NdArrowArray> {
    if sorted.rank() != 1 {
        return Err(nd_err!(
            InvalidArgument,
            "searchsorted needs a 1-D sorted array"
        ));
    }
    let haystack = sorted.values();
    let needles = arrow_cast(values.values().as_ref(), haystack.data_type())?;
    let compare = make_comparator(haystack.as_ref(), needles.as_ref(), ASCENDING)?;
    let out: UInt64Array = (0..needles.len())
        .map(|j| {
            needles.is_valid(j).then(|| {
                partition_point(haystack.len(), |i| match side {
                    SearchSide::Left => compare(i, j) == Ordering::Less,
                    SearchSide::Right => compare(i, j) != Ordering::Greater,
                }) as u64
            })
        })
        .collect();
    NdArrowArray::try_new(Arc::new(out), values.dims().clone())
}

/// First index in `0..len` where `pred` is false. `pred` must be true then false.
pub(crate) fn partition_point(len: usize, pred: impl Fn(usize) -> bool) -> usize {
    let (mut low, mut high) = (0usize, len);
    while low < high {
        let mid = low + (high - low) / 2;
        if pred(mid) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
}

/// Per-row sort indices with `dim` moved last. Returns the moved array and the
/// global take indices.
fn sort_rows(
    a: &NdArrowArray,
    dim: &str,
    descending: bool,
) -> Result<(NdArrowArray, Vec<u32>, usize)> {
    let last = a
        .rank()
        .checked_sub(1)
        .ok_or_else(|| nd_err!(InvalidArgument, "cannot sort a scalar"))?;
    let moved = move_dim(a, dim, last)?;
    let row = moved.dims().as_slice()[last].size();
    let options = SortOptions {
        descending,
        nulls_first: false,
    };
    let mut indices = Vec::with_capacity(moved.len());
    if row > 0 {
        for start in (0..moved.len()).step_by(row) {
            let slice = moved.values().slice(start, row);
            let local = sort_to_indices(slice.as_ref(), Some(options), None)?;
            indices.extend(local.values().iter().map(|&i| i + start as u32));
        }
    }
    Ok((moved, indices, row))
}

/// Sort along `dim`, like `numpy.sort`. Nulls go last. Keeps the attributes.
pub fn sort(a: &NdArrowArray, dim: &str, descending: bool) -> Result<NdArrowArray> {
    let (moved, indices, _) = sort_rows(a, dim, descending)?;
    let taken = arrow_take(moved.values().as_ref(), &UInt32Array::from(indices), None)?;
    let sorted =
        NdArrowArray::try_new(taken, moved.dims().clone())?.with_attributes(a.attributes().clone());
    move_dim(&sorted, dim, a.dims().axis(dim)?)
}

/// Indices that sort along `dim`, like `numpy.argsort`. Type `UInt64`.
pub fn argsort(a: &NdArrowArray, dim: &str, descending: bool) -> Result<NdArrowArray> {
    let (moved, indices, row) = sort_rows(a, dim, descending)?;
    let local: UInt64Array = indices
        .iter()
        .map(|&i| (i as usize % row.max(1)) as u64)
        .collect();
    let out = NdArrowArray::try_new(Arc::new(local), moved.dims().clone())?;
    move_dim(&out, dim, a.dims().axis(dim)?)
}

/// Sorted unique non-null values of all elements, like `numpy.unique`, as a
/// 1-D array along a new dimension `dim`.
pub fn unique(a: &NdArrowArray, dim: &str) -> Result<NdArrowArray> {
    let values = a.values();
    let order = sort_to_indices(values.as_ref(), Some(ASCENDING), None)?;
    let sorted: ArrayRef = arrow_take(values.as_ref(), &order, None)?;
    let compare = make_comparator(sorted.as_ref(), sorted.as_ref(), ASCENDING)?;
    let keep: UInt32Array = (0..sorted.len())
        .filter(|&i| sorted.is_valid(i) && (i == 0 || compare(i - 1, i) != Ordering::Equal))
        .map(|i| i as u32)
        .collect();
    let out = arrow_take(sorted.as_ref(), &keep, None)?;
    let len = out.len();
    NdArrowArray::try_new(out, Dimensions::try_new([(dim, len)])?)
}

#[cfg(test)]
mod tests {
    use arrow::array::{AsArray, Float64Array, Int32Array, StringArray};
    use arrow::datatypes::{Int32Type, UInt64Type};

    use super::*;

    fn nd(values: ArrayRef, dims: &[(&str, usize)]) -> NdArrowArray {
        NdArrowArray::try_new(values, Dimensions::try_new(dims.iter().copied()).unwrap()).unwrap()
    }

    fn u64s(a: &NdArrowArray) -> Vec<Option<u64>> {
        a.values().as_primitive::<UInt64Type>().iter().collect()
    }

    #[test]
    fn searchsorted_sides() {
        let sorted = nd(Arc::new(Int32Array::from(vec![1, 2, 2, 3])), &[("x", 4)]);
        let needles = nd(
            Arc::new(Float64Array::from(vec![
                Some(2.0),
                Some(0.0),
                Some(9.0),
                None,
            ])),
            &[("q", 4)],
        );
        let left = searchsorted(&sorted, &needles, SearchSide::Left).unwrap();
        assert_eq!(u64s(&left), vec![Some(1), Some(0), Some(4), None]);
        let right = searchsorted(&sorted, &needles, SearchSide::Right).unwrap();
        assert_eq!(u64s(&right)[0], Some(3));
    }

    #[test]
    fn sort_and_argsort_along_axis() {
        let a = nd(
            Arc::new(Int32Array::from(vec![3, 1, 2, 9, 7, 8])),
            &[("x", 2), ("y", 3)],
        );
        let s = sort(&a, "y", false).unwrap();
        assert_eq!(
            s.values().as_primitive::<Int32Type>().values(),
            &[1, 2, 3, 7, 8, 9]
        );
        let sx = sort(&a, "x", true).unwrap();
        assert_eq!(sx.dims().names(), vec!["x", "y"]);
        assert_eq!(
            sx.values().as_primitive::<Int32Type>().values(),
            &[9, 7, 8, 3, 1, 2]
        );
        let i = argsort(&a, "y", false).unwrap();
        assert_eq!(u64s(&i), [1, 2, 0, 1, 2, 0].map(Some).to_vec());
    }

    #[test]
    fn unique_strings() {
        let a = nd(
            Arc::new(StringArray::from(vec![
                Some("b"),
                Some("a"),
                None,
                Some("b"),
            ])),
            &[("x", 4)],
        );
        let u = unique(&a, "label").unwrap();
        let v: Vec<&str> = u
            .values()
            .as_string::<i32>()
            .iter()
            .map(Option::unwrap)
            .collect();
        assert_eq!(v, vec!["a", "b"]);
        assert_eq!(u.dims().names(), vec!["label"]);
    }
}
