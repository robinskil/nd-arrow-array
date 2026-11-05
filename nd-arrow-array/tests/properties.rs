//! Property tests: kernel identities that must hold for any shape and data.

use std::sync::Arc;

use arrow::array::{AsArray, Int64Array};
use arrow::datatypes::Int64Type;
use nd_arrow_array::compute::{self, PartialReduction, Reduction, Selector, Slice, merge_all};
use nd_arrow_array::{Dimensions, NdArrowArray, NdColumn};
use proptest::prelude::*;

const NAMES: [&str; 4] = ["a", "b", "c", "d"];

/// A random array: rank 1 to 4, sizes 1 to 5, small integers with some nulls.
fn arrays() -> impl Strategy<Value = NdArrowArray> {
    prop::collection::vec(1usize..=5, 1..=4).prop_flat_map(|shape| {
        let n: usize = shape.iter().product();
        prop::collection::vec(prop::option::weighted(0.9, -50i64..50), n).prop_map(move |values| {
            let dims =
                Dimensions::try_new(NAMES.iter().copied().zip(shape.iter().copied())).unwrap();
            NdArrowArray::try_new(Arc::new(Int64Array::from(values)), dims).unwrap()
        })
    })
}

fn values(a: &NdArrowArray) -> Vec<Option<i64>> {
    a.values().as_primitive::<Int64Type>().iter().collect()
}

/// The value at a multi-index, by direct C-order arithmetic.
fn at(a: &NdArrowArray, index: &[usize]) -> Option<i64> {
    let position: usize = index
        .iter()
        .zip(a.dims().c_strides())
        .map(|(i, s)| i * s)
        .sum();
    values(a)[position]
}

fn indices(shape: &[usize]) -> Vec<Vec<usize>> {
    shape.iter().fold(vec![vec![]], |acc, &size| {
        acc.into_iter()
            .flat_map(|prefix| {
                (0..size).map(move |i| {
                    let mut next = prefix.clone();
                    next.push(i);
                    next
                })
            })
            .collect()
    })
}

proptest! {
    #[test]
    fn reversing_axes_twice_is_identity(a in arrays()) {
        let back = compute::transpose_reversed(&compute::transpose_reversed(&a).unwrap()).unwrap();
        prop_assert_eq!(back, a);
    }

    #[test]
    fn transpose_moves_each_value(a in arrays()) {
        let t = compute::transpose_reversed(&a).unwrap();
        for index in indices(&a.shape()) {
            let reversed: Vec<usize> = index.iter().rev().copied().collect();
            prop_assert_eq!(at(&t, &reversed), at(&a, &index));
        }
    }

    #[test]
    fn isel_slice_matches_direct_indexing(a in arrays(), start in 0isize..5, step in 1isize..3) {
        let dim = a.dims().as_slice()[0].name().to_string();
        let s = compute::isel(&a, &[(&dim, Selector::Slice(Slice::new(Some(start), None, step)))]).unwrap();
        for index in indices(&s.shape()) {
            let mut source = index.clone();
            source[0] = start as usize + index[0] * step as usize;
            prop_assert_eq!(at(&s, &index), at(&a, &source));
        }
    }

    #[test]
    fn concat_of_split_is_identity(a in arrays(), cut in 0isize..5) {
        let dim = a.dims().as_slice()[0].name().to_string();
        let size = a.shape()[0] as isize;
        let cut = cut.min(size);
        let head = compute::isel(&a, &[(&dim, Slice::range(0, cut).into())]).unwrap();
        let tail = compute::isel(&a, &[(&dim, Slice::range(cut, size).into())]).unwrap();
        prop_assert_eq!(compute::concat(&[&head, &tail], &dim).unwrap(), a);
    }

    #[test]
    fn sum_of_broadcast_scales_by_new_size(a in arrays(), extra in 1usize..4) {
        let mut target: Vec<(&str, usize)> = a.dims().iter().map(|d| (d.name(), d.size())).collect();
        target.push(("new", extra));
        let b = compute::broadcast_to(&a, &Dimensions::try_new(target).unwrap()).unwrap();
        let total = |x: &NdArrowArray| values(&compute::sum(x, None).unwrap())[0].unwrap();
        prop_assert_eq!(total(&b), total(&a) * extra as i64);
    }

    #[test]
    fn merged_partials_equal_full_sum(a in arrays()) {
        let dim = a.dims().as_slice()[0].name().to_string();
        let size = a.shape()[0] as isize;
        let partials: Vec<PartialReduction> = (0..size)
            .map(|i| {
                let chunk = compute::isel(&a, &[(&dim, Slice::range(i, i + 1).into())]).unwrap();
                PartialReduction::new(&chunk, None, Reduction::Sum).unwrap()
            })
            .collect();
        let merged = merge_all(&partials).unwrap().finish().unwrap();
        prop_assert_eq!(values(&merged), values(&compute::sum(&a, None).unwrap()));
    }

    #[test]
    fn column_encoding_round_trips(rows in prop::collection::vec(prop::option::of(arrays()), 1..6)) {
        let mut builder = nd_arrow_array::NdColumnBuilder::with_value_type(arrow::datatypes::DataType::Int64);
        for row in &rows {
            builder.append_option(row.as_ref()).unwrap();
        }
        let column = builder.finish().unwrap();
        let again = NdColumn::try_from_array(&column.clone().into_array_ref()).unwrap();
        for (row, original) in rows.iter().enumerate() {
            prop_assert_eq!(again.get(row), original.clone());
        }
    }
}
