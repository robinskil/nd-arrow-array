//! Reductions along named axes, cumulative operations and differences.
//!
//! Nulls are skipped, like the NumPy `nan*` functions. A group without
//! valid values gives `0` for sums, `1` for products, `false` for `any`,
//! `true` for `all`, `0` for counts and null for all other reductions.
//! Floats order NaN above all numbers, like Arrow, so `max` can return NaN.
//! Use [`nan_to_null`](crate::compute::nan_to_null) to treat NaN as missing.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, AsArray, BooleanArray, Int64Array, PrimitiveArray,
    downcast_primitive_array,
};
use arrow::compute::cast as arrow_cast;
use arrow::datatypes::{
    ArrowNativeTypeOp, DataType, Float32Type, Float64Type, Int64Type, UInt64Type,
};

use crate::NdArrowArray;
use crate::compute::elementwise::{ne, sub};
use crate::compute::shape::{Selector, Slice, isel, transpose};
use crate::dimensions::Dimensions;
use crate::error::{Result, nd_err};

/// A reduction for [`reduce`] and [`groupby_reduce`](crate::compute::groupby_reduce).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reduction {
    Sum,
    Prod,
    Mean,
    Min,
    Max,
    /// Number of non-null values.
    Count,
    /// Variance with `ddof` delta degrees of freedom, like `numpy.var`.
    Var {
        ddof: usize,
    },
    /// Standard deviation with `ddof` delta degrees of freedom, like `numpy.std`.
    Std {
        ddof: usize,
    },
    Median,
    /// Quantile `q` in `[0, 1]` with linear interpolation, like `numpy.quantile`.
    Quantile(f64),
    Any,
    All,
    /// Index of the first minimum along the reduced axis (or the flat index for all axes).
    ArgMin,
    /// Index of the first maximum along the reduced axis (or the flat index for all axes).
    ArgMax,
}

/// The values regrouped as `(outer, reduced, inner)` in C order.
struct Blocks {
    values: ArrayRef,
    outer: usize,
    reduced: usize,
    inner: usize,
    out_dims: Dimensions,
}

impl Blocks {
    fn new(a: &NdArrowArray, axes: Option<&[&str]>) -> Result<Self> {
        let dims = a.dims();
        let mut selected: Vec<usize> = match axes {
            None => (0..dims.rank()).collect(),
            Some(names) => names
                .iter()
                .map(|name| dims.axis(name))
                .collect::<Result<_>>()?,
        };
        selected.sort_unstable();
        selected.dedup();
        let sizes = dims.shape();
        let product = |range: &[usize]| range.iter().map(|&i| sizes[i]).product::<usize>();
        let kept: Vec<usize> = (0..dims.rank()).filter(|i| !selected.contains(i)).collect();
        let out_dims = Dimensions::try_new(kept.iter().map(|&i| dims.as_slice()[i].clone()))?;

        let Some((&first, &last)) = selected.first().zip(selected.last()) else {
            return Ok(Self {
                values: a.values().clone(),
                outer: a.len(),
                reduced: 1,
                inner: 1,
                out_dims,
            });
        };
        if last - first + 1 == selected.len() {
            return Ok(Self {
                values: a.values().clone(),
                outer: sizes[..first].iter().product(),
                reduced: product(&selected),
                inner: sizes[last + 1..].iter().product(),
                out_dims,
            });
        }
        let order: Vec<&str> = kept
            .iter()
            .chain(&selected)
            .map(|&i| dims.as_slice()[i].name())
            .collect();
        Ok(Self {
            values: transpose(a, &order)?.values().clone(),
            outer: product(&kept),
            reduced: product(&selected),
            inner: 1,
            out_dims,
        })
    }

    fn groups(&self) -> usize {
        self.outer * self.inner
    }

    /// Visit every valid position as `(group, index in group, position)`.
    fn for_each(
        &self,
        nulls: Option<&arrow::buffer::NullBuffer>,
        mut f: impl FnMut(usize, usize, usize),
    ) {
        let nulls = nulls.filter(|n| n.null_count() > 0);
        for o in 0..self.outer {
            for ri in 0..self.reduced {
                let base = (o * self.reduced + ri) * self.inner;
                let group = o * self.inner;
                for j in 0..self.inner {
                    let p = base + j;
                    if nulls.is_some_and(|n| n.is_null(p)) {
                        continue;
                    }
                    f(group + j, ri, p);
                }
            }
        }
    }

    /// Each group as one contiguous slice, when groups are contiguous and there are no nulls.
    fn contiguous_groups<'a, N>(
        &self,
        raw: &'a [N],
        null_count: usize,
    ) -> Option<std::slice::Chunks<'a, N>> {
        (self.inner == 1 && self.reduced > 0 && null_count == 0).then(|| raw.chunks(self.reduced))
    }

    fn finish(&self, values: ArrayRef) -> Result<NdArrowArray> {
        NdArrowArray::try_new(values, self.out_dims.clone())
    }
}

/// Reduce `a` over the named axes. `None` reduces all axes to a scalar.
pub fn reduce(a: &NdArrowArray, axes: Option<&[&str]>, op: Reduction) -> Result<NdArrowArray> {
    let blocks = Blocks::new(a, axes)?;
    let values = match op {
        Reduction::Sum => sum_prod(&blocks, false)?,
        Reduction::Prod => sum_prod(&blocks, true)?,
        Reduction::Count => count_values(&blocks),
        Reduction::Min => min_max(&blocks, false)?,
        Reduction::Max => min_max(&blocks, true)?,
        Reduction::ArgMin | Reduction::ArgMax => {
            if axes.is_some_and(|names| names.len() != 1) {
                return Err(nd_err!(
                    InvalidArgument,
                    "argmin and argmax reduce one axis or all axes"
                ));
            }
            arg_min_max(&blocks, op == Reduction::ArgMax)?
        }
        Reduction::Mean => float_stat(&blocks, a.data_type(), Stat::Mean)?,
        Reduction::Var { ddof } => float_stat(&blocks, a.data_type(), Stat::Var(ddof, false))?,
        Reduction::Std { ddof } => float_stat(&blocks, a.data_type(), Stat::Var(ddof, true))?,
        Reduction::Median => float_stat(&blocks, a.data_type(), Stat::Quantile(0.5))?,
        Reduction::Quantile(q) => {
            if !(0.0..=1.0).contains(&q) {
                return Err(nd_err!(InvalidArgument, "quantile {q} is outside [0, 1]"));
            }
            float_stat(&blocks, a.data_type(), Stat::Quantile(q))?
        }
        Reduction::Any => any_all(&blocks, a, true)?,
        Reduction::All => any_all(&blocks, a, false)?,
    };
    blocks.finish(values)
}

macro_rules! reductions {
    ($($(#[$doc:meta])* $name:ident => $op:expr),* $(,)?) => {$(
        $(#[$doc])*
        pub fn $name(a: &NdArrowArray, axes: Option<&[&str]>) -> Result<NdArrowArray> {
            reduce(a, axes, $op)
        }
    )*};
}

reductions!(
    /// Sum, like `numpy.nansum`. Integers sum as `Int64` or `UInt64`, with wrapping.
    sum => Reduction::Sum,
    /// Product, like `numpy.nanprod`. Integers multiply as `Int64` or `UInt64`, with wrapping.
    prod => Reduction::Prod,
    /// Mean, like `numpy.nanmean`. Integers give `Float64`.
    mean => Reduction::Mean,
    /// Minimum, like `numpy.nanmin`.
    min => Reduction::Min,
    /// Maximum, like `numpy.nanmax`.
    max => Reduction::Max,
    /// Number of non-null values, like `xarray.DataArray.count`.
    count => Reduction::Count,
    /// Median, like `numpy.nanmedian`.
    median => Reduction::Median,
    /// `true` if any value is `true` (or non-zero), like `numpy.any`.
    any => Reduction::Any,
    /// `true` if all values are `true` (or non-zero), like `numpy.all`.
    all => Reduction::All,
    /// Index of the first minimum, like `numpy.nanargmin`.
    argmin => Reduction::ArgMin,
    /// Index of the first maximum, like `numpy.nanargmax`.
    argmax => Reduction::ArgMax,
);

/// Variance, like `numpy.nanvar`.
pub fn var(a: &NdArrowArray, axes: Option<&[&str]>, ddof: usize) -> Result<NdArrowArray> {
    reduce(a, axes, Reduction::Var { ddof })
}

/// Standard deviation, like `numpy.nanstd`.
pub fn std(a: &NdArrowArray, axes: Option<&[&str]>, ddof: usize) -> Result<NdArrowArray> {
    reduce(a, axes, Reduction::Std { ddof })
}

/// Quantile with linear interpolation, like `numpy.nanquantile`.
pub fn quantile(a: &NdArrowArray, axes: Option<&[&str]>, q: f64) -> Result<NdArrowArray> {
    reduce(a, axes, Reduction::Quantile(q))
}

/// The accumulator type of sums and products, like NumPy.
fn accumulator_type(data_type: &DataType) -> Result<DataType> {
    Ok(match data_type {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64 => DataType::Int64,
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            DataType::UInt64
        }
        DataType::Float16 | DataType::Float32 => DataType::Float32,
        DataType::Float64 => DataType::Float64,
        other => return Err(nd_err!(NotSupported, "sum or product of {other}")),
    })
}

fn sum_prod(blocks: &Blocks, product: bool) -> Result<ArrayRef> {
    let to = accumulator_type(blocks.values.data_type())?;
    let values = arrow_cast(blocks.values.as_ref(), &to)?;
    Ok(match to {
        DataType::Int64 => fold_sum_prod::<Int64Type>(blocks, &values, product),
        DataType::UInt64 => fold_sum_prod::<UInt64Type>(blocks, &values, product),
        DataType::Float32 => fold_sum_prod::<Float32Type>(blocks, &values, product),
        _ => fold_sum_prod::<Float64Type>(blocks, &values, product),
    })
}

fn fold_sum_prod<T: ArrowPrimitiveType>(
    blocks: &Blocks,
    values: &ArrayRef,
    product: bool,
) -> ArrayRef {
    let array = values.as_primitive::<T>();
    let raw = array.values();
    let init = if product {
        T::Native::ONE
    } else {
        T::Native::ZERO
    };
    let op = |a: T::Native, b: T::Native| {
        if product {
            a.mul_wrapping(b)
        } else {
            a.add_wrapping(b)
        }
    };
    if let Some(groups) = blocks.contiguous_groups(raw, array.null_count()) {
        let accs = groups.map(|chunk| lane_fold(chunk, init, op));
        return Arc::new(PrimitiveArray::<T>::from_iter_values(accs));
    }
    let mut accs = vec![init; blocks.groups()];
    blocks.for_each(array.nulls(), |g, _, p| accs[g] = op(accs[g], raw[p]));
    Arc::new(PrimitiveArray::<T>::from_iter_values(accs))
}

/// Fold with eight independent lanes, so that the compiler can vectorize.
fn lane_fold<N: Copy>(chunk: &[N], init: N, op: impl Fn(N, N) -> N) -> N {
    let mut lanes = [init; 8];
    let mut exact = chunk.chunks_exact(8);
    for block in &mut exact {
        for k in 0..8 {
            lanes[k] = op(lanes[k], block[k]);
        }
    }
    let tail = exact.remainder().iter().fold(init, |acc, &v| op(acc, v));
    lanes.into_iter().fold(tail, &op)
}

fn count_values(blocks: &Blocks) -> ArrayRef {
    let mut counts = vec![0i64; blocks.groups()];
    blocks.for_each(blocks.values.logical_nulls().as_ref(), |g, _, _| {
        counts[g] += 1
    });
    Arc::new(Int64Array::from(counts))
}

fn min_max(blocks: &Blocks, is_max: bool) -> Result<ArrayRef> {
    let values = &blocks.values;
    downcast_primitive_array!(
        values => Ok(fold_min_max(blocks, values, is_max)),
        DataType::Boolean => {
            let as_int = arrow_cast(values.as_ref(), &DataType::UInt8)?;
            let out = fold_min_max(blocks, as_int.as_primitive::<arrow::datatypes::UInt8Type>(), is_max);
            Ok(arrow_cast(out.as_ref(), &DataType::Boolean)?)
        }
        other => Err(nd_err!(NotSupported, "min or max of {other}"))
    )
}

fn fold_min_max<T: ArrowPrimitiveType>(
    blocks: &Blocks,
    array: &PrimitiveArray<T>,
    is_max: bool,
) -> ArrayRef {
    let raw = array.values();
    let mut best: Vec<Option<T::Native>> = vec![None; blocks.groups()];
    blocks.for_each(array.nulls(), |g, _, p| {
        let v = raw[p];
        let better = match best[g] {
            None => true,
            Some(b) if is_max => v.is_gt(b),
            Some(b) => v.is_lt(b),
        };
        if better {
            best[g] = Some(v);
        }
    });
    Arc::new(PrimitiveArray::<T>::from_iter(best).with_data_type(array.data_type().clone()))
}

fn arg_min_max(blocks: &Blocks, is_max: bool) -> Result<ArrayRef> {
    let values = &blocks.values;
    downcast_primitive_array!(
        values => Ok(fold_arg(blocks, values, is_max)),
        other => Err(nd_err!(NotSupported, "argmin or argmax of {other}"))
    )
}

fn fold_arg<T: ArrowPrimitiveType>(
    blocks: &Blocks,
    array: &PrimitiveArray<T>,
    is_max: bool,
) -> ArrayRef {
    let raw = array.values();
    let mut best: Vec<Option<(T::Native, usize)>> = vec![None; blocks.groups()];
    blocks.for_each(array.nulls(), |g, ri, p| {
        let v = raw[p];
        let better = match best[g] {
            None => true,
            Some((b, _)) if is_max => v.is_gt(b),
            Some((b, _)) => v.is_lt(b),
        };
        if better {
            best[g] = Some((v, ri));
        }
    });
    Arc::new(Int64Array::from_iter(
        best.into_iter().map(|b| b.map(|(_, i)| i as i64)),
    ))
}

enum Stat {
    Mean,
    Var(usize, bool),
    Quantile(f64),
}

/// Statistics computed in `Float64`. The result is `Float32` for `Float32` input.
fn float_stat(blocks: &Blocks, input: &DataType, stat: Stat) -> Result<ArrayRef> {
    if !(input.is_numeric() || input == &DataType::Boolean) {
        return Err(nd_err!(NotSupported, "statistics of {input}"));
    }
    let values = arrow_cast(blocks.values.as_ref(), &DataType::Float64)?;
    let array = values.as_primitive::<Float64Type>();
    let raw = array.values();
    let groups = blocks.groups();
    let out: Vec<Option<f64>> = match stat {
        Stat::Mean if blocks.contiguous_groups(raw, array.null_count()).is_some() => blocks
            .contiguous_groups(raw, 0)
            .unwrap()
            .map(|chunk| {
                (!chunk.is_empty())
                    .then(|| lane_fold(chunk, 0.0, |a, b| a + b) / chunk.len() as f64)
            })
            .collect(),
        Stat::Mean => {
            let mut acc = vec![(0.0f64, 0usize); groups];
            blocks.for_each(array.nulls(), |g, _, p| {
                acc[g].0 += raw[p];
                acc[g].1 += 1;
            });
            acc.into_iter()
                .map(|(s, n)| (n > 0).then(|| s / n as f64))
                .collect()
        }
        Stat::Var(ddof, sqrt) => {
            // Welford update: count, mean, sum of squared deviations.
            let mut acc = vec![(0usize, 0.0f64, 0.0f64); groups];
            blocks.for_each(array.nulls(), |g, _, p| {
                let (n, mean, m2) = &mut acc[g];
                *n += 1;
                let delta = raw[p] - *mean;
                *mean += delta / *n as f64;
                *m2 += delta * (raw[p] - *mean);
            });
            acc.into_iter()
                .map(|(n, _, m2)| {
                    (n > ddof).then(|| {
                        let v = m2 / (n - ddof) as f64;
                        if sqrt { v.sqrt() } else { v }
                    })
                })
                .collect()
        }
        Stat::Quantile(q) => {
            let mut groups_values: Vec<Vec<f64>> = vec![Vec::new(); groups];
            blocks.for_each(array.nulls(), |g, _, p| groups_values[g].push(raw[p]));
            groups_values
                .into_iter()
                .map(|mut v| linear_quantile(&mut v, q))
                .collect()
        }
    };
    let out: ArrayRef = Arc::new(PrimitiveArray::<Float64Type>::from_iter(out));
    Ok(match input {
        DataType::Float16 | DataType::Float32 => arrow_cast(out.as_ref(), &DataType::Float32)?,
        _ => out,
    })
}

/// NumPy's default `linear` quantile method.
pub(crate) fn linear_quantile(values: &mut [f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f64::total_cmp);
    let position = q * (values.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    Some(values[low] + (values[high] - values[low]) * (position - low as f64))
}

fn any_all(blocks: &Blocks, a: &NdArrowArray, is_any: bool) -> Result<ArrayRef> {
    let truthy = if a.data_type() == &DataType::Boolean {
        blocks.values.clone()
    } else {
        let zero = NdArrowArray::try_new_scalar(arrow_cast(
            &arrow::array::Int8Array::from(vec![0]),
            a.data_type(),
        )?)?;
        let values = NdArrowArray::try_new(
            blocks.values.clone(),
            Dimensions::try_new([("n", blocks.values.len())])?,
        )?;
        ne(&values, &zero)?.values().clone()
    };
    let array = truthy.as_boolean();
    let mut acc = vec![!is_any; blocks.groups()];
    blocks.for_each(array.nulls(), |g, _, p| {
        if array.value(p) == is_any {
            acc[g] = is_any;
        }
    });
    Ok(Arc::new(BooleanArray::from(acc)))
}

/// Cumulative sum along `dim`, like `numpy.nancumsum`. Null positions stay null.
pub fn cumsum(a: &NdArrowArray, dim: &str) -> Result<NdArrowArray> {
    cumulative(a, dim, false)
}

/// Cumulative product along `dim`, like `numpy.nancumprod`. Null positions stay null.
pub fn cumprod(a: &NdArrowArray, dim: &str) -> Result<NdArrowArray> {
    cumulative(a, dim, true)
}

fn cumulative(a: &NdArrowArray, dim: &str, product: bool) -> Result<NdArrowArray> {
    let blocks = Blocks::new(a, Some(&[dim]))?;
    let to = accumulator_type(a.data_type())?;
    let values = arrow_cast(a.values().as_ref(), &to)?;
    let out = match to {
        DataType::Int64 => scan::<Int64Type>(&blocks, &values, product),
        DataType::UInt64 => scan::<UInt64Type>(&blocks, &values, product),
        DataType::Float32 => scan::<Float32Type>(&blocks, &values, product),
        _ => scan::<Float64Type>(&blocks, &values, product),
    };
    NdArrowArray::try_new(out, a.dims().clone())
}

fn scan<T: ArrowPrimitiveType>(blocks: &Blocks, values: &ArrayRef, product: bool) -> ArrayRef {
    let array = values.as_primitive::<T>();
    let raw = array.values();
    let init = if product {
        T::Native::ONE
    } else {
        T::Native::ZERO
    };
    let mut accs = vec![init; blocks.groups()];
    let mut out = vec![init; raw.len()];
    blocks.for_each(array.nulls(), |g, _, p| {
        accs[g] = if product {
            accs[g].mul_wrapping(raw[p])
        } else {
            accs[g].add_wrapping(raw[p])
        };
        out[p] = accs[g];
    });
    Arc::new(PrimitiveArray::<T>::new(out.into(), array.nulls().cloned()))
}

/// The `n`-th discrete difference along `dim`, like `numpy.diff`.
pub fn diff(a: &NdArrowArray, dim: &str, n: usize) -> Result<NdArrowArray> {
    let mut out = a.clone();
    for _ in 0..n {
        let upper = isel(
            &out,
            &[(dim, Selector::Slice(Slice::new(Some(1), None, 1)))],
        )?;
        let lower = isel(
            &out,
            &[(dim, Selector::Slice(Slice::new(None, Some(-1), 1)))],
        )?;
        out = sub(&upper, &lower)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use arrow::array::{Float32Array, Float64Array, Int8Array, Int32Array};
    use arrow::datatypes::{Float64Type, Int64Type};

    use super::*;

    fn nd(values: ArrayRef, dims: &[(&str, usize)]) -> NdArrowArray {
        NdArrowArray::try_new(values, Dimensions::try_new(dims.iter().copied()).unwrap()).unwrap()
    }

    fn grid() -> NdArrowArray {
        nd(
            Arc::new(Int32Array::from_iter_values(0..24)),
            &[("a", 2), ("b", 3), ("c", 4)],
        )
    }

    fn i64s(a: &NdArrowArray) -> Vec<Option<i64>> {
        a.values().as_primitive::<Int64Type>().iter().collect()
    }

    fn f64s(a: &NdArrowArray) -> Vec<Option<f64>> {
        a.values().as_primitive::<Float64Type>().iter().collect()
    }

    #[test]
    fn sum_over_each_axis_layout() {
        let g = grid();
        let all = sum(&g, None).unwrap();
        assert!(all.is_scalar());
        assert_eq!(i64s(&all), vec![Some(276)]);

        let last = sum(&g, Some(&["c"])).unwrap();
        assert_eq!(last.dims().names(), vec!["a", "b"]);
        assert_eq!(i64s(&last), [6, 22, 38, 54, 70, 86].map(Some).to_vec());

        let first = sum(&g, Some(&["a"])).unwrap();
        assert_eq!(first.shape(), vec![3, 4]);
        assert_eq!(i64s(&first)[0], Some(12));

        let outer = sum(&g, Some(&["a", "c"])).unwrap();
        assert_eq!(outer.dims().names(), vec!["b"]);
        assert_eq!(i64s(&outer), vec![Some(60), Some(92), Some(124)]);
    }

    #[test]
    fn nulls_are_skipped() {
        let a = nd(
            Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0), None])),
            &[("x", 2), ("y", 2)],
        );
        assert_eq!(
            f64s(&sum(&a, Some(&["y"])).unwrap()),
            vec![Some(1.0), Some(3.0)]
        );
        assert_eq!(
            f64s(&mean(&a, Some(&["x"])).unwrap()),
            vec![Some(2.0), None]
        );
        assert_eq!(
            i64s(&count(&a, Some(&["x"])).unwrap()),
            vec![Some(2), Some(0)]
        );
        assert_eq!(
            f64s(&sum(&a, Some(&["x"])).unwrap()),
            vec![Some(4.0), Some(0.0)],
            "an all-null group sums to zero"
        );
    }

    #[test]
    fn min_max_and_arg() {
        let a = nd(
            Arc::new(Int32Array::from(vec![3, 1, 4, 1, 5, 9])),
            &[("x", 2), ("y", 3)],
        );
        let m = max(&a, Some(&["y"])).unwrap();
        assert_eq!(
            m.values()
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            &[4, 9]
        );
        assert_eq!(
            i64s(&argmin(&a, Some(&["y"])).unwrap()),
            vec![Some(1), Some(0)]
        );
        assert_eq!(i64s(&argmax(&a, None).unwrap()), vec![Some(5)]);
        assert!(argmax(&grid(), Some(&["a", "b"])).is_err());
    }

    #[test]
    fn var_std_quantile() {
        let a = nd(
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            &[("x", 4)],
        );
        assert_eq!(f64s(&var(&a, None, 0).unwrap()), vec![Some(1.25)]);
        assert_eq!(f64s(&var(&a, None, 1).unwrap())[0].unwrap(), 5.0 / 3.0);
        assert_eq!(f64s(&std(&a, None, 0).unwrap()), vec![Some(1.25f64.sqrt())]);
        assert_eq!(f64s(&median(&a, None).unwrap()), vec![Some(2.5)]);
        assert_eq!(f64s(&quantile(&a, None, 0.25).unwrap()), vec![Some(1.75)]);
        assert!(quantile(&a, None, 1.5).is_err());

        let f = nd(Arc::new(Float32Array::from(vec![1.0, 2.0])), &[("x", 2)]);
        assert_eq!(mean(&f, None).unwrap().data_type(), &DataType::Float32);
    }

    #[test]
    fn any_all_on_numbers_and_bools() {
        let a = nd(
            Arc::new(Int8Array::from(vec![0, 0, 2, 0])),
            &[("x", 2), ("y", 2)],
        );
        let any_y: Vec<bool> = any(&a, Some(&["y"]))
            .unwrap()
            .values()
            .as_boolean()
            .iter()
            .map(Option::unwrap)
            .collect();
        assert_eq!(any_y, vec![false, true]);
        assert!(!all(&a, None).unwrap().values().as_boolean().value(0));
    }

    #[test]
    fn sum_widens_like_numpy() {
        let a = nd(Arc::new(Int8Array::from(vec![100, 100])), &[("x", 2)]);
        assert_eq!(i64s(&sum(&a, None).unwrap()), vec![Some(200)]);
    }

    #[test]
    fn cumsum_and_diff() {
        let a = nd(
            Arc::new(Int32Array::from(vec![Some(1), None, Some(3), Some(4)])),
            &[("x", 2), ("y", 2)],
        );
        assert_eq!(
            i64s(&cumsum(&a, "x").unwrap()),
            vec![Some(1), None, Some(4), Some(4)]
        );
        assert_eq!(
            i64s(&cumprod(&a, "y").unwrap()),
            vec![Some(1), None, Some(3), Some(12)]
        );

        let b = nd(Arc::new(Int32Array::from(vec![1, 4, 9, 16])), &[("x", 4)]);
        let d = diff(&b, "x", 2).unwrap();
        assert_eq!(d.shape(), vec![2]);
        assert_eq!(
            d.values()
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            &[2, 2]
        );
    }
}
