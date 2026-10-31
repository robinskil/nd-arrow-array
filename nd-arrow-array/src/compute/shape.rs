//! Layout kernels: broadcast, transpose, squeeze, expand, select, take, concat.
//!
//! Axes are addressed by dimension name. Layout kernels keep the attributes.

use std::sync::Arc;

use arrow::array::{Array, MutableArrayData, make_array};

use crate::NdArrowArray;
use crate::compute::layout::{Gather, GatherAxis};
use crate::dimensions::{Dimension, Dimensions};
use crate::error::{Result, nd_err};

/// A Python-style slice: `start:stop:step`. Negative indices count from the end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slice {
    pub start: Option<isize>,
    pub stop: Option<isize>,
    pub step: isize,
}

impl Slice {
    pub fn new(start: Option<isize>, stop: Option<isize>, step: isize) -> Self {
        Self { start, stop, step }
    }

    /// `start..stop` with step 1.
    pub fn range(start: isize, stop: isize) -> Self {
        Self::new(Some(start), Some(stop), 1)
    }

    /// `::step`.
    pub fn step(step: isize) -> Self {
        Self::new(None, None, step)
    }

    /// The whole axis, `:`.
    pub fn full() -> Self {
        Self::step(1)
    }

    /// Resolve against an axis of `len`: (first index, count, step), like `slice.indices`.
    pub fn resolve(&self, len: usize) -> Result<(usize, usize, isize)> {
        let step = self.step;
        if step == 0 {
            return Err(nd_err!(InvalidArgument, "slice step cannot be zero"));
        }
        let len = len as isize;
        let clamp = |value: Option<isize>, default: isize, low: isize, high: isize| match value {
            None => default,
            Some(v) if v < 0 => (v + len).max(low),
            Some(v) => v.min(high),
        };
        let (start, stop) = if step > 0 {
            (clamp(self.start, 0, 0, len), clamp(self.stop, len, 0, len))
        } else {
            (
                clamp(self.start, len - 1, -1, len - 1),
                clamp(self.stop, -1, -1, len - 1),
            )
        };
        let count = if step > 0 {
            if stop > start {
                (stop - start + step - 1) / step
            } else {
                0
            }
        } else if start > stop {
            (start - stop - step - 1) / -step
        } else {
            0
        };
        Ok((start.max(0) as usize, count as usize, step))
    }
}

impl From<std::ops::Range<isize>> for Slice {
    fn from(range: std::ops::Range<isize>) -> Self {
        Self::range(range.start, range.end)
    }
}

impl From<std::ops::RangeFrom<isize>> for Slice {
    fn from(range: std::ops::RangeFrom<isize>) -> Self {
        Self::new(Some(range.start), None, 1)
    }
}

impl From<std::ops::RangeTo<isize>> for Slice {
    fn from(range: std::ops::RangeTo<isize>) -> Self {
        Self::new(None, Some(range.end), 1)
    }
}

impl From<std::ops::RangeFull> for Slice {
    fn from(_: std::ops::RangeFull) -> Self {
        Self::full()
    }
}

/// A selection on one axis for [`isel`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selector {
    /// Keep a slice of the axis.
    Slice(Slice),
    /// Keep one index and drop the axis. Negative indices count from the end.
    Index(isize),
    /// Keep the given indices, in order. Negative indices count from the end.
    Take(Vec<isize>),
}

impl From<Slice> for Selector {
    fn from(slice: Slice) -> Self {
        Self::Slice(slice)
    }
}

impl From<isize> for Selector {
    fn from(index: isize) -> Self {
        Self::Index(index)
    }
}

impl From<std::ops::Range<isize>> for Selector {
    fn from(range: std::ops::Range<isize>) -> Self {
        Self::Slice(range.into())
    }
}

impl From<Vec<isize>> for Selector {
    fn from(indices: Vec<isize>) -> Self {
        Self::Take(indices)
    }
}

pub(crate) fn resolve_index(index: isize, len: usize, dim: &str) -> Result<usize> {
    let resolved = if index < 0 {
        index + len as isize
    } else {
        index
    };
    if resolved < 0 || resolved as usize >= len {
        return Err(nd_err!(
            InvalidArgument,
            "index {index} is out of bounds for dimension '{dim}' of size {len}"
        ));
    }
    Ok(resolved as usize)
}

/// Broadcast `a` onto `target` without a copy where possible.
///
/// Axes align by name. Each axis of `a` must be in `target` with the same size
/// or size 1. The result has the axis order of `target`.
pub fn broadcast_to(a: &NdArrowArray, target: &Dimensions) -> Result<NdArrowArray> {
    if a.dims() == target {
        return Ok(a.clone());
    }
    a.dims().check_broadcast_to(target)?;
    let strides = a.dims().c_strides();
    let axes = target
        .iter()
        .map(|dim| {
            let stride = match a.dims().position(dim.name()) {
                Some(axis) if a.dims().as_slice()[axis].size() == dim.size() => strides[axis],
                _ => 0,
            };
            GatherAxis {
                dim: dim.clone(),
                offsets: (0..dim.size()).map(|i| i * stride).collect(),
            }
        })
        .collect();
    Gather { base: 0, axes }.apply(a)
}

/// Broadcast all arrays onto their shared dimensions, see [`Dimensions::broadcast`].
pub fn broadcast_arrays(arrays: &[&NdArrowArray]) -> Result<Vec<NdArrowArray>> {
    let dims: Vec<&Dimensions> = arrays.iter().map(|a| a.dims()).collect();
    let target = Dimensions::broadcast(&dims)?;
    arrays.iter().map(|a| broadcast_to(a, &target)).collect()
}

/// Reorder the axes. `order` must name every dimension once.
pub fn transpose(a: &NdArrowArray, order: &[&str]) -> Result<NdArrowArray> {
    if order.len() != a.rank() {
        return Err(nd_err!(
            InvalidArgument,
            "transpose order {order:?} does not match dimensions {}",
            a.dims()
        ));
    }
    let identity = Gather::identity(a.dims());
    let mut axes = Vec::with_capacity(order.len());
    for name in order {
        let axis = a.dims().axis(name)?;
        axes.push(identity.axes[axis].clone());
    }
    let plan = Gather { base: 0, axes };
    plan.dims()?;
    plan.apply(a)
}

/// Reverse the axis order, like `ndarray.T`.
pub fn transpose_reversed(a: &NdArrowArray) -> Result<NdArrowArray> {
    let names: Vec<&str> = a.dims().names().into_iter().rev().collect();
    transpose(a, &names)
}

/// Move dimension `name` to position `to`.
pub fn move_dim(a: &NdArrowArray, name: &str, to: usize) -> Result<NdArrowArray> {
    let mut names = a.dims().names();
    let from = a.dims().axis(name)?;
    if to >= names.len() {
        return Err(nd_err!(InvalidArgument, "position {to} is out of bounds"));
    }
    let moved = names.remove(from);
    names.insert(to, moved);
    transpose(a, &names)
}

/// Remove size-1 axes without a copy. `None` removes every size-1 axis.
pub fn squeeze(a: &NdArrowArray, dims: Option<&[&str]>) -> Result<NdArrowArray> {
    if let Some(names) = dims {
        for name in names {
            let axis = a.dims().axis(name)?;
            let size = a.dims().as_slice()[axis].size();
            if size != 1 {
                return Err(nd_err!(
                    InvalidArgument,
                    "cannot squeeze dimension '{name}' of size {size}"
                ));
            }
        }
    }
    let kept = a.dims().iter().filter(|d| match dims {
        None => d.size() != 1,
        Some(names) => !names.contains(&d.name()),
    });
    replace_dims(a, Dimensions::try_new(kept.cloned())?)
}

/// Insert a new axis of size 1 at position `axis`, without a copy.
pub fn expand_dims(a: &NdArrowArray, name: &str, axis: usize) -> Result<NdArrowArray> {
    if axis > a.rank() {
        return Err(nd_err!(InvalidArgument, "position {axis} is out of bounds"));
    }
    let mut dims: Vec<Dimension> = a.dims().iter().cloned().collect();
    dims.insert(axis, Dimension::new(name, 1));
    replace_dims(a, Dimensions::try_new(dims)?)
}

/// Rename dimensions without a copy.
pub fn rename_dims(a: &NdArrowArray, renames: &[(&str, &str)]) -> Result<NdArrowArray> {
    for (old, _) in renames {
        a.dims().axis(old)?;
    }
    let dims = a
        .dims()
        .iter()
        .map(|d| match renames.iter().find(|(old, _)| *old == d.name()) {
            Some((_, new)) => Dimension::new(*new, d.size()),
            None => d.clone(),
        });
    replace_dims(a, Dimensions::try_new(dims)?)
}

fn replace_dims(a: &NdArrowArray, dims: Dimensions) -> Result<NdArrowArray> {
    Ok(NdArrowArray::try_new(a.values().clone(), dims)?.with_attributes(a.attributes().clone()))
}

/// Select by position on one or more axes, like xarray `isel`, in one copy.
///
/// A slice keeps the axis, an index drops it, a take keeps the listed indices.
pub fn isel(a: &NdArrowArray, selections: &[(&str, Selector)]) -> Result<NdArrowArray> {
    let mut plan = Gather::identity(a.dims());
    let mut dropped = vec![false; a.rank()];
    for (name, selector) in selections {
        let axis_index = a.dims().axis(name)?;
        let axis = &mut plan.axes[axis_index];
        let len = axis.offsets.len();
        let stride = if len > 1 {
            axis.offsets[1] - axis.offsets[0]
        } else {
            0
        };
        let source: Vec<usize> = match selector {
            Selector::Slice(slice) => {
                let (start, count, step) = slice.resolve(len)?;
                (0..count)
                    .map(|i| (start as isize + i as isize * step) as usize)
                    .collect()
            }
            Selector::Index(index) => {
                dropped[axis_index] = true;
                vec![resolve_index(*index, len, name)?]
            }
            Selector::Take(indices) => indices
                .iter()
                .map(|&i| resolve_index(i, len, name))
                .collect::<Result<_>>()?,
        };
        axis.dim = Dimension::new(axis.dim.name(), source.len());
        axis.offsets = source.into_iter().map(|i| i * stride).collect();
    }
    let mut kept = Vec::with_capacity(plan.axes.len());
    for (axis, dropped) in plan.axes.into_iter().zip(dropped) {
        if dropped {
            plan.base += axis.offsets[0];
        } else {
            kept.push(axis);
        }
    }
    plan.axes = kept;
    plan.apply(a)
}

/// Take `indices` along `dim`, like `numpy.take(a, indices, axis)`.
pub fn take(a: &NdArrowArray, dim: &str, indices: &[isize]) -> Result<NdArrowArray> {
    isel(a, &[(dim, Selector::Take(indices.to_vec()))])
}

/// Reverse the order along `dim`, like `numpy.flip`.
pub fn flip(a: &NdArrowArray, dim: &str) -> Result<NdArrowArray> {
    isel(a, &[(dim, Selector::Slice(Slice::step(-1)))])
}

/// Roll the elements along `dim` by `shift`, like `numpy.roll`.
pub fn roll(a: &NdArrowArray, dim: &str, shift: isize) -> Result<NdArrowArray> {
    let len = a
        .dims()
        .size_of(dim)
        .ok_or_else(|| nd_err!(InvalidArgument, "dimension '{dim}' is not in {}", a.dims()))?
        as isize;
    if len == 0 {
        return Ok(a.clone());
    }
    let indices: Vec<isize> = (0..len).map(|i| (i - shift).rem_euclid(len)).collect();
    take(a, dim, &indices)
}

/// Join arrays along an existing dimension, like `numpy.concatenate`.
///
/// All arrays need the same dimensions, except for the size of `dim`. Arrays
/// with another axis order are transposed to the order of the first array.
/// The result keeps the attributes of the first array.
pub fn concat(arrays: &[&NdArrowArray], dim: &str) -> Result<NdArrowArray> {
    let Some(first) = arrays.first() else {
        return Err(nd_err!(InvalidArgument, "concat needs at least one array"));
    };
    let axis = first.dims().axis(dim)?;
    let order = first.dims().names();
    let mut aligned: Vec<NdArrowArray> = Vec::with_capacity(arrays.len());
    let mut total = 0usize;
    for array in arrays {
        if array.data_type() != first.data_type() {
            return Err(nd_err!(
                SchemaMismatch,
                "concat value types differ: {} and {}",
                first.data_type(),
                array.data_type()
            ));
        }
        let array = if array.dims().names() == order {
            (*array).clone()
        } else {
            transpose(array, &order)?
        };
        for (d, f) in array.dims().iter().zip(first.dims().iter()) {
            if d.name() != dim && d.size() != f.size() {
                return Err(nd_err!(
                    ShapeMismatch,
                    "concat along '{dim}': dimension '{}' has sizes {} and {}",
                    d.name(),
                    f.size(),
                    d.size()
                ));
            }
        }
        total += array.dims().as_slice()[axis].size();
        aligned.push(array);
    }

    let outer: usize = first.dims().as_slice()[..axis]
        .iter()
        .map(Dimension::size)
        .product();
    let inner: usize = first.dims().as_slice()[axis + 1..]
        .iter()
        .map(Dimension::size)
        .product();
    let data: Vec<_> = aligned.iter().map(|a| a.values().to_data()).collect();
    let mut out = MutableArrayData::new(data.iter().collect(), false, outer * total * inner);
    for o in 0..outer {
        for (i, array) in aligned.iter().enumerate() {
            let block = array.dims().as_slice()[axis].size() * inner;
            out.extend(i, o * block, (o + 1) * block);
        }
    }
    let values = make_array(out.freeze());
    let dims = first.dims().iter().map(|d| {
        if d.name() == dim {
            Dimension::new(dim, total)
        } else {
            d.clone()
        }
    });
    Ok(NdArrowArray::try_new(
        Arc::new(values) as Arc<dyn Array>,
        Dimensions::try_new(dims)?,
    )?
    .with_attributes(first.attributes().clone()))
}

/// Join arrays along a new dimension at position `axis`, like `numpy.stack`.
pub fn stack(arrays: &[&NdArrowArray], dim: &str, axis: usize) -> Result<NdArrowArray> {
    let expanded = arrays
        .iter()
        .map(|a| expand_dims(a, dim, axis))
        .collect::<Result<Vec<_>>>()?;
    concat(&expanded.iter().collect::<Vec<_>>(), dim)
}

#[cfg(test)]
mod tests {
    use arrow::array::{AsArray, Int32Array, StringArray};
    use arrow::datatypes::Int32Type;

    use super::*;

    fn arange(dims: &[(&str, usize)]) -> NdArrowArray {
        let dims = Dimensions::try_new(dims.iter().copied()).unwrap();
        let n = dims.num_elements() as i32;
        NdArrowArray::try_new(Arc::new(Int32Array::from_iter_values(0..n)), dims).unwrap()
    }

    fn values(a: &NdArrowArray) -> Vec<i32> {
        a.values().as_primitive::<Int32Type>().values().to_vec()
    }

    fn dims(spec: &[(&str, usize)]) -> Dimensions {
        Dimensions::try_new(spec.iter().copied()).unwrap()
    }

    #[test]
    fn slice_resolve_matches_python() {
        let cases = [
            (Slice::full(), 5, (0, 5, 1)),
            (Slice::range(1, 4), 5, (1, 3, 1)),
            (Slice::range(-2, 100), 5, (3, 2, 1)),
            (Slice::step(2), 5, (0, 3, 2)),
            (Slice::step(-1), 5, (4, 5, -1)),
            (Slice::new(Some(3), Some(0), -2), 5, (3, 2, -2)),
            (Slice::range(4, 1), 5, (4, 0, 1)),
        ];
        for (slice, len, expected) in cases {
            assert_eq!(slice.resolve(len).unwrap(), expected, "{slice:?}");
        }
        assert!(Slice::step(0).resolve(3).is_err());
    }

    #[test]
    fn broadcast_inserts_and_stretches_axes() {
        let a = arange(&[("lat", 3)]);
        let b = broadcast_to(&a, &dims(&[("time", 2), ("lat", 3)])).unwrap();
        assert_eq!(values(&b), vec![0, 1, 2, 0, 1, 2]);

        let col = arange(&[("time", 2), ("lat", 1)]);
        let c = broadcast_to(&col, &dims(&[("lat", 3), ("time", 2)])).unwrap();
        assert_eq!(values(&c), vec![0, 1, 0, 1, 0, 1]);

        assert!(broadcast_to(&a, &dims(&[("lat", 4)])).is_err());
    }

    #[test]
    fn broadcast_arrays_unions_dims() {
        let a = arange(&[("x", 2)]);
        let b = arange(&[("y", 3)]);
        let out = broadcast_arrays(&[&a, &b]).unwrap();
        assert_eq!(out[0].dims(), &dims(&[("x", 2), ("y", 3)]));
        assert_eq!(values(&out[0]), vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(values(&out[1]), vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn transpose_and_move_dim() {
        let a = arange(&[("x", 2), ("y", 3), ("z", 2)]);
        let t = transpose(&a, &["z", "x", "y"]).unwrap();
        assert_eq!(t.shape(), vec![2, 2, 3]);
        assert_eq!(values(&t), vec![0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]);
        assert_eq!(values(&move_dim(&a, "z", 0).unwrap()), values(&t));
        assert!(transpose(&a, &["x", "y"]).is_err());
        assert!(transpose(&a, &["x", "x", "y"]).is_err());
        let r = transpose_reversed(&arange(&[("x", 2), ("y", 3)])).unwrap();
        assert_eq!(values(&r), vec![0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn squeeze_expand_rename() {
        let a = arange(&[("x", 1), ("y", 3), ("z", 1)]);
        assert_eq!(squeeze(&a, None).unwrap().dims(), &dims(&[("y", 3)]));
        assert_eq!(
            squeeze(&a, Some(&["z"])).unwrap().dims(),
            &dims(&[("x", 1), ("y", 3)])
        );
        assert!(squeeze(&a, Some(&["y"])).is_err());
        let e = expand_dims(&a, "w", 3).unwrap();
        assert_eq!(e.dims().names(), vec!["x", "y", "z", "w"]);
        let r = rename_dims(&a, &[("y", "lat")]).unwrap();
        assert_eq!(r.dims().names(), vec!["x", "lat", "z"]);
        assert!(rename_dims(&a, &[("q", "r")]).is_err());
    }

    #[test]
    fn isel_slices_indexes_and_takes() {
        let a = arange(&[("x", 3), ("y", 4)]);
        let s = isel(&a, &[("y", Slice::range(1, 3).into())]).unwrap();
        assert_eq!(values(&s), vec![1, 2, 5, 6, 9, 10]);

        let i = isel(&a, &[("x", Selector::Index(-1))]).unwrap();
        assert_eq!(i.dims(), &dims(&[("y", 4)]));
        assert_eq!(values(&i), vec![8, 9, 10, 11]);

        let both = isel(
            &a,
            &[
                ("x", Slice::step(-2).into()),
                ("y", Selector::Take(vec![3, 0])),
            ],
        )
        .unwrap();
        assert_eq!(values(&both), vec![11, 8, 3, 0]);
        assert!(isel(&a, &[("x", Selector::Index(3))]).is_err());
    }

    #[test]
    fn flip_and_roll() {
        let a = arange(&[("x", 5)]);
        assert_eq!(values(&flip(&a, "x").unwrap()), vec![4, 3, 2, 1, 0]);
        assert_eq!(values(&roll(&a, "x", 2).unwrap()), vec![3, 4, 0, 1, 2]);
        assert_eq!(values(&roll(&a, "x", -1).unwrap()), vec![1, 2, 3, 4, 0]);
    }

    #[test]
    fn concat_and_stack() {
        let a = arange(&[("x", 2), ("y", 2)]);
        let b = arange(&[("x", 1), ("y", 2)]);
        let c = concat(&[&a, &b], "x").unwrap();
        assert_eq!(c.dims(), &dims(&[("x", 3), ("y", 2)]));
        assert_eq!(values(&c), vec![0, 1, 2, 3, 0, 1]);

        let d = concat(&[&a, &a], "y").unwrap();
        assert_eq!(values(&d), vec![0, 1, 0, 1, 2, 3, 2, 3]);

        let t = transpose(&a, &["y", "x"]).unwrap();
        assert_eq!(
            values(&concat(&[&a, &t], "x").unwrap()),
            vec![0, 1, 2, 3, 0, 1, 2, 3]
        );

        let s = stack(&[&a, &a], "member", 0).unwrap();
        assert_eq!(s.dims(), &dims(&[("member", 2), ("x", 2), ("y", 2)]));
        assert!(concat(&[&a, &arange(&[("x", 2), ("y", 3)])], "x").is_err());
    }

    #[test]
    fn layout_kernels_work_for_strings() {
        let a = NdArrowArray::try_new(
            Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
            dims(&[("x", 2), ("y", 2)]),
        )
        .unwrap();
        let t = transpose(&a, &["y", "x"]).unwrap();
        let s: Vec<&str> = t
            .values()
            .as_string::<i32>()
            .iter()
            .map(Option::unwrap)
            .collect();
        assert_eq!(s, vec!["a", "c", "b", "d"]);
    }
}
