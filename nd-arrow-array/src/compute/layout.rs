//! Gather plans: the base of all layout kernels.
//!
//! The values of an [`NdArrowArray`] are always C-contiguous. A [`Gather`]
//! describes an output array as a map from each output axis to positions in
//! the source values. Every layout operation (slice, step, broadcast,
//! transpose, take) only builds a plan, and [`Gather::apply`] copies once.

use arrow::array::{Array, ArrayRef, MutableArrayData, UInt32Array, UInt64Array, make_array};
use arrow::compute::take;

use crate::NdArrowArray;
use crate::dimensions::{Dimension, Dimensions};
use crate::error::Result;

/// Minimum length of a contiguous inner run that is copied as a block.
const MIN_RUN: usize = 8;

/// One output axis: its dimension and, per output index, the offset it adds
/// to the source position.
#[derive(Debug, Clone)]
pub(crate) struct GatherAxis {
    pub dim: Dimension,
    pub offsets: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct Gather {
    /// Offset added to every position, from axes that a selection removed.
    pub base: usize,
    pub axes: Vec<GatherAxis>,
}

impl Gather {
    /// The plan that reproduces `dims` unchanged.
    pub fn identity(dims: &Dimensions) -> Self {
        let axes = dims
            .iter()
            .zip(dims.c_strides())
            .map(|(dim, stride)| GatherAxis {
                dim: dim.clone(),
                offsets: (0..dim.size()).map(|i| i * stride).collect(),
            })
            .collect();
        Self { base: 0, axes }
    }

    pub fn len(&self) -> usize {
        self.axes.iter().map(|a| a.offsets.len()).product()
    }

    pub fn dims(&self) -> Result<Dimensions> {
        Dimensions::try_new(self.axes.iter().map(|a| a.dim.clone()))
    }

    /// Copy the planned positions of `source` into a new array.
    pub fn apply(&self, source: &NdArrowArray) -> Result<NdArrowArray> {
        let dims = self.dims()?;
        let values = self.gather_values(source.values())?;
        Ok(NdArrowArray::new_unchecked(
            values,
            dims,
            source.attributes().clone(),
        ))
    }

    fn gather_values(&self, values: &ArrayRef) -> Result<ArrayRef> {
        let len = self.len();
        if len == 0 {
            return Ok(values.slice(0, 0));
        }
        if let Some(start) = self.contiguous_start() {
            return Ok(values.slice(start, len));
        }
        let run = self.axes.last().map_or(0, |a| a.offsets.len());
        if run >= MIN_RUN && is_consecutive(&self.axes.last().unwrap().offsets) {
            return Ok(self.copy_runs(values, run));
        }
        let positions = self.positions();
        let indices: ArrayRef = if values.len() <= u32::MAX as usize {
            std::sync::Arc::new(UInt32Array::from_iter_values(
                positions.into_iter().map(|p| p as u32),
            ))
        } else {
            std::sync::Arc::new(UInt64Array::from_iter_values(
                positions.into_iter().map(|p| p as u64),
            ))
        };
        Ok(take(values.as_ref(), indices.as_ref(), None)?)
    }

    /// The start position when the output is one contiguous block of the source.
    fn contiguous_start(&self) -> Option<usize> {
        let mut block = 1usize;
        for axis in self.axes.iter().rev() {
            let offsets = &axis.offsets;
            if offsets.len() > 1 {
                let stride = offsets[1].checked_sub(offsets[0])?;
                if stride != block || offsets.windows(2).any(|w| w[1] != w[0] + block) {
                    return None;
                }
            }
            block *= offsets.len();
        }
        Some(self.base + self.axes.iter().map(|a| a.offsets[0]).sum::<usize>())
    }

    fn copy_runs(&self, values: &ArrayRef, run: usize) -> ArrayRef {
        let data = values.to_data();
        let mut out = MutableArrayData::new(vec![&data], false, self.len());
        let outer = &self.axes[..self.axes.len() - 1];
        let first = self.base + self.axes.last().unwrap().offsets[0];
        for_each_position(outer, |start| {
            out.extend(0, start + first, start + first + run)
        });
        make_array(out.freeze())
    }

    /// All source positions in output (C) order.
    pub fn positions(&self) -> Vec<usize> {
        let mut positions = Vec::with_capacity(self.len());
        let Some((last, outer)) = self.axes.split_last() else {
            positions.push(self.base);
            return positions;
        };
        for_each_position(outer, |start| {
            let start = start + self.base;
            positions.extend(last.offsets.iter().map(|o| start + o));
        });
        positions
    }
}

fn is_consecutive(offsets: &[usize]) -> bool {
    offsets.windows(2).all(|w| w[1] == w[0] + 1)
}

/// Call `f` with the summed offset of every index combination of `axes`, in C order.
fn for_each_position(axes: &[GatherAxis], mut f: impl FnMut(usize)) {
    if axes.iter().any(|a| a.offsets.is_empty()) {
        return;
    }
    let mut index = vec![0usize; axes.len()];
    loop {
        f(axes.iter().zip(&index).map(|(a, &i)| a.offsets[i]).sum());
        let mut axis = axes.len();
        loop {
            if axis == 0 {
                return;
            }
            axis -= 1;
            index[axis] += 1;
            if index[axis] < axes[axis].offsets.len() {
                break;
            }
            index[axis] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{AsArray, Int32Array};
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

    #[test]
    fn identity_is_zero_copy() {
        let a = arange(&[("x", 2), ("y", 3)]);
        let plan = Gather::identity(a.dims());
        assert_eq!(plan.contiguous_start(), Some(0));
        assert_eq!(values(&plan.apply(&a).unwrap()), (0..6).collect::<Vec<_>>());
    }

    #[test]
    fn reversed_axes_transpose() {
        let a = arange(&[("x", 2), ("y", 3)]);
        let mut plan = Gather::identity(a.dims());
        plan.axes.reverse();
        let t = plan.apply(&a).unwrap();
        assert_eq!(t.dims().names(), vec!["y", "x"]);
        assert_eq!(values(&t), vec![0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn long_inner_runs_copy_blocks() {
        let a = arange(&[("x", 3), ("y", 10)]);
        let mut plan = Gather::identity(a.dims());
        plan.axes[0].offsets = vec![20, 0];
        plan.axes[0].dim = Dimension::new("x", 2);
        let out = plan.apply(&a).unwrap();
        let expected: Vec<i32> = (20..30).chain(0..10).collect();
        assert_eq!(values(&out), expected);
    }
}
