//! Method forms of common kernels, like NumPy array methods: `a.sum(..)`.

use arrow::datatypes::DataType;

use crate::NdArrowArray;
use crate::compute::{self, Selector};
use crate::dimensions::Dimensions;
use crate::error::Result;

impl NdArrowArray {
    /// See [`compute::broadcast_to`].
    pub fn broadcast_to(&self, target: &Dimensions) -> Result<Self> {
        compute::broadcast_to(self, target)
    }

    /// See [`compute::transpose`].
    pub fn transpose(&self, order: &[&str]) -> Result<Self> {
        compute::transpose(self, order)
    }

    /// See [`compute::squeeze`].
    pub fn squeeze(&self, dims: Option<&[&str]>) -> Result<Self> {
        compute::squeeze(self, dims)
    }

    /// See [`compute::expand_dims`].
    pub fn expand_dims(&self, name: &str, axis: usize) -> Result<Self> {
        compute::expand_dims(self, name, axis)
    }

    /// See [`compute::isel`].
    pub fn isel(&self, selections: &[(&str, Selector)]) -> Result<Self> {
        compute::isel(self, selections)
    }

    /// See [`compute::cast`].
    pub fn cast(&self, to: &DataType) -> Result<Self> {
        compute::cast(self, to)
    }

    /// See [`compute::fill_null`].
    pub fn fill_null(&self, value: &arrow::array::ArrayRef) -> Result<Self> {
        compute::fill_null(self, value)
    }

    /// See [`compute::sum`].
    pub fn sum(&self, axes: Option<&[&str]>) -> Result<Self> {
        compute::sum(self, axes)
    }

    /// See [`compute::mean`].
    pub fn mean(&self, axes: Option<&[&str]>) -> Result<Self> {
        compute::mean(self, axes)
    }

    /// See [`compute::min`].
    pub fn min(&self, axes: Option<&[&str]>) -> Result<Self> {
        compute::min(self, axes)
    }

    /// See [`compute::max`].
    pub fn max(&self, axes: Option<&[&str]>) -> Result<Self> {
        compute::max(self, axes)
    }

    /// See [`compute::count`].
    pub fn count(&self, axes: Option<&[&str]>) -> Result<Self> {
        compute::count(self, axes)
    }

    /// See [`compute::std`].
    pub fn std(&self, axes: Option<&[&str]>, ddof: usize) -> Result<Self> {
        compute::std(self, axes, ddof)
    }

    /// See [`compute::var`].
    pub fn var(&self, axes: Option<&[&str]>, ddof: usize) -> Result<Self> {
        compute::var(self, axes, ddof)
    }

    /// See [`compute::median`].
    pub fn median(&self, axes: Option<&[&str]>) -> Result<Self> {
        compute::median(self, axes)
    }

    /// See [`compute::resample()`].
    pub fn resample(
        &self,
        dim: &str,
        times: &NdArrowArray,
        freq: compute::Frequency,
        op: compute::Reduction,
    ) -> Result<(Self, Self)> {
        compute::resample(self, dim, times, freq, op)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{AsArray, Int32Array};
    use arrow::datatypes::Int64Type;

    use super::*;

    #[test]
    fn methods_chain() {
        let a = NdArrowArray::try_new(
            Arc::new(Int32Array::from_iter_values(0..6)),
            Dimensions::try_new([("x", 2), ("y", 3)]).unwrap(),
        )
        .unwrap();
        let s = a.transpose(&["y", "x"]).unwrap().sum(Some(&["x"])).unwrap();
        assert_eq!(s.values().as_primitive::<Int64Type>().values(), &[3, 5, 7]);
    }
}
