//! Conversion to and from [`ndarray::ArrayD`], with the `ndarray` feature.
//!
//! `ndarray` arrays have no dimension names and no nulls, so a conversion
//! from an nd array needs values without nulls, and a conversion to an nd
//! array needs dimension names.

use std::sync::Arc;

use arrow::array::{Array, ArrowPrimitiveType, PrimitiveArray};
use ndarray::{ArrayD, IxDyn};

use crate::NdArrowArray;
use crate::dimensions::Dimensions;
use crate::error::{Result, nd_err};

impl NdArrowArray {
    /// Copy the values into an `ndarray` array of the same shape.
    /// The values must be of type `T` and without nulls.
    pub fn to_ndarray<T: ArrowPrimitiveType>(&self) -> Result<ArrayD<T::Native>> {
        let values = self
            .values()
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| {
                nd_err!(
                    InvalidArgument,
                    "values have type {}, not {}",
                    self.data_type(),
                    T::DATA_TYPE
                )
            })?;
        if values.null_count() > 0 {
            return Err(nd_err!(
                NotSupported,
                "ndarray has no nulls; fill the nulls first"
            ));
        }
        ArrayD::from_shape_vec(IxDyn(&self.shape()), values.values().to_vec())
            .map_err(|e| nd_err!(ShapeMismatch, "{e}"))
    }

    /// Build an nd array from an `ndarray` array in any memory layout,
    /// with one dimension name per axis.
    pub fn from_ndarray<T: ArrowPrimitiveType>(
        array: &ArrayD<T::Native>,
        names: &[&str],
    ) -> Result<Self> {
        if names.len() != array.ndim() {
            return Err(nd_err!(
                InvalidArgument,
                "{} names for an array of rank {}",
                names.len(),
                array.ndim()
            ));
        }
        let dims = Dimensions::try_new(names.iter().copied().zip(array.shape().iter().copied()))?;
        let values: PrimitiveArray<T> = PrimitiveArray::from_iter_values(array.iter().copied());
        NdArrowArray::try_new(Arc::new(values), dims)
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::Float64Type;
    use ndarray::{Array2, ShapeBuilder};

    use super::*;

    #[test]
    fn round_trip_keeps_logical_order() {
        let fortran = Array2::from_shape_vec((2, 3).f(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap()
            .into_dyn();
        let nd = NdArrowArray::from_ndarray::<Float64Type>(&fortran, &["y", "x"]).unwrap();
        assert_eq!(nd.shape(), vec![2, 3]);
        let back = nd.to_ndarray::<Float64Type>().unwrap();
        assert_eq!(back, fortran);
        assert!(NdArrowArray::from_ndarray::<Float64Type>(&fortran, &["y"]).is_err());
    }
}
