use std::sync::Arc;

use arrow::array::TimestampNanosecondArray;
use chrono::NaiveDateTime;
use ndarray::{Dimension, OwnedRepr};

use super::default::DefaultNdArrowArray;

impl<D: Dimension> From<ndarray::ArrayBase<OwnedRepr<chrono::NaiveDateTime>, D>>
    for DefaultNdArrowArray
{
    fn from(value: ndarray::ArrayBase<OwnedRepr<NaiveDateTime>, D>) -> Self {
        let shape = value.shape().to_vec();
        let array = TimestampNanosecondArray::from_iter_values(
            value
                .into_raw_vec_and_offset()
                .0
                .iter()
                .map(|dt| dt.timestamp_nanos()),
        );

        DefaultNdArrowArray::new(Arc::new(array), shape, None)
    }
}

impl<D: Dimension> From<ndarray::ArrayBase<OwnedRepr<Option<chrono::NaiveDateTime>>, D>>
    for DefaultNdArrowArray
{
    fn from(value: ndarray::ArrayBase<OwnedRepr<Option<NaiveDateTime>>, D>) -> Self {
        let shape = value.shape().to_vec();
        let array = TimestampNanosecondArray::from_iter(
            value
                .into_raw_vec_and_offset()
                .0
                .iter()
                .map(|dt| dt.map(|dt| dt.timestamp_nanos())),
        );

        DefaultNdArrowArray::new(Arc::new(array), shape, None)
    }
}

macro_rules! impl_from_ndarray {
    ($t:ty, $arrow_array:ident) => {
        impl<D: ndarray::Dimension> From<ndarray::ArrayBase<ndarray::OwnedRepr<$t>, D>>
            for DefaultNdArrowArray
        {
            fn from(value: ndarray::ArrayBase<ndarray::OwnedRepr<$t>, D>) -> Self {
                let shape = value.shape().to_vec();
                let array = arrow::array::$arrow_array::from(value.into_raw_vec_and_offset().0);
                DefaultNdArrowArray::new(std::sync::Arc::new(array), shape, None)
            }
        }
    };
}

macro_rules! impl_opt_from_ndarray {
    ($t:ty, $arrow_array:ident) => {
        impl<D: ndarray::Dimension> From<ndarray::ArrayBase<ndarray::OwnedRepr<Option<$t>>, D>>
            for DefaultNdArrowArray
        {
            fn from(value: ndarray::ArrayBase<ndarray::OwnedRepr<Option<$t>>, D>) -> Self {
                let shape = value.shape().to_vec();
                let array = arrow::array::$arrow_array::from(value.into_raw_vec_and_offset().0);
                DefaultNdArrowArray::new(std::sync::Arc::new(array), shape, None)
            }
        }
    };
}

impl_from_ndarray!(i8, Int8Array);
impl_from_ndarray!(i16, Int16Array);
impl_from_ndarray!(i32, Int32Array);
impl_from_ndarray!(i64, Int64Array);
impl_from_ndarray!(u8, UInt8Array);
impl_from_ndarray!(u16, UInt16Array);
impl_from_ndarray!(u32, UInt32Array);
impl_from_ndarray!(u64, UInt64Array);
impl_from_ndarray!(f32, Float32Array);
impl_from_ndarray!(f64, Float64Array);
impl_from_ndarray!(bool, BooleanArray);
impl_from_ndarray!(String, StringArray);

impl_opt_from_ndarray!(i8, Int8Array);
impl_opt_from_ndarray!(i16, Int16Array);
impl_opt_from_ndarray!(i32, Int32Array);
impl_opt_from_ndarray!(i64, Int64Array);
impl_opt_from_ndarray!(u8, UInt8Array);
impl_opt_from_ndarray!(u16, UInt16Array);
impl_opt_from_ndarray!(u32, UInt32Array);
impl_opt_from_ndarray!(u64, UInt64Array);
impl_opt_from_ndarray!(f32, Float32Array);
impl_opt_from_ndarray!(f64, Float64Array);
impl_opt_from_ndarray!(bool, BooleanArray);
impl_opt_from_ndarray!(String, StringArray);

pub fn into_nd_arrow_array<I: Into<DefaultNdArrowArray>>(array: I) -> DefaultNdArrowArray {
    array.into()
}
