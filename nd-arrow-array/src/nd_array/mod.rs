use std::{fmt::Debug, sync::Arc};

use arrow::{
    array::{Array, AsArray, Scalar},
    compute::CastOptions,
};
use dimension::Dimension;

use crate::{
    broadcast::{self, BroadcastResult},
    nd_array::{
        arrow_backed::NdArrowArrayImpl, default::DefaultNdArrowArray, dimension::DimensionRef,
    },
};

pub mod arrow_backed;
pub mod arrow_ext;
pub mod chunk;
pub mod default;
pub mod dimension;

pub trait NdArrowArray: Debug + Send + Sync + 'static {
    ///
    /// Cast the array to a new data type, optionally using cast options.
    /// ///
    /// # Arguments
    /// * `data_type` - The target data type to cast to.
    /// * `cast_options` - Optional cast options to control the casting behavior.
    /// # Returns
    /// A new `NdArrowArray` with the specified data type. (Using the default `DefaultNdArrowArray` implementation)
    fn cast(
        &self,
        data_type: arrow::datatypes::DataType,
        cast_options: Option<CastOptions>,
    ) -> Result<Arc<dyn NdArrowArray>, arrow::error::ArrowError> {
        let inner_array = arrow::compute::kernels::cast::cast_with_options(
            self.array().as_ref(),
            &data_type,
            &cast_options.unwrap_or_default(),
        )?;

        let new_array = DefaultNdArrowArray::new(inner_array, self.dimensions().to_vec());

        Ok(Arc::new(new_array) as Arc<dyn NdArrowArray>)
    }
    /// Returns the shape of the array as a vector of usize.
    fn shape(&self) -> Vec<usize>;

    /// Returns the dimensions of the array.
    fn dimensions_ref<'a>(&'a self) -> std::borrow::Cow<'a, [Dimension]> {
        std::borrow::Cow::Borrowed(self.dimensions())
    }
    #[deprecated(
        note = "This method is deprecated and will be removed in future versions. Use `dimensions_ref` instead."
    )]
    fn dimensions(&self) -> &[Dimension] {
        unimplemented!("Use `dimensions_ref` instead")
    }

    /// Returns the actual values as a flattened array.
    fn values_array(&self) -> Arc<dyn Array> {
        self.array()
    }
    #[deprecated(
        note = "This method is deprecated and will be removed in future versions. Use `values_array` instead."
    )]
    fn array(&self) -> Arc<dyn Array>;
    fn data_type(&self) -> arrow::datatypes::DataType {
        self.array().data_type().clone()
    }
    #[deprecated(note = "Use `data_type` instead. This method will be removed in future versions.")]
    fn dtype(&self) -> arrow::datatypes::DataType {
        self.array().data_type().clone()
    }
    #[deprecated(
        note = "This method is deprecated and will be removed in future versions. Use `data_type` instead."
    )]
    fn generate_field(&self, name: &str) -> arrow::datatypes::Field {
        arrow::datatypes::Field::new(name, self.data_type(), self.is_nullable())
    }
    fn is_nullable(&self) -> bool {
        self.array().is_nullable()
    }
    fn is_scalar(&self) -> bool {
        self.shape().len() == 0
    }

    fn broadcast(&self, target_dimensions: &[Dimension]) -> BroadcastResult<Arc<dyn NdArrowArray>> {
        broadcast::broadcast_array(self, target_dimensions)
    }

    fn to_arrow_array(&self) -> Result<Arc<dyn Array>, arrow_ext::ArrowParseError> {
        arrow_ext::to_arrow_array(self)
    }

    fn arrow_encoded_dtype(&self) -> arrow::datatypes::DataType {
        arrow_ext::arrow_encoded_dtype(self)
    }

    fn generate_arrow_encoded_field(&self, name: &str) -> arrow::datatypes::Field {
        let dtype = self.arrow_encoded_dtype();
        arrow::datatypes::Field::new(name, dtype, self.is_nullable())
    }
}

pub fn new_from_arrow_array<D: Into<Dimension>>(
    array: Arc<dyn Array>,
    dimensions: Vec<D>,
) -> Arc<dyn NdArrowArray> {
    Arc::new(NdArrowArrayImpl::new(
        array,
        dimensions
            .into_iter()
            .map(|d| d.into())
            .map(|d: Dimension| (d.name.clone(), d.size()))
            .collect(),
    ))
}

pub fn new_null_nd_arrow_array(len: usize) -> Arc<dyn NdArrowArray> {
    Arc::new(NdArrowArrayImpl::null(
        arrow::datatypes::DataType::Null,
        len,
    ))
}

pub fn new_null_nd_arrow_array_with_dtype(
    data_type: arrow::datatypes::DataType,
    len: usize,
) -> Arc<dyn NdArrowArray> {
    Arc::new(NdArrowArrayImpl::null(data_type, len))
}

pub fn new_from_arrow_encoded_array(
    array: Scalar<Arc<dyn Array>>,
) -> Result<Arc<dyn NdArrowArray>, arrow_ext::ArrowParseError> {
    NdArrowArrayImpl::try_from_arrow(array)
        .map(|nd_array_impl| Arc::new(nd_array_impl) as Arc<dyn NdArrowArray>)
        .map_err(|e| arrow_ext::ArrowParseError::from(e))
}

pub fn to_arrow_encoded_array(
    array: &dyn NdArrowArray,
) -> Result<Arc<dyn Array>, arrow_ext::ArrowParseError> {
    array.to_arrow_array()
}

#[cfg(test)]
mod tests {
    use crate::nd_array::default::DefaultNdArrowArray;

    use super::*;
    use arrow::array::Int32Array;
    #[test]
    fn test_shape() {
        let dims = vec![Dimension::new("dim1", 2), Dimension::new("dim2", 2)];
        let array = DefaultNdArrowArray::new(
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])) as Arc<dyn Array>,
            dims,
        );
        assert_eq!(array.shape(), &[2, 2]);
    }
}
