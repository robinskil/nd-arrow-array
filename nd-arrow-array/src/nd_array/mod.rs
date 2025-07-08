use std::{fmt::Debug, sync::Arc};

use arrow::{
    array::{Array, AsArray},
    compute::CastOptions,
};
use dimension::Dimension;

use crate::{
    broadcast::{self, BroadcastResult},
    nd_array::default::DefaultNdArrowArray,
};

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
    fn shape(&self) -> Vec<usize>;
    fn dimensions(&self) -> &[Dimension];
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
    Arc::new(DefaultNdArrowArray::new(array, dimensions))
}

pub fn new_null_nd_arrow_array(len: usize) -> Arc<dyn NdArrowArray> {
    let null_array = arrow::array::NullArray::new(len);
    Arc::new(DefaultNdArrowArray::new(
        Arc::new(null_array) as Arc<dyn Array>,
        Vec::<(String, usize)>::new(),
    ))
}

pub fn new_from_arrow_encoded_array(
    array: Arc<dyn Array>,
) -> Result<Arc<dyn NdArrowArray>, arrow_ext::ArrowParseError> {
    let struct_array = array
        .as_struct_opt()
        .ok_or(arrow_ext::ArrowParseError::StructDowncastError)?;

    arrow_ext::try_from_arrow_array(struct_array)
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
