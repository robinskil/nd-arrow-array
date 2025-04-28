use std::{fmt::Debug, sync::Arc};

use arrow::array::Array;
use dimension::Dimension;

use crate::broadcast::{self, BroadcastResult};

pub mod arrow_ext;
pub mod default;
pub mod dimension;

pub trait NdArrowArray: Debug {
    fn shape(&self) -> Vec<usize>;
    fn dimensions(&self) -> &[Dimension];
    fn array(&self) -> Arc<dyn Array>;
    fn dtype(&self) -> arrow::datatypes::DataType {
        self.array().data_type().clone()
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

    fn arrow_encoded_type(&self) -> arrow::datatypes::DataType {
        arrow_ext::arrow_encoded_dtype(self)
    }
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
