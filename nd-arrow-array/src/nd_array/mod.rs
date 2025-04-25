use std::sync::Arc;

use arrow::array::Array;

use crate::broadcast::{self, BroadcastResult};
pub mod arrow_ext;
pub mod default;
pub mod nd_array_ext;

pub trait NdArrowArray {
    fn shape(&self) -> &[usize];

    fn ndim(&self) -> usize;
    fn dim_names<'a>(&'a self) -> Box<dyn Iterator<Item = Option<&'a str>> + 'a>;

    /// Returns the values of the array as an `Arc<dyn Array>`.
    fn values(&self) -> Arc<dyn Array>;
    fn dtype(&self) -> arrow::datatypes::DataType {
        self.values().data_type().clone()
    }
    fn is_nullable(&self) -> bool {
        self.values().is_nullable()
    }

    fn broadcast(&self, target_shape: &[usize]) -> BroadcastResult<Arc<dyn NdArrowArray>> {
        broadcast::broadcast_array(self, target_shape)
    }

    fn to_arrow_array(&self) -> Result<Arc<dyn Array>, arrow_ext::ArrowParseError> {
        arrow_ext::to_arrow_array(self)
    }
}
