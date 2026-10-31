//! N-dimensional arrays as an Arrow extension type.
//!
//! - [`NdArrowArray`]: one N-dimensional array: flat values, named
//!   [`Dimensions`] and [`Attributes`].
//! - [`NdArrayType`]: the `nd_arrow.array` Arrow extension type. Its storage
//!   is `Struct { values, dim_names, dim_sizes, attrs }`.
//! - [`NdColumn`]: an Arrow column of the extension type. Each row is one
//!   [`NdArrowArray`].
//! - [`NdRecordBatch`]: nd arrays over a shared target grid, encoded as a
//!   [`RecordBatch`](arrow::record_batch::RecordBatch) of nd columns.

pub mod array;
pub mod attributes;
pub mod batch;
pub mod column;
pub mod dimensions;
pub mod error;
pub mod extension;

pub use array::NdArrowArray;
pub use attributes::{Attributes, IntoAttributeValue};
pub use batch::{NdRecordBatch, encode_batches, infer_target};
pub use column::{NdColumn, NdColumnBuilder};
pub use dimensions::{Dimension, Dimensions};
pub use error::{NdError, Result};
pub use extension::{NdArrayMetadata, NdArrayType};
