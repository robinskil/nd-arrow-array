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
//! - [`compute`]: kernels with NumPy and xarray semantics (layout,
//!   arithmetic, reductions, search, interpolation, CF decoding).
//! - [`tensor`]: conversion to and from the Arrow canonical tensor types.
//!
//! Features: `parallel` runs large gathers and reductions with rayon;
//! `ndarray` adds conversion to and from `ndarray::ArrayD`.
//!
//! # Example
//!
//! ```
//! use std::sync::Arc;
//!
//! use arrow::array::Float64Array;
//! use nd_arrow_array::compute::{Selector, Slice};
//! use nd_arrow_array::{Attributes, Dimensions, NdArrowArray, NdColumn};
//!
//! # fn main() -> nd_arrow_array::Result<()> {
//! let sst = NdArrowArray::try_new(
//!     Arc::new(Float64Array::from(vec![10.0, 11.0, 12.0, 14.0, 15.0, 16.0])),
//!     Dimensions::try_new([("time", 2), ("lat", 3)])?,
//! )?
//! .with_attributes(Attributes::new().with("units", "degC"));
//!
//! // Anomaly against the time mean: the mean broadcasts back by name.
//! let anomaly = (&sst - &sst.mean(Some(&["time"]))?)?;
//! let north = anomaly.isel(&[("lat", Selector::Slice(Slice::range(1, 3)))])?;
//! assert_eq!(north.shape(), vec![2, 2]);
//! println!("{north}");
//!
//! // One row of an Arrow column with the `nd_arrow.array` extension type.
//! let column = NdColumn::try_from_arrays([&sst])?;
//! assert_eq!(column.value(0), sst);
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

pub mod array;
pub mod attributes;
pub mod batch;
pub mod column;
pub mod compute;
pub mod dimensions;
mod display;
pub mod error;
pub mod extension;
#[cfg(feature = "ndarray")]
mod ndarray_interop;
pub mod tensor;

pub use array::NdArrowArray;
pub use attributes::{Attributes, IntoScalar};
pub use batch::{NdRecordBatch, encode_batches, infer_target};
pub use column::{NdColumn, NdColumnBuilder};
pub use dimensions::{Dimension, Dimensions};
pub use error::{NdError, Result};
pub use extension::{NdArrayMetadata, NdArrayType};
