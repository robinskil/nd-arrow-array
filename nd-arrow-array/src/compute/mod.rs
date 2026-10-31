//! Compute kernels for [`NdArrowArray`](crate::NdArrowArray), with NumPy and xarray semantics.
//!
//! Axes are addressed by dimension name. Broadcasting aligns axes by name.
//! Missing values are Arrow nulls: kernels skip nulls like the NumPy `nan*`
//! functions.
//!
//! Attribute rules follow xarray defaults:
//! - layout kernels, `cast`, `where_`, `fill_null` and `clip` keep the attributes;
//! - arithmetic, comparisons, math functions and reductions drop them;
//! - [`decode_cf`] and [`encode_cf`] move the CF encoding keys.

pub(crate) mod layout;

mod batch;
mod ops;

pub mod cf;
pub mod elementwise;
pub mod groupby;
pub mod interp;
pub mod reduce;
pub mod search;
pub mod shape;
pub mod window;

pub use cf::{CfEncoding, decode_cf, encode_cf};
pub use elementwise::*;
pub use groupby::groupby_reduce;
pub use interp::interp;
pub use reduce::*;
pub use search::{SearchSide, argsort, searchsorted, sort, unique};
pub use shape::*;
pub use window::{RollingOp, rolling};
