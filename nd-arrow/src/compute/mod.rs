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

/// Minimum output length for parallel execution with the `parallel` feature.
#[cfg(feature = "parallel")]
pub(crate) const PARALLEL_MIN_LEN: usize = 1 << 20;

mod batch;
mod methods;
mod ops;

pub mod aggregate;
pub mod cf;
pub mod column;
pub mod elementwise;
pub mod groupby;
pub mod histogram;
pub mod interp;
pub mod partial;
pub mod reduce;
pub mod resample;
pub mod search;
pub mod shape;
pub mod time;
pub mod window;

pub use aggregate::{CoarsenBoundary, coarsen, dot, weighted_mean, weighted_sum};
pub use cf::{CfEncoding, decode_cf, encode_cf};
pub use column::{
    column_binary, column_cast, column_decode_cf, column_unary, row_reduce, row_reduce_axes,
};
pub use elementwise::*;
pub use groupby::groupby_reduce;
pub use histogram::{bincount, histogram};
pub use interp::{InterpMethod, interp, interp_with};
pub use partial::{PartialReduction, merge_all};
pub use reduce::*;
pub use resample::{Frequency, resample};
pub use search::{SearchSide, argsort, argwhere, digitize, isin, searchsorted, sort, unique};
pub use shape::*;
pub use time::{
    CfTimeUnits, DatePart, TimePeriod, decode_cf_time, encode_cf_time, floor_time, time_component,
};
pub use window::{RollingOp, rolling};
