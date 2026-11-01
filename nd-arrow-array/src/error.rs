//! Error type for nd arrays.

use arrow::error::ArrowError;

/// Errors raised by nd array construction, encoding and decoding.
#[derive(Debug, thiserror::Error)]
pub enum NdError {
    /// An error from an Arrow kernel.
    #[error(transparent)]
    Arrow(#[from] ArrowError),
    /// Dimension names or sizes are not valid.
    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),
    /// Shapes do not match or do not broadcast.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    /// An attribute is not valid.
    #[error("invalid attribute: {0}")]
    InvalidAttribute(String),
    /// Stored data does not follow the nd encoding.
    #[error("invalid nd encoding: {0}")]
    InvalidEncoding(String),
    /// Types or schemas do not match.
    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),
    /// An argument is not valid.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// The operation does not support the input.
    #[error("not supported: {0}")]
    NotSupported(String),
}

impl From<NdError> for ArrowError {
    fn from(error: NdError) -> Self {
        match error {
            NdError::Arrow(e) => e,
            other => ArrowError::InvalidArgumentError(other.to_string()),
        }
    }
}

/// A result with [`NdError`] as the default error.
pub type Result<T, E = NdError> = std::result::Result<T, E>;

macro_rules! nd_err {
    ($variant:ident, $($arg:tt)*) => {
        $crate::error::NdError::$variant(format!($($arg)*))
    };
}

pub(crate) use nd_err;
