//! Error type for nd arrays.

use arrow::error::ArrowError;

/// Errors raised by nd array construction, encoding and decoding.
#[derive(Debug, thiserror::Error)]
pub enum NdError {
    #[error(transparent)]
    Arrow(#[from] ArrowError),
    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("invalid attribute: {0}")]
    InvalidAttribute(String),
    #[error("invalid nd encoding: {0}")]
    InvalidEncoding(String),
    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
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

pub type Result<T, E = NdError> = std::result::Result<T, E>;

macro_rules! nd_err {
    ($variant:ident, $($arg:tt)*) => {
        $crate::error::NdError::$variant(format!($($arg)*))
    };
}

pub(crate) use nd_err;
