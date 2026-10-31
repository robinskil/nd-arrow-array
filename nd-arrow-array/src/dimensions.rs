//! Named dimensions for N-dimensional Arrow arrays.
//!
//! A [`Dimensions`] value describes the logical shape of an
//! [`NdArrowArray`](crate::NdArrowArray) in C-order (row-major, first dimension
//! outermost). Dimension names drive xarray-style broadcasting: axes align by
//! name, never by position.

use std::fmt;
use std::sync::Arc;

use crate::error::{Result, nd_err};

/// A single named axis with a fixed size.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dimension {
    name: Arc<str>,
    size: usize,
}

impl Dimension {
    pub fn new(name: impl Into<Arc<str>>, size: usize) -> Self {
        Self {
            name: name.into(),
            size,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.name, self.size)
    }
}

impl<S: Into<Arc<str>>> From<(S, usize)> for Dimension {
    fn from((name, size): (S, usize)) -> Self {
        Self::new(name, size)
    }
}

/// An ordered list of named axes, C-order. Cheap to clone.
///
/// Rank 0 (no axes) describes a scalar: one logical element.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Dimensions {
    dims: Arc<[Dimension]>,
}

impl Dimensions {
    /// Create dimensions, validating that axis names are non-empty and unique.
    pub fn try_new(dims: impl IntoIterator<Item = impl Into<Dimension>>) -> Result<Self> {
        let dims: Vec<Dimension> = dims.into_iter().map(Into::into).collect();
        for (i, dim) in dims.iter().enumerate() {
            if dim.name().is_empty() {
                return Err(nd_err!(
                    InvalidDimensions,
                    "dimension at axis {i} has an empty name"
                ));
            }
            if dims[..i].iter().any(|d| d.name() == dim.name()) {
                return Err(nd_err!(
                    InvalidDimensions,
                    "duplicate dimension name '{}'",
                    dim.name()
                ));
            }
        }
        Ok(Self { dims: dims.into() })
    }

    /// Rank-0 dimensions describing a scalar.
    pub fn scalar() -> Self {
        Self::default()
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Dimension> {
        self.dims.iter()
    }

    pub fn as_slice(&self) -> &[Dimension] {
        &self.dims
    }

    pub fn get(&self, axis: usize) -> Option<&Dimension> {
        self.dims.get(axis)
    }

    pub fn shape(&self) -> Vec<usize> {
        self.dims.iter().map(Dimension::size).collect()
    }

    pub fn names(&self) -> Vec<&str> {
        self.dims.iter().map(Dimension::name).collect()
    }

    /// Total number of logical elements (1 for a scalar, 0 if any axis is empty).
    pub fn num_elements(&self) -> usize {
        self.dims.iter().map(Dimension::size).product()
    }

    /// Like [`Self::num_elements`], but `None` on `usize` overflow.
    pub fn checked_num_elements(&self) -> Option<usize> {
        self.dims
            .iter()
            .try_fold(1usize, |acc, d| acc.checked_mul(d.size()))
    }

    pub fn position(&self, name: &str) -> Option<usize> {
        self.dims.iter().position(|d| d.name() == name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.position(name).is_some()
    }

    /// The axis of `name`, or an error that names the dimensions.
    pub fn axis(&self, name: &str) -> Result<usize> {
        self.position(name)
            .ok_or_else(|| nd_err!(InvalidArgument, "dimension '{name}' is not in {self}"))
    }

    /// The size of dimension `name`, if present.
    pub fn size_of(&self, name: &str) -> Option<usize> {
        self.position(name).map(|axis| self.dims[axis].size())
    }

    /// The broadcast result of several dimension sets, xarray style.
    ///
    /// Axes align by name. The order follows the first appearance of each name.
    /// Sizes must be equal, or one of them must be 1.
    pub fn broadcast(all: &[&Dimensions]) -> Result<Dimensions> {
        let mut out: Vec<Dimension> = Vec::new();
        for dims in all {
            for dim in dims.iter() {
                match out.iter_mut().find(|d| d.name() == dim.name()) {
                    None => out.push(dim.clone()),
                    Some(existing) if existing.size() == dim.size() || dim.size() == 1 => {}
                    Some(existing) if existing.size() == 1 => *existing = dim.clone(),
                    Some(existing) => {
                        return Err(nd_err!(
                            ShapeMismatch,
                            "dimension '{}' has incompatible sizes {} and {}",
                            dim.name(),
                            existing.size(),
                            dim.size()
                        ));
                    }
                }
            }
        }
        Dimensions::try_new(out)
    }

    /// Fails when an array of these dimensions cannot broadcast onto `target`:
    /// each axis must be in `target`, with the same size or with size 1.
    pub fn check_broadcast_to(&self, target: &Dimensions) -> Result<()> {
        for dim in self.iter() {
            let Some(axis) = target.position(dim.name()) else {
                return Err(nd_err!(
                    ShapeMismatch,
                    "cannot broadcast {self} to {target}: dimension '{}' is missing from the target",
                    dim.name()
                ));
            };
            let target_size = target.dims[axis].size();
            if dim.size() != target_size && dim.size() != 1 {
                return Err(nd_err!(
                    ShapeMismatch,
                    "cannot broadcast {self} to {target}: dimension '{}' has size {} but the target expects {target_size}",
                    dim.name(),
                    dim.size(),
                ));
            }
        }
        Ok(())
    }

    /// C-order (row-major) strides, in elements.
    pub fn c_strides(&self) -> Vec<usize> {
        let mut strides = vec![0usize; self.rank()];
        let mut acc = 1usize;
        for axis in (0..self.rank()).rev() {
            strides[axis] = acc;
            acc = acc.saturating_mul(self.dims[axis].size());
        }
        strides
    }
}

impl<'a> IntoIterator for &'a Dimensions {
    type Item = &'a Dimension;
    type IntoIter = std::slice::Iter<'a, Dimension>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl fmt::Display for Dimensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{dim}")?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_c_order() {
        let d = Dimensions::try_new([("time", 2), ("lat", 3), ("lon", 4)]).unwrap();
        assert_eq!(d.c_strides(), vec![12, 4, 1]);
        assert_eq!(d.num_elements(), 24);
        assert_eq!(d.names(), vec!["time", "lat", "lon"]);
    }

    #[test]
    fn scalar_has_one_element() {
        let d = Dimensions::scalar();
        assert_eq!(d.rank(), 0);
        assert_eq!(d.num_elements(), 1);
        assert!(d.c_strides().is_empty());
    }

    #[test]
    fn invalid_names_rejected() {
        assert!(Dimensions::try_new([("x", 2), ("x", 3)]).is_err());
        assert!(Dimensions::try_new([("", 2)]).is_err());
    }

    #[test]
    fn checked_num_elements_detects_overflow() {
        let d = Dimensions::try_new([("a", usize::MAX), ("b", 2)]).unwrap();
        assert_eq!(d.checked_num_elements(), None);
    }

    #[test]
    fn broadcast_check_follows_the_rule() {
        let target = Dimensions::try_new([("time", 2), ("lat", 3)]).unwrap();
        let ok = |spec: &[(&str, usize)]| {
            Dimensions::try_new(spec.iter().copied())
                .unwrap()
                .check_broadcast_to(&target)
        };
        assert!(ok(&[("lat", 3)]).is_ok());
        assert!(ok(&[("time", 1)]).is_ok());
        assert!(Dimensions::scalar().check_broadcast_to(&target).is_ok());
        let missing = ok(&[("depth", 2)]).unwrap_err();
        assert!(
            missing.to_string().contains("'depth' is missing"),
            "{missing}"
        );
        let size = ok(&[("lat", 2)]).unwrap_err();
        assert!(size.to_string().contains("has size 2"), "{size}");
    }
}
