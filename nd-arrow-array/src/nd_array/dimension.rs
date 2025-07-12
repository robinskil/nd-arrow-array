/// A representation of a dimension in an n-dimensional array.
///
/// Each dimension has a name and a size. The name is used to identify the dimension,
/// and the size represents the number of elements along this dimension.
///
/// # Examples
///
/// ```
/// use nd_arrow_array::nd_array::dimension::Dimension;
///
/// let dim = Dimension::new("rows", 10);
/// assert_eq!(dim.size(), 10);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dimension {
    pub name: String,
    pub size: usize,
}

impl Dimension {
    /// Creates a new `Dimension` with the given name and size.
    ///
    /// # Arguments
    ///
    /// * `name` - A string slice that holds the name of the dimension.
    /// * `size` - A usize that holds the size of the dimension.
    /// # Returns
    ///
    /// * A new `Dimension` instance.
    ///
    pub fn new(name: &str, size: usize) -> Self {
        Self {
            name: name.to_string(),
            size,
        }
    }

    /// Returns the size of the dimension.
    ///
    /// # Returns
    ///
    /// * A usize representing the size of the dimension.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the name of the dimension.
    ///
    /// # Returns
    ///
    /// * A string slice that holds the name of the dimension.
    ///
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl From<(String, usize)> for Dimension {
    fn from((name, size): (String, usize)) -> Self {
        Self { name, size }
    }
}

impl From<(&str, usize)> for Dimension {
    fn from((name, size): (&str, usize)) -> Self {
        Self {
            name: name.to_string(),
            size,
        }
    }
}

impl AsRef<Dimension> for Dimension {
    fn as_ref(&self) -> &Dimension {
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DimensionRef<'a> {
    pub name: &'a str,
    pub size: usize,
}

impl<'a> From<&'a Dimension> for DimensionRef<'a> {
    fn from(dim: &'a Dimension) -> Self {
        Self {
            name: dim.name(),
            size: dim.size(),
        }
    }
}

impl<'a> From<(&'a str, usize)> for DimensionRef<'a> {
    fn from((name, size): (&'a str, usize)) -> Self {
        Self { name, size }
    }
}
