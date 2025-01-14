use crate::{dimension::Dimension, explode::ExplodeArgs};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    pub shape_size: Vec<usize>,
    pub dimensions: Vec<Dimension>,
}

impl PartialOrd for Shape {
    /// Compare the dimensions of two shapes
    /// If the dimensions are equal, return Some(Ordering::Equal)
    /// If the dimensions cannot be compared, return None
    /// If the dimensions can be compared, return Some(Ordering::Less) or Some(Ordering::Greater)
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        //Compare the dimensions
        if self.dimensions == other.dimensions {
            return Some(std::cmp::Ordering::Equal);
        } else {
            if self.dimensions.len() < other.dimensions.len() {
                if self.dimensions.len() == 1
                    && other
                        .dimensions
                        .iter()
                        .position(|d| d == &self.dimensions[0])
                        .is_some()
                {
                    return Some(std::cmp::Ordering::Less);
                } else if self.dimensions.is_empty() {
                    return Some(std::cmp::Ordering::Less);
                } else {
                    return None;
                }
            } else if self.dimensions.len() > other.dimensions.len() {
                if other.dimensions.len() == 1
                    && self
                        .dimensions
                        .iter()
                        .position(|d| d == &other.dimensions[0])
                        .is_some()
                {
                    return Some(std::cmp::Ordering::Greater);
                } else if other.dimensions.is_empty() {
                    return Some(std::cmp::Ordering::Greater);
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }
    }
}

impl Shape {
    pub fn new_inferred<V: Into<Vec<D>>, D: Into<Dimension>>(dimensions: V) -> Self {
        let dimensions: Vec<_> = dimensions.into().into_iter().map(|d| d.into()).collect();
        let shape_size = dimensions.iter().map(|d| d.size()).collect();

        Self {
            shape_size,
            dimensions,
        }
    }

    pub fn new(dimensions: Vec<Dimension>) -> Self {
        let shape_size = dimensions.iter().map(|d| d.size()).collect();

        Self {
            shape_size,
            dimensions,
        }
    }

    pub fn dimensions(&self) -> &[Dimension] {
        &self.dimensions
    }

    pub fn flat_size(&self) -> usize {
        self.shape_size.iter().product()
    }

    pub fn scalar_shape() -> Self {
        Self {
            shape_size: vec![],
            dimensions: vec![],
        }
    }

    pub fn is_scalar(&self) -> bool {
        self.dimensions.is_empty()
    }

    pub(crate) fn explode_diff(&self, other: &Shape) -> Result<ExplodeArgs, ()> {
        if self == other {
            return Ok(ExplodeArgs {
                repeat_elems: 1,
                repeat_slices: 1,
            });
        } else if self.is_scalar() {
            return Ok(ExplodeArgs {
                repeat_elems: other.flat_size(),
                repeat_slices: 1,
            });
        } else if self.dimensions.len() == 1 {
            //Find the position of the self dimension in the other dimensions
            let pos = other
                .dimensions
                .iter()
                .position(|d| d == &self.dimensions[0]);
            match pos {
                Some(pos) => {
                    let mut stretch_elems = 1;
                    let mut stretch_slices = 1;
                    for i in 0..pos {
                        stretch_slices *= other.dimensions[i].size();
                    }
                    for i in pos + 1..other.dimensions.len() {
                        stretch_elems *= other.dimensions[i].size();
                    }
                    return Ok(ExplodeArgs {
                        repeat_elems: stretch_elems,
                        repeat_slices: stretch_slices,
                    });
                }
                None => return Err(()),
            }
        } else {
            return Err(());
        }
    }
}
