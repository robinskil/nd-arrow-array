#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dimension {
    pub name: String,
    pub size: usize,
}

impl Dimension {
    pub fn size(&self) -> usize {
        self.size
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
