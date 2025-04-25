use std::sync::Arc;

use arrow::array::Array;

use super::NdArrowArray;

pub struct DefaultNdArrowArray {
    inner_array: Arc<dyn Array>,
    shape: Vec<usize>,
    dim_names: Option<Vec<Option<String>>>,
}

impl DefaultNdArrowArray {
    pub fn new(
        inner_array: Arc<dyn Array>,
        shape: Vec<usize>,
        dim_names: Option<Vec<Option<String>>>,
    ) -> Self {
        // Ensure the shape and length of the inner array match
        let total_size: usize = shape.iter().product();

        assert_eq!(
            total_size,
            inner_array.len(),
            "Shape and inner array length do not match: expected {}, got {}",
            total_size,
            inner_array.len()
        );

        Self {
            inner_array,
            shape,
            dim_names,
        }
    }
}

impl NdArrowArray for DefaultNdArrowArray {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn dim_names<'a>(&'a self) -> Box<dyn Iterator<Item = Option<&'a str>> + 'a> {
        Box::new(
            self.dim_names
                .iter()
                .flat_map(|names| names.iter().map(|name| name.as_deref())),
        )
    }

    fn values(&self) -> Arc<dyn Array> {
        self.inner_array.clone()
    }
}
