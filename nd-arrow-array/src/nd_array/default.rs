use std::sync::Arc;

use arrow::array::Array;

use super::{dimension::Dimension, NdArrowArray};

pub struct DefaultNdArrowArray {
    inner_array: Arc<dyn Array>,
    dimensions: Vec<Dimension>,
}

impl DefaultNdArrowArray {
    pub fn new_scalar(inner_array: Arc<dyn Array>) -> Self {
        if inner_array.len() != 1 {
            panic!("Inner array must be a scalar (length 1)");
        }
        Self {
            inner_array,
            dimensions: vec![],
        }
    }

    pub fn new(inner_array: Arc<dyn Array>, dimensions: Vec<impl Into<Dimension>>) -> Self {
        let dimensions: Vec<Dimension> =
            dimensions.into_iter().map(|d| d.into()).collect::<Vec<_>>();
        // Ensure the shape and length of the inner array match
        let total_size: usize = dimensions.iter().map(|dim| dim.size()).product();

        // Check if the inner array is a scalar (length 1 and no dimensions)
        if !(total_size == 1 && dimensions.is_empty()) {
            assert_eq!(
                total_size,
                inner_array.len(),
                "Shape and inner array length do not match: expected {}, got {}",
                total_size,
                inner_array.len()
            );
        }

        Self {
            inner_array,
            dimensions,
        }
    }
}

impl NdArrowArray for DefaultNdArrowArray {
    fn shape(&self) -> Vec<usize> {
        self.dimensions
            .iter()
            .map(|dim| dim.size())
            .collect::<Vec<_>>()
    }

    fn dimensions(&self) -> &[Dimension] {
        &self.dimensions
    }

    fn array(&self) -> Arc<dyn Array> {
        self.inner_array.clone()
    }
}
