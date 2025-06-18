use std::sync::Arc;

use crate::nd_array::{chunk::NdArrowArrayChunk, dimension::Dimension, NdArrowArray};

// pub struct DefaultChunkedNdArrowArray {
//     pub chunks: Vec<Arc<dyn NdArrowArray>>,
//     pub origin_dimensions: Vec<Dimension>,
// }

// impl ChunkedNdArrowArray for DefaultChunkedNdArrowArray {
//     fn num_chunks(&self) -> usize {
//         self.chunks.len()
//     }

//     fn chunk_shapes(&self) -> Vec<Vec<usize>> {
//         self.chunks.iter().map(|c| c.shape()).collect()
//     }

//     fn origin_dimensions(&self) -> &[Dimension] {
//         &self.origin_dimensions
//     }

//     fn chunks(&self) -> Box<dyn Iterator<Item = Arc<dyn NdArrowArray>> + '_> {
//         Box::new(self.chunks.iter().cloned())
//     }

//     fn origin_chunk_slices(&self) -> Vec<Vec<(usize, usize)>> {
//         todo!()
//     }
// }
