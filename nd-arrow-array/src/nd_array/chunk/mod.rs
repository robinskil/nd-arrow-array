use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use arrow::array::{
    Array, ArrayBuilder, ArrayRef, ArrowPrimitiveType, BinaryArray, BinaryBuilder, BooleanArray,
    BooleanBuilder, PrimitiveArray, PrimitiveBuilder, StringArray, StringBuilder,
};

use crate::nd_array::{dimension::Dimension, NdArrowArray};

pub mod chunker;
pub mod compute;
pub mod default;

pub trait NdArrowArrayChunk {
    /// Returns the number of chunks in the array.
    fn num_chunks(&self) -> usize;

    fn chunk_shapes(&self) -> Vec<Vec<usize>>;
    fn chunk_index(&self) -> Vec<usize>;

    fn origin_chunk_slices(&self) -> Vec<Vec<(usize, usize)>>;
    fn origin_dimensions(&self) -> &[crate::nd_array::dimension::Dimension];

    /// Returns an iterator over the chunks of the array.
    fn chunks(&self) -> Box<dyn Iterator<Item = Arc<dyn NdArrowArray>> + '_>;
}

pub struct ChunkedNdArrowArray {}

pub struct Chunk {
    pub origin: Vec<usize>,
    pub shape: Vec<usize>,
    pub slices: Vec<(usize, usize)>,
    pub array: ArrayRef,
}
