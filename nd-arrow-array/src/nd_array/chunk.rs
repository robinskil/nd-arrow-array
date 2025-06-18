use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::nd_array::{dimension::Dimension, NdArrowArray};

pub trait ChunkedNdArrowArray {
    /// Returns the number of chunks in the array.
    fn num_chunks(&self) -> usize;
    fn chunk_shape(&self) -> Vec<usize>;
    fn dimensions(&self) -> &[crate::nd_array::dimension::Dimension];

    /// Returns an iterator over the chunks of the array.
    fn chunks(&self) -> Box<dyn Iterator<Item = Arc<dyn NdArrowArray>> + '_>;
}

// pub fn generate_chunks(dimensions: &[Dimension]) ->

pub fn compute_aligned_auto_chunks_by_elements(
    columns: &[(String, Vec<Dimension>)],
    target_elements: usize,
) -> HashMap<String, HashMap<String, usize>> {
    let mut dim_sizes: HashMap<String, usize> = HashMap::new();
    let mut dim_to_vars: HashMap<String, HashSet<String>> = HashMap::new();

    // Step 1: Collect dimension sizes and dimension-to-variable mapping
    for column in columns {
        for dimension in column.1.iter() {
            dim_sizes
                .entry(dimension.name().to_string())
                .and_modify(|e| *e = (*e).max(dimension.size()))
                .or_insert(dimension.size());

            dim_to_vars
                .entry(dimension.name().to_string())
                .or_default()
                .insert(column.0.clone());
        }
    }

    // Step 2: Process each variable independently, with partial alignment only
    let mut per_variable_chunks: HashMap<String, HashMap<String, usize>> = HashMap::new();

    for (column_name, dimensions) in columns {
        let dim_list = &var.dims;
        let size_list: Vec<f64> = var.shape.iter().map(|&s| s as f64).collect();

        // Compute aspect ratios
        let max_size = size_list
            .iter()
            .cloned()
            .filter(|&s| s > 1.0)
            .fold(0.0, f64::max)
            .max(1.0);

        let ratios: Vec<f64> = size_list.iter().map(|s| s / max_size).collect();
        let product_of_ratios: f64 = ratios.iter().product();

        let scaling = if product_of_ratios > 0.0 {
            (target_elements as f64 / product_of_ratios).powf(1.0 / ratios.len() as f64)
        } else {
            1.0
        };

        // Initial scaled chunk sizes
        let mut chunks: Vec<usize> = ratios
            .iter()
            .zip(&var.shape)
            .map(|(&r, &dim_len)| {
                let chunk = (r * scaling).floor() as usize;
                chunk.clamp(1, dim_len)
            })
            .collect();

        let mut total_chunk_elems: usize = chunks.iter().product();

        // Rescale up if too small
        if total_chunk_elems < target_elements / 2 {
            let upscale = 2.0;
            chunks = ratios
                .iter()
                .zip(&var.shape)
                .map(|(&r, &dim_len)| {
                    let chunk = (r * scaling * upscale).floor() as usize;
                    chunk.clamp(1, dim_len)
                })
                .collect();
            total_chunk_elems = chunks.iter().product();
        }

        if total_chunk_elems < target_elements / 2 {
            chunks = var.shape.clone();
        }

        let mut chunk_map = HashMap::new();
        for (dim, &chunk) in dim_list.iter().zip(chunks.iter()) {
            chunk_map.insert(dim.clone(), chunk);
        }

        per_variable_chunks.insert(column_name.to_string(), chunk_map);
    }

    // Step 3: Align chunk sizes for shared dimensions only across overlapping variable groups
    let mut final_chunks: HashMap<String, HashMap<String, usize>> = HashMap::new();

    for var in variables {
        let mut chunk_map = per_variable_chunks[&var.name].clone();

        for dim in &var.dims {
            // Find other variables that use this dim
            if let Some(others) = dim_to_vars.get(dim) {
                let mut max_shared_chunk = chunk_map[dim];

                for other in others {
                    if other == &var.name {
                        continue;
                    }
                    if let Some(other_chunk) =
                        per_variable_chunks.get(other).and_then(|m| m.get(dim))
                    {
                        max_shared_chunk = max_shared_chunk.min(*other_chunk);
                    }
                }

                chunk_map.insert(dim.clone(), max_shared_chunk);
            }
        }

        final_chunks.insert(var.name.clone(), chunk_map);
    }

    final_chunks
}
