use std::collections::{HashMap, HashSet};

use crate::nd_array::dimension::Dimension;

pub fn compute_aligned_auto_chunks_by_elements(
    columns: Vec<(String, Vec<Dimension>)>,
    target_elements: usize,
) -> HashMap<String, Vec<Dimension>> {
    compute_aligned_auto_chunks_by_elements_impl(columns.as_slice(), target_elements)
}

pub fn compute_aligned_auto_chunks_by_elements_impl<S: AsRef<str>, D: AsRef<Dimension>>(
    column_shapes: &[(S, Vec<D>)],
    target_elements: usize,
) -> HashMap<String, Vec<Dimension>> {
    let column_shapes = column_shapes
        .iter()
        .map(|(col, dims)| {
            (
                col.as_ref(),
                dims.iter().map(|d| d.as_ref()).collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();

    let mut dim_sizes: HashMap<String, usize> = HashMap::new();
    let mut dim_to_vars: HashMap<String, HashSet<String>> = HashMap::new();

    // Step 1: Collect dimension sizes and dimension-to-variable mapping
    for (column, dimensions) in &column_shapes {
        for dimension in dimensions {
            dim_sizes
                .entry(dimension.name().to_string())
                .and_modify(|e| *e = (*e).max(dimension.size()))
                .or_insert(dimension.size());

            dim_to_vars
                .entry(dimension.name().to_string())
                .or_default()
                .insert(column.to_string());
        }
    }

    // Step 2: Process each variable independently, with partial alignment only
    let mut per_variable_chunks: HashMap<String, HashMap<String, usize>> = HashMap::new();

    for (column_name, dimensions) in &column_shapes {
        let dim_list = dimensions;
        let size_list: Vec<f64> = dim_list.iter().map(|d| d.size() as f64).collect();

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
            .zip(dim_list)
            .map(|(&r, dim)| {
                let chunk = (r * scaling).floor() as usize;
                chunk.clamp(1, dim.size())
            })
            .collect();

        let mut total_chunk_elems: usize = chunks.iter().product();

        // Rescale up if too small
        if total_chunk_elems < target_elements / 2 {
            let upscale = 2.0;
            chunks = ratios
                .iter()
                .zip(dim_list)
                .map(|(&r, dim)| {
                    let chunk = (r * scaling * upscale).floor() as usize;
                    chunk.clamp(1, dim.size())
                })
                .collect();
            total_chunk_elems = chunks.iter().product();
        }

        if total_chunk_elems < target_elements / 2 {
            chunks = dim_list.iter().map(|d| d.size()).collect();
        }

        let mut chunk_map = HashMap::new();
        for (dim, &chunk) in dim_list.iter().zip(chunks.iter()) {
            chunk_map.insert(dim.name().to_string(), chunk);
        }

        per_variable_chunks.insert(column_name.to_string(), chunk_map);
    }

    // Step 3: Align chunk sizes for shared dimensions only across overlapping variable groups
    let mut final_chunks: HashMap<String, Vec<Dimension>> = HashMap::new();

    for (column, dimensions) in column_shapes {
        let mut chunk_map = per_variable_chunks[column].clone();

        for dim in dimensions {
            // Find other variables that use this dim
            if let Some(others) = dim_to_vars.get(dim.name()) {
                let mut max_shared_chunk = chunk_map[dim.name()];

                for other in others {
                    if other == column {
                        continue;
                    }
                    if let Some(other_chunk) = per_variable_chunks
                        .get(other)
                        .and_then(|m| m.get(dim.name()))
                    {
                        max_shared_chunk = max_shared_chunk.min(*other_chunk);
                    }
                }

                chunk_map.insert(dim.name().to_string(), max_shared_chunk);
            }
        }

        final_chunks.insert(
            column.to_string(),
            chunk_map
                .into_iter()
                .map(|(k, v)| Dimension::new(&k, v))
                .collect(),
        );
    }

    final_chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2() {
        let columns: Vec<(String, Vec<Dimension>)> = vec![
            (
                "temp".into(),
                vec![
                    Dimension::new("time", 4),
                    Dimension::new("x", 4),
                    Dimension::new("y", 8),
                ],
            ),
            ("time_series".into(), vec![Dimension::new("time", 4)]),
            ("rand".into(), vec![Dimension::new("lp", 2)]),
            (
                "station".into(),
                vec![Dimension::new("time", 4), Dimension::new("lp", 2)],
            ),
        ];

        let chunk_map = compute_aligned_auto_chunks_by_elements_impl(columns.as_slice(), 40);
        // 8M elements max per chunk

        for (var, chunks) in chunk_map {
            println!("Variable: {var}");
            for dim in chunks {
                println!("  {:#?}", dim);
            }
        }
    }
}
