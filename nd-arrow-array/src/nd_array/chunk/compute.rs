use std::collections::HashMap;

use crate::nd_array::dimension::Dimension;

pub fn compute_aligned_auto_chunks(
    groups: &[Vec<Dimension>],
    target_elements: usize,
) -> Vec<Vec<Dimension>> {
    // Build a mapping from dimension‐name → list of group‐indices that use it
    let mut dim_to_groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (gi, dims) in groups.iter().enumerate() {
        for d in dims {
            dim_to_groups
                .entry(d.name().to_string())
                .or_default()
                .push(gi);
        }
    }

    // Step 1+2+3: per‐group raw chunk computation (same as before)
    let mut per_group_chunks: Vec<HashMap<String, usize>> = Vec::with_capacity(groups.len());
    for dims in groups.iter() {
        let sizes: Vec<f64> = dims.iter().map(|d| d.size() as f64).collect();
        let max_size = sizes
            .iter()
            .cloned()
            .filter(|&s| s > 1.0)
            .fold(0.0, f64::max)
            .max(1.0);
        let ratios: Vec<f64> = sizes.iter().map(|&s| s / max_size).collect();
        let prod_ratios: f64 = ratios.iter().product();

        let scaling = if prod_ratios > 0.0 {
            (target_elements as f64 / prod_ratios).powf(1.0 / (ratios.len() as f64))
        } else {
            1.0
        };

        // initial chunks
        let mut chunks: Vec<usize> = ratios
            .iter()
            .zip(dims.iter())
            .map(|(&r, d)| {
                let c = (r * scaling).floor() as usize;
                c.clamp(1, d.size())
            })
            .collect();

        let mut total: usize = chunks.iter().product();
        // upscale if too small
        if total < target_elements / 2 {
            let upscale = 2.0;
            chunks = ratios
                .iter()
                .zip(dims.iter())
                .map(|(&r, d)| {
                    let c = (r * scaling * upscale).floor() as usize;
                    c.clamp(1, d.size())
                })
                .collect();
            total = chunks.iter().product();
        }
        // fallback to full sizes
        if total < target_elements / 2 {
            chunks = dims.iter().map(|d| d.size()).collect();
        }

        // map by name
        let mut m = HashMap::new();
        for (d, &c) in dims.iter().zip(chunks.iter()) {
            m.insert(d.name().to_string(), c);
        }
        per_group_chunks.push(m);
    }

    // Step 4: align across groups on shared dims
    let mut result = Vec::with_capacity(groups.len());
    for (gi, dims) in groups.iter().enumerate() {
        // clone this group’s chunk map
        let mut cm = per_group_chunks[gi].clone();

        for d in dims {
            if let Some(peers) = dim_to_groups.get(d.name()) {
                // find the minimum chunk across all groups that share this dim
                let min_chunk = peers
                    .iter()
                    .filter_map(|&other_gi| per_group_chunks.get(other_gi))
                    .filter_map(|m| m.get(d.name()))
                    .cloned()
                    .min()
                    .unwrap_or(cm[d.name()]);
                cm.insert(d.name().to_string(), min_chunk);
            }
        }

        // rebuild Vec<Dimension> in original order
        let aligned = dims
            .iter()
            .map(|d| Dimension::new(d.name(), cm[d.name()]))
            .collect();
        result.push(aligned);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2() {
        let columns: Vec<Vec<Dimension>> = vec![
            vec![
                Dimension::new("N_PROF", 44452),
                Dimension::new("N_LEVELS", 3660),
            ],
            vec![Dimension::new("N_PROF", 44452)],
        ];

        let chunk_map = compute_aligned_auto_chunks(columns.as_slice(), 128 * 1024);
        // 8M elements max per chunk

        for (i, chunk) in chunk_map.iter().enumerate() {
            println!("Chunk {}: {:?}", i, chunk);
        }
    }
}
