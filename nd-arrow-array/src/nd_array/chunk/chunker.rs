use std::sync::Arc;

use arrow::array::{
    Array, ArrayBuilder, ArrayRef, ArrowPrimitiveType, BinaryArray, BinaryBuilder, BooleanArray,
    BooleanBuilder, NullArray, NullBuilder, PrimitiveArray, PrimitiveBuilder, StringArray,
    StringBuilder,
};

use crate::nd_array::chunk::Chunk;
/// A little trait to unify `Array` + its matching `ArrayBuilder`.
pub trait Chunkable {
    /// The concrete array type (e.g. `PrimitiveArray<T>`, `BooleanArray`, `StringArray`, …).
    type Array: Array;
    /// The matching builder type.
    type Builder: ArrayBuilder;

    /// Create a builder with capacity `cap`.
    fn builder_with_capacity(cap: usize) -> Self::Builder;
    /// Given a reference to `Self::Array` and an index, push that slot
    /// (value or null) into the builder.
    fn append_index(arr: &Self::Array, idx: usize, builder: &mut Self::Builder);
    /// Finalize the builder into an `ArrayRef`.
    fn finish(builder: Self::Builder) -> ArrayRef;
}

// ——— Implement Chunkable for primitive arrays ———
impl<T: ArrowPrimitiveType> Chunkable for PrimitiveArray<T> {
    type Array = PrimitiveArray<T>;
    type Builder = PrimitiveBuilder<T>;

    fn builder_with_capacity(cap: usize) -> Self::Builder {
        PrimitiveBuilder::<T>::with_capacity(cap)
    }

    fn append_index(arr: &Self::Array, idx: usize, builder: &mut Self::Builder) {
        if arr.is_valid(idx) {
            builder.append_value(arr.value(idx));
        } else {
            builder.append_null();
        }
    }

    fn finish(mut builder: Self::Builder) -> ArrayRef {
        Arc::new(builder.finish())
    }
}

// ——— Implement for booleans ———
impl Chunkable for BooleanArray {
    type Array = BooleanArray;
    type Builder = BooleanBuilder;

    fn builder_with_capacity(cap: usize) -> Self::Builder {
        BooleanBuilder::with_capacity(cap)
    }

    fn append_index(arr: &Self::Array, idx: usize, builder: &mut Self::Builder) {
        if arr.is_valid(idx) {
            builder.append_value(arr.value(idx));
        } else {
            builder.append_null();
        }
    }

    fn finish(mut builder: Self::Builder) -> ArrayRef {
        Arc::new(builder.finish())
    }
}

impl Chunkable for NullArray {
    type Array = NullArray;
    type Builder = NullBuilder; // NullArray is a special case, we can use a PrimitiveBuilder<bool>

    fn builder_with_capacity(_cap: usize) -> Self::Builder {
        NullBuilder::new()
    }

    fn append_index(_arr: &Self::Array, _idx: usize, builder: &mut Self::Builder) {
        builder.append_null();
    }

    fn finish(mut builder: Self::Builder) -> ArrayRef {
        Arc::new(builder.finish())
    }
}

// ——— Implement for UTF-8 strings ———
impl Chunkable for StringArray {
    type Array = StringArray;
    type Builder = StringBuilder;

    fn builder_with_capacity(cap: usize) -> Self::Builder {
        StringBuilder::with_capacity(cap, cap * 4)
    }

    fn append_index(arr: &Self::Array, idx: usize, builder: &mut Self::Builder) {
        if arr.is_valid(idx) {
            builder.append_value(arr.value(idx));
        } else {
            builder.append_null();
        }
    }

    fn finish(mut builder: Self::Builder) -> ArrayRef {
        Arc::new(builder.finish())
    }
}

// ——— Implement for Binary ———
impl Chunkable for BinaryArray {
    type Array = BinaryArray;
    type Builder = BinaryBuilder;

    fn builder_with_capacity(cap: usize) -> Self::Builder {
        BinaryBuilder::with_capacity(cap, cap * 4)
    }

    fn append_index(arr: &Self::Array, idx: usize, builder: &mut Self::Builder) {
        if arr.is_valid(idx) {
            builder.append_value(arr.value(idx));
        } else {
            builder.append_null();
        }
    }

    fn finish(mut builder: Self::Builder) -> ArrayRef {
        Arc::new(builder.finish())
    }
}

/// The one-and-only implementation of the chunking logic.
/// Pulls in *any* `C: Chunkable` — no per-type code needed below!
pub fn chunk_nd_array_with_meta<C: Chunkable>(
    data: &C::Array,
    shape: &[usize],
    chunk_shape: &[usize],
) -> Vec<Chunk> {
    assert_eq!(shape.len(), chunk_shape.len());
    let ndim = shape.len();

    // compute flat strides
    let mut strides = vec![1; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // helper: all starting indices in one dimension
    fn dim_starts(size: usize, chunk: usize) -> Vec<usize> {
        let mut v = Vec::new();
        let mut i = 0;
        while i < size {
            v.push(i);
            i += chunk;
        }
        v
    }

    // cartesian‐product of starts_per_dim
    fn cartesian(vv: &[Vec<usize>]) -> Vec<Vec<usize>> {
        vv.iter().fold(vec![vec![]], |acc, dim| {
            acc.into_iter()
                .flat_map(|base| {
                    dim.iter().map(move |&i| {
                        let mut b = base.clone();
                        b.push(i);
                        b
                    })
                })
                .collect()
        })
    }

    // compute all chunk‐origins
    let starts_per_dim: Vec<Vec<usize>> = shape
        .iter()
        .zip(chunk_shape.iter())
        .map(|(&s, &c)| dim_starts(s, c))
        .collect();
    let origins = cartesian(&starts_per_dim);

    let mut out = Vec::with_capacity(origins.len());

    for origin in origins {
        // figure out this chunk’s shape & slices
        let mut sizes = vec![0; ndim];
        let mut slices = vec![(0, 0); ndim];
        for d in 0..ndim {
            let end = (origin[d] + chunk_shape[d]).min(shape[d]);
            sizes[d] = end - origin[d];
            slices[d] = (origin[d], end);
        }
        // allocate builder
        let total_slots = sizes.iter().product();
        let mut builder = C::builder_with_capacity(total_slots);

        // recursive walker to fill the builder
        fn recurse<C: Chunkable>(
            depth: usize,
            origin: &[usize],
            sizes: &[usize],
            shape: &[usize],
            strides: &[usize],
            data: &C::Array,
            idx: &mut Vec<usize>,
            builder: &mut C::Builder,
        ) {
            if depth == sizes.len() {
                // compute flat index
                let flat: usize = origin
                    .iter()
                    .zip(idx.iter())
                    .map(|(&o, &i)| o + i)
                    .zip(strides.iter())
                    .map(|(x, &s)| x * s)
                    .sum();
                C::append_index(data, flat, builder);
                return;
            }
            for i in 0..sizes[depth] {
                idx.push(i);
                recurse::<C>(depth + 1, origin, sizes, shape, strides, data, idx, builder);
                idx.pop();
            }
        }

        recurse::<C>(
            0,
            &origin,
            &sizes,
            shape,
            &strides,
            data,
            &mut Vec::with_capacity(ndim),
            &mut builder,
        );

        out.push(Chunk {
            chunk_indices: origin.clone(),
            shape: sizes,
            slices,
            array: C::finish(builder),
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name() {}
}
