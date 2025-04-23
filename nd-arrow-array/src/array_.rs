use std::sync::Arc;

use arrow::{
    array::{
        Array, ArrayRef, ArrowPrimitiveType, ListArray, ListBuilder, PrimitiveArray, StringArray,
        StringBuilder, StructArray, StructBuilder, UInt32Array, UInt32Builder,
    },
    buffer::OffsetBuffer,
    datatypes::{ArrowNativeType, DataType, Field, Fields},
    error::ArrowError,
};
use ndarray::{ArrayBase, IntoDimension, OwnedRepr, RawData};

pub trait NdArrowArrayDyn {
    fn shape(&self) -> &[u32];

    fn ndim(&self) -> usize;
    fn dim_names(&self) -> Box<dyn Iterator<Item = Option<&str>>>;

    fn dims<'a>(&'a self) -> Vec<Arc<dyn Dimension + 'a>> {
        let shape = self.shape();
        let mut dim_names = self.dim_names();

        let mut dims = Vec::with_capacity(shape.len());
        for (i, dim) in shape.iter().enumerate() {
            dims.push(Arc::new(BorrowedDimension {
                name: dim_names.next().flatten(),
                size: dim,
            }) as Arc<dyn Dimension>);
        }

        dims
    }
    fn values(&self) -> Arc<dyn Array>;

    /// Return the broadcasted shape of several shapes.
    ///
    /// *Each shape is given as a slice of dimensions (`&[u32]`).*  
    /// If the shapes are incompatible this function **panics**
    /// (change the `panic!` to a `Result` if you prefer fallible behaviour).
    fn find_broadcast_shape(shapes: &[&[u32]]) -> Vec<u32> {
        // longest rank among all inputs
        let max_rank = shapes.iter().map(|s| s.len()).max().unwrap_or(0);

        // start with all‑1s (everything broadcasts to 1)
        let mut result = vec![1u32; max_rank];

        for shape in shapes {
            // walk each shape from the *right* (last dimension) to the left
            for (k, &dim) in shape.iter().rev().enumerate() {
                let idx = max_rank - 1 - k; // aligned position in `result`
                let cur = result[idx];

                if cur == dim || dim == 1 {
                    // nothing to do (either equal or dim==1)
                    continue;
                } else if cur == 1 {
                    // current result is 1 → upgrade to `dim`
                    result[idx] = dim;
                } else {
                    panic!("cannot broadcast dimension {}: {} vs {}", idx, cur, dim);
                }
            }
        }
        result
    }

    fn broadcast(&self, broadcast_shape: &[u32]) -> Result<Arc<dyn Array>, ArrowError> {
        let shape = self.shape();

        // check that the broadcast shape is compatible with the current shape
        todo!()
    }
}

struct BorrowedDimension<'a> {
    name: Option<&'a str>,
    size: &'a u32,
}

impl<'a> Dimension for BorrowedDimension<'a> {
    fn name(&self) -> Option<&str> {
        self.name
    }
    fn size(&self) -> usize {
        *self.size as usize
    }
}

pub struct NdArrowArray {
    inner_array: Arc<dyn Array>,
}

// impl<T, D: ndarray::Dimension> From<ArrayBase<T::Native, D>> for NdArrowArray
// where
//     T::Native: RawData,
//     T: ArrowPrimitiveType,
// {
//     fn from(value: ArrayBase<T::Native, D>) -> Self {
//         todo!()
//     }
// }

pub struct PrimitiveNdArrowArray<T>
where
    T: ArrowPrimitiveType,
{
    phantom: std::marker::PhantomData<T>,
    inner_array: Arc<dyn Array>,
}

impl<A, D: ndarray::Dimension> From<ArrayBase<OwnedRepr<A::Native>, D>> for PrimitiveNdArrowArray<A>
where
    A: ArrowPrimitiveType,
{
    fn from(value: ArrayBase<OwnedRepr<A::Native>, D>) -> Self {
        todo!()
    }
}

pub trait Dimension {
    fn name(&self) -> Option<&str> {
        None
    }
    fn size(&self) -> usize;
}

impl Dimension for usize {
    fn size(&self) -> usize {
        *self
    }
}

fn to_wrapped_list(field: Arc<Field>, array: Arc<dyn Array>) -> Result<ListArray, ArrowError> {
    let offset_buffer = OffsetBuffer::<i32>::from_lengths(vec![array.len()]);

    let list_array = ListArray::try_new(field, offset_buffer, array, None)?;

    Ok(list_array)
}

pub fn create_nd_array<I: Iterator<Item = D>, D: Dimension>(
    array: Arc<dyn Array>,
    dimensions: I,
) -> Result<StructArray, ArrowError> {
    let dimension_names_field = Arc::new(Field::new(
        "dimension_names",
        arrow::datatypes::DataType::Utf8,
        true,
    ));
    let dimension_sizes_field = Arc::new(Field::new(
        "dimension_sizes",
        arrow::datatypes::DataType::UInt32,
        false,
    ));

    let mut dimension_names_builder = StringBuilder::new();
    let mut dimension_sizes_builder = UInt32Builder::new();

    for dim in dimensions.into_iter() {
        if let Some(name) = dim.name() {
            dimension_names_builder.append_value(name);
        } else {
            dimension_names_builder.append_null();
        }
        dimension_sizes_builder.append_value(dim.size() as u32);
    }

    let dimension_names_array = Arc::new(dimension_names_builder.finish());
    let dimension_sizes_array = Arc::new(dimension_sizes_builder.finish());

    let wrapped_dimension_names = Arc::new(to_wrapped_list(
        dimension_names_field.clone(),
        dimension_names_array,
    )?);
    let wrapped_dimension_sizes = Arc::new(to_wrapped_list(
        dimension_sizes_field.clone(),
        dimension_sizes_array,
    )?);

    let data_values_field = Arc::new(Field::new(
        "values",
        array.data_type().clone(),
        array.is_nullable(),
    ));

    // Now create all the list array fields

    let data_array = Arc::new(to_wrapped_list(data_values_field.clone(), array.clone())?);

    let dimension_names_field = Arc::new(Field::new(
        "dimension_names",
        DataType::List(dimension_names_field.clone()),
        true,
    ));
    let dimension_sizes_field = Arc::new(Field::new(
        "dimension_sizes",
        DataType::List(dimension_sizes_field.clone()),
        true,
    ));
    let data_array_field = Arc::new(Field::new(
        "values",
        DataType::List(data_values_field.clone()),
        true,
    ));

    let fields = vec![
        dimension_names_field,
        dimension_sizes_field,
        data_array_field,
    ];
    let arrays: Vec<ArrayRef> = vec![wrapped_dimension_names, wrapped_dimension_sizes, data_array];

    Ok(StructArray::try_new(Fields::from(fields), arrays, None)?)
}

#[cfg(test)]
mod tests {
    use arrow::{
        array::{Int32Array, RecordBatch},
        datatypes::Schema,
    };

    use crate::stream_writer::NdRecordBatchStreamWriter;

    use super::*;

    #[test]
    fn test_name() {
        let values = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let nd_array = create_nd_array(values, vec![2usize, 2usize].into_iter()).unwrap();

        let struct_field = Arc::new(Field::new(
            "temperature",
            DataType::Struct(nd_array.fields().clone()),
            true,
        ));

        let values2 = Arc::new(Int32Array::from(vec![5, 6]));
        let nd_array2 = create_nd_array(values2, vec![2usize].into_iter()).unwrap();

        let struct_field2 = Arc::new(Field::new(
            "longitude",
            DataType::Struct(nd_array2.fields().clone()),
            true,
        ));

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![struct_field, struct_field2])),
            vec![Arc::new(nd_array), Arc::new(nd_array2)],
        )
        .unwrap();

        println!("{:?}", batch);

        //Flatten struct
        let writer = NdRecordBatchStreamWriter::new(batch.schema()).unwrap();
        let flattened_batch = writer.write_batch(batch).unwrap();
        println!("{:?}", flattened_batch);
        // Write to parquet file

        let file = std::fs::File::create("test.parquet").unwrap();
        let mut parquet_writer =
            parquet::arrow::ArrowWriter::try_new(file, flattened_batch.schema(), None).unwrap();

        parquet_writer.write(&flattened_batch).unwrap();
        parquet_writer.close().unwrap();

        // Read from parquet file
        let file = std::fs::File::open("test.parquet").unwrap();
        let parquet_reader =
            parquet::arrow::arrow_reader::ArrowReaderBuilder::try_new(file).unwrap();

        let metadata = parquet_reader.metadata();

        println!("Parquet file metadata: {:?}", metadata);
    }
}
