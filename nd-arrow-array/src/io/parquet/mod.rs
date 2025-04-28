pub mod reader;
pub mod writer;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::AsArray,
        datatypes::{DataType, Field, Int32Type, Schema},
    };
    use futures::StreamExt;
    use parquet::arrow::{arrow_reader::ArrowReaderOptions, arrow_writer::ArrowWriterOptions};
    use tempfile::tempfile;

    use super::{reader::AsyncNdParquetReaderBuilder, writer::NdParquetWriter};
    use crate::{nd_array::default::DefaultNdArrowArray, nd_record_batch::NdRecordBatch};

    #[tokio::test]
    async fn test_encoding_roundtrip_parquet() -> Result<(), Box<dyn std::error::Error>> {
        // Create test data
        let array = Arc::new(
            DefaultNdArrowArray::from_vec::<arrow::datatypes::Int32Type>(
                vec![Some(1), Some(2), Some(3), Some(4)],
                vec![("x", 2), ("y", 2)],
            ),
        );

        let array2 = Arc::new(
            DefaultNdArrowArray::from_vec::<arrow::datatypes::Int32Type>(
                vec![Some(5), Some(6)],
                vec![("x", 2)],
            ),
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("test_column", DataType::Int32, true),
            Field::new("test_column2", DataType::Int32, true),
        ]));

        let batch = NdRecordBatch::new(schema.clone(), vec![array, array2]);

        // Write to temporary file
        let file = tempfile()?;
        {
            let mut writer = NdParquetWriter::new(
                file.try_clone()?,
                Some(ArrowWriterOptions::default()),
                batch.arrow_encoded_schema(),
            );
            writer.write(batch.clone())?;
            writer.finish()?;
        }

        let tokio_f = tokio::fs::File::from_std(file);

        // Read back using async reader
        let builder =
            AsyncNdParquetReaderBuilder::new(tokio_f, Some(ArrowReaderOptions::default()))
                .await?
                .with_projection(vec![0, 1])
                .unwrap();

        let mut stream = builder.build()?;

        // Read and verify the first batch
        let read_batch = stream.next().await.expect("Expected a batch")?;

        // Verify the schema
        assert_eq!(read_batch.schema().fields().len(), 2);
        assert_eq!(read_batch.schema().field(0).name(), "test_column");
        assert_eq!(read_batch.schema().field(0).data_type(), &DataType::Int32);

        // Verify the data
        let array = read_batch.column(0).as_primitive::<Int32Type>();
        assert_eq!(array.len(), 4);
        assert_eq!(array.values(), &[1, 2, 3, 4]);

        println!("Read batch: {:#?}", read_batch);

        Ok(())
    }
}
