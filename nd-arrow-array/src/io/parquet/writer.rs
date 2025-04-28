use std::io::Write;

use arrow::datatypes::SchemaRef;
use parquet::{arrow::arrow_writer::ArrowWriterOptions, errors::ParquetError};

use crate::nd_record_batch::NdRecordBatch;

pub struct NdParquetWriter<W: Write + Send> {
    writer: parquet::arrow::ArrowWriter<W>,
}

impl<W: Write + Send> NdParquetWriter<W> {
    pub fn new(writer: W, writer_options: Option<ArrowWriterOptions>, schema: SchemaRef) -> Self {
        let writer_options = writer_options.unwrap_or_else(ArrowWriterOptions::default);
        let arrow_writer =
            parquet::arrow::ArrowWriter::try_new_with_options(writer, schema, writer_options)
                .unwrap();
        Self {
            writer: arrow_writer,
        }
    }

    pub fn write(&mut self, batch: impl Into<NdRecordBatch>) -> Result<(), ParquetError> {
        let nd_batch: NdRecordBatch = batch.into();
        self.writer.write(
            &nd_batch
                .to_arrow_encoded_record_batch()
                .map_err(|e| ParquetError::External(Box::new(e)))?,
        )?;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<(), ParquetError> {
        self.writer.finish()?; // Finalize the writer and return the underlying writer
        Ok(())
    }
}
