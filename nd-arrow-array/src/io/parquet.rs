use std::sync::Arc;

use arrow::{array::ArrayRef, datatypes::SchemaRef};
use bytes::Bytes;
use parquet::{
    arrow::{
        arrow_reader::{
            statistics::StatisticsConverter, ArrowReaderBuilder, ArrowReaderMetadata,
            ArrowReaderOptions,
        },
        async_reader::{AsyncFileReader, ParquetRecordBatchStream},
        ParquetRecordBatchStreamBuilder,
    },
    errors::ParquetError,
    schema::types::ColumnPath,
};

use crate::{consts, version::validate_version_compatibility};

pub struct AsyncNdParquetReaderBuilder<T: AsyncFileReader + 'static> {
    simplified_schema: SchemaRef,
    builder: ParquetRecordBatchStreamBuilder<T>,
}

impl<T: AsyncFileReader + 'static> AsyncNdParquetReaderBuilder<T> {
    pub async fn new(reader: T) -> Result<Self, ParquetError> {
        let builder = ParquetRecordBatchStreamBuilder::new(reader).await?;
        let simplified_schema =
            Self::validate_translate_arrow_schema(builder.schema().clone()).unwrap();
        Ok(Self {
            builder: builder,
            simplified_schema,
        })
    }

    fn validate_translate_arrow_schema(schema: SchemaRef) -> Result<SchemaRef, String> {
        //Validate the schema key and version
        let encoding_metadata = schema.metadata().get(consts::ND_ARROW_SCHEMA_ENCODING_KEY);
        // Check if the metadata contains the expected key and some version
        if let Some(version) = encoding_metadata {
            if !validate_version_compatibility(version) {
                return Err(format!(
                    "Unsupported encoding version: {}",
                    version.to_string()
                ));
            }
        } else {
            return Err("Missing encoding metadata".to_string());
        }

        let mut simplified_fields = vec![];

        for field in schema.fields() {
            //Expect every field to be a struct
            match field.data_type() {
                arrow::datatypes::DataType::Struct(fields) => {
                    let values_field = fields
                        .iter()
                        .find(|f| f.name() == consts::VALUES)
                        .ok_or(format!("Missing field {}.{}", field.name(), consts::VALUES))?;

                    simplified_fields.push(arrow::datatypes::Field::new(
                        field.name(),
                        values_field.data_type().clone(),
                        values_field.is_nullable(),
                    ));
                }
                _ => return Err(format!("Field {} is not a struct", field.name())),
            }
        }

        Ok(arrow::datatypes::SchemaRef::new(
            arrow::datatypes::Schema::new(simplified_fields),
        ))
    }

    pub fn arrow_schema(&self) -> &arrow::datatypes::SchemaRef {
        &self.simplified_schema
    }

    pub fn underlying_arrow_schema(&self) -> &arrow::datatypes::SchemaRef {
        self.builder.schema()
    }

    pub fn parquet_schema(&self) -> &parquet::schema::types::SchemaDescriptor {
        self.builder.parquet_schema()
    }

    pub fn metadata(&self) -> &Arc<parquet::file::metadata::ParquetMetaData> {
        self.builder.metadata()
    }

    pub async fn row_group_column_mins(&self, column_name: &str) -> Option<ArrayRef> {
        //Verify the column name exists in the schema
        if !self.simplified_schema.field_with_name(column_name).is_ok() {
            return None;
        }
        //Translate the column name to the .values field name
        let underlying_column_name = format!("{}.{}", column_name, consts::VALUES);

        let converter = StatisticsConverter::try_new(
            &underlying_column_name,
            &self.arrow_schema(),
            self.parquet_schema(),
        )
        .unwrap();

        let row_group_mins = self.builder.metadata().row_groups();

        let array_mins = converter.row_group_mins(row_group_mins).unwrap();

        Some(array_mins)
    }

    pub async fn row_group_column_maxes(&self, column_name: &str) -> Option<ArrayRef> {
        //Verify the column name exists in the schema
        if !self.simplified_schema.field_with_name(column_name).is_ok() {
            return None;
        }
        //Translate the column name to the .values field name
        let underlying_column_name = format!("{}.{}", column_name, consts::VALUES);

        let converter = StatisticsConverter::try_new(
            &underlying_column_name,
            &self.arrow_schema(),
            self.parquet_schema(),
        )
        .unwrap();

        let row_group_maxs = self.builder.metadata().row_groups();

        let array_maxs = converter.row_group_maxes(row_group_maxs).unwrap();

        Some(array_maxs)
    }

    // pub fn row_group_filters(&mut self) {
    //     self.builder.with_row_groups()
    // }
}

#[cfg(test)]
mod tests {
    use arrow::{
        array::{Array, Int32Array, ListArray, NullArray, PrimitiveArray, RecordBatch},
        buffer::{Buffer, OffsetBuffer},
        datatypes::Field,
        ipc::{
            convert::fb_to_schema,
            reader::{read_footer_length, FileDecoder, StreamReader},
            root_as_footer,
            writer::IpcWriteOptions,
            Block, CompressionType,
        },
    };
    use parquet::{
        arrow::{
            arrow_reader::{self, ParquetRecordBatchReaderBuilder},
            ArrowWriter,
        },
        file::properties::{WriterProperties, WriterPropertiesBuilder},
    };

    use super::*;

    #[test]
    fn test_2() {
        let data = Arc::new(Int32Array::from(vec![10; 20_000]));
        let list_arr = ListArray::new(
            Arc::new(Field::new(
                "values",
                arrow::datatypes::DataType::Int32,
                true,
            )),
            OffsetBuffer::from_lengths(vec![data.len()]),
            data,
            None,
        );

        let field = Field::new(
            "test",
            arrow::datatypes::DataType::List(Arc::new(Field::new(
                "values",
                arrow::datatypes::DataType::Int32,
                true,
            ))),
            true,
        );

        let field2 = Field::new(
            "test2",
            arrow::datatypes::DataType::List(Arc::new(Field::new(
                "values",
                arrow::datatypes::DataType::Int32,
                true,
            ))),
            true,
        );

        let data3 = Arc::new(NullArray::new(20_000));
        let list_arr3 = ListArray::new(
            Arc::new(Field::new("values", arrow::datatypes::DataType::Null, true)),
            OffsetBuffer::from_lengths(vec![data3.len()]),
            data3.clone(),
            None,
        );

        let field3 = Field::new(
            "test3",
            arrow::datatypes::DataType::List(Arc::new(Field::new(
                "values",
                arrow::datatypes::DataType::Null,
                true,
            ))),
            true,
        );

        let schema = Arc::new(arrow::datatypes::Schema::new(vec![field, field2, field3]));

        let mut buffer = std::io::Cursor::new(vec![]);

        // let writer_props = WriterProperties::builder()
        //     .set_statistics_enabled(parquet::file::properties::EnabledStatistics::Page)
        //     .set_data_page_size_limit(64 * 1024)
        //     .build();

        let mut ipc_writer = arrow::ipc::writer::FileWriter::try_new_with_options(
            &mut buffer,
            &schema.clone(),
            IpcWriteOptions::default()
                .try_with_compression(Some(CompressionType::ZSTD))
                .unwrap(),
        )
        .unwrap();

        for i in 0..200 {
            let rb = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(list_arr.clone()),
                    Arc::new(list_arr.clone()),
                    Arc::new(list_arr3.clone()),
                ],
            )
            .unwrap();
            ipc_writer.write(&rb).unwrap();
            ipc_writer.flush().unwrap();
        }

        ipc_writer.finish().unwrap();

        let bytes = Bytes::from_owner(buffer.get_ref().clone());
        println!("Buffer length: {}", bytes.len());
        let buffer = Buffer::from_bytes(bytes.into());

        let decoder = IPCBufferDecoder::new(buffer.clone());

        println!("Number of batches: {}", decoder.num_batches());

        println!("Batch 0: {:?}", decoder.get_batch(14).unwrap());
    }

    /// Incrementally decodes [`RecordBatch`]es from an IPC file stored in a Arrow
    /// [`Buffer`] using the [`FileDecoder`] API.
    ///
    /// This is a wrapper around the example in the `FileDecoder` which handles the
    /// low level interaction with the Arrow IPC format.
    struct IPCBufferDecoder {
        /// Memory (or memory mapped) Buffer with the data
        buffer: Buffer,
        /// Decoder that reads Arrays that refers to the underlying buffers
        decoder: FileDecoder,
        /// Location of the batches within the buffer
        batches: Vec<Block>,
    }

    impl IPCBufferDecoder {
        fn new(buffer: Buffer) -> Self {
            let trailer_start = buffer.len() - 10;
            let footer_len =
                read_footer_length(buffer[trailer_start..].try_into().unwrap()).unwrap();
            let footer =
                root_as_footer(&buffer[trailer_start - footer_len..trailer_start]).unwrap();

            let schema = fb_to_schema(footer.schema().unwrap());

            let mut decoder = FileDecoder::new(Arc::new(schema), footer.version());

            // Read dictionaries
            for block in footer.dictionaries().iter().flatten() {
                let block_len = block.bodyLength() as usize + block.metaDataLength() as usize;
                let data = buffer.slice_with_length(block.offset() as _, block_len);
                decoder.read_dictionary(block, &data).unwrap();
            }

            // convert to Vec from the flatbuffers Vector to avoid having a direct dependency on flatbuffers
            let batches = footer
                .recordBatches()
                .map(|b| b.iter().copied().collect())
                .unwrap_or_default();

            Self {
                buffer,
                decoder,
                batches,
            }
        }

        /// Return the number of [`RecordBatch`]es in this buffer
        fn num_batches(&self) -> usize {
            self.batches.len()
        }

        /// Return the [`RecordBatch`] at message index `i`.
        ///
        /// This may return `None` if the IPC message was None
        fn get_batch(&self, i: usize) -> arrow::error::Result<Option<RecordBatch>> {
            let block = &self.batches[i];
            let block_len = block.bodyLength() as usize + block.metaDataLength() as usize;
            let data = self
                .buffer
                .slice_with_length(block.offset() as _, block_len);

            println!("Data length: {}", data.len());

            self.decoder.read_record_batch(block, &data)
        }
    }

    #[test]
    fn test_name() {
        let data = Arc::new(Int32Array::from((0..10_000).collect::<Vec<_>>()));
        let list_arr = ListArray::new(
            Arc::new(Field::new(
                "values",
                arrow::datatypes::DataType::Int32,
                true,
            )),
            OffsetBuffer::from_lengths(vec![data.len()]),
            data,
            None,
        );

        let field = Field::new(
            "test",
            arrow::datatypes::DataType::List(Arc::new(Field::new(
                "values",
                arrow::datatypes::DataType::Int32,
                true,
            ))),
            true,
        );

        let schema = Arc::new(arrow::datatypes::Schema::new(vec![field]));

        let mut buffer = vec![];

        let writer_props = WriterProperties::builder()
            .set_statistics_enabled(parquet::file::properties::EnabledStatistics::Page)
            .set_data_page_size_limit(64 * 1024)
            .build();

        let mut test_writer = ArrowWriter::try_new(
            std::io::Cursor::new(&mut buffer),
            schema.clone(),
            Some(writer_props),
        )
        .unwrap();

        for i in 0..200 {
            let rb =
                RecordBatch::try_new(schema.clone(), vec![Arc::new(list_arr.clone())]).unwrap();
            test_writer.write(&rb).unwrap();
        }

        // test_writer.write(&rb).unwrap();

        test_writer.close().unwrap();

        let bytes = Bytes::from(buffer);

        let arrow_reader_options = ArrowReaderOptions::default().with_page_index(true);
        let mut reader =
            ParquetRecordBatchReaderBuilder::try_new_with_options(bytes, arrow_reader_options)
                .unwrap();

        let metadata = reader.metadata();
        let rg_metadata = &metadata.row_groups()[0];

        match metadata.column_index().unwrap()[0][0] {
            parquet::file::page_index::index::Index::INT32(ref index) => {
                println!("Column index: {:?}", index.indexes);
            }
            _ => {
                println!("Column index: {:?}", metadata.column_index().unwrap()[0][0]);
            }
        }

        // rg_metadata.columns().iter().for_each(|column| {
        //     println!(
        //         "Column path: {:?}, stats: {:?}",
        //         column.column_path(),
        //         column.statistics().unwrap().
        //     );
        // });
    }
}
