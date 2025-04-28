use std::sync::Arc;

use arrow::{array::RecordBatch, datatypes::SchemaRef};
use futures::{Stream, StreamExt};
use parquet::{
    arrow::{
        arrow_reader::ArrowReaderOptions,
        async_reader::{AsyncFileReader, ParquetRecordBatchStream},
        ParquetRecordBatchStreamBuilder, ProjectionMask,
    },
    errors::ParquetError,
};

use crate::{consts, nd_record_batch::NdRecordBatch, version::validate_version_compatibility};

pub struct AsyncNdParquetReaderBuilder<T: AsyncFileReader + 'static> {
    simplified_schema: SchemaRef,
    builder: ParquetRecordBatchStreamBuilder<T>,
}

impl<T: AsyncFileReader + 'static> AsyncNdParquetReaderBuilder<T> {
    pub async fn new(
        reader: T,
        reader_options: Option<ArrowReaderOptions>,
    ) -> Result<Self, ParquetError> {
        let reader_options = reader_options.unwrap_or_default();

        let builder =
            ParquetRecordBatchStreamBuilder::new_with_options(reader, reader_options).await?;
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

    pub fn with_projection(mut self, projection: Vec<usize>) -> Result<Self, ParquetError> {
        let projection_mask = ProjectionMask::roots(self.parquet_schema(), projection.into_iter());

        self.builder = self.builder.with_projection(projection_mask);
        Ok(self)
    }

    pub fn build(self) -> Result<NdParquetRecordBatchStream<T>, ParquetError> {
        let stream = self.builder.build()?;
        Ok(NdParquetRecordBatchStream { stream })
    }
}

pub struct NdParquetRecordBatchStream<T: AsyncFileReader> {
    stream: ParquetRecordBatchStream<T>,
}

impl<T: AsyncFileReader> NdParquetRecordBatchStream<T> {
    pub fn schema(&self) -> &SchemaRef {
        self.stream.schema()
    }
}

impl<T: AsyncFileReader + Unpin + 'static> Stream for NdParquetRecordBatchStream<T> {
    type Item = Result<RecordBatch, ParquetError>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let stream = this.stream.poll_next_unpin(cx);
        stream.map(|batch_opt| {
            batch_opt.map(|batch_res| batch_res.and_then(|batch| convert_nd_record_batch(batch)))
        })
    }
}

fn convert_nd_record_batch(batch: RecordBatch) -> Result<RecordBatch, ParquetError> {
    let nd_batch = NdRecordBatch::from_arrow_encoded_record_batch_impl(batch).map_err(|e| {
        ParquetError::General(format!(
            "Failed to parse record batch to NdRecordBatch: {}",
            e
        ))
    })?;

    nd_batch.broadcast_to_record_batch().map_err(|e| {
        ParquetError::General(format!(
            "Failed to broadcast NdRecordBatch to RecordBatch: {}",
            e
        ))
    })
}
