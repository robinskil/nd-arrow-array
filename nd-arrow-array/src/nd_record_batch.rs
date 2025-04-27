use std::{collections::HashMap, sync::Arc};

use arrow::{
    array::RecordBatch,
    datatypes::{Field, Schema, SchemaRef},
};

use crate::{
    broadcast::{self, BroadcastingError},
    consts,
    nd_array::{
        arrow_ext::{self, ArrowParseError},
        NdArrowArray,
    },
    version::validate_version_compatibility,
};

#[derive(Debug, Clone, thiserror::Error)]
pub enum NdRecordBatchError {
    #[error("Failed to parse Arrow Array to Nd Arrow Array: {0}")]
    NdArrowParseError(ArrowParseError),
    #[error("Failed to downcast Arrow Array to Struct Array: {0}")]
    DowncastError(Arc<Field>),
    #[error("Failed to encode Nd Arrow Array to Arrow Array: {0}")]
    ArrowEncodeError(arrow_ext::ArrowParseError),
    #[error("Missing encoding metadata in Arrow RecordBatch")]
    MissingEncodingMetadata,
    #[error("Unsupported encoding version for the current library version: {0}")]
    UnsupportedEncodingVersion(String),
}

pub struct NdRecordBatch {
    schema: SchemaRef,
    arrays: Vec<Arc<dyn NdArrowArray>>,
}

impl NdRecordBatch {
    pub fn to_arrow_encoded_record_batch(&self) -> Result<RecordBatch, NdRecordBatchError> {
        let mut encoding_metadata = HashMap::new();
        encoding_metadata.insert(
            consts::ND_ARROW_SCHEMA_ENCODING_KEY.to_string(),
            consts::ND_ARROW_SCHEMA_ENCODING_VERSION.to_string(),
        );

        let mut encoded_schema_fields = Vec::new();
        let mut encoded_arrays = Vec::new();

        for (idx, array) in self.arrays.iter().enumerate() {
            let arrow_encoded = array
                .to_arrow_array()
                .map_err(|e| NdRecordBatchError::ArrowEncodeError(e))?;

            let field = self.schema.field(idx).clone();
            let updated_field = field.with_data_type(arrow_encoded.data_type().clone());

            encoded_schema_fields.push(updated_field);
            encoded_arrays.push(arrow_encoded);
        }

        Ok(RecordBatch::try_new(
            Arc::new(Schema::new_with_metadata(
                encoded_schema_fields,
                encoding_metadata,
            )),
            encoded_arrays,
        )
        .unwrap())
    }

    pub fn broadcast_to_record_batch(&self) -> Result<RecordBatch, BroadcastingError> {
        let all_dimensions = self
            .arrays
            .iter()
            .map(|array| array.dimensions().to_vec())
            .collect::<Vec<_>>();

        let broadcast_dimensions = broadcast::find_broadcast_dimensions(&all_dimensions);

        match broadcast_dimensions {
            Some(target_dimensions) => {
                let broadcasted_arrays = self
                    .arrays
                    .iter()
                    .map(|array| array.broadcast(&target_dimensions))
                    .collect::<Result<Vec<_>, BroadcastingError>>()?;

                let fields = self.schema.fields().clone();
                let mut arrays = Vec::with_capacity(fields.len());
                for array in broadcasted_arrays {
                    arrays.push(array.array());
                }

                Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap())
            }
            None => {
                // If no broadcast dimensions are found, then everything is already the same shape as they are all scalars. We can just return all the arrays as they are.
                let mut arrays = vec![];
                let mut fields = vec![];

                for (idx, array) in self.arrays.iter().enumerate() {
                    let field = self.schema.field(idx).clone();
                    let updated_field = field.with_data_type(array.dtype().clone());
                    fields.push(updated_field);
                    arrays.push(array.array());
                }

                return Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap());
            }
        }
    }

    pub fn from_arrow_encoded_record_batch(
        record_batch: RecordBatch,
    ) -> Result<Self, NdRecordBatchError> {
        //Check encoding metadata
        let encoding_metadata = record_batch.schema_ref().metadata();
        // Check if the metadata contains the expected key and some version
        if let Some(version) = encoding_metadata.get(consts::ND_ARROW_SCHEMA_ENCODING_KEY) {
            if !validate_version_compatibility(version) {
                return Err(NdRecordBatchError::UnsupportedEncodingVersion(
                    version.to_string(),
                ));
            }
        } else {
            return Err(NdRecordBatchError::MissingEncodingMetadata);
        }

        let array_iter = record_batch.columns().to_vec();
        let field_iter = record_batch.schema().fields().clone();

        //Zip the fields and arrays together
        let zipped = field_iter
            .into_iter()
            .zip(array_iter.into_iter())
            .map(|(field, array)| {
                let struct_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::StructArray>()
                    .ok_or(NdRecordBatchError::DowncastError(field.clone()))?;

                let nd_array = arrow_ext::try_from_arrow_array(struct_array)
                    .map_err(|e| NdRecordBatchError::NdArrowParseError(e))?;

                let updated_field = Field::new(
                    field.name(),
                    nd_array.dtype().clone(),
                    nd_array.is_nullable(),
                );

                Ok((Arc::new(updated_field), nd_array))
            })
            .collect::<Result<Vec<_>, NdRecordBatchError>>()?;

        let fields = zipped
            .iter()
            .map(|(field, _)| field.clone())
            .collect::<Vec<_>>();

        let arrays = zipped
            .into_iter()
            .map(|(_, nd_array)| nd_array)
            .collect::<Vec<_>>();

        let schema = Arc::new(arrow::datatypes::Schema::new(fields));

        Ok(Self { schema, arrays })
    }
}
