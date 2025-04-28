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

#[derive(Debug, Clone)]
pub struct NdRecordBatch {
    schema: SchemaRef,
    arrays: Vec<Arc<dyn NdArrowArray>>,
}

impl NdRecordBatch {
    pub fn new(schema: SchemaRef, arrays: Vec<Arc<dyn NdArrowArray>>) -> Self {
        Self { schema, arrays }
    }

    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    pub fn arrow_encoded_schema(&self) -> SchemaRef {
        let mut encoding_metadata = HashMap::new();
        encoding_metadata.insert(
            consts::ND_ARROW_SCHEMA_ENCODING_KEY.to_string(),
            consts::ND_ARROW_SCHEMA_ENCODING_VERSION.to_string(),
        );

        let encoded_types = self
            .arrays
            .iter()
            .map(|array| array.arrow_encoded_type())
            .collect::<Vec<_>>();

        let encoded_fields = self
            .schema
            .fields()
            .iter()
            .enumerate()
            .map(|(idx, field)| {
                let encoded_field = Field::new(
                    field.name(),
                    encoded_types[idx].clone(),
                    field.is_nullable(),
                );
                encoded_field
            })
            .collect::<Vec<_>>();

        Arc::new(Schema::new_with_metadata(encoded_fields, encoding_metadata))
    }

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

    pub fn from_arrow_encoded_record_batch_impl(
        record_batch: RecordBatch,
    ) -> Result<Self, NdRecordBatchError> {
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

        Self::from_arrow_encoded_record_batch_impl(record_batch)
    }
}
#[cfg(test)]
mod tests {
    use crate::nd_array::{
        default::{DefaultNdArrowArray, SCALAR_DIMENSION},
        NdArrowArray,
    };

    use super::*;

    use arrow::datatypes::{DataType, Float64Type, Int32Type};
    use std::collections::HashMap;

    #[test]
    fn test_to_arrow_encoded_record_batch() {
        // Create a simple NdRecordBatch with two arrays
        let array1 = Arc::new(DefaultNdArrowArray::from_vec::<Int32Type>(
            vec![Some(1), Some(2), Some(3), Some(4)],
            vec![("dim1", 2), ("dim2", 2)],
        )) as Arc<dyn NdArrowArray>;
        let array2 = Arc::new(DefaultNdArrowArray::from_vec::<Float64Type>(
            vec![Some(1.1), Some(2.2), Some(3.3), Some(4.4)],
            vec![("dim1", 2), ("dim2", 2)],
        )) as Arc<dyn NdArrowArray>;

        let field1 = Field::new("field1", DataType::Int32, false);
        let field2 = Field::new("field2", DataType::Float64, false);
        let schema = Arc::new(Schema::new(vec![field1, field2]));

        let nd_record_batch = NdRecordBatch {
            schema,
            arrays: vec![array1, array2],
        };

        // Convert to arrow encoded record batch
        let arrow_batch = nd_record_batch.to_arrow_encoded_record_batch().unwrap();

        // Verify metadata
        assert!(arrow_batch
            .schema_ref()
            .metadata()
            .contains_key(consts::ND_ARROW_SCHEMA_ENCODING_KEY));
        assert_eq!(
            arrow_batch
                .schema_ref()
                .metadata()
                .get(consts::ND_ARROW_SCHEMA_ENCODING_KEY)
                .unwrap(),
            consts::ND_ARROW_SCHEMA_ENCODING_VERSION
        );

        // Verify column count
        assert_eq!(arrow_batch.num_columns(), 2);
        println!("Arrow batch: {:#?}", arrow_batch);
    }

    #[test]
    fn test_round_trip_encoding() {
        // Create a simple NdRecordBatch with two arrays
        let array1 = Arc::new(DefaultNdArrowArray::from_vec::<Int32Type>(
            vec![Some(1), Some(2), Some(3), Some(4)],
            vec![("dim1", 2), ("dim2", 2)],
        )) as Arc<dyn NdArrowArray>;
        let array2 = Arc::new(DefaultNdArrowArray::from_vec::<Float64Type>(
            vec![Some(1.1), Some(2.2), Some(3.3), Some(4.4)],
            vec![("dim1", 2), ("dim2", 2)],
        )) as Arc<dyn NdArrowArray>;

        let field1 = Field::new("field1", DataType::Int32, false);
        let field2 = Field::new("field2", DataType::Float64, false);
        let schema = Arc::new(Schema::new(vec![field1, field2]));

        let original = NdRecordBatch {
            schema,
            arrays: vec![array1, array2],
        };

        // Convert to arrow and back
        let arrow_batch = original.to_arrow_encoded_record_batch().unwrap();
        let roundtrip = NdRecordBatch::from_arrow_encoded_record_batch(arrow_batch).unwrap();

        // Verify schema field count
        assert_eq!(roundtrip.schema.fields().len(), 2);

        // Verify array count
        assert_eq!(roundtrip.arrays.len(), 2);

        // Verify dimensions of arrays
        assert_eq!(
            roundtrip.arrays[0].dimensions(),
            vec![("dim1", 2).into(), ("dim2", 2).into()]
        );
        assert_eq!(
            roundtrip.arrays[1].dimensions(),
            &vec![("dim1", 2).into(), ("dim2", 2).into()]
        );
    }

    #[test]
    fn test_broadcast_to_record_batch() {
        // Create a simple NdRecordBatch with two arrays
        let array1 = Arc::new(DefaultNdArrowArray::from_vec::<Int32Type>(
            vec![Some(1), Some(2), Some(3), Some(4)],
            vec![("dim1", 2), ("dim2", 2)],
        )) as Arc<dyn NdArrowArray>;
        let array2 = Arc::new(DefaultNdArrowArray::from_vec::<Float64Type>(
            vec![Some(3.3), Some(4.4)],
            vec![("dim2", 2)],
        )) as Arc<dyn NdArrowArray>;
        let array3 = Arc::new(DefaultNdArrowArray::from_vec::<Float64Type>(
            vec![Some(0.1)],
            SCALAR_DIMENSION,
        )) as Arc<dyn NdArrowArray>;

        let field1 = Field::new("field1", DataType::Int32, false);
        let field2 = Field::new("field2", DataType::Float64, false);
        let field3 = Field::new("field3", DataType::Float64, false);
        let schema = Arc::new(Schema::new(vec![field1, field2, field3]));

        let nd_record_batch = NdRecordBatch {
            schema,
            arrays: vec![array1, array2, array3],
        };

        // Broadcast to common shape
        let result = nd_record_batch.broadcast_to_record_batch().unwrap();

        // Should have 2 columns
        assert_eq!(result.num_columns(), 3);

        // The result should have 4 elements (2x2)
        assert_eq!(result.num_rows(), 4);

        println!("Broadcasted RecordBatch: {:#?}", result);
    }

    #[test]
    fn test_missing_encoding_metadata() {
        // Create a record batch without the required metadata
        let array = arrow::array::Int32Array::from(vec![1, 2, 3, 4]);

        let field = Field::new("field", DataType::Int32, false);
        let schema = Arc::new(Schema::new(vec![field]));
        let record_batch = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();

        // Should return an error for missing metadata
        let result = NdRecordBatch::from_arrow_encoded_record_batch(record_batch);
        assert!(matches!(
            result,
            Err(NdRecordBatchError::MissingEncodingMetadata)
        ));
    }
}
