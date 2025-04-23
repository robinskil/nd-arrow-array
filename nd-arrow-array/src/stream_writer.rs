use std::sync::Arc;

use arrow::{
    array::RecordBatch,
    datatypes::{DataType, Field, Schema, SchemaRef},
};

pub struct NdRecordBatchStreamWriter {
    input_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl NdRecordBatchStreamWriter {
    fn flatten_schema(input_schema: SchemaRef) -> anyhow::Result<SchemaRef> {
        let mut fields: Vec<Field> = Vec::new();

        for field in input_schema.fields() {
            if let DataType::Struct(struct_fields) = field.data_type() {
                for struct_field in struct_fields {
                    let flattened_field = Field::new(
                        format!("{}.{}", field.name(), struct_field.name()),
                        struct_field.data_type().clone(),
                        struct_field.is_nullable(),
                    );
                    fields.push(flattened_field);
                }
            } else {
                return Err(anyhow::anyhow!(
                    "Expected struct types, found non-struct type in input schema: {} with data type {}",
                    field.name(),
                    field.data_type(),
                ));
            }
        }

        Ok(Arc::new(Schema::new(fields)))
    }

    pub fn new(input_schema: SchemaRef) -> anyhow::Result<Self> {
        let output_schema = Self::flatten_schema(input_schema.clone())?;

        Ok(Self {
            input_schema,
            output_schema,
        })
    }

    pub fn output_schema(&self) -> &SchemaRef {
        &self.output_schema
    }

    pub fn write_batch(
        &self,
        batch: arrow::record_batch::RecordBatch,
    ) -> anyhow::Result<RecordBatch> {
        // Convert the struct arrays of the batch to the flattened output schema
        let mut flattened_arrays = Vec::new();

        for (field_idx, field) in self.input_schema.fields().iter().enumerate() {
            if let DataType::Struct(struct_fields) = field.data_type() {
                let struct_array = batch.column(field_idx);
                let struct_array = struct_array
                    .as_any()
                    .downcast_ref::<arrow::array::StructArray>()
                    .unwrap();
                for struct_field in struct_fields {
                    let field_array = struct_array
                        .column_by_name(struct_field.name())
                        .unwrap()
                        .clone();
                    flattened_arrays.push(field_array);
                }
            } else {
                return Err(anyhow::anyhow!(
                    "Unsupported data type: {}",
                    field.data_type()
                ));
            }
        }

        Ok(RecordBatch::try_new(
            self.output_schema.clone(),
            flattened_arrays,
        )?)
    }
}
