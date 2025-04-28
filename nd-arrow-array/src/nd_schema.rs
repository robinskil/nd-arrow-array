use std::sync::Arc;

use arrow::datatypes::{Field, SchemaRef};

use crate::consts;

pub fn translate_simplified_to_nd_schema(arrow_schema: &arrow::datatypes::Schema) -> SchemaRef {
    let mut nd_fields = Vec::new();
    for field in arrow_schema.fields() {
        let field_name = field.name().to_string();
        let field_type = field.data_type().clone();
        let is_nullable = field.is_nullable();

        let dimension_names_field = Field::new(
            consts::DIMENSION_NAMES,
            arrow::datatypes::DataType::Utf8,
            false,
        );
        let list_dimension_names_array_field = Field::new(
            consts::DIMENSION_NAMES,
            arrow::datatypes::DataType::List(Arc::new(dimension_names_field.clone())),
            false,
        );

        let dimension_sizes_field = Field::new(
            consts::DIMENSION_SIZES,
            arrow::datatypes::DataType::UInt32,
            false,
        );
        let list_dimension_sizes_array_field = Field::new(
            consts::DIMENSION_SIZES,
            arrow::datatypes::DataType::List(Arc::new(dimension_sizes_field.clone())),
            false,
        );

        let inner_array_field = Field::new(consts::VALUES, field_type, is_nullable);
        let list_array_field = Field::new(
            consts::VALUES,
            arrow::datatypes::DataType::List(Arc::new(inner_array_field.clone())),
            false,
        );

        let sub_fields = vec![
            list_dimension_names_array_field,
            list_dimension_sizes_array_field,
            list_array_field,
        ];

        let struct_dtype = arrow::datatypes::DataType::Struct(sub_fields.into());
        let nd_field = Field::new(field_name, struct_dtype, false);
        nd_fields.push(nd_field);
    }

    let nd_schema = arrow::datatypes::Schema::new(nd_fields);
    Arc::new(nd_schema)
}

pub fn translate_from_nd_schema_to_simplified(nd_schema: &SchemaRef) -> arrow::datatypes::Schema {
    let mut simplified_fields = Vec::new();
    for field in nd_schema.fields() {
        if let arrow::datatypes::DataType::Struct(fields) = field.data_type() {
            let values_field = fields.iter().find(|f| f.name() == consts::VALUES);
            if let Some(values_field) = values_field {
                if let arrow::datatypes::DataType::List(values_dtype) = values_field.data_type() {
                    simplified_fields.push(arrow::datatypes::Field::new(
                        field.name(),
                        values_dtype.data_type().clone(),
                        values_dtype.is_nullable(),
                    ));
                }
            }
        }
    }

    let simplified_schema = arrow::datatypes::Schema::new(simplified_fields);
    simplified_schema
}
#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema};

    #[test]
    fn test_round_trip_schema() {
        // Create a simplified schema with two fields.
        let simplified_schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, true),
        ]);

        // Convert to ND schema and then recover the simplified schema.
        let nd_schema = super::translate_simplified_to_nd_schema(&simplified_schema);

        println!("ND schema: {:?}", nd_schema);

        let recovered_schema = super::translate_from_nd_schema_to_simplified(&nd_schema);

        println!("Recovered schema: {:?}", recovered_schema);

        // Check that both schemas have the same field names, types, and nullability.
        assert_eq!(
            simplified_schema.fields().len(),
            recovered_schema.fields().len()
        );
        for (orig, rec) in simplified_schema
            .fields()
            .iter()
            .zip(recovered_schema.fields().iter())
        {
            assert_eq!(orig.name(), rec.name());
            assert_eq!(orig.data_type(), rec.data_type());
            assert_eq!(orig.is_nullable(), rec.is_nullable());
        }
    }
}
