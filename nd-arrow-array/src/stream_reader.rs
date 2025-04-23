use arrow::{
    array::{RecordBatch, StructArray},
    datatypes::{DataType, Field, Fields, SchemaRef},
};

pub struct NdRecordBatchStreamReader {
    input_schema: SchemaRef,
    output_schema: SchemaRef,
    struct_array_indexes: Vec<Vec<usize>>,
}

impl NdRecordBatchStreamReader {
    pub fn new(input_schema: SchemaRef) -> Self {
        //For each field, find the .dimension_name & .dimension_size array and re-create struct array
        let chunk_size = 4;
        let field_slices = input_schema.fields.chunks_exact(chunk_size);
        let mut schema_struct_fields = vec![];
        let mut schema_struct_array_indexes: Vec<Vec<usize>> = vec![];
        for (chunk_index, slice) in field_slices.enumerate() {
            let index = chunk_index * chunk_size;
            let struct_name = slice[0].name().split(".").next().unwrap();

            let dimension_names_field = slice[0]
                .as_ref()
                .clone()
                .with_name(format!("{}.dimension_names", struct_name));
            let dimension_sizes_field = slice[1]
                .as_ref()
                .clone()
                .with_name(format!("{}.dimension_sizes", struct_name));
            let data_array_field = slice[2]
                .as_ref()
                .clone()
                .with_name(format!("{}.values", struct_name));

            let struct_fields = Fields::from(vec![
                dimension_names_field,
                dimension_sizes_field,
                data_array_field,
            ]);
            let struct_dtype = DataType::Struct(struct_fields);

            schema_struct_fields.push(Field::new(struct_name, struct_dtype, true));

            let struct_array_indexes = vec![index, index + 1, index + 2];
            schema_struct_array_indexes.push(struct_array_indexes);
        }

        todo!()
    }

    pub fn ouput_schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    pub fn process_batch(
        &self,
        batch: arrow::record_batch::RecordBatch,
    ) -> Result<RecordBatch, ()> {
        // Convert the flattened struct arrays of the batch to the original schema
        // StructArray::todo!()
        todo!()
    }
}
