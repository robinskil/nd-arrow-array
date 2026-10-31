use std::sync::Arc;

use arrow::array::{Float32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use nd_arrow_array::extension::is_nd_field;
use nd_arrow_array::{
    Attributes, Dimensions, NdArrayType, NdArrowArray, NdColumn, NdRecordBatch, encode_batches,
};

fn ipc_round_trip(batch: &RecordBatch) -> RecordBatch {
    let mut bytes = Vec::new();
    let mut writer = StreamWriter::try_new(&mut bytes, &batch.schema()).unwrap();
    writer.write(batch).unwrap();
    writer.finish().unwrap();
    drop(writer);

    let mut reader = StreamReader::try_new(bytes.as_slice(), None).unwrap();
    let read = reader.next().unwrap().unwrap();
    assert!(reader.next().is_none());
    read
}

fn sample_batch() -> NdRecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("time", DataType::Int64, true),
        Field::new("station", DataType::Utf8, true),
        Field::new("temp", DataType::Float32, true),
    ]));
    let time = NdArrowArray::try_new(
        Arc::new(Int64Array::from(vec![0, 3600])),
        Dimensions::try_new([("time", 2)]).unwrap(),
    )
    .unwrap()
    .with_attributes(Attributes::new().with("units", "seconds since 1970-01-01"));
    let station = NdArrowArray::try_new(
        Arc::new(StringArray::from(vec!["A", "B", "C"])),
        Dimensions::try_new([("station", 3)]).unwrap(),
    )
    .unwrap();
    let temp = NdArrowArray::try_new(
        Arc::new(Float32Array::from(vec![
            Some(1.0),
            None,
            Some(3.0),
            Some(4.0),
            Some(5.0),
            None,
        ])),
        Dimensions::try_new([("time", 2), ("station", 3)]).unwrap(),
    )
    .unwrap()
    .with_attributes(
        Attributes::new()
            .with("units", "degC")
            .with("valid_range", vec![-50.0f32, 60.0]),
    );
    NdRecordBatch::try_new_inferred(schema, vec![time, station, temp]).unwrap()
}

#[test]
fn nd_batch_survives_ipc() {
    let nd = sample_batch();
    let encoded = encode_batches(&[nd.clone(), nd.clone()]).unwrap();
    let read = ipc_round_trip(&encoded);

    assert_eq!(read.schema(), encoded.schema());
    for field in read.schema().fields() {
        assert!(
            is_nd_field(field),
            "{} lost the extension tag",
            field.name()
        );
        let ext: NdArrayType = field.try_extension_type().unwrap();
        assert_eq!(field.data_type(), &ext.storage_type());
    }

    let decoded = NdRecordBatch::try_from_record_batch(&read).unwrap();
    assert_eq!(decoded.len(), 2);
    for batch in decoded {
        assert_eq!(batch.target(), nd.target());
        assert_eq!(batch.columns(), nd.columns());
    }
}

#[test]
fn column_without_attributes_survives_ipc() {
    let array = NdArrowArray::try_new_scalar(Arc::new(Int64Array::from(vec![42]))).unwrap();
    let column = NdColumn::try_from_arrays([&array]).unwrap();
    assert!(column.extension_type().attribute_fields().is_empty());

    let schema = Arc::new(Schema::new(vec![column.to_field("x")]));
    let batch = RecordBatch::try_new(schema, vec![column.into_array_ref()]).unwrap();
    let read = ipc_round_trip(&batch);

    let decoded = NdColumn::try_from_array(read.column(0).as_ref()).unwrap();
    assert_eq!(decoded.value(0), array);
}

#[test]
fn non_nd_batch_rejected() {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1]))]).unwrap();
    assert!(NdRecordBatch::try_from_record_batch(&batch).is_err());
}
