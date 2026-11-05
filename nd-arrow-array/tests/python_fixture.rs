//! Read an nd record batch that the Python bindings wrote.
//! Generate tests/fixtures/python.arrow with `pytest` in nd-arrow-py.

use std::fs::File;
use std::path::Path;

use arrow::array::{Array, AsArray};
use arrow::datatypes::{DataType, Float32Type, Float64Type, Int32Type, TimeUnit};
use arrow::ipc::reader::FileReader;
use nd_arrow_array::NdRecordBatch;

#[test]
fn reads_python_fixture() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/python.arrow");
    if !path.exists() {
        eprintln!("skipped: {} is absent", path.display());
        return;
    }
    let mut reader = FileReader::try_new(File::open(path).unwrap(), None).unwrap();
    let batch = reader.next().unwrap().unwrap();
    let decoded = NdRecordBatch::try_from_record_batch(&batch).unwrap();
    assert_eq!(decoded.len(), 1);
    let ds = &decoded[0];

    assert_eq!(ds.attributes().get_str("title"), Some("python fixture"));
    assert_eq!(ds.target().names(), vec!["time", "lat"]);
    assert_eq!(ds.target().shape(), vec![2, 3]);

    let sst = ds.column_by_name("sst").unwrap();
    assert_eq!(sst.dims().names(), vec!["time", "lat"]);
    let values = sst.values().as_primitive::<Float64Type>();
    assert_eq!(values.value(0), 1.5);
    assert!(values.is_null(2));
    assert_eq!(values.value(5), 6.5);
    assert_eq!(sst.attributes().get_str("units"), Some("K"));
    let range = sst
        .attributes()
        .get("valid_range")
        .unwrap()
        .as_list::<i32>()
        .value(0);
    assert_eq!(range.as_primitive::<Float32Type>().values(), &[0.0, 400.0]);

    let lat = ds.column_by_name("lat").unwrap();
    assert_eq!(
        lat.values().as_primitive::<Int32Type>().values(),
        &[-10, 0, 10]
    );

    let time = ds.column_by_name("time").unwrap();
    assert_eq!(
        time.data_type(),
        &DataType::Timestamp(TimeUnit::Second, None)
    );
}
