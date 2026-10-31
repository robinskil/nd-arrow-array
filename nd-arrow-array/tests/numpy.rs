//! Compare the kernels with NumPy results from tests/fixtures/numpy.json.
//! Regenerate the fixtures with `python3 tests/fixtures/gen_numpy.py`.

use std::sync::Arc;

use arrow::array::{Array, AsArray, Float64Array};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type};
use nd_arrow_array::compute::{self, SearchSide};
use nd_arrow_array::{Dimensions, NdArrowArray};
use serde_json::Value;

fn array(value: &Value) -> NdArrowArray {
    let dims: Vec<(String, usize)> = value["dims"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| {
            (
                d[0].as_str().unwrap().to_string(),
                d[1].as_u64().unwrap() as usize,
            )
        })
        .collect();
    let data: Float64Array = value["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(Value::as_f64)
        .collect();
    NdArrowArray::try_new(Arc::new(data), Dimensions::try_new(dims).unwrap()).unwrap()
}

fn strings(value: &Value) -> Vec<&str> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect()
}

fn run(case: &Value) -> NdArrowArray {
    let input = array(&case["input"]);
    let params = &case["params"];
    let axes: Option<Vec<&str>> = params.get("axes").filter(|v| !v.is_null()).map(strings);
    let axes = axes.as_deref();
    let dim = || params["dim"].as_str().unwrap();
    match case["op"].as_str().unwrap() {
        "sum" => compute::sum(&input, axes),
        "mean" => compute::mean(&input, axes),
        "min" => compute::min(&input, axes),
        "max" => compute::max(&input, axes),
        "median" => compute::median(&input, axes),
        "count" => compute::count(&input, axes),
        "var0" => compute::var(&input, axes, 0),
        "var1" => compute::var(&input, axes, 1),
        "std1" => compute::std(&input, axes, 1),
        "quantile30" => compute::quantile(&input, axes, 0.3),
        "argmin" => compute::argmin(&input, axes),
        "argmax" => compute::argmax(&input, axes),
        "cumsum" => compute::cumsum(&input, dim()),
        "diff" => compute::diff(&input, dim(), params["n"].as_u64().unwrap() as usize),
        "transpose" => compute::transpose(&input, &strings(&params["order"])),
        "sort" => compute::sort(&input, dim(), false),
        "argsort" => compute::argsort(&input, dim(), false),
        "searchsorted" => {
            let side = match params["side"].as_str().unwrap() {
                "left" => SearchSide::Left,
                _ => SearchSide::Right,
            };
            compute::searchsorted(&input, &array(&params["values"]), side)
        }
        "interp" => compute::interp(&input, "x", &array(&params["x"]), &array(&params["new_x"])),
        "add" => compute::add(&input, &array(&params["other"])),
        "sub" => compute::sub(&input, &array(&params["other"])),
        "mul" => compute::mul(&input, &array(&params["other"])),
        "div" => compute::div(&input, &array(&params["other"])),
        other => panic!("unknown op {other}"),
    }
    .unwrap_or_else(|e| panic!("{}: {e}", case["name"]))
}

fn close(actual: Option<f64>, expected: Option<f64>) -> bool {
    match (actual, expected) {
        (None, None) => true,
        (Some(a), Some(e)) => (a - e).abs() <= 1e-9 * e.abs().max(1.0),
        _ => false,
    }
}

#[test]
fn kernels_match_numpy() {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/numpy.json"),
    )
    .unwrap();
    let fixtures: Value = serde_json::from_str(&text).unwrap();
    let cases = fixtures["cases"].as_array().unwrap();
    assert!(
        cases.len() >= 90,
        "fixture file has only {} cases",
        cases.len()
    );
    let mut failures = Vec::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let expected = array(&case["expected"]);
        let actual = run(case);
        if actual.dims() != expected.dims() {
            failures.push(format!(
                "{name}: dims {} != {}",
                actual.dims(),
                expected.dims()
            ));
            continue;
        }
        let values = cast(actual.values().as_ref(), &DataType::Float64).unwrap();
        let values = values.as_primitive::<Float64Type>();
        let wanted = expected.values().as_primitive::<Float64Type>();
        for i in 0..values.len() {
            let a = values.is_valid(i).then(|| values.value(i));
            let e = wanted.is_valid(i).then(|| wanted.value(i));
            if !close(a, e) {
                failures.push(format!("{name}[{i}]: {a:?} != {e:?}"));
                break;
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}
