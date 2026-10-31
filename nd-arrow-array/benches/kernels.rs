use std::hint::black_box;
use std::sync::Arc;

use arrow::array::Float64Array;
use criterion::{Criterion, criterion_group, criterion_main};
use nd_arrow_array::compute::{self, Selector, Slice};
use nd_arrow_array::{Dimensions, NdArrowArray};

fn grid(time: usize, lat: usize, lon: usize) -> NdArrowArray {
    let n = time * lat * lon;
    NdArrowArray::try_new(
        Arc::new(Float64Array::from_iter_values((0..n).map(|i| i as f64))),
        Dimensions::try_new([("time", time), ("lat", lat), ("lon", lon)]).unwrap(),
    )
    .unwrap()
}

fn kernels(c: &mut Criterion) {
    let a = grid(100, 180, 360);
    let lat = NdArrowArray::try_new(
        Arc::new(Float64Array::from_iter_values((0..180).map(f64::from))),
        Dimensions::try_new([("lat", 180)]).unwrap(),
    )
    .unwrap();

    c.bench_function("broadcast lat to grid", |b| {
        b.iter(|| compute::broadcast_to(black_box(&lat), a.dims()).unwrap())
    });
    c.bench_function("transpose time last", |b| {
        b.iter(|| compute::transpose(black_box(&a), &["lat", "lon", "time"]).unwrap())
    });
    c.bench_function("isel lat slice", |b| {
        b.iter(|| {
            compute::isel(
                black_box(&a),
                &[("lat", Selector::Slice(Slice::range(10, 90)))],
            )
            .unwrap()
        })
    });
    c.bench_function("sum over time (leading)", |b| {
        b.iter(|| compute::sum(black_box(&a), Some(&["time"])).unwrap())
    });
    c.bench_function("sum over lon (trailing)", |b| {
        b.iter(|| compute::sum(black_box(&a), Some(&["lon"])).unwrap())
    });
    c.bench_function("mean over time and lon", |b| {
        b.iter(|| compute::mean(black_box(&a), Some(&["time", "lon"])).unwrap())
    });
    c.bench_function("add with broadcast", |b| {
        b.iter(|| compute::add(black_box(&a), black_box(&lat)).unwrap())
    });
}

criterion_group!(benches, kernels);
criterion_main!(benches);
