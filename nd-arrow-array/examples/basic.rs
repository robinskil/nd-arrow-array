use std::sync::Arc;

use arrow::array::Float64Array;
use nd_arrow_array::compute::{Selector, Slice};
use nd_arrow_array::{Attributes, Dimensions, NdArrowArray};

fn main() -> nd_arrow_array::Result<()> {
    // A (time=2, lat=3) array of temperatures, flat in C order.
    let sst = NdArrowArray::try_new(
        Arc::new(Float64Array::from(vec![10.0, 12.0, 14.0, 11.0, 13.0, 15.0])),
        Dimensions::try_new([("time", 2), ("lat", 3)])?,
    )?
    .with_attributes(Attributes::new().with("units", "degC"));

    // Subtract the mean over time. Axes align by name.
    let anomaly = (&sst - &sst.mean(Some(&["time"]))?)?;

    // Keep the first two latitudes.
    let north = anomaly.isel(&[("lat", Selector::Slice(Slice::range(0, 2)))])?;
    println!("{north}");
    Ok(())
}
