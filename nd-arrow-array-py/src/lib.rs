//! Python bindings for nd-arrow-array.

mod error;
mod numpy_conv;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("NdFormatError", m.py().get_type::<error::NdFormatError>())?;
    m.add_function(wrap_pyfunction!(numpy_conv::_roundtrip_numpy, m)?)?;
    Ok(())
}
