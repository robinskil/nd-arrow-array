//! Python bindings for nd-arrow-array.

mod error;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("NdFormatError", m.py().get_type::<error::NdFormatError>())?;
    Ok(())
}
