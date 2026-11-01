//! A NumPy-like text form of an [`NdArrowArray`].
//!
//! ```text
//! <NdArrowArray Float64 (time: 2, lat: 3)>
//! [[0.0, 0.1, 0.2],
//!  [1.0, 1.1, 1.2]]
//! Attributes:
//!     units: K
//! ```
//!
//! Like NumPy, an axis longer than [`EDGE_ITEMS`] * 2 shows only its first
//! and last items when the array has more than [`SUMMARY_THRESHOLD`] values.

use std::fmt;

use arrow::util::display::{ArrayFormatter, FormatOptions};

use crate::NdArrowArray;

/// Arrays with more values than this are summarized, like NumPy's `threshold`.
pub const SUMMARY_THRESHOLD: usize = 1000;
/// Items shown at each end of a summarized axis, like NumPy's `edgeitems`.
pub const EDGE_ITEMS: usize = 3;

impl fmt::Display for NdArrowArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims: Vec<String> = self
            .dims()
            .iter()
            .map(|d| format!("{}: {}", d.name(), d.size()))
            .collect();
        writeln!(
            f,
            "<NdArrowArray {} ({})>",
            self.data_type(),
            dims.join(", ")
        )?;

        let options = FormatOptions::default().with_null("null");
        let formatter =
            ArrayFormatter::try_new(self.values().as_ref(), &options).map_err(|_| fmt::Error)?;
        let value = |i: usize| formatter.value(i).to_string();
        let summarize = self.len() > SUMMARY_THRESHOLD;
        let shape = self.shape();
        let strides = self.dims().c_strides();
        if shape.is_empty() {
            write!(f, "{}", value(0))?;
        } else {
            write_axis(f, &value, &shape, &strides, 0, 0, summarize)?;
        }

        if !self.attributes().is_empty() {
            write!(f, "\nAttributes:")?;
            for (key, attr) in self.attributes().iter() {
                let attr_formatter =
                    ArrayFormatter::try_new(attr.as_ref(), &options).map_err(|_| fmt::Error)?;
                write!(f, "\n    {key}: {}", attr_formatter.value(0))?;
            }
        }
        Ok(())
    }
}

fn write_axis(
    f: &mut fmt::Formatter<'_>,
    value: &impl Fn(usize) -> String,
    shape: &[usize],
    strides: &[usize],
    axis: usize,
    offset: usize,
    summarize: bool,
) -> fmt::Result {
    let size = shape[axis];
    let indices: Vec<Option<usize>> = if summarize && size > 2 * EDGE_ITEMS {
        (0..EDGE_ITEMS)
            .map(Some)
            .chain(std::iter::once(None))
            .chain((size - EDGE_ITEMS..size).map(Some))
            .collect()
    } else {
        (0..size).map(Some).collect()
    };
    let last = axis + 1 == shape.len();
    write!(f, "[")?;
    for (k, index) in indices.iter().enumerate() {
        if k > 0 {
            if last {
                write!(f, ", ")?;
            } else {
                write!(f, ",{}", "\n".repeat(shape.len() - axis - 1))?;
                write!(f, "{}", " ".repeat(axis + 1))?;
            }
        }
        match index {
            None => write!(f, "...")?,
            Some(i) if last => write!(f, "{}", value(offset + i * strides[axis]))?,
            Some(i) => write_axis(
                f,
                value,
                shape,
                strides,
                axis + 1,
                offset + i * strides[axis],
                summarize,
            )?,
        }
    }
    write!(f, "]")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int32Array};

    use crate::{Attributes, Dimensions, NdArrowArray};

    #[test]
    fn prints_like_numpy() {
        let a = NdArrowArray::try_new(
            Arc::new(Float64Array::from(vec![
                Some(0.0),
                Some(0.5),
                None,
                Some(1.0),
                Some(1.5),
                Some(2.0),
            ])),
            Dimensions::try_new([("time", 2), ("lat", 3)]).unwrap(),
        )
        .unwrap()
        .with_attributes(Attributes::new().with("units", "K"));
        assert_eq!(
            a.to_string(),
            "<NdArrowArray Float64 (time: 2, lat: 3)>\n[[0.0, 0.5, null],\n [1.0, 1.5, 2.0]]\nAttributes:\n    units: K"
        );
    }

    #[test]
    fn summarizes_large_arrays() {
        let a = NdArrowArray::try_new(
            Arc::new(Int32Array::from_iter_values(0..2000)),
            Dimensions::try_new([("x", 2000)]).unwrap(),
        )
        .unwrap();
        assert!(a.to_string().ends_with("[0, 1, 2, ..., 1997, 1998, 1999]"));
        let s = NdArrowArray::try_new_scalar(Arc::new(Int32Array::from(vec![7]))).unwrap();
        assert_eq!(s.to_string(), "<NdArrowArray Int32 ()>\n7");
    }
}
