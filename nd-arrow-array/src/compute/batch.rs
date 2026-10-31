//! Kernels on [`NdRecordBatch`]: materialize, select by position or by
//! coordinate value, and interpolate.
//!
//! A coordinate is a 1-D column with the same name as its dimension, like xarray.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, make_comparator};
use arrow::compute::{SortOptions, cast as arrow_cast};
use arrow::datatypes::{DataType, Field, Float64Type, Schema};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};

use crate::compute::interp::interp;
use crate::compute::search::partition_point;
use crate::compute::shape::{Selector, Slice, broadcast_to, isel, squeeze};
use crate::dimensions::{Dimension, Dimensions};
use crate::error::{Result, nd_err};
use crate::{NdArrowArray, NdRecordBatch};

impl NdRecordBatch {
    /// Broadcast every column onto the target grid and return a flat
    /// [`RecordBatch`] with one row per grid cell, in C order.
    pub fn materialize(&self) -> Result<RecordBatch> {
        let columns = self
            .columns()
            .iter()
            .map(|c| Ok(broadcast_to(c, self.target())?.values().clone()))
            .collect::<Result<Vec<ArrayRef>>>()?;
        let options = RecordBatchOptions::new().with_row_count(Some(self.num_rows()));
        Ok(RecordBatch::try_new_with_options(
            self.schema().clone(),
            columns,
            &options,
        )?)
    }

    /// The coordinate of `dim`: a column named `dim` with dimensions `(dim)`.
    pub fn coordinate(&self, dim: &str) -> Option<&NdArrowArray> {
        self.column_by_name(dim)
            .filter(|c| c.rank() == 1 && c.dims().as_slice()[0].name() == dim)
    }

    /// Select by position on target dimensions, like xarray `Dataset.isel`.
    ///
    /// Each column gets the selections of its own dimensions. A column with
    /// size 1 on a selected dimension keeps broadcasting, and an index
    /// selection removes that axis.
    pub fn isel(&self, selections: &[(&str, Selector)]) -> Result<Self> {
        let mut target: Vec<Dimension> = self.target().iter().cloned().collect();
        let mut resolved: Vec<(&str, Selector, usize)> = Vec::new();
        for (name, selector) in selections {
            let axis = self.target().axis(name)?;
            let size = target[axis].size();
            let count = match selector {
                Selector::Slice(slice) => slice.resolve(size)?.1,
                Selector::Index(_) => 0,
                Selector::Take(indices) => indices.len(),
            };
            resolved.push((name, selector.clone(), count));
            target[axis] = Dimension::new(*name, count);
        }
        let dropped: Vec<&str> = resolved
            .iter()
            .filter(|(_, s, _)| matches!(s, Selector::Index(_)))
            .map(|(n, _, _)| *n)
            .collect();
        target.retain(|d| !dropped.contains(&d.name()));

        let columns = self
            .columns()
            .iter()
            .map(|column| {
                let mut own = Vec::new();
                let mut squeezed = Vec::new();
                for (name, selector, count) in &resolved {
                    let Some(size) = column.dims().size_of(name) else {
                        continue;
                    };
                    let full = self.target().size_of(name) == Some(size);
                    if full {
                        own.push((*name, selector.clone()));
                    } else if matches!(selector, Selector::Index(_)) {
                        squeezed.push(*name);
                    } else if *count == 0 {
                        own.push((*name, Selector::Take(Vec::new())));
                    }
                }
                let selected = isel(column, &own)?;
                if squeezed.is_empty() {
                    Ok(selected)
                } else {
                    squeeze(&selected, Some(&squeezed))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        NdRecordBatch::try_new(self.schema().clone(), columns, Dimensions::try_new(target)?)
    }

    /// Select the positions whose coordinate lies in `[start, end]`, like
    /// xarray `sel` with a label slice. An open end is `None`.
    ///
    /// The coordinate must be sorted, ascending or descending, without nulls.
    pub fn sel_range(
        &self,
        dim: &str,
        start: Option<&ArrayRef>,
        end: Option<&ArrayRef>,
    ) -> Result<Self> {
        let coordinate = self.require_coordinate(dim)?;
        let values = coordinate.values();
        let n = values.len();
        let options = SortOptions::default();
        let self_compare = make_comparator(values.as_ref(), values.as_ref(), options)?;
        let ascending = (1..n).all(|i| self_compare(i - 1, i) != Ordering::Greater);
        let descending = (1..n).all(|i| self_compare(i - 1, i) != Ordering::Less);
        if values.null_count() > 0 || !(ascending || descending) {
            return Err(nd_err!(
                InvalidArgument,
                "coordinate '{dim}' must be sorted and without nulls"
            ));
        }
        let bound = |value: Option<&ArrayRef>| -> Result<Option<ArrayRef>> {
            value
                .map(|v| Ok(arrow_cast(v.slice(0, 1).as_ref(), values.data_type())?))
                .transpose()
        };
        let (start, end) = (bound(start)?, bound(end)?);
        let position = |value: &Option<ArrayRef>,
                        pred: fn(Ordering) -> bool,
                        default: usize|
         -> Result<usize> {
            Ok(match value {
                None => default,
                Some(v) => {
                    let compare = make_comparator(values.as_ref(), v.as_ref(), options)?;
                    partition_point(n, |i| pred(compare(i, 0)))
                }
            })
        };
        let (first, stop) = if ascending {
            (
                position(&start, |o| o == Ordering::Less, 0)?,
                position(&end, |o| o != Ordering::Greater, n)?,
            )
        } else {
            (
                position(&end, |o| o == Ordering::Greater, 0)?,
                position(&start, |o| o != Ordering::Less, n)?,
            )
        };
        let stop = stop.max(first);
        self.isel(&[(
            dim,
            Selector::Slice(Slice::range(first as isize, stop as isize)),
        )])
    }

    /// Select the position whose coordinate is nearest to `value`, like xarray
    /// `sel(method="nearest")`. The dimension is removed. Ties pick the first.
    pub fn sel_nearest(&self, dim: &str, value: &ArrayRef) -> Result<Self> {
        let coordinate = self.require_coordinate(dim)?;
        let coords = arrow_cast(coordinate.values().as_ref(), &DataType::Float64)?;
        let target = arrow_cast(value.slice(0, 1).as_ref(), &DataType::Float64)?;
        let target = target.as_primitive::<Float64Type>();
        if target.is_null(0) {
            return Err(nd_err!(InvalidArgument, "nearest selection value is null"));
        }
        let t = target.value(0);
        let best = coords
            .as_primitive::<Float64Type>()
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.map(|c| (i, (c - t).abs())))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or_else(|| nd_err!(InvalidArgument, "coordinate '{dim}' has no values"))?;
        self.isel(&[(dim, Selector::Index(best.0 as isize))])
    }

    /// Interpolate every column along `dim` onto `new_coords`, see [`interp`].
    ///
    /// Columns that span `dim` must be numeric and become floating point.
    /// The coordinate becomes `new_coords`. Other columns stay unchanged.
    pub fn interp(&self, dim: &str, new_coords: &NdArrowArray) -> Result<Self> {
        let coordinate = self.require_coordinate(dim)?.clone();
        if new_coords.rank() != 1 || new_coords.dims().as_slice()[0].name() != dim {
            return Err(nd_err!(
                InvalidArgument,
                "new coordinates must be 1-D along '{dim}'"
            ));
        }
        let size = self.target().size_of(dim).unwrap_or(0);
        let mut fields: Vec<Field> = Vec::with_capacity(self.num_columns());
        let mut columns = Vec::with_capacity(self.num_columns());
        for (field, column) in self.schema().fields().iter().zip(self.columns()) {
            let new_column = if field.name() == dim {
                new_coords.clone()
            } else if column.dims().size_of(dim) == Some(size) && size > 1 {
                interp(column, dim, &coordinate, new_coords)?
            } else {
                column.clone()
            };
            fields.push(
                field
                    .as_ref()
                    .clone()
                    .with_data_type(new_column.data_type().clone()),
            );
            columns.push(new_column);
        }
        let target = self.target().iter().map(|d| {
            if d.name() == dim {
                Dimension::new(dim, new_coords.len())
            } else {
                d.clone()
            }
        });
        let schema = Schema::new_with_metadata(fields, self.schema().metadata().clone());
        NdRecordBatch::try_new(Arc::new(schema), columns, Dimensions::try_new(target)?)
    }

    fn require_coordinate(&self, dim: &str) -> Result<&NdArrowArray> {
        self.target().axis(dim)?;
        self.coordinate(dim).ok_or_else(|| {
            nd_err!(
                InvalidArgument,
                "dimension '{dim}' has no coordinate column"
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{Float64Type, Int32Type};

    use super::*;
    use crate::IntoScalar;

    fn batch() -> NdRecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("time", DataType::Int32, true),
            Field::new("lat", DataType::Float64, true),
            Field::new("sst", DataType::Float64, true),
            Field::new("flag", DataType::Int32, true),
        ]));
        let d = |spec: &[(&str, usize)]| Dimensions::try_new(spec.iter().copied()).unwrap();
        let time =
            NdArrowArray::try_new(Arc::new(Int32Array::from(vec![10, 20])), d(&[("time", 2)]))
                .unwrap();
        let lat = NdArrowArray::try_new(
            Arc::new(Float64Array::from(vec![30.0, 20.0, 10.0])),
            d(&[("lat", 3)]),
        )
        .unwrap();
        let sst = NdArrowArray::try_new(
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
            d(&[("time", 2), ("lat", 3)]),
        )
        .unwrap();
        let flag =
            NdArrowArray::try_new(Arc::new(Int32Array::from(vec![7])), d(&[("time", 1)])).unwrap();
        NdRecordBatch::try_new_inferred(schema, vec![time, lat, sst, flag]).unwrap()
    }

    #[test]
    fn materialize_broadcasts_columns() {
        let rb = batch().materialize().unwrap();
        assert_eq!(rb.num_rows(), 6);
        let time: Vec<i32> = rb.column(0).as_primitive::<Int32Type>().values().to_vec();
        assert_eq!(time, vec![10, 10, 10, 20, 20, 20]);
        let lat: Vec<f64> = rb.column(1).as_primitive::<Float64Type>().values().to_vec();
        assert_eq!(lat, vec![30.0, 20.0, 10.0, 30.0, 20.0, 10.0]);
        let flag: Vec<i32> = rb.column(3).as_primitive::<Int32Type>().values().to_vec();
        assert_eq!(flag, vec![7; 6]);
    }

    #[test]
    fn isel_updates_target_and_columns() {
        let b = batch()
            .isel(&[
                ("time", Selector::Index(1)),
                ("lat", Slice::range(0, 2).into()),
            ])
            .unwrap();
        assert_eq!(b.target(), &Dimensions::try_new([("lat", 2)]).unwrap());
        assert!(b.column(0).is_scalar());
        assert!(
            b.column(3).is_scalar(),
            "the broadcast axis of flag is squeezed"
        );
        let sst: Vec<f64> = b
            .column(2)
            .values()
            .as_primitive::<Float64Type>()
            .values()
            .to_vec();
        assert_eq!(sst, vec![3.0, 4.0]);
    }

    #[test]
    fn sel_range_on_descending_coordinate() {
        let b = batch()
            .sel_range(
                "lat",
                Some(&15.0f64.into_scalar()),
                Some(&30.0f64.into_scalar()),
            )
            .unwrap();
        let lat: Vec<f64> = b
            .column(1)
            .values()
            .as_primitive::<Float64Type>()
            .values()
            .to_vec();
        assert_eq!(lat, vec![30.0, 20.0]);
        let t = batch()
            .sel_range("time", Some(&15i32.into_scalar()), None)
            .unwrap();
        assert_eq!(t.target().size_of("time"), Some(1));
    }

    #[test]
    fn sel_nearest_drops_dim() {
        let b = batch().sel_nearest("lat", &22.0f64.into_scalar()).unwrap();
        assert!(!b.target().contains("lat"));
        let sst: Vec<f64> = b
            .column(2)
            .values()
            .as_primitive::<Float64Type>()
            .values()
            .to_vec();
        assert_eq!(sst, vec![1.0, 4.0]);
    }

    #[test]
    fn interp_on_time() {
        let new_time = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![15])),
            Dimensions::try_new([("time", 1)]).unwrap(),
        )
        .unwrap();
        let b = batch().interp("time", &new_time).unwrap();
        let sst: Vec<f64> = b
            .column(2)
            .values()
            .as_primitive::<Float64Type>()
            .values()
            .to_vec();
        assert_eq!(sst, vec![1.5, 2.5, 3.5]);
        assert_eq!(b.schema().field(0).data_type(), &DataType::Int32);
    }
}
