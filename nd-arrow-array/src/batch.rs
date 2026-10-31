//! Grid-shaped record batches: columns on subsets of one shared target grid.

use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};

use crate::array::NdArrowArray;
use crate::column::NdColumn;
use crate::dimensions::{Dimension, Dimensions};
use crate::error::{Result, nd_err};
use crate::extension::{is_nd_field, logical_schema, with_base_metadata};

/// A record batch whose columns are [`NdArrowArray`]s over a shared target grid.
///
/// Each column can use a subset of the target dimensions: a scalar attribute,
/// a 1-D coordinate, a full-rank data variable. Columns stay un-broadcast.
/// The schema is the logical schema: one field per column, typed by its values.
#[derive(Debug, Clone)]
pub struct NdRecordBatch {
    schema: SchemaRef,
    columns: Vec<NdArrowArray>,
    target: Dimensions,
}

impl NdRecordBatch {
    pub fn try_new(
        schema: SchemaRef,
        columns: Vec<NdArrowArray>,
        target: Dimensions,
    ) -> Result<Self> {
        if schema.fields().len() != columns.len() {
            return Err(nd_err!(
                SchemaMismatch,
                "nd batch has {} columns but the schema declares {} fields",
                columns.len(),
                schema.fields().len()
            ));
        }
        for (field, column) in schema.fields().iter().zip(&columns) {
            if field.data_type() != column.data_type() {
                return Err(nd_err!(
                    SchemaMismatch,
                    "column '{}' has type {} but the schema declares {}",
                    field.name(),
                    column.data_type(),
                    field.data_type()
                ));
            }
            column.dims().check_broadcast_to(&target)?;
        }
        Ok(Self {
            schema,
            columns,
            target,
        })
    }

    /// A batch with the target grid inferred by [`infer_target`].
    pub fn try_new_inferred(schema: SchemaRef, columns: Vec<NdArrowArray>) -> Result<Self> {
        let target = infer_target(&columns)?;
        Self::try_new(schema, columns, target)
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub fn columns(&self) -> &[NdArrowArray] {
        &self.columns
    }

    pub fn column(&self, index: usize) -> &NdArrowArray {
        &self.columns[index]
    }

    pub fn column_by_name(&self, name: &str) -> Option<&NdArrowArray> {
        let index = self.schema.index_of(name).ok()?;
        Some(&self.columns[index])
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn target(&self) -> &Dimensions {
        &self.target
    }

    /// Rows after a broadcast onto the full target grid.
    pub fn num_rows(&self) -> usize {
        self.target.num_elements()
    }

    /// Encode as a single-row [`RecordBatch`] of nd columns.
    pub fn to_record_batch(&self) -> Result<RecordBatch> {
        encode_batches(std::slice::from_ref(self))
    }

    /// Decode every row of an nd-encoded [`RecordBatch`]: one nd batch per row.
    pub fn try_from_record_batch(batch: &RecordBatch) -> Result<Vec<Self>> {
        let decoder = Decoder::try_new(batch)?;
        (0..batch.num_rows()).map(|row| decoder.row(row)).collect()
    }

    /// Decode row `row` of an nd-encoded [`RecordBatch`].
    pub fn try_from_record_batch_row(batch: &RecordBatch, row: usize) -> Result<Self> {
        if row >= batch.num_rows() {
            return Err(nd_err!(
                InvalidEncoding,
                "row {row} is out of bounds for a batch of {} rows",
                batch.num_rows()
            ));
        }
        Decoder::try_new(batch)?.row(row)
    }
}

/// Encode several nd batches with one shared logical schema, one row each.
///
/// The `attrs` struct of each column holds the union of the attribute keys of all batches.
pub fn encode_batches(batches: &[NdRecordBatch]) -> Result<RecordBatch> {
    let Some(first) = batches.first() else {
        return Err(nd_err!(SchemaMismatch, "no nd batches to encode"));
    };
    let schema = first.schema();
    if let Some(other) = batches
        .iter()
        .find(|b| b.schema().fields() != schema.fields())
    {
        return Err(nd_err!(
            SchemaMismatch,
            "nd batches disagree on the schema: {} vs {}",
            schema,
            other.schema()
        ));
    }

    let mut fields: Vec<Field> = Vec::with_capacity(schema.fields().len());
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for (index, logical) in schema.fields().iter().enumerate() {
        let column = NdColumn::try_from_arrays(batches.iter().map(|b| b.column(index)))?;
        fields.push(with_base_metadata(
            column.to_field(logical.name()),
            logical.metadata(),
        ));
        columns.push(column.into_array_ref());
    }

    let schema = Schema::new_with_metadata(fields, schema.metadata().clone());
    let options = RecordBatchOptions::new().with_row_count(Some(batches.len()));
    Ok(RecordBatch::try_new_with_options(
        Arc::new(schema),
        columns,
        &options,
    )?)
}

struct Decoder {
    schema: SchemaRef,
    columns: Vec<NdColumn>,
}

impl Decoder {
    fn try_new(batch: &RecordBatch) -> Result<Self> {
        if let Some(field) = batch.schema().fields().iter().find(|f| !is_nd_field(f)) {
            return Err(nd_err!(
                InvalidEncoding,
                "column '{}' is not an nd column",
                field.name()
            ));
        }
        let columns = batch
            .columns()
            .iter()
            .map(|c| NdColumn::try_from_array(c.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        let schema = Arc::new(logical_schema(batch.schema_ref())?);
        Ok(Self { schema, columns })
    }

    fn row(&self, row: usize) -> Result<NdRecordBatch> {
        let columns: Vec<NdArrowArray> = self.columns.iter().map(|c| c.value(row)).collect();
        NdRecordBatch::try_new_inferred(self.schema.clone(), columns)
    }
}

/// Infer the target grid from columns. The highest-rank column sets the axis
/// order. Axes that only lower-rank columns carry are appended. A size-1 axis
/// takes a larger size from another column.
pub fn infer_target(columns: &[NdArrowArray]) -> Result<Dimensions> {
    let mut order: Vec<Dimension> = Vec::new();
    if let Some(widest) = columns.iter().max_by_key(|c| c.rank()) {
        order.extend(widest.dims().iter().cloned());
    }
    for dim in columns.iter().flat_map(|c| c.dims().iter()) {
        match order.iter_mut().find(|d| d.name() == dim.name()) {
            Some(existing) if existing.size() == 1 => *existing = dim.clone(),
            Some(_) => {}
            None => order.push(dim.clone()),
        }
    }
    Dimensions::try_new(order)
}

#[cfg(test)]
mod tests {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::DataType;

    use super::*;
    use crate::attributes::Attributes;

    fn dims(spec: &[(&str, usize)]) -> Dimensions {
        Dimensions::try_new(spec.iter().copied()).unwrap()
    }

    fn grid_batch() -> NdRecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("time", DataType::Int32, true),
            Field::new("lat", DataType::Int32, true),
            Field::new("sst", DataType::Float64, true),
        ]));
        let time =
            NdArrowArray::try_new(Arc::new(Int32Array::from(vec![7, 8])), dims(&[("time", 2)]))
                .unwrap();
        let lat = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![10, 20, 30])),
            dims(&[("lat", 3)]),
        )
        .unwrap()
        .with_attributes(Attributes::new().with("units", "degrees_north"));
        let sst = NdArrowArray::try_new(
            Arc::new(Float64Array::from(vec![0.0, 0.1, 0.2, 1.0, 1.1, 1.2])),
            dims(&[("time", 2), ("lat", 3)]),
        )
        .unwrap()
        .with_attributes(
            Attributes::new()
                .with("units", "K")
                .with("_FillValue", -999.0),
        );
        NdRecordBatch::try_new(
            schema,
            vec![time, lat, sst],
            dims(&[("time", 2), ("lat", 3)]),
        )
        .unwrap()
    }

    #[test]
    fn record_batch_round_trip() {
        let nd = grid_batch();
        let encoded = nd.to_record_batch().unwrap();
        assert_eq!(encoded.num_rows(), 1);
        assert!(encoded.schema().fields().iter().all(|f| is_nd_field(f)));

        let decoded = NdRecordBatch::try_from_record_batch_row(&encoded, 0).unwrap();
        assert_eq!(decoded.target(), nd.target());
        assert_eq!(decoded.schema().fields(), nd.schema().fields());
        assert_eq!(decoded.columns(), nd.columns());
    }

    #[test]
    fn every_row_of_a_concatenated_batch_decodes() {
        let other = NdRecordBatch::try_new_inferred(
            grid_batch().schema().clone(),
            vec![
                NdArrowArray::try_new(Arc::new(Int32Array::from(vec![1])), dims(&[("time", 1)]))
                    .unwrap(),
                NdArrowArray::try_new(Arc::new(Int32Array::from(vec![5, 6])), dims(&[("lat", 2)]))
                    .unwrap(),
                NdArrowArray::try_new(
                    Arc::new(Float64Array::from(vec![3.0, 4.0])),
                    dims(&[("time", 1), ("lat", 2)]),
                )
                .unwrap(),
            ],
        )
        .unwrap();

        let encoded = encode_batches(&[grid_batch(), other.clone()]).unwrap();
        let decoded = NdRecordBatch::try_from_record_batch(&encoded).unwrap();
        let rows: Vec<usize> = decoded.iter().map(NdRecordBatch::num_rows).collect();
        assert_eq!(rows, vec![6, 2]);
        assert_eq!(decoded[1].columns(), other.columns());
    }

    #[test]
    fn target_appends_axes_of_narrow_columns() {
        let sst = NdArrowArray::try_new(
            Arc::new(Float64Array::from(vec![0.0; 6])),
            dims(&[("time", 2), ("lat", 3)]),
        )
        .unwrap();
        let depth = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![5, 10])),
            dims(&[("depth", 2)]),
        )
        .unwrap();
        let target = infer_target(&[sst, depth]).unwrap();
        assert_eq!(target, dims(&[("time", 2), ("lat", 3), ("depth", 2)]));
    }

    #[test]
    fn schema_mismatch_rejected() {
        let schema = Arc::new(Schema::new(vec![Field::new("t", DataType::Float64, true)]));
        let t = NdArrowArray::try_new(Arc::new(Int32Array::from(vec![7, 8])), dims(&[("t", 2)]))
            .unwrap();
        assert!(NdRecordBatch::try_new(schema, vec![t], dims(&[("t", 2)])).is_err());
    }

    #[test]
    fn incompatible_column_dims_rejected() {
        let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Int32, true)]));
        let d = NdArrowArray::try_new(Arc::new(Int32Array::from(vec![1, 2])), dims(&[("d", 2)]))
            .unwrap();
        assert!(NdRecordBatch::try_new(schema, vec![d], dims(&[("t", 2)])).is_err());
    }
}
