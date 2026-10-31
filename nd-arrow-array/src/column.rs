//! An Arrow column of nd arrays: the storage array of [`NdArrayType`].

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, ListArray, StringArray, StringBuilder, StructArray, UInt32Array,
    UInt32Builder, new_empty_array, new_null_array,
};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::compute::concat;
use arrow::datatypes::{DataType, Field, FieldRef, Fields};

use crate::array::NdArrowArray;
use crate::attributes::{Attributes, attribute_field};
use crate::dimensions::{Dimension, Dimensions};
use crate::error::{NdError, Result, nd_err};
use crate::extension::NdArrayType;

/// A validated nd column. Each row holds one [`NdArrowArray`] or a null.
///
/// Construction checks every row once, so [`Self::value`] is infallible and
/// slices the storage without a copy.
#[derive(Debug, Clone)]
pub struct NdColumn {
    storage: StructArray,
    extension: NdArrayType,
}

impl NdColumn {
    /// Wrap and validate a storage array.
    pub fn try_new(storage: StructArray) -> Result<Self> {
        let extension = NdArrayType::try_from_storage_type(storage.data_type())?;
        validate_rows(&storage)?;
        Ok(Self { storage, extension })
    }

    /// Wrap and validate a storage array given as an [`ArrayRef`].
    pub fn try_from_array(array: &dyn Array) -> Result<Self> {
        let storage = array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                nd_err!(
                    InvalidEncoding,
                    "nd column must be a Struct array, got {}",
                    array.data_type()
                )
            })?;
        Self::try_new(storage.clone())
    }

    /// Encode `arrays` as one row each. `arrays` must not be empty.
    pub fn try_from_arrays<'a>(arrays: impl IntoIterator<Item = &'a NdArrowArray>) -> Result<Self> {
        let mut builder = NdColumnBuilder::new();
        for array in arrays {
            builder.append(array)?;
        }
        builder.finish()
    }

    /// A column of `len` null rows.
    pub fn new_null(extension: NdArrayType, len: usize) -> Self {
        let storage = new_null_array(&extension.storage_type(), len)
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("storage type is a struct")
            .clone();
        Self { storage, extension }
    }

    pub fn extension_type(&self) -> &NdArrayType {
        &self.extension
    }

    pub fn value_type(&self) -> &DataType {
        self.extension.value_type()
    }

    /// A field named `name` with the storage type and the extension tag.
    pub fn to_field(&self, name: impl Into<String>) -> Field {
        self.extension.to_field(name, true)
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    pub fn is_null(&self, row: usize) -> bool {
        self.storage.is_null(row)
    }

    pub fn null_count(&self) -> usize {
        self.storage.null_count()
    }

    /// The nd array at `row`, or `None` for a null row.
    pub fn get(&self, row: usize) -> Option<NdArrowArray> {
        if self.storage.is_null(row) {
            return None;
        }
        let values = self.values_list().value(row);
        let dims = self.dimensions_at(row);
        let attrs = Attributes::from_struct_row(self.attrs_struct(), row);
        Some(NdArrowArray::new_unchecked(values, dims, attrs))
    }

    /// The nd array at `row`. A null row gives a rank-0 null scalar,
    /// which broadcasts to an all-null array.
    pub fn value(&self, row: usize) -> NdArrowArray {
        self.get(row)
            .unwrap_or_else(|| NdArrowArray::new_null_scalar(self.value_type()))
    }

    /// The dimensions at `row`. A null row gives scalar dimensions.
    pub fn dimensions_at(&self, row: usize) -> Dimensions {
        if self.storage.is_null(row) {
            return Dimensions::scalar();
        }
        let (names, sizes) = self.dims_at(row);
        let dims: Vec<Dimension> = names
            .iter()
            .zip(sizes.values())
            .map(|(name, &size)| Dimension::new(name.expect("validated"), size as usize))
            .collect();
        Dimensions::try_new(dims).expect("validated")
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = Option<NdArrowArray>> + '_ {
        (0..self.len()).map(|row| self.get(row))
    }

    pub fn slice(&self, offset: usize, len: usize) -> Self {
        Self {
            storage: self.storage.slice(offset, len),
            extension: self.extension.clone(),
        }
    }

    /// The flat values of all rows (the child of the `values` list).
    pub fn values_list(&self) -> &ListArray {
        list_column(&self.storage, 0)
    }

    pub fn attrs_struct(&self) -> &StructArray {
        self.storage
            .column(3)
            .as_any()
            .downcast_ref()
            .expect("validated")
    }

    pub fn storage(&self) -> &StructArray {
        &self.storage
    }

    pub fn into_storage(self) -> StructArray {
        self.storage
    }

    pub fn into_array_ref(self) -> ArrayRef {
        Arc::new(self.storage)
    }

    fn dims_at(&self, row: usize) -> (StringArray, UInt32Array) {
        let names = list_column(&self.storage, 1).value(row);
        let sizes = list_column(&self.storage, 2).value(row);
        (
            names
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .clone(),
            sizes
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap()
                .clone(),
        )
    }
}

impl From<NdColumn> for ArrayRef {
    fn from(column: NdColumn) -> Self {
        column.into_array_ref()
    }
}

impl TryFrom<StructArray> for NdColumn {
    type Error = NdError;

    fn try_from(storage: StructArray) -> Result<Self> {
        Self::try_new(storage)
    }
}

fn list_column(storage: &StructArray, index: usize) -> &ListArray {
    storage
        .column(index)
        .as_any()
        .downcast_ref()
        .expect("validated")
}

/// Check each non-null row: list validity, dimension names and sizes, value count.
fn validate_rows(storage: &StructArray) -> Result<()> {
    let values = list_column(storage, 0);
    let names = list_column(storage, 1);
    let sizes = list_column(storage, 2);
    let name_values = names
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| nd_err!(InvalidEncoding, "'dim_names' items must be Utf8"))?;
    let size_values = sizes
        .values()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| nd_err!(InvalidEncoding, "'dim_sizes' items must be UInt32"))?;

    for row in 0..storage.len() {
        if storage.is_null(row) {
            continue;
        }
        if values.is_null(row) || names.is_null(row) || sizes.is_null(row) {
            return Err(nd_err!(
                InvalidEncoding,
                "row {row}: a non-null nd row has a null list"
            ));
        }
        let name_range = list_range(names, row);
        let size_range = list_range(sizes, row);
        if name_range.len() != size_range.len() {
            return Err(nd_err!(
                InvalidEncoding,
                "row {row}: {} dimension names but {} dimension sizes",
                name_range.len(),
                size_range.len()
            ));
        }
        let mut dims = Vec::with_capacity(name_range.len());
        for (n, s) in name_range.zip(size_range) {
            if name_values.is_null(n) || size_values.is_null(s) {
                return Err(nd_err!(
                    InvalidEncoding,
                    "row {row}: null dimension name or size"
                ));
            }
            dims.push(Dimension::new(
                name_values.value(n),
                size_values.value(s) as usize,
            ));
        }
        let dims = Dimensions::try_new(dims)?;
        let expected = dims.checked_num_elements();
        let actual = values.value_length(row) as usize;
        if expected != Some(actual) {
            return Err(nd_err!(
                ShapeMismatch,
                "row {row}: {actual} values do not match dimensions {dims}"
            ));
        }
    }
    Ok(())
}

fn list_range(list: &ListArray, row: usize) -> std::ops::Range<usize> {
    let offsets = list.value_offsets();
    offsets[row] as usize..offsets[row + 1] as usize
}

/// Collects nd arrays and encodes them into an [`NdColumn`].
///
/// The `attrs` struct of the column holds the union of all attribute keys.
/// A row without a key gets a null for it. The same key with two different
/// types is an error.
#[derive(Debug, Default)]
pub struct NdColumnBuilder {
    value_field: Option<FieldRef>,
    rows: Vec<Option<NdArrowArray>>,
}

impl NdColumnBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// A builder with a known value type, so that all-null and empty columns work.
    pub fn with_value_type(value_type: DataType) -> Self {
        Self::with_value_field(Arc::new(Field::new(
            crate::extension::ITEM_FIELD,
            value_type,
            true,
        )))
    }

    /// A builder with a custom item field for the values list.
    pub fn with_value_field(value_field: FieldRef) -> Self {
        Self {
            value_field: Some(value_field),
            rows: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn append(&mut self, array: &NdArrowArray) -> Result<()> {
        match &self.value_field {
            Some(field) if field.data_type() != array.data_type() => {
                return Err(nd_err!(
                    SchemaMismatch,
                    "nd array has value type {} but the column has {}",
                    array.data_type(),
                    field.data_type()
                ));
            }
            Some(_) => {}
            None => {
                self.value_field = Some(Arc::new(Field::new(
                    crate::extension::ITEM_FIELD,
                    array.data_type().clone(),
                    true,
                )));
            }
        }
        self.rows.push(Some(array.clone()));
        Ok(())
    }

    pub fn append_null(&mut self) {
        self.rows.push(None);
    }

    pub fn append_option(&mut self, array: Option<&NdArrowArray>) -> Result<()> {
        match array {
            Some(array) => self.append(array),
            None => {
                self.append_null();
                Ok(())
            }
        }
    }

    pub fn finish(self) -> Result<NdColumn> {
        let value_field = self.value_field.ok_or_else(|| {
            nd_err!(
                SchemaMismatch,
                "cannot infer the value type of an nd column without non-null rows"
            )
        })?;
        let rows = self.rows;
        let present: Vec<&NdArrowArray> = rows.iter().flatten().collect();

        let attr_fields = union_attribute_fields(&present)?;
        let extension = NdArrayType::with_value_field(value_field.clone(), attr_fields.clone());

        let values = concat_or_empty(
            &present
                .iter()
                .map(|a| a.values().as_ref())
                .collect::<Vec<_>>(),
            value_field.data_type(),
        )?;
        let values_list = ListArray::try_new(
            value_field,
            offsets_from_lengths(rows.iter().map(|r| r.as_ref().map_or(0, NdArrowArray::len)))?,
            values,
            None,
        )?;

        let rank_offsets = offsets_from_lengths(
            rows.iter()
                .map(|r| r.as_ref().map_or(0, NdArrowArray::rank)),
        )?;
        let total_rank = *rank_offsets.last().unwrap() as usize;
        let mut names = StringBuilder::with_capacity(total_rank, total_rank * 8);
        let mut sizes = UInt32Builder::with_capacity(total_rank);
        for dim in present.iter().flat_map(|a| a.dims().iter()) {
            let size = u32::try_from(dim.size()).map_err(|_| {
                nd_err!(
                    InvalidDimensions,
                    "dimension '{}' of size {} does not fit in UInt32",
                    dim.name(),
                    dim.size()
                )
            })?;
            names.append_value(dim.name());
            sizes.append_value(size);
        }
        let storage_fields = extension.storage_fields();
        let item_field = |index: usize| match storage_fields[index].data_type() {
            DataType::List(item) => item.clone(),
            _ => unreachable!(),
        };
        let names_list = ListArray::try_new(
            item_field(1),
            rank_offsets.clone(),
            Arc::new(names.finish()),
            None,
        )?;
        let sizes_list =
            ListArray::try_new(item_field(2), rank_offsets, Arc::new(sizes.finish()), None)?;

        let attrs = encode_attributes(&rows, &attr_fields)?;

        let nulls = rows
            .iter()
            .any(Option::is_none)
            .then(|| NullBuffer::from_iter(rows.iter().map(Option::is_some)));

        let storage = StructArray::try_new(
            storage_fields,
            vec![
                Arc::new(values_list),
                Arc::new(names_list),
                Arc::new(sizes_list),
                Arc::new(attrs),
            ],
            nulls,
        )?;
        Ok(NdColumn { storage, extension })
    }
}

fn offsets_from_lengths(lengths: impl Iterator<Item = usize>) -> Result<OffsetBuffer<i32>> {
    let lengths: Vec<usize> = lengths.collect();
    let total: usize = lengths.iter().sum();
    if i32::try_from(total).is_err() {
        return Err(nd_err!(
            InvalidEncoding,
            "{total} list items exceed the i32 offset range"
        ));
    }
    Ok(OffsetBuffer::from_lengths(lengths))
}

fn concat_or_empty(arrays: &[&dyn Array], data_type: &DataType) -> Result<ArrayRef> {
    Ok(match arrays {
        [] => new_empty_array(data_type),
        [single] => single.slice(0, single.len()),
        many => concat(many)?,
    })
}

fn union_attribute_fields(arrays: &[&NdArrowArray]) -> Result<Fields> {
    let mut fields: Vec<FieldRef> = Vec::new();
    for (name, value) in arrays.iter().flat_map(|a| a.attributes().iter()) {
        match fields.iter().find(|f| f.name() == name) {
            Some(existing) if existing.data_type() != value.data_type() => {
                return Err(nd_err!(
                    InvalidAttribute,
                    "attribute '{name}' has types {} and {}",
                    existing.data_type(),
                    value.data_type()
                ));
            }
            Some(_) => {}
            None => fields.push(attribute_field(name, value.data_type())),
        }
    }
    Ok(fields.into())
}

fn encode_attributes(rows: &[Option<NdArrowArray>], fields: &Fields) -> Result<StructArray> {
    if fields.is_empty() {
        return Ok(StructArray::new_empty_fields(rows.len(), None));
    }
    let columns = fields
        .iter()
        .map(|field| {
            let null = new_null_array(field.data_type(), 1);
            let parts: Vec<&dyn Array> = rows
                .iter()
                .map(|row| {
                    row.as_ref()
                        .and_then(|a| a.attributes().get(field.name()))
                        .map_or(null.as_ref(), |v| v.as_ref())
                })
                .collect();
            concat_or_empty(&parts, field.data_type())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StructArray::try_new(fields.clone(), columns, None)?)
}

#[cfg(test)]
mod tests {
    use arrow::array::{AsArray, Float64Array, Int32Array};
    use arrow::datatypes::{Float64Type, Int32Type};

    use super::*;

    fn nd(values: Vec<i32>, dims: &[(&str, usize)]) -> NdArrowArray {
        NdArrowArray::try_new(
            Arc::new(Int32Array::from(values)),
            Dimensions::try_new(dims.iter().copied()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn rows_round_trip_with_attribute_union() {
        let a = nd(vec![1, 2, 3, 4, 5, 6], &[("time", 2), ("lat", 3)])
            .with_attributes(Attributes::new().with("units", "K"));
        let b = nd(vec![7, 8], &[("lat", 2)])
            .with_attributes(Attributes::new().with("scale_factor", 2.0f64));
        let c = nd(vec![9], &[]);

        let column = NdColumn::try_from_arrays([&a, &b, &c]).unwrap();
        assert_eq!(column.len(), 3);
        assert_eq!(
            column.extension_type().attribute_fields().len(),
            2,
            "units and scale_factor"
        );

        let decoded: Vec<NdArrowArray> = column.iter().map(Option::unwrap).collect();
        assert_eq!(decoded, vec![a, b, c]);
        assert_eq!(
            decoded[1]
                .attributes()
                .get_primitive::<Float64Type>("scale_factor"),
            Some(2.0)
        );
        assert!(!decoded[1].attributes().contains("units"));
    }

    #[test]
    fn null_rows_decode_to_null_scalars() {
        let mut builder = NdColumnBuilder::new();
        builder.append_null();
        builder.append(&nd(vec![1, 2], &[("x", 2)])).unwrap();
        let column = builder.finish().unwrap();

        assert_eq!(column.null_count(), 1);
        assert!(column.get(0).is_none());
        let null = column.value(0);
        assert!(null.is_scalar());
        assert_eq!(null.values().null_count(), 1);
        assert_eq!(
            column
                .value(1)
                .values()
                .as_primitive::<Int32Type>()
                .values(),
            &[1, 2]
        );
    }

    #[test]
    fn empty_column_needs_a_value_type() {
        assert!(NdColumnBuilder::new().finish().is_err());
        let column = NdColumnBuilder::with_value_type(DataType::Float64)
            .finish()
            .unwrap();
        assert!(column.is_empty());
        assert_eq!(column.value_type(), &DataType::Float64);
    }

    #[test]
    fn value_type_mismatch_rejected() {
        let mut builder = NdColumnBuilder::new();
        builder.append(&nd(vec![1], &[])).unwrap();
        let float = NdArrowArray::try_new_scalar(Arc::new(Float64Array::from(vec![1.0]))).unwrap();
        assert!(builder.append(&float).is_err());
    }

    #[test]
    fn conflicting_attribute_types_rejected() {
        let a = nd(vec![1], &[]).with_attributes(Attributes::new().with("units", "K"));
        let b = nd(vec![1], &[]).with_attributes(Attributes::new().with("units", 1i32));
        assert!(NdColumn::try_from_arrays([&a, &b]).is_err());
    }

    #[test]
    fn slice_keeps_rows() {
        let arrays = [nd(vec![1], &[]), nd(vec![2, 3], &[("x", 2)])];
        let column = NdColumn::try_from_arrays(&arrays).unwrap().slice(1, 1);
        assert_eq!(column.len(), 1);
        assert_eq!(column.value(0), arrays[1]);
    }

    #[test]
    fn invalid_storage_rejected() {
        let column = NdColumn::try_from_arrays([&nd(vec![1, 2], &[("x", 2)])]).unwrap();
        let (fields, mut children, nulls) = column.into_storage().into_parts();
        children[2] = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::UInt32, false)),
            OffsetBuffer::from_lengths([1]),
            Arc::new(UInt32Array::from(vec![3])),
            None,
        ));
        let broken = StructArray::new(fields, children, nulls);
        let err = NdColumn::try_new(broken).unwrap_err();
        assert!(err.to_string().contains("do not match"), "{err}");
    }
}
