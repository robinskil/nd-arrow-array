//! Per-array attributes (CF/netCDF style metadata such as `units`).
//!
//! Each attribute is a named Arrow value of length 1, so an attribute can be
//! any Arrow type: a string, a number, a list (`valid_range`), and so on. In the
//! encoding, the attributes of a column become the `attrs` struct: one child
//! field per attribute key.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, ListArray, PrimitiveArray, StringArray, StructArray,
    new_empty_array, new_null_array,
};
use arrow::buffer::OffsetBuffer;
use arrow::compute::concat;
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Field, FieldRef, Fields, Float32Type, Float64Type, Int8Type,
    Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};

use crate::error::{Result, nd_err};

/// An ordered set of named, length-1 Arrow values.
#[derive(Debug, Clone, Default)]
pub struct Attributes {
    entries: Vec<(Arc<str>, ArrayRef)>,
}

impl Attributes {
    /// Empty attributes.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of attributes.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when there are no attributes.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The attribute value as a length-1 array.
    pub fn get(&self, name: &str) -> Option<&ArrayRef> {
        self.entries
            .iter()
            .find(|(key, _)| key.as_ref() == name)
            .map(|(_, value)| value)
    }

    /// True when attribute `name` exists.
    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Convenience access for a string attribute.
    pub fn get_str(&self, name: &str) -> Option<&str> {
        let value = self.get(name)?.as_any().downcast_ref::<StringArray>()?;
        value.is_valid(0).then(|| value.value(0))
    }

    /// Convenience access for a primitive attribute of type `T`.
    pub fn get_primitive<T: ArrowPrimitiveType>(&self, name: &str) -> Option<T::Native> {
        let value = self
            .get(name)?
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()?;
        value.is_valid(0).then(|| value.value(0))
    }

    /// Insert or replace an attribute.
    pub fn insert(&mut self, name: impl Into<Arc<str>>, value: impl IntoScalar) {
        self.insert_unchecked(name.into(), value.into_scalar());
    }

    /// Insert or replace an attribute from an Arrow array of length 1.
    pub fn try_insert_array(&mut self, name: impl Into<Arc<str>>, value: ArrayRef) -> Result<()> {
        let name = name.into();
        if value.len() != 1 {
            return Err(nd_err!(
                InvalidAttribute,
                "attribute '{name}' must have length 1, got {}",
                value.len()
            ));
        }
        self.insert_unchecked(name, value);
        Ok(())
    }

    /// Builder form of [`Self::insert`].
    pub fn with(mut self, name: impl Into<Arc<str>>, value: impl IntoScalar) -> Self {
        self.insert(name, value);
        self
    }

    /// Remove attribute `name` and return its value.
    pub fn remove(&mut self, name: &str) -> Option<ArrayRef> {
        let index = self
            .entries
            .iter()
            .position(|(key, _)| key.as_ref() == name)?;
        Some(self.entries.remove(index).1)
    }

    /// The attributes in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ArrayRef)> {
        self.entries
            .iter()
            .map(|(key, value)| (key.as_ref(), value))
    }

    /// The `attrs` struct fields for these attributes. All fields are nullable.
    pub fn fields(&self) -> Fields {
        self.entries
            .iter()
            .map(|(key, value)| attribute_field(key, value.data_type()))
            .collect()
    }

    /// Read row `row` of an `attrs` struct. A null child value is an absent attribute.
    pub fn from_struct_row(attrs: &StructArray, row: usize) -> Self {
        let entries = attrs
            .fields()
            .iter()
            .zip(attrs.columns())
            .filter(|(_, column)| column.is_valid(row))
            .map(|(field, column)| (Arc::from(field.name().as_str()), column.slice(row, 1)))
            .collect();
        Self { entries }
    }

    /// The struct fields for the union of the keys in `all`. One key with two
    /// types is an error.
    pub fn union_fields<'a>(all: impl IntoIterator<Item = &'a Attributes>) -> Result<Fields> {
        let mut fields: Vec<FieldRef> = Vec::new();
        for attrs in all {
            for (name, value) in attrs.iter() {
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
        }
        Ok(fields.into())
    }

    /// Encode one struct row per entry of `rows`, with `fields` from
    /// [`Self::union_fields`]. A missing key or a `None` row gives nulls.
    pub fn encode_rows(rows: &[Option<&Attributes>], fields: &Fields) -> Result<StructArray> {
        let columns = fields
            .iter()
            .map(|field| {
                let null = new_null_array(field.data_type(), 1);
                let parts: Vec<&dyn Array> = rows
                    .iter()
                    .map(|row| {
                        row.and_then(|a| a.get(field.name()))
                            .map_or(null.as_ref(), |v| v.as_ref())
                    })
                    .collect();
                Ok(if parts.is_empty() {
                    new_empty_array(field.data_type())
                } else {
                    concat(&parts)?
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if fields.is_empty() {
            return Ok(StructArray::new_empty_fields(rows.len(), None));
        }
        Ok(StructArray::try_new(fields.clone(), columns, None)?)
    }

    fn insert_unchecked(&mut self, name: Arc<str>, value: ArrayRef) {
        match self.entries.iter_mut().find(|(key, _)| *key == name) {
            Some(entry) => entry.1 = value,
            None => self.entries.push((name, value)),
        }
    }
}

impl PartialEq for Attributes {
    /// Equal when both hold the same keys with equal values, in any order.
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .all(|(key, value)| other.get(key).is_some_and(|v| v == value))
    }
}

pub(crate) fn attribute_field(name: &str, data_type: &DataType) -> FieldRef {
    Arc::new(Field::new(name, data_type.clone(), true))
}

/// Conversion of a Rust value into a length-1 Arrow array (a scalar).
pub trait IntoScalar {
    /// The value as an Arrow array of length 1.
    fn into_scalar(self) -> ArrayRef;
}

impl IntoScalar for &str {
    fn into_scalar(self) -> ArrayRef {
        Arc::new(StringArray::from(vec![self]))
    }
}

impl IntoScalar for String {
    fn into_scalar(self) -> ArrayRef {
        self.as_str().into_scalar()
    }
}

impl IntoScalar for bool {
    fn into_scalar(self) -> ArrayRef {
        Arc::new(BooleanArray::from(vec![self]))
    }
}

impl IntoScalar for Vec<&str> {
    fn into_scalar(self) -> ArrayRef {
        list_of(Arc::new(StringArray::from(self)))
    }
}

impl IntoScalar for Vec<String> {
    fn into_scalar(self) -> ArrayRef {
        list_of(Arc::new(StringArray::from(self)))
    }
}

macro_rules! primitive_attribute {
    ($($native:ty => $arrow:ty),* $(,)?) => {$(
        impl IntoScalar for $native {
            fn into_scalar(self) -> ArrayRef {
                Arc::new(PrimitiveArray::<$arrow>::from(vec![self]))
            }
        }

        impl IntoScalar for Vec<$native> {
            fn into_scalar(self) -> ArrayRef {
                list_of(Arc::new(PrimitiveArray::<$arrow>::from(self)))
            }
        }
    )*};
}

primitive_attribute!(
    i8 => Int8Type,
    i16 => Int16Type,
    i32 => Int32Type,
    i64 => Int64Type,
    u8 => UInt8Type,
    u16 => UInt16Type,
    u32 => UInt32Type,
    u64 => UInt64Type,
    f32 => Float32Type,
    f64 => Float64Type,
);

fn list_of(values: ArrayRef) -> ArrayRef {
    let field = Arc::new(Field::new("item", values.data_type().clone(), true));
    Arc::new(ListArray::new(
        field,
        OffsetBuffer::from_lengths([values.len()]),
        values,
        None,
    ))
}

#[cfg(test)]
mod tests {
    use arrow::array::AsArray;

    use super::*;

    #[test]
    fn insert_replaces_and_keeps_order() {
        let mut attrs = Attributes::new()
            .with("units", "degC")
            .with("scale_factor", 0.5f64)
            .with("valid_range", vec![-5.0f32, 40.0]);
        attrs.insert("units", "K");

        assert_eq!(attrs.len(), 3);
        assert_eq!(attrs.get_str("units"), Some("K"));
        assert_eq!(
            attrs.get_primitive::<Float64Type>("scale_factor"),
            Some(0.5)
        );
        let names: Vec<&str> = attrs.iter().map(|(k, _)| k).collect();
        assert_eq!(names, vec!["units", "scale_factor", "valid_range"]);

        let range = attrs.get("valid_range").unwrap().as_list::<i32>().value(0);
        assert_eq!(range.as_primitive::<Float32Type>().values(), &[-5.0, 40.0]);
    }

    #[test]
    fn array_attribute_must_have_length_one() {
        let mut attrs = Attributes::new();
        let two: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        assert!(attrs.try_insert_array("bad", two).is_err());
    }

    #[test]
    fn equality_ignores_order() {
        let a = Attributes::new().with("a", 1i32).with("b", "x");
        let b = Attributes::new().with("b", "x").with("a", 1i32);
        assert_eq!(a, b);
        assert_ne!(a, Attributes::new().with("a", 2i32).with("b", "x"));
    }
}
