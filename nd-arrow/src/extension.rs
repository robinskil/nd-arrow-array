//! The `nd_arrow.array` Arrow extension type.
//!
//! An nd column is an Arrow `Struct` with this storage layout. Each row holds
//! one complete N-dimensional array:
//!
//! ```text
//! Struct {
//!   values:    List<T>,        // flat C-order values
//!   dim_names: List<Utf8>,     // one name per axis
//!   dim_sizes: List<UInt32>,   // one size per axis
//!   attrs:     Struct { .. },  // optional: one nullable child per attribute key
//! }
//! ```
//!
//! The field carries `ARROW:extension:name = nd_arrow.array` and
//! `ARROW:extension:metadata = {"version":1}`, so the intent survives IPC,
//! Parquet and any engine that keeps field metadata.
//!
//! # Rules
//!
//! `FORMAT.md` in the repository is the full, language-neutral specification.
//! This section is a summary.
//!
//! - The field order is fixed: `values`, `dim_names`, `dim_sizes`, then `attrs`
//!   if present.
//! - A column without attributes leaves out the `attrs` field, so no storage
//!   format needs a struct with zero fields. Readers also accept an `attrs`
//!   struct with zero fields and treat it as no attributes.
//! - `dim_names` and `dim_sizes` have the same length. This length is the rank.
//! - The length of `values` is the product of `dim_sizes`.
//! - Rank 0 (empty dimension lists) is a scalar with one value.
//! - Dimension names are not empty and are unique in a row.
//! - Validity of elements is in the null buffer of `values`. There is no fill value.
//! - An attribute can be any Arrow type. The `attrs` struct holds the union of
//!   the keys of all rows. A null child means that the row does not have that key.
//! - A null row decodes to a rank-0 null scalar.
//!
//! # Example
//!
//! A `(time=2, lat=3)` array of `Float64` with `units = "K"`:
//!
//! ```text
//! values:    [0.0, 0.1, 0.2, 1.0, 1.1, 1.2]
//! dim_names: ["time", "lat"]
//! dim_sizes: [2, 3]
//! attrs:     { units: "K" }
//! ```
//!
//! # Global attributes
//!
//! An encoded record batch (see [`encode_batches`](crate::encode_batches))
//! stores the global attributes of each nd batch in one extra column,
//! [`GLOBAL_ATTRIBUTES_FIELD`], tagged with [`GLOBAL_ATTRIBUTES_EXTENSION`].
//! It has one row per nd batch and the union of keys, with the same null rule
//! as `attrs`. The column is left out when no batch has global attributes.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef, Fields, Schema};
use arrow::error::ArrowError;
use arrow_schema::extension::{
    EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY, ExtensionType,
};
use serde::{Deserialize, Serialize};

use crate::error::{Result, nd_err};

/// Storage field with the flat values.
pub const VALUES_FIELD: &str = "values";
/// Storage field with the dimension names.
pub const DIM_NAMES_FIELD: &str = "dim_names";
/// Storage field with the dimension sizes.
pub const DIM_SIZES_FIELD: &str = "dim_sizes";
/// Optional storage field with the attributes.
pub const ATTRS_FIELD: &str = "attrs";

/// Name of the list item field in `values`, `dim_names` and `dim_sizes`.
pub const ITEM_FIELD: &str = "item";

/// The encoding version this crate writes and reads.
pub const ENCODING_VERSION: u32 = 1;

/// Extension metadata, serialized as JSON in `ARROW:extension:metadata`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NdArrayMetadata {
    /// The encoding version.
    pub version: u32,
}

impl Default for NdArrayMetadata {
    fn default() -> Self {
        Self {
            version: ENCODING_VERSION,
        }
    }
}

/// The nd extension type: the element field of the values, plus the attribute fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdArrayType {
    value_field: FieldRef,
    attrs: Fields,
    metadata: NdArrayMetadata,
}

impl NdArrayType {
    /// An nd type with nullable values of `value_type` and the given attribute fields.
    pub fn new(value_type: DataType, attrs: Fields) -> Self {
        Self::with_value_field(Arc::new(Field::new(ITEM_FIELD, value_type, true)), attrs)
    }

    /// An nd type with a custom item field for the values list,
    /// for example to keep nested extension metadata on the elements.
    pub fn with_value_field(value_field: FieldRef, attrs: Fields) -> Self {
        let attrs = attrs
            .iter()
            .map(|f| {
                if f.is_nullable() {
                    f.clone()
                } else {
                    Arc::new(f.as_ref().clone().with_nullable(true))
                }
            })
            .collect();
        Self {
            value_field,
            attrs,
            metadata: NdArrayMetadata::default(),
        }
    }

    /// The item field of the `values` list.
    pub fn value_field(&self) -> &FieldRef {
        &self.value_field
    }

    /// The Arrow type of the array values.
    pub fn value_type(&self) -> &DataType {
        self.value_field.data_type()
    }

    /// One field per attribute key. Empty when there are no attributes.
    pub fn attribute_fields(&self) -> &Fields {
        &self.attrs
    }

    /// The storage type of this extension type.
    pub fn storage_type(&self) -> DataType {
        DataType::Struct(self.storage_fields())
    }

    /// True when the storage has an `attrs` field.
    pub fn has_attributes(&self) -> bool {
        !self.attrs.is_empty()
    }

    pub(crate) fn storage_fields(&self) -> Fields {
        let mut fields = vec![
            Field::new(
                VALUES_FIELD,
                DataType::List(self.value_field.clone()),
                false,
            ),
            Field::new(DIM_NAMES_FIELD, dim_names_type(), false),
            Field::new(DIM_SIZES_FIELD, dim_sizes_type(), false),
        ];
        if self.has_attributes() {
            fields.push(Field::new(
                ATTRS_FIELD,
                DataType::Struct(self.attrs.clone()),
                false,
            ));
        }
        fields.into()
    }

    /// Parse the storage layout. This ignores field metadata.
    /// An `attrs` struct with zero fields reads as no attributes.
    pub fn try_from_storage_type(data_type: &DataType) -> Result<Self> {
        let DataType::Struct(fields) = data_type else {
            return Err(nd_err!(
                InvalidEncoding,
                "storage type must be a Struct, got {data_type}"
            ));
        };
        let names: Vec<&str> = fields.iter().map(|f| f.name().as_str()).collect();
        let expected = [VALUES_FIELD, DIM_NAMES_FIELD, DIM_SIZES_FIELD, ATTRS_FIELD];
        if !(names == expected[..3] || names == expected) {
            return Err(nd_err!(
                InvalidEncoding,
                "storage fields must be {expected:?}, got {names:?}"
            ));
        }

        let DataType::List(value_field) = fields[0].data_type() else {
            return Err(nd_err!(
                InvalidEncoding,
                "'{VALUES_FIELD}' must be a List, got {}",
                fields[0].data_type()
            ));
        };
        expect_list_of(&fields[1], &DataType::Utf8)?;
        expect_list_of(&fields[2], &DataType::UInt32)?;
        let attrs = match fields.get(3).map(|f| f.data_type()) {
            None => Fields::empty(),
            Some(DataType::Struct(attrs)) => attrs.clone(),
            Some(other) => {
                return Err(nd_err!(
                    InvalidEncoding,
                    "'{ATTRS_FIELD}' must be a Struct, got {other}"
                ));
            }
        };

        Ok(Self::with_value_field(value_field.clone(), attrs))
    }

    /// A field named `name` with this storage type and the extension tag.
    pub fn to_field(&self, name: impl Into<String>, nullable: bool) -> Field {
        Field::new(name, self.storage_type(), nullable).with_extension_type(self.clone())
    }
}

fn dim_names_type() -> DataType {
    DataType::List(Arc::new(Field::new(ITEM_FIELD, DataType::Utf8, false)))
}

fn dim_sizes_type() -> DataType {
    DataType::List(Arc::new(Field::new(ITEM_FIELD, DataType::UInt32, false)))
}

fn expect_list_of(field: &Field, item_type: &DataType) -> Result<()> {
    match field.data_type() {
        DataType::List(item) if item.data_type() == item_type => Ok(()),
        other => Err(nd_err!(
            InvalidEncoding,
            "'{}' must be List<{item_type}>, got {other}",
            field.name()
        )),
    }
}

impl ExtensionType for NdArrayType {
    const NAME: &'static str = "nd_arrow.array";

    type Metadata = NdArrayMetadata;

    fn metadata(&self) -> &Self::Metadata {
        &self.metadata
    }

    fn serialize_metadata(&self) -> Option<String> {
        Some(serde_json::to_string(&self.metadata).expect("metadata serializes"))
    }

    fn deserialize_metadata(metadata: Option<&str>) -> Result<Self::Metadata, ArrowError> {
        let metadata = match metadata.map(str::trim) {
            None | Some("") | Some("{}") => NdArrayMetadata::default(),
            Some(json) => serde_json::from_str(json).map_err(|e| {
                ArrowError::InvalidArgumentError(format!("invalid nd extension metadata: {e}"))
            })?,
        };
        if metadata.version != ENCODING_VERSION {
            return Err(ArrowError::InvalidArgumentError(format!(
                "unsupported nd encoding version {}, expected {ENCODING_VERSION}",
                metadata.version
            )));
        }
        Ok(metadata)
    }

    fn supports_data_type(&self, data_type: &DataType) -> Result<(), ArrowError> {
        let parsed = Self::try_from_storage_type(data_type)?;
        if parsed.value_field != self.value_field || parsed.attrs != self.attrs {
            return Err(ArrowError::InvalidArgumentError(format!(
                "nd extension type does not match storage type {data_type}"
            )));
        }
        Ok(())
    }

    fn try_new(data_type: &DataType, metadata: Self::Metadata) -> Result<Self, ArrowError> {
        let mut parsed = Self::try_from_storage_type(data_type)?;
        parsed.metadata = metadata;
        Ok(parsed)
    }
}

/// Extension name of the column that holds the global attributes of each nd batch.
pub const GLOBAL_ATTRIBUTES_EXTENSION: &str = "nd_arrow.global_attributes";

/// Name of the global attributes column in an encoded record batch.
pub const GLOBAL_ATTRIBUTES_FIELD: &str = "__global_attrs__";

/// The global attributes column: a `Struct` with one nullable child per key,
/// tagged with [`GLOBAL_ATTRIBUTES_EXTENSION`].
pub fn global_attributes_field(fields: Fields) -> Field {
    Field::new(GLOBAL_ATTRIBUTES_FIELD, DataType::Struct(fields), false).with_metadata(
        HashMap::from([
            (
                EXTENSION_TYPE_NAME_KEY.to_string(),
                GLOBAL_ATTRIBUTES_EXTENSION.to_string(),
            ),
            (
                EXTENSION_TYPE_METADATA_KEY.to_string(),
                serde_json::to_string(&NdArrayMetadata::default()).expect("metadata serializes"),
            ),
        ]),
    )
}

/// True when `field` is a global attributes column.
pub fn is_global_attributes_field(field: &Field) -> bool {
    field.extension_type_name() == Some(GLOBAL_ATTRIBUTES_EXTENSION)
        && matches!(field.data_type(), DataType::Struct(_))
}

/// True when `field` carries the nd extension tag.
pub fn is_nd_field(field: &Field) -> bool {
    field.extension_type_name() == Some(NdArrayType::NAME)
}

/// The nd field for a logical field: same name and metadata, values of the
/// logical type, no attributes. The field is nullable.
pub fn encoded_field(logical: &Field) -> Field {
    let field = NdArrayType::new(logical.data_type().clone(), Fields::empty())
        .to_field(logical.name(), true);
    with_base_metadata(field, logical.metadata())
}

/// The logical field of an nd field: the element type of the values,
/// with the field metadata minus the extension keys.
pub fn logical_field(encoded: &Field) -> Result<Field> {
    let ext: NdArrayType = encoded.try_extension_type()?;
    let mut metadata = encoded.metadata().clone();
    metadata.remove(EXTENSION_TYPE_NAME_KEY);
    metadata.remove(EXTENSION_TYPE_METADATA_KEY);
    Ok(Field::new(
        encoded.name(),
        ext.value_type().clone(),
        encoded.is_nullable(),
    )
    .with_metadata(metadata))
}

/// Every field of `logical` as an nd field, see [`encoded_field`].
pub fn encoded_schema(logical: &Schema) -> Schema {
    let fields: Vec<Field> = logical.fields().iter().map(|f| encoded_field(f)).collect();
    Schema::new_with_metadata(fields, logical.metadata().clone())
}

/// Every field of `encoded` as its logical field, see [`logical_field`].
pub fn logical_schema(encoded: &Schema) -> Result<Schema> {
    let fields = encoded
        .fields()
        .iter()
        .map(|f| logical_field(f))
        .collect::<Result<Vec<_>>>()?;
    Ok(Schema::new_with_metadata(
        fields,
        encoded.metadata().clone(),
    ))
}

/// `field` with `base` metadata added under its own (extension) keys.
pub(crate) fn with_base_metadata(field: Field, base: &HashMap<String, String>) -> Field {
    let mut metadata = base.clone();
    metadata.extend(field.metadata().clone());
    field.with_metadata(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> NdArrayType {
        NdArrayType::new(
            DataType::Float64,
            Fields::from(vec![Field::new("units", DataType::Utf8, true)]),
        )
    }

    #[test]
    fn storage_type_round_trips() {
        let ext = sample();
        let parsed = NdArrayType::try_from_storage_type(&ext.storage_type()).unwrap();
        assert_eq!(parsed, ext);
    }

    #[test]
    fn field_is_tagged_and_parses_back() {
        let field = sample().to_field("sst", true);
        assert!(is_nd_field(&field));
        let ext: NdArrayType = field.try_extension_type().unwrap();
        assert_eq!(ext, sample());
        assert_eq!(
            field.metadata().get(EXTENSION_TYPE_METADATA_KEY).unwrap(),
            r#"{"version":1}"#
        );
    }

    #[test]
    fn logical_and_encoded_fields_round_trip() {
        let logical = Field::new("sst", DataType::Float32, true)
            .with_metadata(HashMap::from([("k".to_string(), "v".to_string())]));
        let encoded = encoded_field(&logical);
        assert!(is_nd_field(&encoded));
        assert_eq!(encoded.metadata().get("k").unwrap(), "v");
        assert_eq!(logical_field(&encoded).unwrap(), logical);
    }

    #[test]
    fn invalid_storage_rejected() {
        assert!(NdArrayType::try_from_storage_type(&DataType::Float64).is_err());
        let wrong_order = DataType::Struct(Fields::from(vec![
            Field::new(DIM_NAMES_FIELD, dim_names_type(), false),
            Field::new(
                VALUES_FIELD,
                DataType::new_list(DataType::Int32, true),
                false,
            ),
            Field::new(DIM_SIZES_FIELD, dim_sizes_type(), false),
            Field::new(ATTRS_FIELD, DataType::Struct(Fields::empty()), false),
        ]));
        assert!(NdArrayType::try_from_storage_type(&wrong_order).is_err());
    }

    #[test]
    fn unknown_version_rejected() {
        assert!(NdArrayType::deserialize_metadata(Some(r#"{"version":2}"#)).is_err());
        assert!(NdArrayType::deserialize_metadata(None).is_ok());
    }
}
