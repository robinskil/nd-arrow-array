use std::sync::Arc;

use arrow::{
    array::{Array, StructArray},
    datatypes::{DataType, Field},
};

use crate::consts;

use super::{default::DefaultNdArrowArray, dimension::Dimension, NdArrowArray};

#[derive(Debug, Clone, thiserror::Error)]
pub enum ArrowParseError {
    #[error("No dimension names found.")]
    NoDimensionNames,
    #[error("No dimension sizes found.")]
    NoDimensionSizes,
    #[error("No inner values array found.")]
    NoValues,
    #[error("Shape and inner array length do not match: expected {expected}, got {got}")]
    ShapeAndArrayLengthMismatch { expected: usize, got: usize },
    #[error("Failed to downcast Arrow Array to Encoded ListArray.")]
    ListDowncastError,
    #[error("Number of dimension sizes does not match the number of dimension names.")]
    DimensionSizeMismatch,
}

pub fn try_from_arrow_array(array: &StructArray) -> Result<Arc<dyn NdArrowArray>, ArrowParseError> {
    let dimension_names_list = array
        .column_by_name(consts::DIMENSION_NAMES)
        .ok_or(ArrowParseError::NoDimensionSizes)?
        .as_any()
        .downcast_ref::<arrow::array::ListArray>()
        .ok_or(ArrowParseError::ListDowncastError)?;

    let dimension_names = dimension_names_list
        .values()
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .ok_or(ArrowParseError::NoDimensionNames)?;

    let dimension_names = dimension_names
        .iter()
        .filter_map(|v| v.map(|s| s.to_string()))
        .collect::<Vec<_>>();

    let dimension_sizes_list = array
        .column_by_name(consts::DIMENSION_SIZES)
        .ok_or(ArrowParseError::NoDimensionSizes)?
        .as_any()
        .downcast_ref::<arrow::array::ListArray>()
        .ok_or(ArrowParseError::ListDowncastError)?;

    let dimension_sizes = dimension_sizes_list
        .values()
        .as_any()
        .downcast_ref::<arrow::array::UInt32Array>()
        .ok_or(ArrowParseError::NoDimensionSizes)?;

    let shape = dimension_sizes
        .iter()
        .filter_map(|v| v.map(|u| u as usize))
        .collect::<Vec<_>>();

    if dimension_names.len() != shape.len() {
        return Err(ArrowParseError::DimensionSizeMismatch);
    }

    let list_data_array = array
        .column_by_name(consts::VALUES)
        .ok_or(ArrowParseError::NoValues)?;

    let data_array = list_data_array
        .as_any()
        .downcast_ref::<arrow::array::ListArray>()
        .ok_or(ArrowParseError::ListDowncastError)?
        .value(0);

    // Check if the data array is a scalar (length 1 and no dimensions)
    // If the data array is not a scalar, we need to check if the shape
    // and the data array length match
    if shape.iter().product::<usize>() != data_array.len()
        || (data_array.len() == 1 && shape.len() == 0)
    {
        return Err(ArrowParseError::ShapeAndArrayLengthMismatch {
            expected: shape.iter().product(),
            got: data_array.len(),
        });
    }

    let dimensions = dimension_names
        .iter()
        .zip(shape.iter())
        .map(|(name, &size)| Dimension::from((name.as_str(), size)))
        .collect::<Vec<_>>();

    Ok(Arc::new(DefaultNdArrowArray::new(
        data_array.clone(),
        dimensions,
    )))
}

pub fn arrow_encoded_dtype<A: NdArrowArray + ?Sized>(array: &A) -> DataType {
    let inner_array_field = Field::new(consts::VALUES, array.dtype().clone(), false);
    let list_array_field = Field::new(
        consts::VALUES,
        arrow::datatypes::DataType::List(Arc::new(inner_array_field.clone())),
        false,
    );

    let dimension_names_field = Field::new(
        consts::DIMENSION_NAMES,
        arrow::datatypes::DataType::Utf8,
        false,
    );
    let list_dimension_names_array_field = Field::new(
        consts::DIMENSION_NAMES,
        arrow::datatypes::DataType::List(Arc::new(dimension_names_field.clone())),
        false,
    );

    let dimension_sizes_field = Field::new(
        consts::DIMENSION_SIZES,
        arrow::datatypes::DataType::UInt32,
        false,
    );
    let list_dimension_sizes_array_field = Field::new(
        consts::DIMENSION_SIZES,
        arrow::datatypes::DataType::List(Arc::new(dimension_sizes_field.clone())),
        false,
    );

    DataType::Struct(
        vec![
            list_dimension_names_array_field,
            list_dimension_sizes_array_field,
            list_array_field,
        ]
        .into(),
    )
}

pub fn to_arrow_array<A: NdArrowArray + ?Sized>(
    array: &A,
) -> Result<Arc<dyn Array>, ArrowParseError> {
    // Create inner list data array
    let inner_array = array.array();
    let inner_array_field = Field::new(consts::VALUES, inner_array.data_type().clone(), false);
    let list_array_field = Field::new(
        consts::VALUES,
        arrow::datatypes::DataType::List(Arc::new(inner_array_field.clone())),
        false,
    );
    let inner_list_array = Arc::new(arrow::array::ListArray::new(
        Arc::new(inner_array_field.clone()),
        arrow::buffer::OffsetBuffer::<i32>::from_lengths(vec![inner_array.len()]),
        inner_array,
        None,
    ));

    //Create dimension names array
    let dimension_names = array
        .dimensions()
        .iter()
        .map(|dim| dim.name().to_string())
        .collect::<Vec<_>>();
    let dimension_names_array = Arc::new(arrow::array::StringArray::from_iter(
        dimension_names.iter().map(|s| Some(s.as_str())),
    ));
    let dimension_names_field = Field::new(
        consts::DIMENSION_NAMES,
        arrow::datatypes::DataType::Utf8,
        false,
    );
    let list_dimension_names_array = Arc::new(arrow::array::ListArray::new(
        Arc::new(dimension_names_field.clone()),
        arrow::buffer::OffsetBuffer::<i32>::from_lengths(vec![dimension_names.len()]),
        dimension_names_array,
        None,
    ));
    let list_dimension_names_array_field = Field::new(
        consts::DIMENSION_NAMES,
        arrow::datatypes::DataType::List(Arc::new(dimension_names_field.clone())),
        false,
    );
    // Create dimension sizes array

    let dimension_sizes = array
        .dimensions()
        .iter()
        .map(|dim| dim.size() as u32)
        .collect::<Vec<_>>();

    let dimension_sizes_array = Arc::new(arrow::array::UInt32Array::from_iter(
        dimension_sizes.iter().map(|&s| Some(s)),
    ));
    let dimension_sizes_field = Field::new(
        consts::DIMENSION_SIZES,
        arrow::datatypes::DataType::UInt32,
        false,
    );
    let list_dimension_sizes_array_field = Field::new(
        consts::DIMENSION_SIZES,
        arrow::datatypes::DataType::List(Arc::new(dimension_sizes_field.clone())),
        false,
    );
    let list_dimension_sizes_array = Arc::new(arrow::array::ListArray::new(
        Arc::new(dimension_sizes_field.clone()),
        arrow::buffer::OffsetBuffer::<i32>::from_lengths(vec![dimension_sizes.len()]),
        dimension_sizes_array,
        None,
    ));

    let parsed_arrow_array = StructArray::new(
        vec![
            list_dimension_names_array_field,
            list_dimension_sizes_array_field,
            list_array_field,
        ]
        .into(),
        vec![
            list_dimension_names_array,
            list_dimension_sizes_array,
            inner_list_array,
        ],
        None,
    );

    Ok(Arc::new(parsed_arrow_array))
}
#[cfg(test)]
mod tests {
    use super::*;

    use crate::nd_array::default::DefaultNdArrowArray;
    use arrow::array::Float64Array;
    use std::sync::Arc;

    #[test]
    fn test_to_arrow_array_and_back() {
        // Create a simple NdArrowArray
        let values = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let dimensions = vec![Dimension::new("dim1", 2), Dimension::new("dim2", 3)];
        let array = DefaultNdArrowArray::new(values, dimensions);

        // Convert to Arrow array
        let arrow_array = to_arrow_array(&array).unwrap();
        let struct_array = arrow_array.as_any().downcast_ref::<StructArray>().unwrap();

        // Convert back to NdArrowArray
        let nd_array = try_from_arrow_array(struct_array).unwrap();

        // Verify
        assert_eq!(nd_array.shape(), array.shape());
        assert_eq!(nd_array.array().len(), array.array().len());
        assert_eq!(nd_array.dimensions(), array.dimensions());
    }
}
