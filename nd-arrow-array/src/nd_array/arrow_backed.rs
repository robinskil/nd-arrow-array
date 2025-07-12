use std::sync::Arc;

use arrow::{
    array::{
        Array, ArrayData, ArrayRef, AsArray, ListArray, NullArray, Scalar, StringBuilder,
        StructArray, UInt32Builder,
    },
    buffer::Buffer,
    datatypes::{DataType, Field, FieldRef, UInt32Type},
};

use crate::{
    broadcast::{broadcast_array_impl, broadcast_reshape_args, BroadcastingError},
    nd_array::{arrow_ext::ArrowParseError, dimension::Dimension, NdArrowArray},
};

#[derive(Debug, Clone)]
pub struct NdArrowArrayImpl {
    arrow_backed: StructArray,
}

impl NdArrowArrayImpl {
    pub fn try_from_arrow(array: Scalar<ArrayRef>) -> Result<Self, ArrowParseError> {
        let inner_array = array.into_inner();
        let struct_array = inner_array
            .as_struct_opt()
            .ok_or(ArrowParseError::StructDowncastError)?;

        // Validate the fields of the struct array
        struct_array
            .column_by_name("dimension_names")
            .ok_or(ArrowParseError::NoDimensionNames)?;
        struct_array
            .column_by_name("dimension_sizes")
            .ok_or(ArrowParseError::NoDimensionSizes)?;
        struct_array
            .column_by_name("values")
            .ok_or(ArrowParseError::NoValues)?;

        Ok(Self {
            arrow_backed: struct_array.clone(),
        })
    }

    pub fn new<S: AsRef<str>>(values: ArrayRef, dimensions: Vec<(S, usize)>) -> Self {
        // --- 1) build and wrap the names list ---
        let mut name_builder = StringBuilder::new();
        for (name, _) in &dimensions {
            name_builder.append_value(name.as_ref());
        }
        let name_array: ArrayRef = Arc::new(name_builder.finish());
        let names_list = Self::wrap_in_list(name_array);

        // --- 2) build and wrap the sizes list ---
        let mut size_builder = UInt32Builder::new();
        for (_, size) in &dimensions {
            size_builder.append_value(*size as u32);
        }
        let size_array: ArrayRef = Arc::new(size_builder.finish());
        let sizes_list = Self::wrap_in_list(size_array);

        // --- 3) wrap your `values` array exactly as before ---
        let values_list = Self::wrap_in_list(values);

        let struct_fields = vec![
            FieldRef::new(Field::new(
                "dimension_names",
                names_list.data_type().clone(),
                true,
            )),
            FieldRef::new(Field::new(
                "dimension_sizes",
                sizes_list.data_type().clone(),
                true,
            )),
            FieldRef::new(Field::new("values", values_list.data_type().clone(), true)),
        ];

        Self {
            arrow_backed: StructArray::new(
                struct_fields.into(),
                vec![names_list, sizes_list, values_list],
                None,
            ),
        }
    }

    /// Wrap any ArrayRef into a 1-element ListArray (zero-copy).
    fn wrap_in_list(array: ArrayRef) -> ArrayRef {
        let child_data = array.to_data().clone();
        let field = Field::new("item", array.data_type().clone(), true);
        let list_type = DataType::List(Arc::new(field));
        let offsets = Buffer::from_slice_ref(&[0i32, array.len() as i32]);

        let data = ArrayData::builder(list_type)
            .len(1)
            .add_buffer(offsets)
            .add_child_data(child_data)
            .build()
            .unwrap();

        Arc::new(ListArray::from(data))
    }

    pub fn null(data_type: DataType, len: usize) -> Self {
        let struct_fields = vec![
            FieldRef::new(Field::new(
                "dimension_names",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            )),
            FieldRef::new(Field::new(
                "dimension_sizes",
                DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
                true,
            )),
            FieldRef::new(Field::new(
                "values",
                DataType::List(Arc::new(Field::new("item", data_type, true))),
                true,
            )),
        ];
        Self {
            arrow_backed: StructArray::new_null(struct_fields.into(), len),
        }
    }

    pub fn is_null(&self) -> bool {
        self.arrow_backed.is_null(0)
    }
}

impl NdArrowArray for NdArrowArrayImpl {
    fn shape(&self) -> Vec<usize> {
        let sizes_list_arr = self.arrow_backed.column_by_name("dimension_sizes").unwrap();
        let sizes_list_arr = sizes_list_arr.as_list::<i32>();
        let sizes_arry = sizes_list_arr.value(0);
        sizes_arry
            .as_primitive::<UInt32Type>()
            .iter()
            .map(|opt_size| opt_size.map(|size| size as usize).unwrap())
            .collect::<Vec<_>>()
    }

    fn dimensions(&self) -> &[super::dimension::Dimension] {
        unimplemented!()
    }

    fn array(&self) -> Arc<dyn arrow::array::Array> {
        let dyn_values_list_arr = self.arrow_backed.column_by_name("values").unwrap();
        let values_list_arr = dyn_values_list_arr.as_list::<i32>();
        values_list_arr.value(0)
    }

    fn dimensions_ref<'a>(&'a self) -> std::borrow::Cow<'a, [Dimension]> {
        let dimension_names = self.arrow_backed.column_by_name("dimension_names").unwrap();
        let dimension_names = dimension_names.as_list::<i32>();
        let dimension_names = dimension_names.value(0);

        let sizes = self.shape();

        // Create DimensionRef objects from names and sizes
        let dimension_refs = dimension_names
            .as_string::<i32>()
            .iter()
            .zip(sizes.into_iter())
            .map(|(name, size)| Dimension::from((name.unwrap(), size)))
            .collect::<Vec<_>>();

        std::borrow::Cow::Owned(dimension_refs)
    }

    fn to_arrow_array(
        &self,
    ) -> Result<Arc<dyn arrow::array::Array>, super::arrow_ext::ArrowParseError> {
        Ok(Arc::new(self.arrow_backed.clone()))
    }

    fn broadcast(
        &self,
        target_dimensions: &[Dimension],
    ) -> crate::broadcast::BroadcastResult<Arc<dyn NdArrowArray>> {
        let dimensions = self.dimensions_ref();

        if dimensions == target_dimensions {
            return Ok(Arc::new(self.clone()));
        }

        let (repeat_slice_count, repeat_element_count) =
            broadcast_reshape_args(dimensions.as_ref(), target_dimensions).ok_or(
                BroadcastingError::InvalidShapes(dimensions.to_vec(), target_dimensions.to_vec()),
            )?;

        let values = self.values_array();

        let broadcasted_array = if self.is_null() {
            // Return a null array with the target dimensions
            let null_array = Arc::new(NullArray::new(1)) as Arc<dyn arrow::array::Array>;
            let broadcasted_array = broadcast_array_impl(
                null_array.as_ref(),
                repeat_element_count,
                repeat_slice_count,
            )?;

            broadcasted_array
        } else {
            broadcast_array_impl(values.as_ref(), repeat_element_count, repeat_slice_count)?
        };

        let broadcasted_nd_array = NdArrowArrayImpl::new(
            broadcasted_array,
            target_dimensions
                .iter()
                .map(|dim| (dim.name(), dim.size()))
                .collect(),
        );

        Ok(Arc::new(broadcasted_nd_array))
    }
}
