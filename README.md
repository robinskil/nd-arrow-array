# N-Dimensional Arrow Arrays

## ND Arrow Specification

### Overview

This document provides the specifications for N-Dimensional Arrow Arrays, detailing their structure, usage, and examples.

### Structure

- **Dimensions**: Each array can have multiple dimensions.
- **Elements**: The elements within the array can be of any arrow logical types.

### N-Dimensional Arrow Array Representation

- **Data Type**: The array is represented using a struct array type.
  - **Fields**:
    - `dimension_names`: The names of each dimension.
    - `dimension_sizes`: The size of each dimension.
    - `values`: The actual data stored in the array.

- **Example**: A 3D array with dimensions (2, 3, 4) would have:
  - `dimension_names`: ["dim1", "dim2", "dim3"]
  - `dimension_sizes`: [2, 3, 4]
  - `values`: The actual data in a flattened format.
- **Flattening**: The data is stored in a flattened format, meaning that the multi-dimensional data is represented as a single array.

### ND Record Batch

- **ND Record Batch**: A collection of N-Dimensional Arrow Arrays.
  - **Metadata**: Encoding defining that a record batch contains N-Dimensional Arrow Arrays.
  - **Fields**:
    - `schema`: The schema of the record batch.
    - `arrays`: A list of N-Dimensional Arrow Arrays.

#### Broadcasting

- **Broadcasting**: Broadcasting is the process of expanding the dimensions of an array to match another array's dimensions.
  - **Example**: If you have a 2D array and a 1D array, the 1D array can be broadcasted to match the dimensions of the 2D array.
- **Broadcasting Rules**:
  - The dimensions of the arrays must be compatible.
  - If the dimensions are not compatible, an error will be raised.
- **Example**: Broadcasting a 1D array of shape (3,) to a 2D array of shape (2, 3) will result in a new array of shape (2, 3).

This will also apply work for record batches with N-Dimensional Arrow Arrays. The broadcasting rules will be the same as above.
