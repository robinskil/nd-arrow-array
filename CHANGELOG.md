# Changelog

All notable changes to `nd-arrow-array` are in this file.
One version covers the Rust crate and the Python package.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The project follows [Semantic Versioning](https://semver.org/).
Before 1.0, a minor version can break the API.

The release workflow copies the section of each version into the GitHub release.
Keep the heading format `## [x.y.z] - YYYY-MM-DD`.

## [Unreleased]

## [0.5.0] - 2026-11-05

First release on crates.io and PyPI. This version replaces the old API completely.

### Added

- The `nd_arrow.array` Arrow extension type, format version 1. See `FORMAT.md`.
- `NdArrowArray`, `NdColumn`, `NdColumnBuilder` and `NdRecordBatch`.
- Global attributes for record batches.
- NumPy-style kernels: layout, arithmetic, reductions, sort, search,
  interpolation, rolling windows, group-by and time resampling.
- CF decoding of fill values, packing and time.
- Conversion to and from the Arrow canonical tensor types.
- Conversion to and from `ndarray::ArrayD` (feature `ndarray`).
- Parallel kernels with rayon (feature `parallel`).
- Python package `nd-arrow-array` (`import nd_arrow_array`) with `NdArray`,
  `NdColumn` and `NdRecordBatch`.
- Python conversion to and from NumPy, pyarrow and xarray. cftime dates
  become timestamps in the standard calendar.

[Unreleased]: https://github.com/robinskil/nd-arrow-array/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/robinskil/nd-arrow-array/releases/tag/v0.5.0
