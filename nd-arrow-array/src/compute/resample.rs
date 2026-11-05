//! Resampling along a time dimension, like `xarray.DataArray.resample`.
//!
//! Each time goes into one bin of a regular [`Frequency`]. Bins are closed on
//! the left and are labeled by their start, like the pandas defaults:
//!
//! - Fixed periods (day, hour, minute, second) count from midnight of the day
//!   of the first time, like pandas `origin="start_day"`.
//! - Calendar periods (month, year) count from the month or year of the first
//!   time, like pandas `MS` and `YS`.

use std::str::FromStr;
use std::sync::Arc;

use arrow::array::{TimestampMicrosecondArray, new_null_array};

use crate::NdArrowArray;
use crate::compute::reduce::{Reduction, reduce};
use crate::compute::shape::{Selector, Slice, concat, expand_dims, isel};
use crate::compute::time::{
    MICROS_PER_DAY, MICROS_PER_SECOND, TimePeriod, civil_from_days, days_from_civil, to_micros,
};
use crate::error::{Result, nd_err};

/// Upper limit of the number of bins, against memory exhaustion.
pub const MAX_BINS: usize = u32::MAX as usize;

/// A regular time step: `step` times a calendar `period`, for example 6 hours.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frequency {
    step: u32,
    period: TimePeriod,
}

impl Frequency {
    /// A frequency of `step` periods. `step` must be at least 1.
    pub fn new(step: u32, period: TimePeriod) -> Result<Self> {
        if step == 0 {
            return Err(nd_err!(
                InvalidArgument,
                "frequency step must be at least 1"
            ));
        }
        Ok(Self { step, period })
    }

    /// Parse a pandas frequency string, for example `"6h"`, `"1D"` or `"3MS"`.
    ///
    /// The aliases are `YS`, `MS`, `D`, `h`, `min` and `s`, with an optional
    /// step in front. End-anchored aliases such as `M` and `Y` are not supported.
    pub fn parse(text: &str) -> Result<Self> {
        let text = text.trim();
        let split = text
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(text.len());
        let (digits, alias) = text.split_at(split);
        let step = match digits {
            "" => 1,
            _ => digits
                .parse()
                .map_err(|_| nd_err!(InvalidArgument, "invalid frequency step in '{text}'"))?,
        };
        let period = match alias {
            "YS" => TimePeriod::Year,
            "MS" => TimePeriod::Month,
            "D" => TimePeriod::Day,
            "h" => TimePeriod::Hour,
            "min" => TimePeriod::Minute,
            "s" => TimePeriod::Second,
            _ => {
                return Err(nd_err!(
                    InvalidArgument,
                    "unsupported frequency '{text}': use YS, MS, D, h, min or s"
                ));
            }
        };
        Self::new(step, period)
    }

    /// The number of periods per bin.
    pub fn step(&self) -> u32 {
        self.step
    }

    /// The calendar period.
    pub fn period(&self) -> TimePeriod {
        self.period
    }
}

impl FromStr for Frequency {
    type Err = crate::NdError;

    fn from_str(text: &str) -> Result<Self> {
        Self::parse(text)
    }
}

/// Reduce `a` per time bin along `dim`.
///
/// `times` is 1-D with one timestamp (or date) per position along `dim`, in
/// any order. Null times are left out. The result replaces `dim` with one
/// position per bin, from the bin of the first time to the bin of the last.
/// A bin without times gives null, like xarray. The second value holds the
/// bin starts as `Timestamp(Microsecond)` along `dim`, with the attributes of
/// `times`.
pub fn resample(
    a: &NdArrowArray,
    dim: &str,
    times: &NdArrowArray,
    freq: Frequency,
    op: Reduction,
) -> Result<(NdArrowArray, NdArrowArray)> {
    let axis = a.dims().axis(dim)?;
    let n = a.dims().as_slice()[axis].size();
    if times.rank() != 1 || times.len() != n {
        return Err(nd_err!(
            InvalidArgument,
            "resample times must be 1-D with {n} values along '{dim}'"
        ));
    }
    let stamps = to_micros(times)?;
    let (bin_of, labels) = assign_bins(&stamps, freq)?;

    // Counting sort: the positions of bin b are positions[offsets[b]..offsets[b + 1]].
    let mut offsets = vec![0usize; labels.len() + 1];
    for bin in bin_of.iter().flatten() {
        offsets[bin + 1] += 1;
    }
    for b in 0..labels.len() {
        offsets[b + 1] += offsets[b];
    }
    let mut cursor = offsets.clone();
    let mut positions = vec![0isize; offsets[labels.len()]];
    for (position, bin) in bin_of.iter().enumerate() {
        if let Some(bin) = bin {
            positions[cursor[*bin]] = position as isize;
            cursor[*bin] += 1;
        }
    }

    let mut parts: Vec<Option<NdArrowArray>> = Vec::with_capacity(labels.len());
    for b in 0..labels.len() {
        let members = &positions[offsets[b]..offsets[b + 1]];
        let (Some(&first), Some(&last)) = (members.first(), members.last()) else {
            parts.push(None);
            continue;
        };
        let selector = if (last - first) as usize + 1 == members.len() {
            Selector::Slice(Slice::range(first, last + 1))
        } else {
            Selector::Take(members.to_vec())
        };
        let group = isel(a, &[(dim, selector)])?;
        let reduced = reduce(&group, Some(&[dim]), op)?;
        parts.push(Some(expand_dims(&reduced, dim, axis)?));
    }

    let template = parts
        .iter()
        .flatten()
        .next()
        .ok_or_else(|| nd_err!(InvalidArgument, "resample times are all null"))?;
    let empty = NdArrowArray::try_new(
        new_null_array(template.data_type(), template.len()),
        template.dims().clone(),
    )?;
    let parts: Vec<&NdArrowArray> = parts.iter().map(|p| p.as_ref().unwrap_or(&empty)).collect();
    let result = concat(&parts, dim)?;

    let labels = NdArrowArray::try_new(
        Arc::new(TimestampMicrosecondArray::from(labels)),
        crate::Dimensions::try_new([(dim, parts.len())])?,
    )?
    .with_attributes(times.attributes().clone());
    Ok((result, labels))
}

/// The bin of each time (`None` for a null time) and the start of each bin.
fn assign_bins(
    stamps: &TimestampMicrosecondArray,
    freq: Frequency,
) -> Result<(Vec<Option<usize>>, Vec<i64>)> {
    let (Some(min), Some(max)) = (arrow::compute::min(stamps), arrow::compute::max(stamps)) else {
        return Err(nd_err!(InvalidArgument, "resample times are all null"));
    };
    let step = i128::from(freq.step);
    let unit = match freq.period {
        TimePeriod::Second => Some(MICROS_PER_SECOND),
        TimePeriod::Minute => Some(60 * MICROS_PER_SECOND),
        TimePeriod::Hour => Some(3_600 * MICROS_PER_SECOND),
        TimePeriod::Day => Some(MICROS_PER_DAY),
        TimePeriod::Month | TimePeriod::Year => None,
    };
    let scheme = match unit {
        Some(unit) => Bins::Fixed {
            origin: i128::from(min.div_euclid(MICROS_PER_DAY) * MICROS_PER_DAY),
            width: step * i128::from(unit),
        },
        None => {
            let first = month_of(min);
            let year = freq.period == TimePeriod::Year;
            Bins::Calendar {
                base: if year {
                    first - first.rem_euclid(12)
                } else {
                    first
                },
                months: if year { 12 * step } else { step },
            }
        }
    };
    let index = |t: i64| scheme.index(t);
    let label = |k: i128| scheme.label(k);

    // The origin can lie bins before the first time; the output starts at its bin.
    let first = index(min);
    let count = usize::try_from(index(max) - first + 1)
        .ok()
        .filter(|&c| c <= MAX_BINS)
        .ok_or_else(|| nd_err!(InvalidArgument, "resample needs more than {MAX_BINS} bins"))?;
    let bins = stamps
        .iter()
        .map(|t| t.map(|t| (index(t) - first) as usize))
        .collect();
    let labels = (0..count as i128).map(|k| label(first + k)).collect();
    Ok((bins, labels))
}

/// Bin layout: fixed-width bins from an origin, or whole months from a base month.
enum Bins {
    Fixed { origin: i128, width: i128 },
    Calendar { base: i128, months: i128 },
}

impl Bins {
    /// The bin index of time `t`, counted from the origin.
    fn index(&self, t: i64) -> i128 {
        match *self {
            Bins::Fixed { origin, width } => (i128::from(t) - origin) / width,
            Bins::Calendar { base, months } => (month_of(t) - base) / months,
        }
    }

    /// The start of bin `k`.
    fn label(&self, k: i128) -> i64 {
        match *self {
            Bins::Fixed { origin, width } => (origin + k * width) as i64,
            Bins::Calendar { base, months } => {
                let month = base + k * months;
                let (year, month) = (month.div_euclid(12), month.rem_euclid(12) as u32 + 1);
                days_from_civil(year as i64, month, 1) * MICROS_PER_DAY
            }
        }
    }
}

/// Months since year 0 of time `t`.
fn month_of(t: i64) -> i128 {
    let (year, month, _) = civil_from_days(t.div_euclid(MICROS_PER_DAY));
    i128::from(year) * 12 + i128::from(month) - 1
}

#[cfg(test)]
mod tests {
    use arrow::array::{AsArray, Float64Array, Int32Array};
    use arrow::datatypes::{Float64Type, Int64Type, TimestampMicrosecondType};

    use super::*;
    use crate::{Attributes, Dimensions};

    fn at(year: i64, month: u32, day: u32, hour: i64) -> i64 {
        days_from_civil(year, month, day) * MICROS_PER_DAY + hour * 3_600 * MICROS_PER_SECOND
    }

    fn time_axis(stamps: Vec<Option<i64>>) -> NdArrowArray {
        let n = stamps.len();
        NdArrowArray::try_new(
            Arc::new(TimestampMicrosecondArray::from(stamps)),
            Dimensions::try_new([("time", n)]).unwrap(),
        )
        .unwrap()
    }

    fn f64s(a: &NdArrowArray) -> Vec<Option<f64>> {
        a.values().as_primitive::<Float64Type>().iter().collect()
    }

    fn stamps(a: &NdArrowArray) -> Vec<i64> {
        a.values()
            .as_primitive::<TimestampMicrosecondType>()
            .values()
            .to_vec()
    }

    // Expected values come from xarray 2026.9 `DataArray.resample`.
    fn hourly() -> (NdArrowArray, NdArrowArray) {
        let times = time_axis(vec![
            Some(at(2021, 1, 1, 1)),
            Some(at(2021, 1, 1, 2)),
            Some(at(2021, 1, 1, 5)),
            Some(at(2021, 1, 3, 23)),
            Some(at(2021, 1, 4, 0)),
        ]);
        let values = NdArrowArray::try_new(
            Arc::new(Float64Array::from(vec![
                Some(1.0),
                None,
                Some(2.0),
                Some(4.0),
                Some(8.0),
            ])),
            times.dims().clone(),
        )
        .unwrap();
        (values, times)
    }

    #[test]
    fn daily_bins_match_xarray() {
        let (values, times) = hourly();
        let freq = Frequency::parse("1D").unwrap();
        let (sum, labels) = resample(&values, "time", &times, freq, Reduction::Sum).unwrap();
        assert_eq!(f64s(&sum), vec![Some(3.0), None, Some(4.0), Some(8.0)]);
        assert_eq!(
            stamps(&labels),
            vec![
                at(2021, 1, 1, 0),
                at(2021, 1, 2, 0),
                at(2021, 1, 3, 0),
                at(2021, 1, 4, 0)
            ]
        );
        let (mean, _) = resample(&values, "time", &times, freq, Reduction::Mean).unwrap();
        assert_eq!(f64s(&mean), vec![Some(1.5), None, Some(4.0), Some(8.0)]);
        let (count, _) = resample(&values, "time", &times, freq, Reduction::Count).unwrap();
        let count: Vec<Option<i64>> = count.values().as_primitive::<Int64Type>().iter().collect();
        assert_eq!(count, vec![Some(2), None, Some(1), Some(1)]);

        let two_days = Frequency::parse("2D").unwrap();
        let (sum, _) = resample(&values, "time", &times, two_days, Reduction::Sum).unwrap();
        assert_eq!(f64s(&sum), vec![Some(3.0), Some(12.0)]);
    }

    #[test]
    fn fixed_bins_start_at_midnight_of_the_first_day() {
        let times = time_axis(vec![Some(at(1969, 12, 31, 23)), Some(at(1970, 1, 1, 1))]);
        let values = NdArrowArray::try_new(
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            times.dims().clone(),
        )
        .unwrap();
        let freq = Frequency::parse("7h").unwrap();
        let (sum, labels) = resample(&values, "time", &times, freq, Reduction::Sum).unwrap();
        assert_eq!(f64s(&sum), vec![Some(3.0)]);
        assert_eq!(stamps(&labels), vec![at(1969, 12, 31, 21)]);
    }

    #[test]
    fn calendar_bins_match_xarray() {
        let times = time_axis(vec![
            Some(at(2020, 11, 15, 0)),
            Some(at(2021, 2, 10, 0)),
            Some(at(2021, 3, 1, 0)),
            Some(at(2022, 7, 4, 0)),
        ]);
        let values = NdArrowArray::try_new(
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            times.dims().clone(),
        )
        .unwrap();
        let sum = |freq: &str| {
            let freq = Frequency::parse(freq).unwrap();
            let (sum, labels) = resample(&values, "time", &times, freq, Reduction::Sum).unwrap();
            (f64s(&sum), stamps(&labels))
        };
        let (v, l) = sum("3MS");
        assert_eq!(
            v,
            vec![Some(1.0), Some(5.0), None, None, None, None, Some(4.0)]
        );
        assert_eq!(l[0], at(2020, 11, 1, 0));
        assert_eq!(l[1], at(2021, 2, 1, 0));
        let (v, l) = sum("YS");
        assert_eq!(v, vec![Some(1.0), Some(5.0), Some(4.0)]);
        assert_eq!(l[2], at(2022, 1, 1, 0));
        let (v, l) = sum("2YS");
        assert_eq!(v, vec![Some(6.0), Some(4.0)]);
        assert_eq!(l, vec![at(2020, 1, 1, 0), at(2022, 1, 1, 0)]);
    }

    #[test]
    fn unsorted_and_null_times_on_a_grid() {
        let times = time_axis(vec![Some(at(2021, 1, 2, 6)), None, Some(at(2021, 1, 1, 6))])
            .with_attributes(Attributes::new().with("long_name", "time"));
        let values = NdArrowArray::try_new(
            Arc::new(Int32Array::from_iter_values(0..6)),
            Dimensions::try_new([("y", 2), ("time", 3)]).unwrap(),
        )
        .unwrap();
        let freq = Frequency::new(1, TimePeriod::Day).unwrap();
        let (sum, labels) = resample(&values, "time", &times, freq, Reduction::Sum).unwrap();
        assert_eq!(sum.dims().names(), vec!["y", "time"]);
        assert_eq!(
            sum.values().as_primitive::<Int64Type>().values(),
            &[2, 0, 5, 3]
        );
        assert_eq!(labels.attributes().get_str("long_name"), Some("time"));
    }

    #[test]
    fn invalid_input_rejected() {
        assert!(Frequency::parse("M").is_err());
        assert!(Frequency::parse("0D").is_err());
        assert_eq!(
            "15min".parse::<Frequency>().unwrap(),
            Frequency::new(15, TimePeriod::Minute).unwrap()
        );
        let times = time_axis(vec![None, None]);
        let values = NdArrowArray::try_new(
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            times.dims().clone(),
        )
        .unwrap();
        let freq = Frequency::parse("D").unwrap();
        assert!(resample(&values, "time", &times, freq, Reduction::Sum).is_err());
        let seconds = time_axis(vec![Some(i64::MIN / 2), Some(i64::MAX / 2)]);
        let freq = Frequency::parse("s").unwrap();
        assert!(resample(&values, "time", &seconds, freq, Reduction::Sum).is_err());
    }
}
