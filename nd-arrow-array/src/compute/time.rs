//! CF time coordinates and time labels for resampling.
//!
//! [`decode_cf_time`] turns numbers with `units = "<unit> since <reference>"`
//! into an Arrow `Timestamp(Microsecond)` without time zone, like xarray. The
//! microsecond unit covers ±292,000 years. Only the `standard`, `gregorian` and
//! `proleptic_gregorian` calendars are supported, all as proleptic Gregorian.

use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, Float64Array, TimestampMicrosecondArray};
use arrow::compute::cast as arrow_cast;
pub use arrow::compute::kernels::temporal::DatePart;
use arrow::compute::kernels::temporal::date_part;
use arrow::datatypes::{DataType, Float64Type, TimeUnit, TimestampMicrosecondType};

use crate::compute::cf::decode_cf;
use crate::error::{Result, nd_err};
use crate::{Attributes, NdArrowArray};

/// CF attribute: the time unit and reference, for example `days since 1950-01-01`.
pub const UNITS: &str = "units";
/// CF attribute: the calendar of the time values.
pub const CALENDAR: &str = "calendar";

pub(crate) const MICROS_PER_SECOND: i64 = 1_000_000;
pub(crate) const MICROS_PER_DAY: i64 = 86_400 * MICROS_PER_SECOND;

/// The decoded `Timestamp` type.
pub fn timestamp_type() -> DataType {
    DataType::Timestamp(TimeUnit::Microsecond, None)
}

/// A parsed CF time unit: microseconds per step and the reference instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CfTimeUnits {
    /// Microseconds per unit step.
    pub step_micros: i64,
    /// The reference instant, in microseconds since 1970-01-01.
    pub reference_micros: i64,
}

impl CfTimeUnits {
    /// Parse `"<unit> since <reference>"`, for example `"days since 1950-01-01"`.
    pub fn parse(units: &str) -> Result<Self> {
        let invalid = || nd_err!(InvalidAttribute, "invalid CF time units '{units}'");
        let (unit, reference) = units.split_once(" since ").ok_or_else(invalid)?;
        let step_micros = match unit.trim().to_ascii_lowercase().as_str() {
            "microseconds" | "microsecond" | "us" => 1,
            "milliseconds" | "millisecond" | "ms" | "msec" => 1_000,
            "seconds" | "second" | "secs" | "sec" | "s" => MICROS_PER_SECOND,
            "minutes" | "minute" | "mins" | "min" => 60 * MICROS_PER_SECOND,
            "hours" | "hour" | "hrs" | "hr" | "h" => 3_600 * MICROS_PER_SECOND,
            "days" | "day" | "d" => MICROS_PER_DAY,
            "weeks" | "week" => 7 * MICROS_PER_DAY,
            _ => return Err(invalid()),
        };
        let reference_micros = parse_datetime(reference.trim()).ok_or_else(invalid)?;
        Ok(Self {
            step_micros,
            reference_micros,
        })
    }
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's algorithm).
pub(crate) fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let yoe = year - era * 400;
    let month = month as i64;
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The proleptic Gregorian date of a day count since 1970-01-01.
pub(crate) fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// Parse `YYYY-M-D[( |T)h:m[:s[.frac]]][Z|UTC|±hh[:mm]]` to microseconds since the epoch.
fn parse_datetime(text: &str) -> Option<i64> {
    let text = text.trim();
    let (date, rest) = match text.find(['T', ' ']) {
        Some(i) => (&text[..i], text[i + 1..].trim()),
        None => (text, ""),
    };
    let negative_year = date.starts_with('-');
    let mut parts = date.trim_start_matches('-').splitn(3, '-');
    let year: i64 = parts.next()?.parse().ok()?;
    let year = if negative_year { -year } else { year };
    let month: u32 = parts.next().map_or(Some(1), |m| m.parse().ok())?;
    let day: u32 = parts.next().map_or(Some(1), |d| d.parse().ok())?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    let mut micros = days_from_civil(year, month, day) * MICROS_PER_DAY;
    let rest = rest.trim_end_matches("UTC").trim();
    if rest.is_empty() {
        return Some(micros);
    }
    let (time, zone) = match rest.find(['Z', '+', '-']) {
        Some(i) => (&rest[..i], rest[i..].trim()),
        None => (rest, ""),
    };
    let mut fields = time.trim().split(':');
    let hour: i64 = fields.next()?.trim().parse().ok()?;
    let minute: i64 = fields.next().map_or(Some(0), |m| m.parse().ok())?;
    let seconds: f64 = fields.next().map_or(Some(0.0), |s| s.parse().ok())?;
    micros += hour * 3_600 * MICROS_PER_SECOND
        + minute * 60 * MICROS_PER_SECOND
        + (seconds * MICROS_PER_SECOND as f64).round() as i64;

    if let Some(offset) = zone.strip_prefix(['+', '-']) {
        let sign = if zone.starts_with('-') { -1 } else { 1 };
        let digits: String = offset.chars().filter(char::is_ascii_digit).collect();
        let (h, m) = match digits.len() {
            1 | 2 => (digits.parse::<i64>().ok()?, 0),
            4 => (
                digits[..2].parse::<i64>().ok()?,
                digits[2..].parse::<i64>().ok()?,
            ),
            _ => return None,
        };
        micros -= sign * (h * 3_600 + m * 60) * MICROS_PER_SECOND;
    } else if !(zone.is_empty() || zone == "Z") {
        return None;
    }
    Some(micros)
}

fn check_calendar(attrs: &Attributes) -> Result<()> {
    match attrs
        .get_str(CALENDAR)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        None | Some("standard" | "gregorian" | "proleptic_gregorian") => Ok(()),
        Some(other) => Err(nd_err!(NotSupported, "CF calendar '{other}'")),
    }
}

/// Decode a CF time variable into `Timestamp(Microsecond)`.
///
/// Fill values become null first (see [`decode_cf`]). Fractional steps round
/// to the nearest microsecond. The `units` and `calendar` attributes move to
/// the returned attributes, for [`encode_cf_time`].
pub fn decode_cf_time(a: &NdArrowArray) -> Result<(NdArrowArray, Attributes)> {
    let units = a
        .attributes()
        .get_str(UNITS)
        .ok_or_else(|| nd_err!(InvalidAttribute, "time variable has no 'units' attribute"))?;
    let parsed = CfTimeUnits::parse(units)?;
    check_calendar(a.attributes())?;
    let (masked, _) = decode_cf(a)?;

    let numbers = arrow_cast(masked.values().as_ref(), &DataType::Float64)?;
    let numbers = numbers.as_primitive::<Float64Type>();
    let stamps: TimestampMicrosecondArray = numbers
        .iter()
        .map(|v| {
            v.map(|v| parsed.reference_micros + (v * parsed.step_micros as f64).round() as i64)
        })
        .collect();

    let mut kept = Attributes::new();
    let mut moved = Attributes::new();
    for (key, value) in masked.attributes().iter() {
        let target = if key == UNITS || key == CALENDAR {
            &mut moved
        } else {
            &mut kept
        };
        target.try_insert_array(key, value.clone())?;
    }
    Ok((
        NdArrowArray::try_new(Arc::new(stamps), a.dims().clone())?.with_attributes(kept),
        moved,
    ))
}

/// Encode timestamps as `Float64` numbers of `units`, the inverse of [`decode_cf_time`].
/// Adds the `units` attribute.
pub fn encode_cf_time(a: &NdArrowArray, units: &str) -> Result<NdArrowArray> {
    let parsed = CfTimeUnits::parse(units)?;
    let stamps = to_micros(a)?;
    let numbers: Float64Array = stamps
        .iter()
        .map(|v| v.map(|v| (v - parsed.reference_micros) as f64 / parsed.step_micros as f64))
        .collect();
    let attrs = a.attributes().clone().with(UNITS, units);
    Ok(NdArrowArray::try_new(Arc::new(numbers), a.dims().clone())?.with_attributes(attrs))
}

pub(crate) fn to_micros(a: &NdArrowArray) -> Result<TimestampMicrosecondArray> {
    if !matches!(
        a.data_type(),
        DataType::Timestamp(_, _) | DataType::Date32 | DataType::Date64
    ) {
        return Err(nd_err!(
            InvalidArgument,
            "expected timestamps, got {}",
            a.data_type()
        ));
    }
    let cast = arrow_cast(a.values().as_ref(), &timestamp_type())?;
    Ok(cast.as_primitive::<TimestampMicrosecondType>().clone())
}

/// Calendar periods for [`floor_time`] and [`Frequency`](crate::compute::Frequency).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimePeriod {
    /// The start of the year.
    Year,
    /// The start of the month.
    Month,
    /// The start of the day.
    Day,
    /// The start of the hour.
    Hour,
    /// The start of the minute.
    Minute,
    /// The start of the second.
    Second,
}

/// Round timestamps down to the start of their `period`, like pandas `floor`.
/// For regular bins with a reduction, see [`resample`](fn@crate::compute::resample).
pub fn floor_time(a: &NdArrowArray, period: TimePeriod) -> Result<NdArrowArray> {
    let stamps = to_micros(a)?;
    let fixed = |step: i64| move |v: i64| v.div_euclid(step) * step;
    let out: TimestampMicrosecondArray = match period {
        TimePeriod::Second => stamps.unary(fixed(MICROS_PER_SECOND)),
        TimePeriod::Minute => stamps.unary(fixed(60 * MICROS_PER_SECOND)),
        TimePeriod::Hour => stamps.unary(fixed(3_600 * MICROS_PER_SECOND)),
        TimePeriod::Day => stamps.unary(fixed(MICROS_PER_DAY)),
        TimePeriod::Month | TimePeriod::Year => stamps.unary(|v| {
            let (year, month, _) = civil_from_days(v.div_euclid(MICROS_PER_DAY));
            let month = if period == TimePeriod::Year { 1 } else { month };
            days_from_civil(year, month, 1) * MICROS_PER_DAY
        }),
    };
    Ok(NdArrowArray::try_new(Arc::new(out), a.dims().clone())?
        .with_attributes(a.attributes().clone()))
}

/// A calendar component of timestamps as `Int32`, for example the month.
pub fn time_component(a: &NdArrowArray, part: DatePart) -> Result<NdArrowArray> {
    let stamps: ArrayRef = Arc::new(to_micros(a)?);
    let out = date_part(stamps.as_ref(), part)?;
    NdArrowArray::try_new(out, a.dims().clone())
}

#[cfg(test)]
mod tests {
    use arrow::array::{Float32Array, Int32Array};
    use arrow::datatypes::Int32Type;

    use super::*;
    use crate::Dimensions;
    use crate::compute::cf::FILL_VALUE;

    fn micros(year: i64, month: u32, day: u32, hour: i64) -> i64 {
        days_from_civil(year, month, day) * MICROS_PER_DAY + hour * 3_600 * MICROS_PER_SECOND
    }

    #[test]
    fn civil_conversion_round_trips() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 3, 1), 11_017);
        for days in [-1_000_000, -719_468, -1, 0, 59, 60, 11_016, 1_000_000] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
        }
    }

    #[test]
    fn parses_cf_units() {
        let u = CfTimeUnits::parse("days since 1950-01-01").unwrap();
        assert_eq!(u.step_micros, MICROS_PER_DAY);
        assert_eq!(u.reference_micros, micros(1950, 1, 1, 0));
        let h = CfTimeUnits::parse("hours since 2000-1-1 06:30:00.5 +01:00").unwrap();
        assert_eq!(
            h.reference_micros,
            micros(2000, 1, 1, 5) + 30 * 60 * MICROS_PER_SECOND + 500_000
        );
        assert!(CfTimeUnits::parse("days after 1950-01-01").is_err());
        assert!(CfTimeUnits::parse("fortnights since 1950-01-01").is_err());
        assert!(CfTimeUnits::parse("seconds since 1970-01-01T00:00:00Z").is_ok());
    }

    #[test]
    fn decode_and_encode_round_trip() {
        let a = NdArrowArray::try_new(
            Arc::new(Float32Array::from(vec![0.0, 1.5, -999.0])),
            Dimensions::try_new([("time", 3)]).unwrap(),
        )
        .unwrap()
        .with_attributes(
            Attributes::new()
                .with(UNITS, "days since 1950-01-01")
                .with(CALENDAR, "standard")
                .with(FILL_VALUE, -999.0f32)
                .with("long_name", "time"),
        );
        let (decoded, encoding) = decode_cf_time(&a).unwrap();
        assert_eq!(decoded.data_type(), &timestamp_type());
        let v: Vec<Option<i64>> = decoded
            .values()
            .as_primitive::<TimestampMicrosecondType>()
            .iter()
            .collect();
        assert_eq!(
            v,
            vec![
                Some(micros(1950, 1, 1, 0)),
                Some(micros(1950, 1, 2, 12)),
                None
            ]
        );
        assert_eq!(decoded.attributes().get_str("long_name"), Some("time"));
        assert_eq!(encoding.get_str(UNITS), Some("days since 1950-01-01"));

        let encoded = encode_cf_time(&decoded, "days since 1950-01-01").unwrap();
        let back: Vec<Option<f64>> = encoded
            .values()
            .as_primitive::<Float64Type>()
            .iter()
            .collect();
        assert_eq!(back, vec![Some(0.0), Some(1.5), None]);
    }

    #[test]
    fn rejects_other_calendars() {
        let a = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![0])),
            Dimensions::try_new([("time", 1)]).unwrap(),
        )
        .unwrap()
        .with_attributes(
            Attributes::new()
                .with(UNITS, "days since 2000-01-01")
                .with(CALENDAR, "noleap"),
        );
        assert!(decode_cf_time(&a).is_err());
    }

    #[test]
    fn floor_and_components() {
        let stamps = TimestampMicrosecondArray::from(vec![
            micros(2021, 3, 15, 13),
            micros(1969, 12, 31, 23),
        ]);
        let a = NdArrowArray::try_new(
            Arc::new(stamps),
            Dimensions::try_new([("time", 2)]).unwrap(),
        )
        .unwrap();
        let floor = |p| -> Vec<i64> {
            floor_time(&a, p)
                .unwrap()
                .values()
                .as_primitive::<TimestampMicrosecondType>()
                .values()
                .to_vec()
        };
        assert_eq!(
            floor(TimePeriod::Month),
            vec![micros(2021, 3, 1, 0), micros(1969, 12, 1, 0)]
        );
        assert_eq!(
            floor(TimePeriod::Year),
            vec![micros(2021, 1, 1, 0), micros(1969, 1, 1, 0)]
        );
        assert_eq!(
            floor(TimePeriod::Day),
            vec![micros(2021, 3, 15, 0), micros(1969, 12, 31, 0)]
        );
        let month = time_component(&a, DatePart::Month).unwrap();
        assert_eq!(
            month.values().as_primitive::<Int32Type>().values(),
            &[3, 12]
        );
    }
}
