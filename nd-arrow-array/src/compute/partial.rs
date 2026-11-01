//! Reduction states that merge exactly across chunks.
//!
//! Reduce each chunk to a [`PartialReduction`], merge the partials in any
//! order or tree shape, then [`finish`](PartialReduction::finish). The result
//! equals one reduction over all chunks (up to float rounding). This is the
//! base for chunked and lazy execution.

use std::sync::Arc;

use arrow::array::{Array, AsArray, Float64Array};
use arrow::compute::cast;
use arrow::compute::kernels::{boolean, cmp, zip::zip};
use arrow::datatypes::{DataType, Float64Type, Int64Type};

use crate::NdArrowArray;
use crate::compute::elementwise::{add, logical_and, logical_or, mul};
use crate::compute::reduce::{Reduction, count, reduce};
use crate::dimensions::Dimensions;
use crate::error::{Result, nd_err};

#[derive(Debug, Clone)]
enum State {
    /// Sum, product, any or all: one array that merges with one kernel.
    Single(NdArrowArray),
    /// Minimum or maximum, with nulls for empty groups.
    Extreme(NdArrowArray),
    /// Sum and count, both `Float64`.
    Mean { sum: Vec<f64>, count: Vec<u64> },
    /// Count, mean and sum of squared deviations (Chan et al.).
    Moments {
        count: Vec<u64>,
        mean: Vec<f64>,
        m2: Vec<f64>,
    },
}

/// The mergeable state of one reduction over one chunk.
#[derive(Debug, Clone)]
pub struct PartialReduction {
    op: Reduction,
    dims: Dimensions,
    input_type: DataType,
    state: State,
}

impl PartialReduction {
    /// Reduce one chunk. Median, quantiles and arg reductions cannot merge.
    pub fn new(a: &NdArrowArray, axes: Option<&[&str]>, op: Reduction) -> Result<Self> {
        let f64_input = || crate::compute::cast(a, &DataType::Float64);
        let (state, dims) = match op {
            Reduction::Sum
            | Reduction::Prod
            | Reduction::Count
            | Reduction::Any
            | Reduction::All => {
                let r = reduce(a, axes, op)?;
                let dims = r.dims().clone();
                (State::Single(r), dims)
            }
            Reduction::Min | Reduction::Max => {
                let r = reduce(a, axes, op)?;
                let dims = r.dims().clone();
                (State::Extreme(r), dims)
            }
            Reduction::Mean => {
                let x = f64_input()?;
                let s = reduce(&x, axes, Reduction::Sum)?;
                let n = count(&x, axes)?;
                let dims = s.dims().clone();
                (
                    State::Mean {
                        sum: f64_values(&s),
                        count: u64_values(&n),
                    },
                    dims,
                )
            }
            Reduction::Var { .. } | Reduction::Std { .. } => {
                let x = f64_input()?;
                let n = count(&x, axes)?;
                let mean = reduce(&x, axes, Reduction::Mean)?;
                let var = reduce(&x, axes, Reduction::Var { ddof: 0 })?;
                let dims = n.dims().clone();
                let count = u64_values(&n);
                let m2 = f64_values(&var)
                    .iter()
                    .zip(&count)
                    .map(|(v, &c)| if c == 0 { 0.0 } else { v * c as f64 })
                    .collect();
                (
                    State::Moments {
                        count,
                        mean: f64_values(&mean),
                        m2,
                    },
                    dims,
                )
            }
            Reduction::Median | Reduction::Quantile(_) | Reduction::ArgMin | Reduction::ArgMax => {
                return Err(nd_err!(NotSupported, "{op:?} has no mergeable state"));
            }
        };
        Ok(Self {
            op,
            dims,
            input_type: a.data_type().clone(),
            state,
        })
    }

    pub fn dims(&self) -> &Dimensions {
        &self.dims
    }

    /// Combine with the partial of another chunk.
    pub fn merge(&self, other: &Self) -> Result<Self> {
        if self.op != other.op || self.dims != other.dims {
            return Err(nd_err!(
                InvalidArgument,
                "cannot merge {:?} over {} with {:?} over {}",
                self.op,
                self.dims,
                other.op,
                other.dims
            ));
        }
        let state = match (&self.state, &other.state) {
            (State::Single(a), State::Single(b)) => State::Single(match self.op {
                Reduction::Prod => mul(a, b)?,
                Reduction::Any => logical_or(a, b)?,
                Reduction::All => logical_and(a, b)?,
                _ => add(a, b)?,
            }),
            (State::Extreme(a), State::Extreme(b)) => {
                State::Extreme(merge_extreme(a, b, self.op == Reduction::Max)?)
            }
            (State::Mean { sum: sa, count: ca }, State::Mean { sum: sb, count: cb }) => {
                State::Mean {
                    sum: sa.iter().zip(sb).map(|(a, b)| a + b).collect(),
                    count: ca.iter().zip(cb).map(|(a, b)| a + b).collect(),
                }
            }
            (
                State::Moments {
                    count: na,
                    mean: ma,
                    m2: qa,
                },
                State::Moments {
                    count: nb,
                    mean: mb,
                    m2: qb,
                },
            ) => {
                let mut count = Vec::with_capacity(na.len());
                let mut mean = Vec::with_capacity(na.len());
                let mut m2 = Vec::with_capacity(na.len());
                for i in 0..na.len() {
                    let (n1, n2) = (na[i] as f64, nb[i] as f64);
                    let n = n1 + n2;
                    let (c, m, q) = if n2 == 0.0 {
                        (na[i], ma[i], qa[i])
                    } else if n1 == 0.0 {
                        (nb[i], mb[i], qb[i])
                    } else {
                        let delta = mb[i] - ma[i];
                        (
                            na[i] + nb[i],
                            ma[i] + delta * n2 / n,
                            qa[i] + qb[i] + delta * delta * n1 * n2 / n,
                        )
                    };
                    count.push(c);
                    mean.push(m);
                    m2.push(q);
                }
                State::Moments { count, mean, m2 }
            }
            _ => unreachable!("same op gives the same state"),
        };
        Ok(Self {
            op: self.op,
            dims: self.dims.clone(),
            input_type: self.input_type.clone(),
            state,
        })
    }

    /// The reduction result.
    pub fn finish(&self) -> Result<NdArrowArray> {
        let float_out = |values: Vec<Option<f64>>| -> Result<NdArrowArray> {
            let mut out: Arc<dyn Array> = Arc::new(Float64Array::from(values));
            if matches!(self.input_type, DataType::Float32 | DataType::Float16) {
                out = cast(out.as_ref(), &DataType::Float32)?;
            }
            NdArrowArray::try_new(out, self.dims.clone())
        };
        match &self.state {
            State::Single(a) | State::Extreme(a) => Ok(a.clone()),
            State::Mean { sum, count } => float_out(
                sum.iter()
                    .zip(count)
                    .map(|(&s, &n)| (n > 0).then(|| s / n as f64))
                    .collect(),
            ),
            State::Moments { count, m2, .. } => {
                let (ddof, sqrt) = match self.op {
                    Reduction::Var { ddof } => (ddof, false),
                    Reduction::Std { ddof } => (ddof, true),
                    _ => unreachable!(),
                };
                float_out(
                    count
                        .iter()
                        .zip(m2)
                        .map(|(&n, &q)| {
                            (n as usize > ddof).then(|| {
                                let v = q / (n as usize - ddof) as f64;
                                if sqrt { v.sqrt() } else { v }
                            })
                        })
                        .collect(),
                )
            }
        }
    }
}

/// Merge partials in a balanced tree, which keeps float error low.
pub fn merge_all(partials: &[PartialReduction]) -> Result<PartialReduction> {
    match partials {
        [] => Err(nd_err!(InvalidArgument, "no partial reductions to merge")),
        [single] => Ok(single.clone()),
        many => {
            let (left, right) = many.split_at(many.len() / 2);
            merge_all(left)?.merge(&merge_all(right)?)
        }
    }
}

fn f64_values(a: &NdArrowArray) -> Vec<f64> {
    let values = cast(a.values().as_ref(), &DataType::Float64).expect("numeric");
    values
        .as_primitive::<Float64Type>()
        .iter()
        .map(|v| v.unwrap_or(0.0))
        .collect()
}

fn u64_values(a: &NdArrowArray) -> Vec<u64> {
    a.values()
        .as_primitive::<Int64Type>()
        .values()
        .iter()
        .map(|&v| v as u64)
        .collect()
}

/// Elementwise minimum or maximum that skips nulls.
fn merge_extreme(a: &NdArrowArray, b: &NdArrowArray, is_max: bool) -> Result<NdArrowArray> {
    let (x, y) = (a.values(), b.values());
    let better = if is_max {
        cmp::gt(y, x)?
    } else {
        cmp::lt(y, x)?
    };
    let a_missing = arrow::compute::is_null(x.as_ref())?;
    let b_present = arrow::compute::is_not_null(y.as_ref())?;
    let take_b = boolean::or_kleene(&a_missing, &boolean::and_kleene(&b_present, &better)?)?;
    let values = zip(&take_b, y, x)?;
    NdArrowArray::try_new(values, a.dims().clone())
}

#[cfg(test)]
mod tests {
    use arrow::array::Float64Array;

    use super::*;
    use crate::compute::shape::{Selector, Slice, isel};

    fn data() -> NdArrowArray {
        let values: Vec<Option<f64>> = (0..60)
            .map(|i| (i % 7 != 3).then(|| ((i * 37) % 23) as f64 - 11.0))
            .collect();
        NdArrowArray::try_new(
            Arc::new(Float64Array::from(values)),
            Dimensions::try_new([("time", 12), ("x", 5)]).unwrap(),
        )
        .unwrap()
    }

    fn chunks(a: &NdArrowArray) -> Vec<NdArrowArray> {
        [(0, 5), (5, 6), (6, 12)]
            .iter()
            .map(|&(s, e)| isel(a, &[("time", Selector::Slice(Slice::range(s, e)))]).unwrap())
            .collect()
    }

    fn close(a: &NdArrowArray, b: &NdArrowArray) -> bool {
        let x = cast(a.values().as_ref(), &DataType::Float64).unwrap();
        let y = cast(b.values().as_ref(), &DataType::Float64).unwrap();
        let (x, y) = (
            x.as_primitive::<Float64Type>(),
            y.as_primitive::<Float64Type>(),
        );
        x.len() == y.len()
            && x.iter().zip(y.iter()).all(|(p, q)| match (p, q) {
                (None, None) => true,
                (Some(p), Some(q)) => (p - q).abs() < 1e-9,
                _ => false,
            })
    }

    #[test]
    fn merged_chunks_equal_full_reduction() {
        let a = data();
        let ops = [
            Reduction::Sum,
            Reduction::Prod,
            Reduction::Count,
            Reduction::Mean,
            Reduction::Min,
            Reduction::Max,
            Reduction::Var { ddof: 0 },
            Reduction::Std { ddof: 1 },
        ];
        for op in ops {
            for axes in [Some(&["time"][..]), None] {
                let partials: Vec<PartialReduction> = chunks(&a)
                    .iter()
                    .map(|c| PartialReduction::new(c, axes, op).unwrap())
                    .collect();
                let merged = merge_all(&partials).unwrap().finish().unwrap();
                let full = reduce(&a, axes, op).unwrap();
                assert!(close(&merged, &full), "{op:?} over {axes:?}");
            }
        }
    }

    #[test]
    fn unsupported_and_mismatched() {
        let a = data();
        assert!(PartialReduction::new(&a, None, Reduction::Median).is_err());
        let p = PartialReduction::new(&a, None, Reduction::Sum).unwrap();
        let q = PartialReduction::new(&a, Some(&["x"]), Reduction::Sum).unwrap();
        assert!(p.merge(&q).is_err());
    }
}
