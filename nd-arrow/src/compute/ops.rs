//! Operator overloads: `&a + &b` calls [`add`](crate::compute::add) and returns a `Result`.

use std::ops;

use crate::NdArrowArray;
use crate::compute::elementwise;
use crate::error::Result;

macro_rules! binary_operator {
    ($($trait:ident :: $method:ident => $kernel:path),* $(,)?) => {$(
        impl ops::$trait<&NdArrowArray> for &NdArrowArray {
            type Output = Result<NdArrowArray>;

            fn $method(self, rhs: &NdArrowArray) -> Self::Output {
                $kernel(self, rhs)
            }
        }
    )*};
}

binary_operator!(
    Add::add => elementwise::add,
    Sub::sub => elementwise::sub,
    Mul::mul => elementwise::mul,
    Div::div => elementwise::div,
    Rem::rem => elementwise::rem,
    BitAnd::bitand => elementwise::logical_and,
    BitOr::bitor => elementwise::logical_or,
    BitXor::bitxor => elementwise::logical_xor,
);

impl ops::Neg for &NdArrowArray {
    type Output = Result<NdArrowArray>;

    fn neg(self) -> Self::Output {
        elementwise::neg(self)
    }
}

impl ops::Not for &NdArrowArray {
    type Output = Result<NdArrowArray>;

    fn not(self) -> Self::Output {
        elementwise::logical_not(self)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{AsArray, Int32Array};
    use arrow::datatypes::Int32Type;

    use crate::{Dimensions, IntoScalar, NdArrowArray};

    #[test]
    fn operators_call_kernels() {
        let a = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Dimensions::try_new([("x", 3)]).unwrap(),
        )
        .unwrap();
        let two = NdArrowArray::try_new_scalar(2i32.into_scalar()).unwrap();
        let b = (&(&a * &two).unwrap() - &a).unwrap();
        assert_eq!(b.values().as_primitive::<Int32Type>().values(), &[1, 2, 3]);
        let n = (-&a).unwrap();
        assert_eq!(
            n.values().as_primitive::<Int32Type>().values(),
            &[-1, -2, -3]
        );
    }
}
