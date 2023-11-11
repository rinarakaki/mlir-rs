/*!
# Arbitrary Precision Floating Point

This file declares a class to represent arbitrary precision floating point values and provide a variety of arithmetic operations on them.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/APFloat.h>
*/

pub use rustc_apfloat::*;
pub use rustc_apfloat::ieee::*;
pub use rustc_apfloat::ieee::{
    SingleS as IEEEsingle
};

// Float8E5M2,
// Float8E4M3FNType => ap_float::Float8E4M3FN,
// Float16Type => ap_float::BFloat,
// BFloat16Type => ap_float::IEEEhalf,
// Float64Type => IEEEdouble,
// Float80Type => x87DoubleExtended,
// Float128Type => IEEEquad
