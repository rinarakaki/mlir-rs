//! Useful math functions
//! 
//! - include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/MathExtras.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/MathExtras.cpp>

use crate::{
    adt::bit,
    support::compiler
};

/// Create a bitmask with the N right-most bits set to 1, and all other
/// bits set to 0.  Only unsigned types are allowed.
pub fn mask_trailing_ones<T>(n: u32) -> T {
    static_assert(std::is_unsigned_v<T>, "Invalid type!");
    let bits: u32 = CHAR_BIT * sizeof::<T>();
    assert!(n <= bits, "Invalid bit index");
    if n == 0 { 0 } else { (T(-1) >> (bits - n)) }
}
