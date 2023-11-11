/*!
This file contains types to represent alignments.
They are instrumented to guarantee some invariants are preserved and prevent
invalid manipulations.

- Align represents an alignment in bytes, it is always set and always a valid
power of two, its minimum value is 1 which means no alignment requirements.

- MaybeAlign is an optional type, it may be undefined or set. When it's set
you can get the underlying Align type by using the getValue() method.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/Alignment.h>
*/

use crate::llvm::support::math_extras;

/**
This struct is a compact representation of a valid (non-zero power of two) alignment.
It is suitable for use as static global constants.
*/
pub struct Alignment {
    /// The log2 of the required alignment.
    /// ShiftValue is less than 64 by construction.
    shift_value: u8  // = 0;
}

/// Returns a multiple of A needed to store `Size` bytes.
#[inline]
pub fn align_to(size: u64, alignment: Alignment) -> u64 {
    let value = alignment.value();
    // The following line is equivalent to `(Size + Value - 1) / Value * Value`.
  
    // The division followed by a multiplication can be thought of as a right
    // shift followed by a left shift which zeros out the extra bits produced in
    // the bump; `~(Value - 1)` is a mask where all those bits being zeroed out
    // are just zero.
  
    // Most compilers can generate this code but the pattern may be missed when
    // multiple functions gets inlined.
    (size + value - 1) & !(value - 1usize)
}

/**
A trivial type to allow construction of constexpr Align.
This is currently needed to workaround a bug in GCC 5.3 which prevents
definition of constexpr assign operators.
<https://stackoverflow.com/questions/46756288/explicitly-defaulted-function-cannot-be-declared-as-constexpr-because-the-implic>
FIXME: Remove this, make all assign operators constexpr and introduce user
defined literals when we don't have to support GCC 5.3 anymore.
<https://llvm.org/docs/GettingStarted.html#getting-a-modern-host-c-toolchain>
*/
struct LogValue {
    log: u8
}
