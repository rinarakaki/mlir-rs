/*!
This file defines the AlignedCharArrayUnion class.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/AlignOf.h>
*/

use std::mem::{align_of, size_of};

/**
A suitably aligned and sized character array member which can hold elements
of any type.

This template is equivalent to std::aligned_union_t<1, ...>, but we cannot
use it due to a bug in the MSVC x86 compiler:
<https://github.com/microsoft/STL/issues/1533>
Using `alignas` here works around the bug.
*/
pub struct AlignedCharArrayUnion<T, U> {
    // #[repr(align(align_of_union::<T, U>()))]
    // buffer: [char; size_of_union::<T, U>()]
    _marker: std::marker::PhantomData<T>,
    _marker1: std::marker::PhantomData<U>,
}

// const fn align_of_union<T, U>() -> usize {
//     [1, align_of::<T>(), align_of::<U>()].iter().max()
// }

// const fn size_of_union<T, U>() -> usize {
//     [1, size_of::<T>(), size_of::<U>()].iter().max()
// }
