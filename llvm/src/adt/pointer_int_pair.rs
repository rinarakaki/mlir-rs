/*!
# Pair for Pointer and Int

This file defines the PointerIntPair class.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/PointerIntPair.h>
*/

use core::{
    cmp::PartialEq,
    mem::transmute_copy
};

use crate::llvm::support::{
    compiler,
    pointer_like_type_traits::PointerLikeTypeTraits,
    type_traits
};

/**
This class implements a pair of a pointer and small integer. It is designed to represent this in the space required by one pointer by bitmangling the integer into the low part of the pointer. This can only be done for small integers: typically up to 3 bits, but it depends on the number of bits available according to `PointerLikeTypeTraits` for the type.

Note that PointerIntPair always puts the IntVal part in the highest bits possible. For example, PointerIntPair<void*, 1, bool> will put the bit for the bool into bit #2, not bit #0, which allows the low two bits to be used for something else. For example, this allows:

```cpp
PointerIntPair<PointerIntPair<void*, 1, bool>, 1, bool>
```

and the two bools will land in different bits.
*/
pub struct PointerIntPair<Pointer, const INT_BITS: usize, Int = usize> {
    value: isize, // = 0;
    _unimplemented: std::marker::PhantomData<Pointer>,
    _unimplemented1: std::marker::PhantomData<Int>,
}

impl<Pointer, const INT_BITS: usize, Int>
    PointerIntPair<Pointer, INT_BITS, Int>
{
    type PtrTraits = PointerLikeTypeTraits<Pointer>;
    type Info = PointerIntPairInfo<Pointer, INT_BITS, PtrTraits>;

    /// The bits that come from the pointer.
    const POINTER_BIT_MASK: usize
        = !(usize)(((isize)1 << PtrTraits::NumLowBitsAvailable) - 1);

    pub const fn pointer(&self) -> Pointer {
        PtrTraits::from_void_pointer(
            reinterpret_cast::<*mut ()>(self.value & POINTER_BIT_MASK));
    }

    // pub const fn int(&self) -> Int {
    //     Info::get_int(self.value) as Int
    // }
}

impl<Pointer, const INT_BITS: usize, Int, PtrTraits>
    PartialEq for PointerIntPair<Pointer, INT_BITS, Int, PtrTraits> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }

    fn ne(&self, other: &Self) -> bool {
        self.value != other.value
    }
}

pub struct PointerIntPairInfo<Pointer, const INT_BITS: usize, PtrTraits> {
    _unimplemented: std::marker::PhantomData<Pointer>,
    _unimplemented1: std::marker::PhantomData<PtrTraits>
}

impl<Pointer, const INT_BITS: usize, PtrTraits>
    PointerIntPairInfo<Pointer, INT_BITS, PtrTraits>
{
    // /// PointerBitMask - The bits that come from the pointer.
    // const POINTER_BIT_MASK: usize =
    //     !((1isize << PtrTraits::NumLowBitsAvailable) - 1);

    // /// IntShift - The number of low bits that we reserve for other uses, and
    // /// keep zero.
    // const INT_SHIFT: usize = PtrTraits::NumLowBitsAvailable - INT_BITS;

    // /// IntMask - This is the unshifted mask for valid bits of the int type.
    // const INT_MASK: usize = (1usize << INT_BITS) - 1;


    // pub fn get_pointer(value: isize) -> Pointer {
    //     unsafe {
    //         let value = Self::POINTER_BIT_MASK;  // TODO delete
    //         PtrTraits::get_from_void_pointer(
    //             transmute_copy(value & Self::POINTER_BIT_MASK))
            
    //     }
    // }

    // pub fn get_int(value: isize) -> isize {
    //     (value >> Self::INT_SHIFT) & Self::INT_MASK
    // }
}

#[repr(usize)]
enum MaskAndShiftConstants {
    /// The bits that come from the pointer.
    PointerBitMask,

    /// The number of low bits that we reserve for other uses, and keep zero.
    IntShift,

    /// This is the unshifted mask for valid bits of the int type.
    IntMask,

    /// This is the bits for the integer shifted in place.
    ShiftedIntMask
}

struct DenseMapInfo<T, Enable> {
    _unimplemented: std::marker::PhantomData<T>,
    _unimplemented1: std::marker::PhantomData<Enable>
}
