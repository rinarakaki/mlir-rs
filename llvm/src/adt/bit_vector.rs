//! # Bit Vectors
//! 
//! This file implements the BitVector class.
//! 
//! - include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/BitVector.h>

use crate::{
    adt::{
        array_ref,
        dense_map_info,
        iterator_range,
        small_vector::SmallVector
    },
    support::math_extras::{
        count_population, count_trailing_zeros, mask_trailing_ones
    }
};

pub struct BitVector {
    /// Actual bits.
    bits: SmallVector<[usize]>,
    /// Size of bitvector in bits.
    len: usize,  // = 0; 
}

const BITWORD_SIZE: usize = core::mem::size_of::<usize> * 8;

impl BitVector {
    /// Tests whether there are no bits in this bitvector.
    pub const fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns the number of bits in this bitvector.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns the number of bits which are set.
    pub const fn count(&self) -> usize {
        let mut output = 0;
        for bit in self.bits {
            output += count_population(bit);
        }
        output
    }

    /// find_first_in - Returns the index of the first set/unset bit, depending on `set`, in the range [begin, end).
    /// Returns -1 if all bits in the range are unset/set.
    pub const fn find_first_in(
        &self, begin: usize, end: usize, set: bool, // = true
    ) -> isize {
        assert!(begin <= end && end <= self.len);
        if begin == end {
            return -1;
        }

        let first_word = begin / BITWORD_SIZE;
        let last_word = (end - 1) / BITWORD_SIZE;

        // Check subsequent words.
        // The code below is based on search for the first _set_ bit. If we're searching for the first _unset_, we just take the complement of each word before we use it and apply the same method.
        for  i in first_word..=last_word {
            let copy = self.bits[i];
            if !set {
                copy = !copy;
            }

            if i == first_word {
                let first_bit = begin % BITWORD_SIZE;
                copy &= mask_trailing_zeros::<usize>(first_bit);
            }

            if i == last_word {
                let last_bit = (end - 1) % BITWORD_SIZE;
                copy &= mask_trailing_ones::<usize>(last_bit + 1);
            }
            if copy != 0 {
                return i * BITWORD_SIZE + count_trailing_zeros(copy);
            }
        }
        return -1;
    }

    /**
    Returns the index of the last set bit in the range [begin, end). Returns -1 if all bits in the range are unset.
    */
    pub const fn find_last_in(
        &self, begin: usize, end: usize
    ) -> isize
    {
        assert!(begin <= end && end <= self.len);
        if begin == end {
            return -1;
        }

        let last_word = (end - 1) / BITWORD_SIZE;
        let first_word = begin / BITWORD_SIZE;

        for i in (first_word..=last_word).rev() {
            let current_word = i - 1;

            let copy = self.bits[current_word];
            if current_word == last_word {
                let last_bit = (end - 1) % BITWORD_SIZE;
                copy &= mask_trailing_ones::<usize>(last_bit + 1);
            }

            if current_word == first_word {
                let first_bit = begin % BITWORD_SIZE;
                copy &= mask_trailing_zeros::<usize>(first_bit);
            }

            if copy != 0 {
                return (current_word + 1) * BITWORD_SIZE
                    - count_leading_zeros(copy) - 1;
            }
        }

        return -1;
    }

    /**
    Returns the index of the first set bit, -1 if none of the bits are set.
    */
    pub const fn find_first(&self) -> isize {
        self.find_first_in(0, self.len, true)
    }

    /**
    Returns the index of the last set bit, -1 if none of the bits are set.
    */
    pub const fn find_last(&self) -> isize {
        self.find_last_in(0, self.len)
    }
}
