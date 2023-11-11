/*!
- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/ArrayRef.h>
- manual <https://llvm.org/docs/ProgrammersManual.html#dss-arrayref>
*/

use std::ops::Index;

use crate::llvm::{
    adt::{
        hashing,
        small_vector,
        stl_extras
    },
    support::compiler
};

pub type ArrayRef<'a, T> = &'a [T];
pub type MutableArrayRef<'a, T> = &'a mut [T];

// /**
// ArrayRef - Represent a constant reference to an array (0 or more elements consecutively in memory), i.e. a start pointer and a length. It allows various APIs to take consecutive elements easily and conveniently.

// This class does not own the underlying data, it is expected to be used in situations where the data resides in some other buffer, whose lifetime extends past that of the ArrayRef. For this reason, it is not in general safe to store an ArrayRef.

// This is intended to be trivially copyable, so it should be passed by value.
// */
// #[derive(Default)]
// #[must_use]
// pub struct ArrayRef<T> {
//     /// The start of the array, in an external buffer.
//     data: Option<T>,

//     /// The number of elements.
//     length: usize

// }

// /**
// MutableArrayRef - Represent a mutable reference to an array (0 or more elements consecutively in memory), i.e. a start pointer and a length. It allows various APIs to take and modify consecutive elements easily and conveniently.

// This class does not own the underlying data, it is expected to be used in
// situations where the data resides in some other buffer, whose lifetime
// extends past that of the MutableArrayRef. For this reason, it is not in
// general safe to store a MutableArrayRef.

// This is intended to be trivially copyable, so it should be passed by
// value.
// */
// #[must_use]
// pub struct MutableArrayRef<T> {
//     _unimplemented: std::marker::PhantomData<T>,
// }

// impl<T> MutableArrayRef<T> {
//     pub const fn data(&self) -> Option<T> {
//         const_cast<Option<T>>(ArrayRef<T>::data())
//     }

//     /// front - Get the first element.
//     pub const fn front(&self) -> &T {
//         assert!(!self.is_empty());
//         self.data()[0]
//     }

//     /// back - Get the last element.
//     pub const fn back(&self) -> &T {
//         assert!(!self.is_empty());
//         self.data()[self.len() - 1];
//     }

//     /// Drop the first `n` elements of the array.
//     pub const fn drop_front(&self, n: usize) -> Self {
//         let len = self.len();
//         assert!(len >= n, "Dropping more elements than exist");
//         self.slice(n, len - n)
//     }
    
//     pub const fn drop_back(&self, n: usize) -> Self {
//         let len = self.len();
//         assert!(len >= n, "Dropping more elements than exist");
//         self.slice(0, len - n)
//     }

//     /// Return a copy of *this with the first N elements satisfying the
//     /// given predicate removed.
//     pub const fn drop_while<P>(&mut self, predicate: P) -> Self {
//         return Self(self.find_if_not(*self, predicate), end());
//     }

//     /// Return a copy of *this with the first N elements not satisfying
//     /// the given predicate removed.
//     pub const fn drop_until<P>(&mut self, predicate: P) -> Self {
//         return Self(self.find_if(*self, predicate), end());
//     }

//     /// Return a copy of *this with only the first \p N elements.
//     pub const fn take_front(&self, n: usize) -> Self {
//         if n >= self.len() {
//             return *self;
//         }
//         self.drop_back(self.len() - n)
//     }

//     /// Return a copy of *this with only the last \p N elements.
//     pub const fn take_back(&self, n: usize) -> Self {
//         if n >= self.len() {
//             return *self;
//         }
//         return self.drop_front(self.len() - n);
//     }

//     /// Return the first N elements of this Array that satisfy the given
//     /// predicate.
//     pub const fn take_while<P>(&mut self, predicate: P) -> Self {
//         return Self(begin(), self.find_if_not(*self, predicate));
//     }

//     /// Return the first N elements of this Array that don't satisfy the
//     /// given predicate.
//     pub const fn take_until<P>(&mut self, predicate: P) -> Self {
//         return Self(begin(), self.find_if(*self, predicate));
//     }
// }

// impl<T> Index<usize> for MutableArrayRef<T> {
//     type Output = T;

//     fn index(&self, index: usize) -> &Self::Output {
//         assert!(index < self.len(), "Invalid index!");
//         self.data()[index]
//     }
// }
