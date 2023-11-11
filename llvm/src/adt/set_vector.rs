/*!
# Set with Insert Order Iteration

This file implements a set that has insertion order iteration characteristics. This is useful for keeping a set of things that need to be visited later but in a deterministic order (insertion order). The interface is purposefully minimal.

This file defines SetVector and SmallSetVector, which performs no
allocations if the SetVector has less than a certain number of elements.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/SetVector.h>
*/

use crate::llvm::{
    adt::{
        dense_set::{DenseSet, SmallDenseSet},
        stl_extras
    },
    support::compiler
};

/**
A vector that has set insertion semantics.

This adapter class provides a way to keep a set of things that also has the
property of a deterministic iteration order. The order of iteration is the
order of insertion.
*/
pub struct SetVector<T, Vector = Vec<T>, Set = DenseSet<T>> {
    _unimplemented: std::marker::PhantomData<T>,
    /// The set.
    set: set_type,
    /// The vector.
    vector: vector_type   
}

/**
A SetVector that performs no allocations if smaller than a certain size.
*/
pub type SmallSetVector<T, const N: usize>
    = SetVector<T, SmallVector<[T; N]>, SmallDenseSet<T, N>>;
