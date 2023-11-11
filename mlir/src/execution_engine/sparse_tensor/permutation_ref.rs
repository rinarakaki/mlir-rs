/*!
# Permutation reference wrapper

This header is not part of the public API.  It is placed in the includes directory only because that's required by the implementations of template-classes.

This file is part of the lightweight runtime support library for sparse tensor manipulations.  The functionality of the support library is meant to simplify benchmarking, testing, and debugging MLIR code operating on sparse tensors.  However, the provided functionality is **not** part of core MLIR itself.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/ExecutionEngine/SparseTensor/PermutationRef.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/ExecutionEngine/SparseTensor/PermutationRef.cpp>
*/

use std::ops::Index;

/// Checks whether the `perm` array is a permutation of `[0 .. size)`.
// MLIR_SPARSETENSOR_PURE
pub fn is_permutation<const N: usize>(perm: [u64; N]) -> bool {
    // TODO: If we ever depend on LLVMSupport, then use `llvm::BitVector` instead.
    let len = perm.len();
    let seen = Vec::with_capacity(len);
    seen.fill(false);
    for i in 0..len {
        let j = perm[i];
        if j >= len || seen[j] {
            return false;
        }
        seen[j] = true;
    }
    true
}

/*
TODO: To implement things like `inverse` and `compose` while preserving
the knowledge that `is_permutation` is true, we'll need to also have
an owning version of `PermutationRef`.  (Though ideally we'll really
want to defunctionalize those methods so that we can avoid intermediate
arrays/copies and only materialize the data on request.)
*/

/**
A non-owning class for capturing the knowledge that `is_permutation` is true, to avoid needing to assert it repeatedly.
*/
// MLIR_SPARSETENSOR_GSL_POINTER
#[must_use]
pub type PermutationRef<const N: usize> = [u64; N];

impl<const N: usize> PermutationRef<N> {
    // NOTE: This form of the method is required by `to_mlir_sparse_tensor`,
    // so it can reuse the `out` buffer for each iteration of a loop.
    #[inline]
    pub const fn push_forward<T>(
        &mut self,
        values: &[T; N],
        out: &mut [T; N]
    ) {
        for i in 0..self.len() {
            out[self[i]] = values[i];
        }
    }

    /*
    NOTE: this is only needed by `to_mlir_sparse_tensor`, which in turn only needs it as a vector to hand off to `new_sparse_tensor`.
    Otherwise we would want the result to be an owning-permutation, to retain the knowledge that `isPermutation` is true.
    */
    /**
    Constructs the inverse permutation. This is equivalent to calling `push_forward` with `std::iota` for the values.
    */
    pub const fn inverse(&self) -> Vec<u64> {
        let len = self.len();
        let out = Vec::with_capacity(len);
        for i in 0..len {
            out[self[i]] = i;
        }
        out
    }

    #[inline]
    pub const fn permute<T>(
        &mut self,
        values: &[T; N],
        out: &mut [T; N]
    ) {
        for i in 0..self.len() {
            out[i] = values[self[i]];
        }
    }
}
