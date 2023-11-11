/*!
# Pointer Traits 

This file defines the PointerLikeTypeTraits class.  This allows data structures to reason about pointers and other things that are pointer sized.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/PointerLikeTypeTraits.h>
*/
  
use crate::llvm::support::data_types;

/**
A traits type that is used to handle pointer types and things that are just wrappers for pointers as a uniform entity.
*/
pub struct PointerLikeTypeTraits<T> {
    _unimplemented: std::marker::PhantomData<T>
}
