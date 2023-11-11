/*!
This file defines MallocAllocator. MallocAllocator conforms to the LLVM
`Allocator` concept which consists of an Allocate method accepting a size
and alignment, and a Deallocate accepting a pointer and size. Further, the
LLVM "Allocator" concept has overloads of Allocate and Deallocate for
setting size and alignment based on the final type. These overloads are
typically provided by a base class template `AllocatorBase`.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/AllocatorBase.h>
*/

type VoidP = *mut u8;

use crate::llvm::support::{
    compiler,
    mem_alloc::{allocate_buffer, deallocate_buffer}
};

/**
CRTP base class providing obvious overloads for the core
`Allocate()` methods of LLVM-style allocators.

This base class both documents the full public interface exposed by all
LLVM-style allocators, and redirects all of the overloads to a single core
set of methods which the derived class must define.
*/
pub trait Allocator {
    /**
    Allocate `size` bytes of `alignment` aligned memory. This method
    must be implemented by `Derived`.
    */
    fn allocate(&self, size: usize, alignment: usize) -> VoidP;

    /**
    Deallocate `ptr` to `size` bytes of memory allocated by this allocator.
    */
    fn deallocate(&self, ptr: VoidP, size: usize, alignment: usize);
}

pub struct MallocAllocator;

impl MallocAllocator {
    // LLVM_ATTRIBUTE_RETURNS_NONNULL
    pub fn allocate(&self, size: usize, alignment: usize) -> VoidP {
        allocate_buffer(size, alignment)
    }

    pub fn deallocate(&self, ptr: VoidP, size: usize, alignment: usize) {
        deallocate_buffer(ptr, size, alignment);
    }
}
