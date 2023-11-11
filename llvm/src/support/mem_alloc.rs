/*!
This file defines counterparts of C library allocation functions defined in
the namespace 'std'. The new allocation functions crash on allocation
failure instead of returning null pointer.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/MemAlloc.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/MemAlloc.cpp>
*/

type VoidP = *mut u8;

use std::alloc::{alloc, dealloc, Layout};

use crate::llvm::support::{
    compiler,
    error_handling
};

/**
Allocate a buffer of memory with the given size and alignment.

When the compiler supports aligned operator new, this will use it to
handle even over-aligned allocations.

However, this doesn't make any attempt to leverage the fancier techniques
like posix_memalign due to portability. It is mostly intended to allow
compatibility with platforms that, after aligned allocation was added, use
reduced default alignment.
*/
// LLVM_ATTRIBUTE_RETURNS_NONNULL LLVM_ATTRIBUTE_RETURNS_NOALIAS
pub fn allocate_buffer(size: usize, alignment: usize) -> VoidP {
    let layout = Layout::from_size_align(size, alignment).unwrap();
    unsafe { alloc(layout) }
}

/**
Deallocate a buffer of memory with the given size and alignment.

If supported, this will used the sized delete operator. Also if supported,
this will pass the alignment to the delete operator.

The pointer must have been allocated with the corresponding new operator,
most likely using the above helper.
*/
pub fn deallocate_buffer(ptr: VoidP, size: usize, alignment: usize) {
    let layout = Layout::from_size_align(size, alignment).unwrap();
    unsafe { dealloc(ptr, layout); }
}
