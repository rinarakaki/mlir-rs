/*!
This file defines the BumpPtrAllocator interface. BumpPtrAllocator conforms
to the LLVM "Allocator" concept and is similar to MallocAllocator, but
objects cannot be deallocated. Their lifetime is tied to the lifetime of the
allocator.

This file implements the BumpPtrAllocator interface.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/Allocator.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/Allocator.cpp>
*/

type VoidPOut = std::ptr::NonNull<[u8]>;
type VoidPIn = std::ptr::NonNull<u8>;

use crate::llvm::{
    adt::small_vector::SmallVector,
    support::{
        alignment,
        allocator_base::MallocAllocator,
        compiler,
        math_extras,
        raw_ostream
    }
};

/**
Allocate memory in an ever growing pool, as if by bump-pointer.

This isn't strictly a bump-pointer allocator as it uses backing slabs of
memory rather than relying on a boundless contiguous heap. However, it has
bump-pointer semantics in that it is a monotonically growing pool of memory
where every allocation is found by merely allocating the next N bytes in
the slab, or the next N bytes in the next slab.

Note that this also has a threshold for forcing allocations above a certain
size into their own slab.

The BumpPtrAllocatorImpl template defaults to using a MallocAllocator
object, which wraps malloc, to allocate memory, but it can be changed to
use a custom allocator.

The GrowthDelay specifies after how many allocated slabs the allocator
increases the size of the slabs.
*/
pub struct BumpPtrAllocator<
    Allocator = MallocAllocator,
    // const SLAB_SIZE: usize = 4096,
    // const SIZE_THRESHOLD: usize = SLAB_SIZE,
    // const GROWTH_DELAY: usize = 128
> {
    _marker: std::marker::PhantomData<Allocator>,
    /**
    The current pointer into the current slab.

    This points to the next free byte in the slab.
    */
    cur_ptr: *mut u8,

    /// The end of the current slab.
    end: *mut u8,

    /// The slabs allocated so far.
    slabs: SmallVector<VoidPIn>,

    /// Custom-sized slabs allocated for too-large allocation requests.
    custom_sized_slabs: SmallVector<(VoidPIn, usize)>,

    /**
    How many bytes we've allocated.

    Used so that we can compute how much space was wasted.
    */
    bytes_allocated: usize,

    /**
    The number of bytes to put between allocations when running under
    a sanitiser.
    */
    red_zone_size: usize
}

impl<Allocator> Default for BumpPtrAllocator<Allocator> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData::default(),
            cur_ptr: None,  // char * = nullptr,
            end: None,  // char * = nullptr,
            slabs: SmallVector::new(4),
            custom_sized_slabs: SmallVector::new(0),
            bytes_allocated: 0,
            red_zone_size: 1
        }
    }
}

/**
A BumpPtrAllocator that allows only elements of a specific type to be allocated.

This allows calling the destructor in DestroyAll() and when the allocator is destroyed.
*/
pub struct SpecificBumpPtrAllocator<T> {
    allocator: BumpPtrAllocator
}
