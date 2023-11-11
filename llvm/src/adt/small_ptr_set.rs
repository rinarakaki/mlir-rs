/*!
# 'Normally Small' Pointer Set

This file defines the SmallPtrSet class. See the doxygen comment for SmallPtrSetImplBase for more details on the algorithm used.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/SmallPtrSet.h>
*/

use core::{
    cmp::PartialEq,
    mem::size_of
};

use crate::llvm::{
    adt::epoch_tracker::increment_epoch,
    support::{
        compiler,
        reverse_iteration,
        type_traits
    }
};

/**
SmallPtrSetImplBase - This is the common code shared among all the
SmallPtrSet<>'s, which is almost everything. SmallPtrSet has two modes, one for small and one for large sets.

Small sets use an array of pointers allocated in the SmallPtrSet object, which is treated as a simple array of pointers.  When a pointer is added to the set, the array is scanned to see if the element already exists, if not the element is 'pushed back' onto the array.  If we run out of space in the array, we grow into the 'large set' case.  SmallSet should be used when the sets are often small.  In this case, no memory allocation is used, and only light-weight and cache-efficient scanning is used.

Large sets use a classic exponentially-probed hash table.  Empty buckets are represented with an illegal pointer value (-1) to allow null pointers to be inserted.  Tombstones are represented with another illegal pointer value (-2), to allow deletion.  The hash table is resized when the table is 3/4 or more.  When this happens, the table is doubled in size.
*/
pub struct SmallPtrSetImplBase {}

/**
This class implements a set which is optimized for holding
SmallSize or less elements.  This internally rounds up SmallSize to the next
power of two if it is not already a power of two.  See the comments above
SmallPtrSetImplBase for details of the algorithm.
*/
pub struct SmallPtrSet<Pointer, const SMALL_SIZE: usize> {
    _unimplemented: std::marker::PhantomData<Pointer>,

    // ====================================================================
    // SmallPtrSetImplBase
    // ====================================================================

    // const
    /// Points to a fixed size set of buckets, used in 'small mode'.
    small_array: [*mut (); SMALL_SIZE],

    // const
    /**
    Current set of buckets. If equal to small_array, then the set is in 'small mode'.
    */
    cur_array: [*mut (); SMALL_SIZE],

    /// The allocated size of cur_array, always a power of two.
    cur_array_size: usize,

    /**
    Number of elements in cur_array that contain a value or are a tombstone.
    If small, all these elements are at the beginning of cur_array and the rest
    is uninitialised.
    */
    num_non_empty: usize,
    
    /// Number of tombstones in cur_array.
    num_tombstones: usize
}

impl<Pointer, SMALL_SIZE> SmallPtrSet<Pointer, SMALL_SIZE> {
    // ====================================================================
    // SmallPtrSetImplBase
    // ====================================================================

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub const fn len(&self) -> usize {
        self.num_non_empty - self.num_tombstones
    }

    pub fn clear(&mut self) {
        increment_epoch();
        /*
        If the capacity of the array is huge, and the # elements used is small,
        shrink the array.
        */
        if !self.is_small() {
            if
                self.len() * 4 < self.cur_array_size
                && self.cur_array_size > 32
            {
                return self.shrink_and_clear();
            }
                
            // Fill the array with empty markers.
            memset(self.cur_array, -1, self.cur_array_size * size_of::<*mut ()>());
        }
    
        self.num_non_empty = 0;
        self.num_tombstones = 0;
    }

    fn tombstone_marker() -> *mut () {
        reinterpret_cast::<*mut ()>(-2)
    }

    fn empty_marker() -> *mut () {
        // Note that -1 is chosen to make clear() efficiently implementable with
        // memset and because it's not a valid pointer value.
        reinterpret_cast::<*mut ()>(-1)
    }

    const fn end_pointer(&self) -> Option<*mut ()> {
        if self.is_small() {
            self.cur_array + self.num_non_empty
        } else {
            self.cur_array + self.cur_array_size
        }
    }

    /**
    insert_imp - This returns true if the pointer was new to the set, false if it was already in the set.  This is hidden from the client so that the derived class can check that the right type of pointer is passed in.
    */
    fn insert(&mut self, ptr: *mut ()) -> (Option<*mut ()>, bool) {
        if self.is_small() {
            // Check to see if it is already in the set.
            let mut last_tombstone = None;
            let E = self.small_array + self.num_non_empty;
            for a_ptr in self.small_array..E {
                let value = *a_ptr;
                if value == ptr {
                    return (a_ptr, false);
                }
                
                if value == Self::tombstone_marker() {
                    last_tombstone = a_ptr;
                }
            }

            // Did we find any tombstone marker?
            if last_tombstone.is_some() {
                *last_tombstone = ptr;
                num_tombstones -= 1;
                increment_epoch();
                return (last_tombstone, true);
            }

            // Nope, there isn't.  If we stay small, just 'pushback' now.
            if self.num_non_empty < self.cur_array_size {
                self.small_array[self.num_non_empty] = ptr;
                self.num_non_empty += 1;
                increment_epoch();
                return (self.small_array + (self.num_non_empty - 1), true);
            }
            // Otherwise, hit the big set case, which will call grow.
        }
        return insert_imp_big(ptr);
    }

    /**
    erase_imp - If the set contains the specified pointer, remove it and
    return true, otherwise return false.  This is hidden from the client so
    that the derived class can check that the right type of pointer is passed
    in.
    */
    fn erase(&mut self, ptr: *mut ()) -> bool {
        let p = self.find(ptr);
        if p == self.end_pointer() {
            return false;
        }

        let loc = const_cast::<[*mut ()]>(p);
        assert!(*loc == ptr, "Broken find!");
        *loc = Self::tombstone_marker();
        self.num_tombstones += 1;
        true
    }

    /**
    Returns the raw pointer needed to construct an iterator.  If element not
    found, this will be self.end_pointer.  Otherwise, it will be a pointer to the slot which stores ptr;
    */
    const fn find(&self, ptr: *mut ()) -> Option<*mut ()> {
        if self.is_small() {
            // Linear search for the item.
            for a_ptr in self.small_array..(self.small_array + self.num_non_empty)
            {
                if *a_ptr == ptr {
                    return a_ptr;
                }
            }
            return self.end_pointer();
        }

        // Big set case.
        let bucket = self.find_bucket_for(ptr);
        if *bucket == ptr {
            return bucket;
        }
        return self.end_pointer();
    }

    const fn is_small(&self) -> bool {
        self.cur_array == self.small_array
    }

    fn insert_imp_big(&mut self, ptr: *mut ()) -> (Option<*mut ()>, bool) {

    }

    const fn find_bucket_for(&self, ptr: *mut ()) -> Option<*mut ()> {

    }
    
    fn shrink_and_clear(&mut self) {

    }

    /// grow - Allocate a larger backing store for the buckets and move it over.
    fn grow(&mut self, new_size: usize) {

    }

    /// swap - Swaps the elements of two sets.
    /// Note: This method assumes that both sets have the same small size.
    fn swap(&self, other: &Self) {
    }

    // void CopyFrom(const SmallPtrSetImplBase &RHS);
    // void MoveFrom(usize SmallSize, SmallPtrSetImplBase &&RHS);

    // /// Code shared by MoveFrom() and move constructor.
    // void MoveHelper(usize SmallSize, SmallPtrSetImplBase &&RHS);
    // /// Code shared by CopyFrom() and copy constructor.
    // void CopyHelper(const SmallPtrSetImplBase &RHS);
}
