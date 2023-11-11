/*!
This file defines the StringMap class.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/StringMap.h>
*/

use crate::llvm::{
    adt::{
        string_map_entry,
        iterator
    },
    support::{
        allocator_base::MallocAllocator,
        pointer_like_type_traits
    }
};

/**
StringMap - This is an unconventional map that is specialised for handling keys that are 'strings', which are basically ranges of bytes. This does some funky memory allocation and hashing things to make it extremely efficient, storing the string data *after* the value in the map.
*/
// pub struct StringMap<V, Allocator = MallocAllocator> {

// }
pub type StringMap<V, Allocator = MallocAllocator>
    = alloc::collections::HashMap<String, V, Allocator>;
