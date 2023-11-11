/*!
This file defines the MemoryBuffer interface.
This file implements the MemoryBufferRef interface.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/MemoryBufferRef.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/MemoryBufferRef.cpp>
*/

use crate::llvm::{
    support::memory_buffer::MemoryBuffer
};

pub struct MemoryBufferRef {
    buffer: &'static str,
    identifier: &'static str
}
