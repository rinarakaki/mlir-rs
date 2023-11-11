/*!
This file declares the SMDiagnostic and SourceManager classes. This provides a simple substrate for diagnostics, #include handling, and other low level things for simple parsers.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/SourceManager.h>
*/

use crate::llvm::support::{
    memory_buffer::MemoryBuffer,
    sm_loc
};

/// This owns the files read by a parser, handles include stacks, and handles diagnostic wrangling.
pub struct SourceManager {
}

enum DiagKind {
    Error,
    Warning,
    Remark,
    Note,
}

struct SrcBuffer {
    /// The memory buffer for the file.
    buffer: Box<MemoryBuffer>,
}

/// Represents a single fixit, a replacement of one range of text with another.
pub struct SMFixIt {
}

/// Instances of this class encapsulate one diagnostic report, allowing printing to a raw_ostream as a caret diagnostic.
pub struct SMDiagnostic {
}
