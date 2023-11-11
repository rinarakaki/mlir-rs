//!  This header defines interfaces to write MLIR bytecode files/streams.
//! 
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Bytecode/BytecodeWriter.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Bytecode/Writer/BytecodeWriter.cpp>

pub mod ir_numbering;

use llvm::{
    adt::{
        cached_hash_string,
        map_vector,
        small_string
    },
    support::{
        debug,
        raw_ostream::RawOStream
    }
};
use crate::{
    bytecode::{
        encoding,
        implementation
    },
    ir::{
        asm_state,
        builtins::dialect,
        operation::{
            Operation,
            // implementation
        },
    }
};

/// This class contains the configuration used for the bytecode writer. It
/// controls various aspects of bytecode generation, and contains all of the
/// various bytecode writer hooks.
pub struct BytecodeWriterConfig {
}

/// Write the bytecode for the given operation to the provided output stream.
/// For streams where it matters, the given stream should be in "binary" mode.
fn write_bytecode_to_file(op: Option<Operation>, os: &mut RawOStream,
    config: Option<&BytecodeWriterConfig>) {
}
