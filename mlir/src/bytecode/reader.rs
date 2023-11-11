/*!
This header defines interfaces to read MLIR bytecode files/streams.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Bytecode/BytecodeReader.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Bytecode/Reader/BytecodeReader.cpp>
*/

use crate::{
    mlir::{
        asm_parser::asm_parser,
        bytecode::{
            encoding,
            // implementation
        },
        ir::{
            asm_state::ParserConfig,
            block::Block,
            builtins::{
                dialect,
                operations
            },
            operation::implementation,
            verifier
        },
        support::llvm,
    },
    llvm::{
        adt::{
            map_vector,
            scope_exit,
            small_string,
            string_extras
        },
        support::{
            memory_buffer_ref::MemoryBufferRef,
            save_and_restore,
            source_manager::SourceManager
        }
    }
};

/**
Returns true if the given buffer starts with the magic bytes that signal MLIR bytecode.
*/
pub fn is_bytecode(buffer: MemoryBufferRef) -> bool {
    buffer.buffer().starts_with(r"ML\xefR")
}

/**
Read the operations defined within the given memory buffer, containing MLIR bytecode, into the provided block.
*/
pub fn read_bytecode_file(
    buffer: MemoryBufferRef, block: Option<Block>, config: &ParserConfig
) -> Result<(), Box<dyn std::error::Error>>
{
    Err(())
}

/**
An overload with a source manager whose main file buffer is used for parsing. The lifetime of the source manager may be freely extended during parsing such that the source manager is not destroyed before the parsed IR.
*/
pub fn read_bytecode_file(
    source_manager: &Box<SourceManager>, block: Option<Block>, config: &ParserConfig
) -> Result<(), Box<dyn std::error::Error>> {
    Err(())
}
