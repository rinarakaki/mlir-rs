/*!
This file is contains the interface to the MLIR assembly parser library.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/AsmParser/AsmParser.h>
*/

use crate::{
    mlir::{
        asm_parser::{
            asm_parser_state::AsmParserState,
            code_complete::AsmParserCodeCompleteContext
        },
        ir::{
            asm_state::ParserConfig,
            block::Block
        }
    },
    llvm::support::source_manager::SourceManager
};

/**
This parses the file specified by the indicated `SourceManager` and appends parsed operations to the given block. If the block is non-empty, the operations are placed before the current terminator. If parsing is successful, success is returned. Otherwise, an error message is emitted through the error handler registered in the context, and failure is returned. If `sourceFileLoc` is non-null, it is populated with a file location representing the start of the source file that is being parsed. If `asmState` is non-null, it is populated with detailed information about the parsed IR (including exact locations for SSA uses and definitions). `asmState` should only be provided if this detailed information is desired. If `codeCompleteContext` is non-null, it is used to signal tracking of a code completion event (generally only ever useful for LSP or other high level language tooling).
*/
pub fn parse_asm_source_file(
    source_manager: &SourceManager,
    block: *mut Block,
    config: &ParserConfig,
    asm_state: Option<AsmParserState>, // = nullptr,
    code_complete_context: Option<AsmParserCodeCompleteContext> //= nullptr
) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

/**
This parses a single MLIR attribute to an MLIR context if it was valid. If not, an error message is emitted through a new SourceManagerDiagnosticHandler constructed from a new SourceManager with a single a MemoryBuffer wrapping
`attrStr`. If the passed `attrStr` has additional tokens that were not part
of the type, an error is emitted.
*/
// TODO: Improve diagnostic reporting.
Attribute parseAttribute(llvm::StringRef attrStr, MLIRContext *context);

/**
This parses a single MLIR type to an MLIR context if it was valid.  If not,
an error message is emitted through a new SourceManagerDiagnosticHandler
constructed from a new SourceManager with a single a MemoryBuffer wrapping
`typeStr`. If the passed `typeStr` has additional tokens that were not part
of the type, an error is emitted.
*/
// TODO: Improve diagnostic reporting.
Type parseType(llvm::StringRef typeStr, MLIRContext *context);


/**
This parses a single IntegerSet/AffineMap to an MLIR context if it was
valid. If not, an error message is emitted through a new
SourceManagerDiagnosticHandler constructed from a new SourceManager with a single
MemoryBuffer wrapping `str`. If the passed `str` has additional tokens that
were not part of the IntegerSet/AffineMap, a failure is returned.
*/
AffineMap parseAffineMap(llvm::StringRef str, MLIRContext *context);
