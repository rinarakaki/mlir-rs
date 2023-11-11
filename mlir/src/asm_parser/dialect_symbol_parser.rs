/*!
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/AsmParser/DialectSymbolParser.cpp>
*/

use crate::{
    mlir::{
        asm_parser::asm_parser_impl,
        ir::{
            builtins::types,
            dialect::{
                self,
                implementation
            },
        }
    },
    llvm::support::source_manager
};

/**
This class provides the main implementation of the DialectAsmParser that allows for dialects to parse attributes and types. This allows for dialect hooking into the main MLIR parsing logic.
*/
pub struct CustomDialectAsmParser {
    /// The full symbol specification.
    full_spec: &'static str
}
