/*!
This file declares the MLIR Lexer class.

- lib
  - <https://github.com/llvm/llvm-project/blob/main/mlir/lib/AsmParser/Parser.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/lib/AsmParser/Parser.cpp>
*/

use crate::{
    mlir::{
        asm_parser::{
            asm_parser,
            asm_parser_impl,
            asm_parser_state,
            code_complete,
            parser_state::ParserState
        },
        ir::{
            affine_map,
            builders,
            builtins::operations,
            dialect,
            operation::implementation,
            verifier,
        }
    },
    llvm::{
        adt::{
            bit,
            dense_map,
            scope_exit,
            string_set
        },
        support::{
            endian,
            pretty_stack_trace,
            source_manager
        }
    }
};

/**
This class implement support for parsing global entities like attributes and types. It is intended to be subclassed by specialised subparsers that include state.
*/
pub struct Parser<'a> {
    /**
    The Parser is subclassed and reinstantiated. Do not add additional non-trivial state here, add it to the ParserState class.
    */
    state: &'a ParserState<'a>
}
