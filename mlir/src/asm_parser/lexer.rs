/*!
This file declares the MLIR Lexer class.

- lib
  - <https://github.com/llvm/llvm-project/blob/main/mlir/lib/AsmParser/Lexer.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/lib/AsmParser/Lexer.cpp>
*/

use crate::{
    mlir::{
        asm_parser::{
            asm_parser,
            code_complete,
            token::Token
        },
        ir::{
            diagnostics,
            location,
            mlir_context::MLIRContext
        }
    },
    llvm::{
        adt::{
            string_extras,
            string_switch
        },
        support::source_manager::SourceManager
    }
};

/// This class breaks up the current file into a token stream.
pub struct Lexer<'a> {
    source_manager: &'a SourceManager,
    context: Option<MLIRContext>,
    current_buffer: &'a str,
    current_ptr: Option<char>,
    /**
    An optional code completion point within the input file, used to indicate the position of a code completion token.
    */
    code_complete_loc: Option<char>
}

impl<'a> Lexer<'a> {
    pub fn get_source_manager(&self) -> &SourceManager {
        &self.source_manager
    }

    /**
    Change the position of the lexer cursor.  The next token we lex will start
    at the designated point in the input.
    */
    pub fn reset_pointer(&mut self, new_pointer: char) {
        self.cur_ptr = Some(new_pointer);
    }

    /**
    Return the code completion location of the lexer, or nullptr if there is
    none.
    */
    pub const fn get_code_complete_loc(&self) -> Option<char> {
        self.code_complete_loc
    }
}
