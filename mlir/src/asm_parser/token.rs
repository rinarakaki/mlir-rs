/*!
This file declares the MLIR Lexer class.

- lib
  - <https://github.com/llvm/llvm-project/blob/main/mlir/lib/AsmParser/Token.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/lib/AsmParser/Token.cpp>
*/

use crate::{
    mlir::support::llvm,
    llvm::{
        adt::{
            // optional,
            string_extras,
        },
        support::sm_loc
    }
};

/// This represents a token in the MLIR syntax.
pub struct Token {
    /// Discriminator that indicates the sort of token this is.
    kind: TokenKind,

    /**
    A reference to the entire token contents; this is always a pointer into a memory buffer owned by the source manager.
    */
    spelling: &'static str
}

#[derive(Clone, Copy)]
pub enum TokenKind {
  // #define TOK_MARKER(NAME) NAME,
  // #define TOK_IDENTIFIER(NAME) NAME,
  // #define TOK_LITERAL(NAME) NAME,
  // #define TOK_PUNCTUATION(NAME, SPELLING) NAME,
  // #define TOK_KEYWORD(SPELLING) kw_##SPELLING,
  // #include "TokenKinds.def"
}

impl Token {
    /// Return the bytes that make up this token.
    pub const fn spelling(&self) -> &'static str {
        self.spelling
    }

    // Token classification.
    pub const fn kind(&self) -> TokenKind {
        self.kind
    }
    
    pub const fn is(&self, kind: TokenKind) -> bool {
        match self.kind {
            kind => true,
            _ => false
        }
    }
}
