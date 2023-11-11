/*!
This file declares the MLIR Lexer class.

- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/AsmParser/ParserState.h>
*/

use crate::{
    mlir::{
        asm_parser::{
            asm_parser_state::AsmParserState,
            code_complete::AsmParserCodeCompleteContext,
            lexer::Lexer,
            token::Token
        },
        ir::{
            attribute::Attribute,
            operation::implementation::{
                AsmDialectResourceHandle, OpAsmDialectInterface
            },
            r#type::Type
        }
    },
    llvm::adt::{
        dense_map::DenseMap,
        string_map::StringMap,
    }
};

/// This class contains record of any parsed top-level symbols.
pub struct SymbolState {
    /// A map from attribute alias identifier to Attribute.
    attribute_alias_definitions: StringMap<dyn Attribute>,

    /// A map from type alias identifier to Type.
    type_alias_definitions: StringMap<dyn Type>,

    /**
    A map of dialect resource keys to the resolved resource name and handle to use during parsing.
    */
    dialect_resources: DenseMap<
        Option<OpAsmDialectInterface>,
        StringMap<(String, AsmDialectResourceHandle)>
    >
}

/**
This class refers to all of the state maintained globally by the parser, such as the current lexer position etc.
*/
pub struct ParserState<'a> {
    /// The lexer for the source file we're parsing.
    lexer: Lexer<'a>,

    /// This is the next token that hasn't been consumed yet.
    current_token: Token,

    /// The current state for symbol parsing.
    symbols: &'a SymbolState,

    /// An optional pointer to a struct containing high level parser state to be
    /// populated during parsing.
    asm_state: Option<AsmParserState>,

    /// An optional code completion context.
    code_complete_context: Option<AsmParserCodeCompleteContext>,

    /**
    Contains the stack of default dialect to use when parsing regions.
    A new dialect get pushed to the stack before parsing regions nested
    under an operation implementing `OpAsmOpInterface`, and
    popped when done. At the top-level we start with "builtin" as the
    default, so that the top-level `module` operation parses as-is.
    */
    default_dialect_stack: Vec<&'static str>  // {"builtin"};
}

impl<'a> ParserState<'a> {
    // /// The configuration used to setup the parser.
    // const config: &'a ParserConfig;
}
