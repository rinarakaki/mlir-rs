/*!
This file implements a parser for Affine structures.

- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/AsmParser/AffineParser.cpp>
*/

#[allow(non_camel_case_types)]
type unsigned = u16;

use crate::{
    mlir::{
        asm_parser::parser,
        ir::{
            affine_expr::AffineExpr,
            affine_map,
            integer_set
        },
        support::logical_result::LogicalResult
    },
    llvm::support::source_manager
};

/// Lower precedence ops (all at the same precedence level). LNoOp is false in
/// the boolean sense.
enum AffineLowPrecOp {
    /// Null value.
    LNoOp,
    Add,
    Sub
}

/// Higher precedence ops - all at the same precedence level. HNoOp is false
/// in the boolean sense.
enum AffineHighPrecOp {
    /// Null value.
    HNoOp,
    Mul,
    FloorDiv,
    CeilDiv,
    Mod
}

/**
This is a specialised parser for affine structures (affine maps, affine expressions, and integer sets), maintaining the state transient to their bodies.
*/
pub struct AffineParser {
    allow_parsing_ssa_ids: bool,
    parse_element: fn(bool) -> LogicalResult,  // function_ref
    num_dim_operands: unsigned,  // = 0;
    num_symbol_operands: unsigned, //= 0;
    dims_and_symbols: Vec<(&'static str, AffineExpr)>
}
