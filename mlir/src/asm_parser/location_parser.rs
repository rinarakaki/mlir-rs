/*!
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/AsmParser/LocationParser.cpp>
*/

use crate::mlir::{
    asm_parser::{
        parser::Parser,
        token::Token
    },
    ir::location::{Location, LocationAttribute},
    support::logical_result::LogicalResult
};

impl<'a> Parser<'a> {
    /// Parse a raw location instance.
    pub fn parse_location_instance(&self, location: &Location) -> LogicalResult {
        LogicalResult {}  // unimlemented
    }

    /// Parse a callsite location instance.
    pub fn parse_call_site_location(&self, location: &Location) -> LogicalResult {
        LogicalResult {}  // unimlemented
    }
  
    /// Parse a fused location instance.
    pub fn parse_fused_location(&self, location: &Location) -> LogicalResult {
        LogicalResult {} // unimlemented
    }
  
    /// Parse a name or FileLineCol location instance.
    pub fn parse_name_or_file_line_col_location(&self, location: &Location) -> LogicalResult {
        LogicalResult {}  // unimlemented
    }
}
