//! sInterfaces for Symbol Operation
//!
//! This file contains a set of interfaces and traits that can be used to define properties of symbol and symbol table operations.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/SymbolInterfaces.td>

use crate::mlir::{
    ir::symbol_table::SymbolTableCollection,
    support::logical_result::LogicalResult
};

/// This interface describes an operation that may define a `Symbol`. A `Symbol` operation resides immediately within a region that defines a `SymbolTable`.
/// See [Symbols and SymbolTables](../SymbolsAndSymbolTables.md) for more details and constraints on `Symbol` operations.
pub trait Symbol: OperationInterface {
}

/// This interface describes an operation that may use a `Symbol`. This interface allows for users of symbols to hook into verification and other symbol related utilities that are either costly or otherwise disallowed within a traditional operation.
pub trait SymbolUserOpInterface: OperationInterface {
    /// Verify the symbol uses held by this operation.
    fn verify_symbol_uses(&self, symbol_table: SymbolTableCollection) -> LogicalResult;
}

// /// Op defines a symbol table.
// pub trait SymbolTable : NativeOpTrait<"SymbolTable"> {
  
// }
