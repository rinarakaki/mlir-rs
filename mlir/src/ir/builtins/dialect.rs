/*!
# MLIR Builtin Dialect

This file contains the definition of the Builtin dialect. This dialect contains all of the attributes, operations, and types that are core to MLIR.

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BuiltinDialect.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BuiltinDialect.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/BuiltinDialect.cpp>
*/

pub mod bytecode;

use crate::{
    mlir::ir::{
        block_and_value_mapping,
        builders,
        builtins::{
            // dialect::bytecode,
            operations,
            types
        },
        dialect::Dialect,
        dialect_resource_blob_manager,
        operation::implementation,
        pattern_match,
        type_range
    }
};

/// A dialect containing the builtin Attributes, Operations, and Types
pub trait BuiltinDialect: Dialect {
}
