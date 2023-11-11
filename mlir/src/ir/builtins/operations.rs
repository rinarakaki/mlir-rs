//! # MLIR Builtin Operations
//!
//! This file contains the Builtin dialect's operations.
//!
//! Defines the set of builtin MLIR operations, or the set of operations necessary for the validity of and defining the IR.
//!
//! - include
//!   - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BuiltinOps.h>
//!   - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BuiltinOps.td>

use llvm::support::pointer_like_type_traits;
use crate::{
    ir::{
        operation::implementation,
        owning_op_ref,
        region_kind_interface,
        symbol_table
    },
    interfaces::{
        cast_interfaces,
        data_layout_interfaces,
        side_effect_interfaces
    }
};

/// A top level container operation.
///
/// A `module` represents a top-level container operation. It contains a single [graph region](../LangRef.md#control-flow-and-ssacfg-regions) containing a single block which can contain any operations and does not have a terminator. Operations within this region cannot implicitly capture values defined outside the module, i.e. Modules are [IsolatedFromAbove](../Traits.md#isolatedfromabove). Modules have an optional [symbol name](../SymbolsAndSymbolTables.md) which can be used to refer to them in operations.
///
/// Example:
///
/// ```mlir
/// module {
///     func.func @foo()
/// }
/// ```
pub struct Module {

}
