//! This file defines a utility class for maintaining a mapping for multiple value types.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BlockAndValueMapping.h>

use crate::{
    mlir::ir::{
        block::Block,
        value::Value
    },
    llvm::adt::dense_map::DenseMap
};

// This is a utility class for mapping one set of values to another. New mappings can be inserted via 'map'. Existing mappings can be found via the 'lookup*' functions. There are two variants that differ only in return value when an existing is not found for the provided key.
// 'lookupOrNull' returns nullptr where as 'lookupOrDefault' will return the lookup key.
pub struct BlockAndValueMapping {
    value_map: DenseMap<Value, Value>,
    block_map: DenseMap<*mut Block, *mut Block>
}
