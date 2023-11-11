//! llvm::GraphTraits for CFGs
//!
//! This file implements specialisations of llvm::GraphTraits for various MLIR CFG data types. This allows the generic LLVM graph algorithms to be applied to CFGs.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/RegionGraphTraits.h>

use crate::{
    mlir::ir::{
        block::Block,
        region::Region
    },
    llvm::adt::graph_traits
};
