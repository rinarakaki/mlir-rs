//! This file defines utilities for walking and visiting operations.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Visitors.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/Visitors.cpp>

#[allow(non_camel_case_types)]
type int = i32;

use crate::{
    mlir::{
        ir::{
            operation,
            visitors
        },
        support::{
            llvm,
            logical_result
        }
    },
    llvm::adt::stl_extras
};

/// A utility result that is used to signal how to proceed with an ongoing walk:
///
/// - Interrupt: the walk will be interrupted and no more operations, regions
/// or blocks will be visited.
/// - Advance: the walk will continue.
/// - Skip: the walk of the current operation, region or block and their
/// nested elements that haven't been visited already will be skipped and will
/// continue with the next operation, region or block.
pub enum WalkResult {
    Interrupt,
    Advance,
    Skip
}

impl WalkResult {
    
}

/// Traversal order for region, block and operation walk utilities.
pub enum WalkOrder {
    PreOrder,
    PostOrder
}

/// A utility class to encode the current walk stage for "generic" walkers.
/// When walking an operation, we can either choose a Pre/Post order walker
/// which invokes the callback on an operation before/after all its attached
/// regions have been visited, or choose a "generic" walker where the callback
/// is invoked on the operation N+1 times where N is the number of regions
/// attached to that operation. The `WalkStage` class below encodes the current
/// stage of the walk, i.e., which regions have already been visited, and the
/// callback accepts an additional argument for the current stage. Such
/// generic walkers that accept stage-aware callbacks are only applicable when
/// the callback operates on an operation (i.e., not applicable for callbacks
/// on Blocks or Regions).
pub struct WalkStage {
    // const int numRegions;
    next_region: int
}
