//! Region Kind Interfaces
//!
//! This file contains the definitions of the infer op interfaces defined in `RegionKindInterface.td`.
//!
//! - include
//!   - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/RegionKindInterface.h>
//!   - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/RegionKindInterface.td>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/RegionKindInterface.cpp>

use crate::mlir::ir::operation::base::OpInterface;

/// The kinds of regions contained in an operation. SSACFG regions require the SSA-Dominance property to hold. Graph regions do not require SSA-Dominance. If a registered operation does not implement RegionKindInterface, then any regions it contains are assumed to be SSACFG regions.
pub enum RegionKind {
    SSACFG,
    Graph,
}

/// Interface for operations to describe the abstract semantics of their regions. Currently, two kinds of regions are supported. RegionKind::Graph represents a graph region without control flow semantics. RegionKind::SSACFG represents an [SSA-style control flow](../LangRef.md/#modeling-control-flow) region with basic blocks, sequential semantics, and reachability.
// OpInterface to query the properties of regions in an operation
pub trait RegionKindInterface: OpInterface {
    /// Return the kind of the region with the given index inside this operation.
    fn region_kind(index: usize) -> RegionKind;

    /// Return true if the kind of the given region requires the SSA-Dominance property
    fn has_ssa_dominance(index: usize) -> bool {
    }
}
