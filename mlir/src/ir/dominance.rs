//! The DominanceInfo and PostDominanceInfo class provide routines for performimg simple dominance checks, and expose dominator trees for advanced clients.
//! These classes provide fully region-aware functionality, lazily constructing dominator information for any multi-block regions that need it.
//!
//! For more information about the theory behind dominance in graphs algorithms, see: <https://en.wikipedia.org/wiki/Dominator_(graph_theory)>
//!
//! Implementation of dominance related classes and instantiations of extern templates.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Dominance.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/Dominance.cpp>

use std::cell::Cell;
use llvm::adt::{
    dense_map::DenseMap,
    pointer_int_pair::PointerIntPair
};
use crate::ir::region::Region;

pub struct DominanceInfoBase {
    //! A mapping of regions to their base dominator tree and a cached "hasSSADominance" bit. This map does not contain dominator trees for single block CFG regions, but we do want to cache the "hasSSADominance" bit for them. We may also not have computed the DomTree yet. In either case, the DomTree is just null.
    dominance_infos: Cell<
        DenseMap<Option<Region>, PointerIntPair<Option<DomTree>, 1, bool>>>
}

//! A class for computing basic dominance information. Note that this
//! class is aware of different types of regions and returns a
//! region-kind specific concept of dominance. See RegionKindInterface.
pub struct DominanceInfo {
}
