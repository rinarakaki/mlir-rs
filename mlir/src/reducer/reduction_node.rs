/*!
This file defines the reduction nodes which are used to track of the metadata for a specific generated variant within a reduction pass and are the building blocks of the reduction tree structure. A reduction tree is used to keep track of the different generated variants throughout a reduction pass in the MLIR Reduce tool.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Reducer/ReductionNode.h>
*/

#[allow(non_camel_case_types)]
type int = i32;

use std::collections::VecDeque;

use crate::{
    mlir::{
        ir::{
            builtins::operations::Module,
            owning_op_ref::OwningOpRef,
            region::Region
        },
        reducer::tester::Interestingness,
    },
    llvm::{
        support::{
            allocator::SpecificBumpPtrAllocator,
            tool_output_file
        }
    }
};

/// Defines the traversal method options to be used in the reduction tree
/// traversal.
enum TraversalMode {
    SinglePath,
    Backtrack,
    MultiPath
}

/**
ReductionTreePass will build a reduction tree during module reduction and the ReductionNode represents the vertex of the tree. A ReductionNode records the information such as the reduced module, how this node is reduced from the parent node, etc. This information will be used to construct a reduction path to reduce the certain module.
*/
pub struct ReductionNode<'a> {
    /// This is a copy of module from parent node. All the reducer patterns will
    /// be applied to this instance.
    module: OwningOpRef<Module>,

    /// The region of certain operation we're reducing in the module
    region: *mut Region,  // = nullptr;

    /// The node we are reduced from. It means we will be in variants of parent node.
    parent: Option<Box<ReductionNode<'a>>>,  // = nullptr;

    /// The size of module after applying the reducer patterns with range constraints. This is only valid while the interestingness has been tested.
    size: usize, // = 0;

    /// This is true if the module has been evaluated and it exhibits the
    /// interesting behaviour.
    interesting: Interestingness,  //= Tester::Interestingness::Untested;

    /**
    `ranges` represents the selected subset of operations in the region. We implicitly number each operation in the region and ReductionTreePass will apply reducer patterns on the operation falls into the `ranges`. We will generate new ReductionNode with subset of `ranges` to see if we can do further reduction. we may split the element in the `ranges` so that we can have more subset variants from `ranges`.
    Note that after applying the reducer patterns the number of operation in the region may have changed, we need to update the `ranges` after that.
    */
    ranges: Vec<Range>,

    /**
    `startRanges` records the ranges of operations selected from the parent node to produce this ReductionNode. It can be used to construct the reduction path from the root. I.e., if we apply the same reducer patterns and `startRanges` selection on the parent region, we will get the same module as this node.
    */
    start_ranges: Vec<Range>,

    /// This points to the child variants that were created using this node as a starting point.
    variants: Vec<Option<ReductionNode<'a>>>,

    allocator: &'a SpecificBumpPtrAllocator<ReductionNode<'a>>
}

type Range = (int, int);

/**
A custom BFS iterator. The difference between llvm/ADT/BreadthFirstIterator.h is the graph we're exploring is dynamic.
We may explore more neighbors at certain node if we didn't find interested event. As a result, we defer pushing adjacent nodes until poping the last visited node. The graph exploration strategy will be put in getNeighbors().

Subclass BaseIterator and implement traversal strategy in getNeighbors().
*/
struct BaseIterator<'a, T> {
    visit_queue: VecDeque<Option<ReductionNode<'a>>>
}
