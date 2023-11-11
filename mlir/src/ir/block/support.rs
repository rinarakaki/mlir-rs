/*!
This file defines a number of support types for the Block class.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BlockSupport.h>
*/

use crate::{
    mlir::ir::{
        use_def_lists::IROperand,
        value,
    },
    llvm::adt::{
        pointer_union,
        ilist,
        ilist_node
    }
};


/**
A block operand represents an operand that holds a reference to a Block, e.g. for terminator operations.
*/
pub struct BlockOperand {

}

/**
Implement a predecessor iterator for blocks. This works by walking the use lists of the blocks. The entries on this list are the BlockOperands that are embedded into terminator operations. From the operand, we can get the terminator that contains it, and its parent block is the predecessor.
*/
pub struct PredecessorIterator {

}

/// This class implements the successor iterators for Block.
pub struct SuccessorRange {}

/**
This class provides an abstraction over the different types of ranges over Blocks. In many cases, this prevents the need to explicitly materialise a SmallVector/std::vector. This class should be used in places that are not suitable for a more derived type (e.g. ArrayRef) or a template range parameter.
*/
pub struct BlockRange {

}
