//! MLIR Block Class
//!
//! This file defines the Block class.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Block.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/Block.cpp>

pub mod support;

use core::{
    ptr::null,
    slice::IterMut
};
use rustc_data_structures::graph::{DirectedGraph, With};
use rustc_index::vec::Idx;
use llvm::adt::{
    bit_vector,
    ilist::{IList, IListIterator},
    iterator_range::IteratorRange,
    pointer_int_pair::PointerIntPair
};
use crate::ir::{
    block::support::Successors,
    builders,
    location::Location,
    operation::Operation,
    region::Region,
    type_range::{TypeRange, ValueTypeRange},
    r#type::Type,
    value::Input,
    visitors
};

//! [`Block`] represents an ordered list of [`Operation`]s.
#[derive(Copy, Debug, Eq, PartialEq, Hash)]
pub struct Block {
    //! Pair of the parent object that owns this block and a bit that signifies if the operations within this block have a valid ordering.
    parent_valid_op_order_pair: PointerIntPair<*mut Region, 1, bool>,

    //! This is the list of operations in the block.
    operations: IList<Operation>,

    //! This is the list of inputs to the block.
    inputs: Vec<Input>
}

impl Block {
    pub fn clear(&mut self) {
        // Drop all references from within this block.
        self.drop_all_references();
    
        // Clear operations in the reverse order so that uses are destroyed before their defs.
        while !self.is_empty() {
           self.operations.pop_back();
        }
    }

    //! Provide a 'parent' method for ilist_node_with_parent methods.
    //! We mark it as a const function because ilist_node_with_parent specifically requires a 'parent() const' method. Once ilist_node removes this constraint, we should drop the const to fit the rest of the MLIR const model.
    pub const fn parent(&self) -> *mut Region {
        self.parent_valid_op_order_pair.pointer()
    }

    //! Returns the closest surrounding operation that contains this block.
    pub fn parent_operation(&self) -> *mut Operation {
        if self.parent().is_null() {
            return null();
        }
        unsafe {
            (*self.parent()).parent_operation()
        }
    }

    //! Returns if this block is the entry block in the parent region.
    pub fn is_entry_block(&self) -> bool {
        self == self.parent().front()
    }

    //! Insert this block (which must not already be in a region) right before the specified block.
    pub fn insert_before(&self, block: *mut Block) {
        assert!(self.parent().is_null(), "Already inserted into a block!");
        assert!(
            !block.parent().is_null(),
            "Cannot insert before a block without a parent");
        block.parent().blocks().insert(block.iterator(), self);
    }

    //! Unlink this block from its current region and insert it right before the specific block.
    pub fn move_before(&self, block: *mut Block) {
        assert!(
            !block.parent().is_null(),
            "Cannot insert before a block without a parent");
        block.parent().blocks().splice(
            block.iterator(), self.parent().blocks(), self.iterator());
    }

    //! Unlink this Block from its parent region and delete it.
    pub fn erase(&self) {
        assert!(!self.parent().is_null(), "Block has no parent");
        self.parent().block().erase(self);
    }

    // --------------------------------------------------------------------
    // Block Inputs Management
    // --------------------------------------------------------------------

    pub fn inputs(&self) -> &mut [Input] {
        self.inputs.as_mut_slice()
    }

    //! Return a range containing the types of the inputs for this block.
    pub fn input_types(&self) -> ValueTypeRange<&mut [Input]> {
        ValueTypeRange::<&mut [Input]>::new(self.inputs())
    }

    pub fn inputs_empty(&self) -> bool {

    }

    //! Add one value to the input list.
    pub fn add_input(&mut self, r#type: impl Type, location: Location)
    -> Input
    {
        let input = Input::create(r#type, self, self.inputs.len(), location);
        self.inputs.push(input);
        input
    }

    // /**
    // Insert one value to the position in the input list indicated by the
    // given iterator. The existing inputs are shifted. The block is expected
    // not to have predecessors.
    // */
    // pub fn insert_input() {}

    //! Add one input to the input list for each type specified in the list.
    //! `locations` is required to have the same number of elements as `types`.
    pub fn add_inputs(
        &mut self,
        types: TypeRange,
        locations: &[Location]
    ) -> IterMut<Input>
    {
        assert!(types.len() == locations.len(),
                "Incorrect number of block argument locations");
        let initial_size = self.inputs.len();
        self.inputs.reserve(initial_size + types.len());

        for (r#type, location) in llvm::zip(types, locations) {
            self.add_input(r#type, location);
        }
        self.inputs.iter_mut()[initial_size..self.inputs.len()]
    }

    //! Add one value to the input list at the specified position.
    pub fn insert_input(
        &mut self, index: usize, r#type: impl Type, location: Location
    ) -> Input
    {
        assert!(index <= self.inputs.len(), "Invalid insertion index");

        let input = Input::create(r#type, self, index, location);
        self.inputs.insert(self.inputs.begin() + index, self.inputs);
        // Update the cached position for all the inputs after the newly inserted one.
        index += 1;
        for input in self.inputs.iter().skip(index) {
            input.index = index;
            index += 1;
        }
        input
    }

    //! Erase the input at `index` and remove it from the input list.
    pub fn erase_input(&mut self, index: usize) {
        assert!(index < self.inputs.len());
        drop(self.inputs[index]);
        self.inputs.erase(index);
        for input in self.inputs.iter().skip(index) {
            input.index = index;
            index += 1;
        }
    }

    //! Erases `n` inputs from the index `start`.
    pub fn erase_inputs(&mut self, start: usize, n: usize) {
        assert!(start + n <= self.inputs.len());
        for i in 0..n {
            drop(self.inputs[start + i]);
        }
        self.inputs.drain(start..(start + n));
        for input in self.inputs.iter().skip(start) {
            input.input = start;
            start += 1;
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    pub fn input(&self, index: usize) -> Input {
        self.inputs[index]
    }

    // --------------------------------------------------------------------
    // Operation List Management
    // --------------------------------------------------------------------

    pub fn operations(&self) -> &IList<Operation> {
        &self.operations
    }

    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    pub fn push_back(&mut self, operation: *mut Operation) {
        self.operations.push_back(operation);
    }

    pub fn push_front(&mut self, operation: *mut Operation) {
        self.operations.push_front(operation);
    }

    pub fn back(&self) -> &Operation {
        self.operations.back()
    }

    pub fn front(&self) -> &Operation {
        self.operations.front()
    }

    //! Returns 'op' if 'op' lies in this block, or otherwise finds the ancestor operation of 'op' that lies in this block. Returns nullptr if the latter fails.
    //! TODO: This is very specific functionality that should live somewhere else, probably in Dominance.cpp.
    pub fn find_ancestor_op_in_block(&self, operation: &Operation)
        -> *mut Operation
    {
        // Traverse up the operation hierarchy starting from the owner of operand to find the ancestor operation that resides in the block of 'forOp'.
        let current: *mut Operation = operation;
        while current.block() != self {
            current = current.parent();
            if !current {
                return null();
            }
        }
        current
    }

    //! This drops all operand uses from operations within this block, which is an essential step in breaking cyclic dependences between references when they are to be deleted.
    pub fn drop_all_references(&mut self) {
        for operation in *self {
            operation.drop_all_references();
        }
    }

    //! This drops all uses of values defined in this block or in the blocks of nested regions wherever the uses are located.
    pub fn drop_all_defined_value_uses(&mut self) {
        for input in self.inputs() {
            input.drop_all_uses();
        }
        for operation in *self {
            operation.drop_all_defined_value_uses();
        }
        self.drop_all_uses();
    }

    //! Returns true if the ordering of the child operations is valid, false otherwise.
    pub fn is_op_order_valid(&self) -> bool {
        self.parent_valid_op_order_pair.int()
    }

    //! Invalidates the current ordering of operations.
    pub fn invalidate_op_order(&self) {
        // Validate the current ordering.
        assert!(!self.verify_op_order());
        self.parent_valid_op_order_pair.set_int(false);
    }

    //! Verifies the current ordering of child operations matches the validOpOrder flag. Returns false if the order is valid, true otherwise.
    pub fn verify_op_order(&self) -> bool {
        // The order is already known to be invalid.
        if !self.is_op_order_valid() {
            return false;
        }
        // The order is valid if there are less than 2 operations.
        if self.operations.len() < 2 {
            return false;
        }
        let prev = null();
        for operation in *self {
            // The previous operation must have a smaller order index than the next as it appears earlier in the list.
            if !prev.is_nul()
            && prev.order_index != Operation::kInvalidOrderIdx
            && prev.order_index >= operation.order_index
            {
                return true;
            }
            prev = operation;
        }
        false
    }

    //! Recomputes the ordering of child operations within the block.
    pub fn recompute_op_order(&self) {
        self.parent_valid_op_order_pair.set_int(true);

        let order_index = 0;
        for operation in *self {
            order_index += Operation::kOrderStride;
            operation.order_index = order_index;
        }
    }

    /* Terminator Management */

    //! Get the terminator operation of this block. This function asserts that
    //! the block has a valid terminator operation.
    pub fn terminator(&self) -> *mut Operation {
        assert(
            !self.is_empty()
            && back().might_have_trait::<IsTerminator>());
        self.back()
    }

    /* Predecessors and Successors */

    //! Return true if this block has no predecessors.
    pub fn has_no_predecessors(&self) -> bool {
        self.predecessors().is_empty()
    }

    //! Returns true if this blocks has no successors.
    pub fn has_no_successors(&self) -> bool {
        self.successors().is_empty()
    }

    //! If this block has exactly one predecessor, return it.  Otherwise, return null.
    //!
    //! Note that if a block has duplicate predecessors from a single block (e.g. if you have a conditional branch with the same block as the true/false destinations) is not considered to be a single predecessor.
    pub fn single_predecessor(&self) -> *mut Block {
        let predecessors = self.predecessors();
        match predecessors.next() {
            None => null(),
            Some(predecessor) => {
                match predecessors.next() {
                    None => predecessor,
                    Some(_) => null()
                }
            }
        }
    }

    //! If this block has a unique predecessor, i.e., all incoming edges originate from one block, return it. Otherwise, return null.
    pub fn unique_predecessor(&self) -> *mut Block {
        let predecessors = self.predecessors();
        match predecessors.next() {
            None => null(),
            Some(first_predecessor) => {
                // Check for any conflicting predecessors.
                if predecessors.all(
                    |predecessor| predecessor == first_predecessor)
                {
                    first_predecessor
                } else {
                    null()
                }
            }
        }
    }

    // Indexed successor access.
    pub fn num_successors(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            self.back().num_successors()
        }
    }

    pub fn successor(&self, index: usize) -> *mut Block {
        assert!(index < self.num_successors());
        self.terminator().successor(index)
    }

    pub fn successors(&self) -> SuccessorRange {
        SuccessorRange::new(self)
    }

    /* Operation Walkers */

    /* Other */

    // /**
    // Split the block into two blocks before the specified operation or iterator.

    // Note that all operations BEFORE the specified iterator stay as part of the original basic block, and the rest of the operations in the original block are moved to the new block, including the old terminator. The original block is left without a terminator.

    // The newly formed Block is returned, and the specified iterator is invalidated.
    // */
    // pub fn split_block(&mut self, split_before: iterator) -> *mut Block {

    // }
}

impl Idx for Block {
    fn new(index: usize) -> Self {
    }

    fn index(self) -> usize {
    }
}

impl DirectedGraph for Block {
    type Node = Self;
}

impl GraphSuccessors<'graph> for Block {
    type Item = Self;
    type Iter = Successors;
}

impl WithSuccessors for Block {
    fn successors(&self, node: Self::Node) -> Self::Iter {
    }
}

pub type Operations = IListIterator;

/* Predecessors and Successors */
