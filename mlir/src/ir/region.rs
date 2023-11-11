//! MLIR Region Class
//!
//! This file defines the Region class.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Region.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/Region.cpp>

use core::{
    default::Default,
    ops::Drop,
    ptr::null,
    slice::IterMut
};

use crate::{
    mlir::ir::{
        block::Block,
        block_and_value_mapping,
        location::Location,
        mlir_context::MLIRContext,
        operation::Operation,
        r#type::Type,
        type_range::ValueTypeRange,
        value::Input
    },
    llvm::adt::ilist::IList
};

/// This class contains a list of basic blocks and a link to the parent operation it is attached to.
#[derive(Default)]
pub struct Region {
    blocks: IList<Block>,

    /// This is the object we are part of.
    container: *mut Operation
}

impl Region {
    pub fn new(container: *mut Operation) -> Self {
        Self { container, ..Default::default() }
    }

    /// Return the context this region is inserted in.  The region must have a valid parent container.
    pub fn context(&self) -> Option<MLIRContext> {
        match self.container {
            None => panic!("Region is not attached to a container"),
            Some(operation) => operation.context()
        }
    }

    /// Return a location for this region. This is the location attached to the parent container. The region must have a valid parent container.
    pub fn location(&self) -> Location {
        match self.container {
            None => panic!("Region is not attached to a container"),
            Some(operation) => operation.location()
        }
    }

    /*
    Block List Management
    */

    pub fn blocks(&self) -> &IList<Block> {
        &self.blocks
    }

    pub fn emplace_block(&self) -> &Block {
        self.push_back(Block::new());
        self.back()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn push_back(&mut self, block: Block) {
        self.blocks.push_back(block);
    }

    pub fn push_front(&mut self, block: Block) {
        self.blocks.push_front(block);
    }

    pub fn back(&self) -> &Block {
        self.blocks.back()
    }
    pub fn front(&self) -> &Block {
        self.blocks.front()
    }

    /// Return true if this region has exactly one block.
    pub fn has_one_block(&self) -> bool {
        self.blocks.len() == 1
    }

    /*
    Argument Handling
    */

    pub fn arguments(&self) -> &mut [Input] {
        if self.is_empty() {
            &mut []
        } else {
            self.front().arguments()
        }
    }

    /// Returns the argument types of the first block within the region.
    pub fn input_types(&self) -> ValueTypeRange<&mut [Input]> {
        self.arguments().into()
    }

    /// Add one value to the argument list.
    pub fn add_input(&mut self, r#type: impl Type, location: Location) -> Input
    {
        self.front().add_input(r#type, location)
    }

    /// Insert one value to the position in the argument list indicated by the given iterator. The existing arguments are shifted. The block is expected not to have predecessors.
    pub fn insert_input(
        &mut self, it: IterMut<Input>, r#type: impl Type, location: Location
    ) -> Input {
        self.front().insert_input(it, r#type, location);
    }

    // /// Add one argument to the argument list for each type specified in the list.
    // /// `locs` contains the locations for each of the new arguments, and must be of equal size to `types`.
    // pub fn add_inputs(&mut self, types: TypeRange, locations: &[Location]) -> iterator_range<args_iterator>;
    
    /// Add one value to the argument list at the specified position.
    pub fn insert_input(
        &mut self, index: usize, r#type: impl Type, location: Location
    ) -> Input {
        self.front().insert_input(index, r#type, location)
    }

    /// Erase the argument at 'index' and remove it from the argument list.
    pub fn erase_argument(&mut self, index: usize) {
        self.front().erase_argument(index);
    }

    pub fn num_inputs(&self) -> usize {
        self.arguments().len()
    }
    
    pub fn argument(&self, index: usize) -> Input {
        self.arguments()[index]
    }

    /*
    Operation List Utilities
    */
    
    /*
    Misc. Utilities
    */
    
    /// Return the region containing this region or nullptr if the region is attached to a top-level operation.
    pub fn parent_region(&self) -> *mut Region {
    }

    /// Return the parent operation this region is attached to.
    pub fn parent_operation(&self) -> *mut Operation {
        self.container
    }
    
//     /// Find the first parent operation of the given type, or nullptr if there is no ancestor operation.
//     pub fn parent_of_type<Parent>(&self) -> Parent {
//         auto *region = self;
//         do {
//             if (auto parent = dyn_cast_or_null<ParentT>(region->container)) {
//                 return parent;
//             }
//         } while (region = region.parent_region());
//         parent::new();
//     }

    // /// Return the number of this region in the parent operation.
    // pub fn region_number(&self) -> unsigned {
    //     // Regions are always stored consecutively, so use pointer subtraction to igure out what number this is.
    //     self - &self.parent_operation().regions()[0];
    // }

    /// Return true if this region is a proper ancestor of the `other` region.
    pub fn is_proper_ancestor(&self, other: *mut Self) -> bool {
        if self == other {
            return false;
        }
  
        while let Some(other) = other.parent_region() {
            if self == other {
                return true;
            }
        }
        false
    }

    /// Return true if this region is ancestor of the `other` region.  A region is considered as its own ancestor, use `isProperAncestor` to avoid this.
    pub fn is_ancestor(&self, other: *mut Self) -> bool {
        self == other || self.is_proper_ancestor(other)
    }
    
    /// Takes body of another region (that region will have no body after this operation completes).  The current body of this region is cleared.
    pub fn take_body(&self, other: &Self) {
        self.drop_all_references();
        self.blocks.clear();
        self.blocks.splice(self.blocks.end(), other.blocks());
    }

    /// Returns 'block' if 'block' lies in this region, or otherwise finds the ancestor of 'block' that lies in this region. Returns nullptr if the latter fails.
    pub fn find_ancestor_block_in_region(&self, block: &Block) -> *mut Block {
        let mut current_block = block;
        while Some(self) != current_block.parent() {
            let parent_operation = current_block.parent_operation();
            if parent_operation.is_none()
            || parent_operation.unwrap().block().is_none() {
                return None;
            }
            current_block = &parent_operation.block();
        }
        current_block
    }

    /// Returns 'operation' if 'operation' lies in this region, or otherwise finds the ancestor of 'operation' that lies in this region. Returns nullptr if the latter fails.
    pub fn find_ancestor_op_in_region(&self, operation: &Operation)
        -> *mut Operation
    {
        let mut current: *mut Operation = operation;
        while !current.parent_region().is_null() {
            if current.parent_region() == self {
                return current;
            }

            current = current.parent_region().parent_operation();
            if current.is_none() {
                return null();
            }
        }
        null()
    }

    /// Drop all operand uses from operations within this region, which is an essential step in breaking cyclic dependences between references when they are to be deleted.
    pub fn drop_all_references(&mut self) {
        for block in *self {
            block.drop_all_references();
        }
    }
    
    /*
    Operation Walkers
    */
    
    /*
    CFG View Utilities
    */
}

impl Drop for Region {
    fn drop(&mut self) {
        // Operations may have cyclic references, which need to be dropped before we can start deleting them.
        self.drop_all_references();
    }
}

// impl IList for Block {
//     fn parent_region(&self) -> *mut Region {
//         let offset = 
//             size_t(&((*mut Region)nullptr->*Region::get_sublist_access(nullptr)));
//         let anchor = static_cast<iplist<Block> *>(self);
//         reinterpret_cast<*mut Region>(reinterpret_cast<char *>(anchor) - offset)
//     }

//     /// This is a trait method invoked when a basic block is added to a region.
//     /// We keep the region pointer up to date.
//     fn add_node_to_list(&mut self, block: Option<Block>) {
//         assert!(!block.parent(), "Already in a region!");
//         block.parent_valid_op_order_pair.set_pointer(self.parent_region());
//     }
  
//     /// This is a trait method invoked when an operation is removed from a
//     /// region.  We keep the region pointer up to date.
//     fn remove_node_from_list(&mut self, block: Option<Block>) {
//         assert!(block.parent(), "Not already in a region!");
//         block.parent_valid_op_order_pair.set_pointer(nullptr);
//     }
  
//     /// This is a trait method invoked when an operation is moved from one block
//     /// to another.  We keep the block pointer up to date.
//     fn transfer_nodes_from_list(
//         &self,
//         other: &Self,
//         first: block_iterator,
//         last: block_iterator
//     ) {
//         // If we are transferring operations within the same function, the parent pointer doesn't need to be updated.
//         let cur_parent = self.parent_region();
//         if cur_parent == other.parent_region() {
//             return;
//         }
    
//         // Update the 'parent' member of each Block.
//         while first != last {
//             first.parent_valid_op_order_pair.set_pointer(cur_parent);
//             first += 1;
//         }
//     }
// }

/// This class provides iteration over the held operations of blocks directly within a region.
pub struct OpIterator {
    /// The region whose operations are being iterated over.
    region: *mut Region,
    // /// The block of `region` whose operations are being iterated over.
    // block: Region::iterator,
    // /// The current operation within `block`.
    // operation: Block::iterator
}

/// This class provides an abstraction over the different types of ranges over Regions. In many cases, this prevents the need to explicitly materialise a SmallVector/std::vector. This class should be used in places that are not suitable for a more derived type (e.g. ArrayRef) or a template range parameter.
pub struct RegionRange {
}
