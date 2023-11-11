//! Helpers for Constructing MLIR Classes
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Builders.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/Builders.cpp>

use core::default::Default;

use crate::{
    mlir::ir::{
        affine_expr::AffineExpr,
        affine_map::AffineMap,
        block::Block,
        block_and_value_mapping::BlockAndValueMapping,
        builtins::types,
        dialect,
        integer_set,
        matchers,
        mlir_context::MLIRContext,
        symbol_table,
        r#type::Type,
        value::{Input, Value}
    },
    llvm::support::{
        compiler,
        raw_ostream
    }
};

/// This class is a general helper class for creating context-global objects like types, attributes, and affine expressions.
pub struct Builder {
    context: *mut MLIRContext,

    /* === Builder === */
    
    /// The current block this builder is inserting into.
    block: *mut Block, // = nullptr; 

    /// The insertion point within the block that this builder is inserting before.
    insert_point: Block::iterator,

    /// The optional listener for events of this builder.
    pub listener: *mut Listener
}

impl Builder {
    /* === Builder === */
    
    /* === OpBuilder === */

    /* Listeners */

    /// Sets the listener of this builder to the one provided.
    pub fn set_listener(&self, listener: *mut Listener) {
        self.listener = listener;
    }

    /// Returns the current listener of this builder, or nullptr if this builder doesn't have a listener.
    pub const fn listener(&self) -> *mut Listener {
        self.listener
    }

    /* Insertion Point Management */

    /// Sets the insertion point to the node after the specified operation, which will cause subsequent insertions to go right after it.
    pub fn set_insertion_point_after(&self, operation: *mut Operation) {
        self.set_insertion_point(operation.block(), Block::iterator(operation).next())
    }

    /// Sets the insertion point to the node after the specified value. If value has a defining operation, sets the insertion point to the node after such defining operation. This will cause subsequent insertions to go right after it. Otherwise, value is a BlockArgument. Sets the insertion point to the start of its block.
    pub fn set_insertion_point_after_value(&self, value: Value) {
        let operation = value.defining_operation();
        if !operation.is_null()  {
            self.set_insertion_point_after(operation);
        } else {
            let input: Input = value;
            self.set_insertion_point_to_start(input.owner());
        }
    }

    /// Sets the insertion point to the start of the specified block.
    pub fn set_insertion_point_to_start(&self, block: *mut Block) {
        self.set_insertion_point(block, block.begin());
    }

    /// Sets the insertion point to the end of the specified block.
    pub fn set_insertion_point_to_end(&self, block: *mut Block) {
        self.set_insertion_point(block, block.end());
    }

    /// Return the block the current insertion point belongs to.  Note that the insertion point is not necessarily the end of the block.
    pub const fn insertion_block(&self) -> *mut Block {
        self.block
    }

    /// Returns the current insertion point of the builder.
    pub fn insertion_point(&self) -> Block::iterator {
        self.insert_point
    }

    /// Returns the current block of the builder.
    pub const fn block(&self) -> *mut Block {
        self.block
    }

    /* Block Creation */

    /* Operation Creation */
}

/// This class represents a listener that may be used to hook into various actions within an Builder.
pub struct Listener {
}

/// This class represents a saved insertion point.
pub struct InsertPoint {  
    block: *mut Block,
    point: Block::iterator,
}

impl Default for InsertPoint {
    /// Creates a new insertion point which doesn't point to anything.
    fn default() -> Self {
        Self {
            block: None,
            point: Block::iterator::default()
        }
    }
}

impl InsertPoint {
    /// Creates a new insertion point at the given location.
    pub fn new(block: *mut Block, point: Block::iterator) -> Self {
        Self { block, point }
    }

    /// Returns true if this insert point is set.
    pub const fn is_set(&self) -> bool {
        self.block.is_some()
    }

    pub const fn block(&self) -> Option<Block> {
        self.block
    }

    pub const fn point(&self) -> Block::iterator {
        self.point
    }
}

/// RAII guard to reset the insertion point of the builder when destroyed.
pub struct InsertionGuard {
    builder: *mut Builder,
    insert_point: InsertPoint
}
