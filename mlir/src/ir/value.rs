//! MLIR Value Classes
//!
//! This file defines generic Value type and manipulation utilities.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Value.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/Value.cpp>

use core::{
    cmp::PartialEq,
    hash::Hash,
    fmt::Display,
    ptr::null
};

use crate::{
    mlir::{
        ir::{
            block::{Block, support::BlockOperand},
            builtins::types,
            location::Location,
            mlir_context::MLIRContext,
            operation::Operation,
            region::Region,
            r#type::Type,
            use_def_lists::{
                IRObjectWithUseList, IROperand, ValueUseIterator, ValueUserIterator
            },
        },
        support::llvm
    },
    llvm::{
        adt::{
            small_ptr_set,
            pointer_int_pair::PointerIntPair
        },
        support::pointer_like_type_traits
    }
};

/// This class represents an instance of an SSA value in the MLIR system, representing a computable value that has a type and a set of users. An SSA value is either a `Input` or the `Output` of an operation. Note: This class has value-type semantics and is just a simple wrapper around a ValueImpl that is either owned by a block (in the case of a `Input`) or an Operation (in the case of an `Output`).
#[repr(align(8))]
pub struct Value {
    /// The type of this result and the kind.
    type_and_kind: PointerIntPair<dyn Type, 3, Kind>
}

impl Value {
    /*
    ====================================================================
    ValueImpl
    ====================================================================

    The base class for all derived Value classes. It contains all of the components that are shared across Value classes.
    */

    pub fn new(r#type: impl Type, kind: Kind) -> Self {
        Self {
            type_and_kind: PointerIntPair::<dyn Type, 3, Kind>::new(r#type, kind)
        }
    }

    /*
    ====================================================================
    ValueImpl
    ====================================================================
    */

    /// Return the type of this value.
    pub const fn r#type(&self) -> impl Type {
        self.type_and_kind.pointer()
    }

    // /**
    // Set the type of this value.
    // Mutate the type of this Value to be of the specified type.
    // 
    // NOTE: This is an extremely dangerous operation which can create completely invalid IR very easily.  It is strongly recommended that you recreate IR objects with the right types instead of mutating them in place.
    // */
    // pub fn set_type(&self, r#type: impl Type) {
    //     self.type_and_kind.set_pointer(r#type);
    // }

    /// Return the kind of this value.
    pub const fn kind(&self) -> Kind {
        self.type_and_kind.int()
    }

    /*
    Value
    */

    /// Utility to get the associated MLIRContext that this value is defined in.
    pub const fn context(&self) -> *mut MLIRContext {
        self.r#type().context()
    }

    /// If this value is the output of an operation, return the operation that defines it.
    pub const fn defining_operation(&self) -> *mut Operation {
        match dyn_cast::<Output>() {
            Ok(output) => output.owner(),
            Err(_) => null(),
        }
    }

    /// Return the location of this value.
    pub const fn location(&self) -> Location {
        match self.defining_operation() {
            None => cast::<Input>().location(),
            Some(operation) => operation.location()
        }
    }
    
    pub fn set_location(&mut self, location: Location) {
        match self.defining_operation() {
            None => cast::<Input>().set_location(location),
            Some(operation) => operation.set_location(location)()
        };
    }

    /// Return the Region in which this Value is defined.
    pub fn parent_region(&self) -> *mut Region {
        match self.defining_operation() {
            None => cast::<Input>().owner().parent(),
            Some(operation) => operation.parent_region()
        }
    }

    /// Return the Block in which this Value is defined.
    pub fn parent_block(&self) -> *mut Block {
        match self.defining_operation() {
            None => cast::<Input>().owner(),
            Some(operation) => operation.block()
        }
    }

    /*
    UseLists
    */

    // /// Drop all uses of this object from their respective owners.
    // pub const fn drop_all_uses(&self) {
    //     impl->dropAllUses()
    // }

    /// Replace all uses of 'this' value with the new value, updating anything in the IR that uses 'this' to use the other value instead.  When this returns there are zero uses of 'this'.
    pub const fn replace_all_uses_with(&self, value: Value) {
        
    }

    /// Replace all uses of 'this' value with 'new_value' if the given callback returns true.
    pub fn replace_uses_with_if(
        &self,
        value: Value,
        should_replace: fn(&Operand) -> bool  // function_ref
    ) {
        for r#use in self.uses().iter_mut() {
            if should_replace(r#use) {
                r#use.set(value);
            }
        }
            
    }

    /// Returns true if the value is used outside of the given block.
    pub fn is_used_outside_of_block(&self, block: *mut Block) -> bool {
        self.users().any(|user: *mut Operation| user.block() != block)
    }

    /*
    Uses
    */

    // /// Returns a range of all uses, which is useful for iterating over all uses.
    // pub const fn uses(&self) -> ValueUseIterator<Operand>;

    // /// Returns true if this value has exactly one use.
    // pub const fn has_one_use(&self) -> bool;

    // /// Returns true if this value has no uses.
    // pub const fn use_empty(&self) -> bool;
    
    /*
    Users
    */

    pub const fn users(&self) -> ValueUserIterator {
        // {user_begin(), user_end()}
    }

    /*
    Utilities
    */
}

// : Display + Hash + PartialEq

/// The enumeration represents the various different kinds of values the internal representation may take. We use all of the bits from Type that we can to store indices inline.
pub enum Kind {
    /// The first N kinds are all inline operation results. An inline operation result means that the kind represents the result number. This removes the need to store an additional index value. The derived class here is an `OpResultImpl`.
    InlineOutput = 0,

    /// The next kind represents a 'out-of-line' operation result. This is for results with numbers larger than we can represent inline. The derived class here is an `OpResultImpl`.
    OutOfLineOutput = 6,

    /// The last kind represents a block input. The derived class here is an `BlockArgumentImpl`.
    Input = 7
}

/// This class represents an operand of an operation. Instances of this class contain a reference to a specific `Value`.
pub trait Operand: IROperand {

}

/// This class represents an input of a Block.
pub struct Input {
    /*
    ====================================================================
    BlockArgumentImpl
    ====================================================================
    */

    /// The owner of this input.
    owner: *mut Block,

    /// The position in the input list.
    pub index: usize,  // i64,

    /// The source location of this input.
    pub location: Location
}

impl Input {
    /// Allocate a new input with the given type and owner.
    fn new(
        r#type: impl Type,
        owner: *mut Block,
        index: usize,  // i64,
        location: Location
    ) -> Self
    {
        Self::new(r#type, owner, index, location)
    }

    /// Returns the block that owns this input.
    pub const fn owner(&self) -> *mut Block {
        self.owner
    }

    /// Returns the number of this argument.
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Destroy and deallocate this input.
    pub(crate) fn destroy(self) {
    }
}

/// This is a value defined by a result of an operation.
#[repr(align(8))]
pub struct Output {

}

/// This class provides the implementation for an operation result.
impl Output {
    /*
    ====================================================================
    OpResultImpl
    ====================================================================
    */

    /// Returns the parent operation of this result.
    /// Returns the parent operation of this trailing result.
    pub const fn owner(&self) -> *mut Operation {
        /*
        We need to do some arithmetic to get the operation pointer. Results are stored in reverse order before the operation, so move the trailing owner up to the start of the array. A rough diagram of the memory layout is:

        | Out-of-Line results | Inline results | Operation |

        Given that the results are reverse order we use the result number to know how far to jump to get to the operation. So if we are currently the 0th result, the layout would be:

        | Inline result 0 | Operation

        ^-- To get the base address of the operation, we add the result count + 1.
        */
        let mut result = dyn_cast::<InlineOutput>(self);
        if result {
            result += result.output_number() + 1;
            return reinterpret_cast::<*mut Operation>(const_cast::<*mut InlineOutput>(result));
        }

        // Out-of-line results are stored in an array just before the inline results.
        let mut out_of_line_it = (*mut OutOfLineOutput)(self);
        out_of_line_it += (out_of_line_it.out_of_line_index + 1);

        // Move the owner past the inline results to get to the operation.
        let mut inline_it = reinterpret_cast::<*mut InlineOutput>(out_of_line_it);
        inline_it += Self::max_inline_results();
        reinterpret_cast::<*mut Operation>(const_cast::<*mut InlineOutput>(inline_it))
    }

    /// Returns the result number of this op result.
    pub const fn output_number(&self) -> usize {
        let out_of_line_result = dyn_cast::<OutOfLineOutput>(self);
        if out_of_line_result {
            return out_of_line_result.output_number();
        }
        return cast::<InlineOutput>(self).output_number();
    }

    /// Returns the next operation result at `offset` after this result. This method is useful when indexing the result storage of an operation, given that there is more than one kind of operation result (with the different kinds having different sizes) and that operations are stored in reverse order.
    pub fn next_result_at_offset(&self, offset: isize) -> *mut Self {
        if offset == 0 {
            return self;
        }
        
        /*
        We need to do some arithmetic to get the next result given that results are in reverse order, and that we need to account for the different types of results. As a reminder, the rough diagram of the memory layout is:

        | Out-of-Line results | Inline results | Operation |

        So an example operation with two results would look something like:

        | Inline result 1 | Inline result 0 | Operation |
        */
    
        // Handle the case where this result is an inline result.
        let result = self;
        let inline_result = dyn_cast::<InlineOutput>(self);
        if inline_result {
            /*
            Check to see how many results there are after this one before the start of the out-of-line results. If the desired offset is less than the number remaining, we can directly use the offset from the current result pointer. The following diagrams highlight the two situations.

            | Out-of-Line results | Inline results | Operation |
                                               ^- Say we are here.
                                      ^- If our destination is here, we can use the
                                         offset directly.

            */
            let left_before_trailing =
                Self::max_inline_results() - inline_result.output_number() - 1;
            if left_before_trailing >= offset {
                return inline_result - offset;
            }
        
            /*
            Otherwise, adjust the current result pointer to the end (start in memory) of the inline result array.

            | Out-of-Line results | Inline results | Operation |
                                               ^- Say we are here.
                             ^- If our destination is here, we need to first jump to
                                the end (start in memory) of the inline result array.

            */
            result = inline_result - left_before_trailing;
            offset -= left_before_trailing;
        }
    
        // If we land here, the current result is an out-of-line result and we can offset directly.
        reinterpret_cast::<*mut OutOfLineOutput>(result) - offset
    }

    /// Returns the maximum number of results that can be stored inline.
    pub fn max_inline_results() -> usize {
        Kind::OutOfLineOutput as usize
    }

    /*
    Output
    */

    /// Given a number of operation results, returns the number that need to be stored inline.
    fn num_inline(num_outputs: usize) -> usize {
        std::cmp::min(num_outputs, Self::max_inline_results())
    }

    /// Given a number of operation results, returns the number that need to be stored as trailing.
    fn num_trailing(num_outputs: usize) -> usize {
        // If we can pack all of the results, there is no need for additional storage.
        let max_inline = Self::max_inline_results();
        if num_outputs <= max_inline {
            0
        } else {
            num_outputs - max_inline
        }
    }
}

/// This class provides the implementation for an operation result whose index can be represented 'inline' in the underlying `Value`.
pub struct InlineOutput {

}

/// This class provides the implementation for an operation result whose index cannot be represented 'inline', and thus requires an additional index field.
pub struct OutOfLineOutput {

}

/// TypedValue is a Value with a statically know type.
/// TypedValue can be null/empty
pub struct TypedValue<T> {
    _unimplemented: std::marker::PhantomData<T>
}

impl BlockOperand {
    /// Provide the use list that is attached to the given block.
    pub fn use_list(value: *mut Block) -> *mut IRObjectWithUseList<Self> {
        value
    }

    /// Return which operand this is in the BlockOperand list of the Operation.
    pub fn operand_number(&self) -> usize {
        self - &self.owner().block_operands()[0];
    }
}
