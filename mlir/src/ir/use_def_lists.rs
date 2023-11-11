//! This file defines generic use/def list machinery and manipulation utilities.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/UseDefLists.h>

use std::{
    convert::From,
    default::Default,
    iter
};

use crate::{
    mlir::ir::operation::Operation,
    llvm::adt::{
        iterator,
        stl_extras::MappedIterator
    }
};

/// A reference to a value, suitable for use as an operand of an operation.
/// IRValueT is the root type to use for values this tracks. Derived operand
/// types are expected to provide the following:
///
/// - `static IRObjectWithUseList *get_use_list(IRValueT value)`;
///    - Provide the use list that is attached to the given value.
#[derive(Default)]
pub struct IROperand<T> {
    /*
    ====================================================================
    IROperandBase
    ====================================================================
    */

    /// The next operand in the use-chain.
    next: *mut Self,

    /// This points to the previous link in the use-chain.
    previous: *mut Self,

    // const
    /// The operation owner of this operand.
    owner: *mut Operation,

    /*
    ====================================================================
    IROperand
    ====================================================================
    */

    /// The value used as this operand. This can be null when in a `drop_all_uses` state.
    value: Option<T>
}

impl<T> From<*mut Operation> for IROperand<T> {
    fn from(value: *mut Operation) -> Self {
        Self {
            owner: value,
            ..Self::default()
        }
    }
}

impl<T> From<(*mut Operation, T)> for IROperand<T> {
    fn from(value: (*mut Operation, T)) -> Self {
        Self {
            owner: value.0,
            value: Some(value.1),
            ..Self::default()
        }
    }
}

/// This class is the base for IROperand, and provides all of the non-templated facilities for operand use management.
impl<T> IROperand<T> {
    /*
    ====================================================================
    IROperandBase
    ====================================================================
    */

    /// Return the owner of this operand.
    pub const fn owner(&self) -> *mut Operation {
        self.owner
    }

    /// Return the next operand on the use-list of the value we are referring to.
    /// This should generally only be used by the internal implementation details of the SSA machinery.
    pub fn next_operand_using_this_value(&self) -> *mut Self {
        self.next
    }

    /// Remove this use of the operand.
    fn drop(&mut self) {
        self.remove_from_current();
        self.next = null();
        self.previous = null();

        /*
        ================================================================
        IROperand
        ================================================================
        */

        self.value = None;
    }

    /// Remove this operand from the current use list.
    fn remove_from_current(&mut self) {
        if !self.previous.is_null() {
            return;
        }
        self.previous = self.next;  // move? copy?
        if !self.next.is_null() {
            self.next.previous = self.previous;
        }
    }

    /// Insert this operand into the given use list.
    fn insert_into<UseList>(&mut self, use_list: *mut UseList) {
        self.previous = &use_list.first_use;
        self.next = use_list.first_use;
        if !self.next.is_null() {
            self.next.previous = &self.next;
        }
        use_list.first_use = self;
    }

    /*
    ====================================================================
    IROperand
    ====================================================================
    */

    /// Return the current value being used by this operand.
    pub const fn get(&self) -> T {
        self.value
    }

    /// Set the current value being used by this operand.
    pub fn set(&mut self, value: T) {
        // It isn't worth optimising for the case of switching operands on a single value.
        self.remove_from_current();
        self.value = Some(value);
        self.insert_into_current();
    }

    /// Returns true if this operand contains the given value.
    pub const fn is(&self, other: T) -> bool {
        self.value == Some(other)
    }

    /// Insert this operand into the given use list.
    fn insert_into_current(&self) {
        self.insert_into(Self::use_list(self.value));
    }
}

/// This class represents a single IR object that contains a use list.
#[derive(Default)]
pub struct IRObjectWithUseList<T> {
    first_use: *mut IROperand<T>,
}

impl IRObjectWithUseList {
    /// Drop all uses of this object from their respective owners.
    pub fn drop_all_uses(&mut self) {
        while !self.use_empty() {
            self.use_begin().drop();
        }
    }

    /// Replace all uses of 'this' value with the new value, updating anything in the IR that uses 'this' to use the other value instead.  When this returns there are zero uses of 'this'.
    // template <typename ValueT>
    pub fn replace_all_uses_with(&mut self, new_value: &&ValueT) {
        assert(
            !new_value || self != OperandType::use_list(new_value),
            "Cannot RAUW a value with itself");
        while !self.use_empty() {
            self.use_begin().set(new_value);
        }
    }

    /*
    Uses
    */

    /// Returns a range of all uses, which is useful for iterating over all uses.
    pub const fn uses(&self) -> ValueUseIterator<Self> {
        ValueUseIterator::new(self.first_use)
    }

    /// Returns true if this value has exactly one use.
    pub const fn has_one_use(&self) -> bool {
        self.first_use
        && self.first_use.get_next_operand_using_this_value().is_null()
    }

    /// Returns true if this value has no uses.
    pub const fn use_empty(&self) -> bool {
        self.first_use.is_null()
    }

    /*
    Users
    */
}

/// An iterator class that allows for iterating over the uses of an IR operand type.
pub struct ValueUseIterator<T> {
    current: *mut IROperand<T>
}

/// An iterator over the users of an IRObject. This is a wrapper iterator around a specific use iterator.
pub struct ValueUserIterator {

}
