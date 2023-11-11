//! # MLIR Operation Class
//!
//! This file defines the Operation class.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Operation.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/Operation.cpp>

pub mod asm_interface;
pub mod base;
pub mod definition;
pub mod implementation;
pub mod support;

use core::cell::Cell;
use proc_macro::{Diagnostic, Level};

use crate::{
    mlir::{
        ir::{
            attribute::{Attribute, NamedAttribute},
            block::{
                Block,
                support::BlockOperand,
            },
            block_and_value_mapping,
            builtins::{
                attributes::DictionaryAttribute,
                types
            },
            diagnostics::append_op,
            dialect::Dialect,
            location::Location,
            mlir_context::MLIRContext,
            operation::{
                definition::FoldResult,
                // implementation,
                support::{
                    OperandStorage, OperationName, PrintingFlags,
                    RegisteredOperationName,
                }
            },
            pattern_match,
            region::Region,
            type_range::TypeRange,
            type_utilities,
            use_def_lists::ValueUserIterator,
            value::{
                InlineOutput, Operand, Output, OutOfLineOutput, Value},
            value_range::{OutputRange, UseIterator, UseRange, ValueRange}
        },
        interfaces::fold_interfaces, support::logical_result::LogicalResult
    },
    llvm::{
        adt::{
            small_vector::SmallVector,
            string_extras,
            twine::Twine, stl_extras::has_single_element
        },
        support::trailing_objects::TrailingObjects
    }
};

/**
Operation is the basic unit of execution within MLIR.

The following documentation are recommended to understand this class:
- <https://mlir.llvm.org/docs/LangRef/#operations>
- <https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/>

An Operation is defined first by its name, which is a unique string. The name is interpreted so that if it contains a '.' character, the part before is the dialect name this operation belongs to, and everything that follows is this operation name within the dialect.

An Operation defines zero or more SSA `Value` that we refer to as the Operation outputs. This array of Value is actually stored in memory before the Operation itself in reverse order. That is for an Operation with 3 outputs we allocate the following memory layout:

 [Result2, Result1, Result0, Operation]
                             ^ this is where `Operation*` pointer points to.

A consequence of this is that this class must be heap allocated, which is handled by the various `create` methods. Each result contains:

- one pointer to the first use (see `Operand`)
- the type of the SSA Value this result defines.
- the index for this result in the array.

The outputs are defined as subclass of `ValueImpl`, and more precisely as the only two subclasses of `OpResultImpl`: `InlineOutput` and `OutOfLineOutput`. The former is used for the first 5 outputs and the latter for the subsequent ones. They differ in how they store their index: the first 5 outputs only need 3 bits and thus are packed with the Type pointer, while the subsequent one have an extra `usize` value and thus need more space.

An Operation also has zero or more inputs: these are uses of SSA Value, which can be the outputs of other operations or Block arguments. Each of these uses is an instance of `Operand`. This optional array is initially tail allocated with the operation class itself, but can be dynamically moved out-of-line in a dynamic allocation as needed.

An Operation may contain optionally one or multiple `Region`s, stored in a tail allocated array. Each `Region` is a list of `Block`s. Each `Block` is itself a list of `Operation`s. This structure is effectively forming a tree.

Some operations like branches also refer to other `Block`, in which case they would have an array of `BlockOperand`.

Finally an Operation also contain an optional `DictionaryAttribute`, a `Location`, and a pointer to its parent `Block` (if any).
*/
#[repr(align(8))]
pub struct Operation {
    /// The operation block that contains this operation.
    block: *mut Block,  // = nullptr;

    /**
    This holds information about the source location the operation was defined or derived from.
    */
    pub location: Location,

    /// Relative order of this operation in its parent block. Used for O(1) local dominance checks between operations.
    order_index: Cell<usize>,  // = 0;

    num_outputs: usize,  // const
    num_succs: usize,  // const
    num_regions: usize,  // const : 31;

    /// This bit signals whether this operation has an operand storage or not. The operand storage may be elided for operations that are known to never have inputs.
    has_operand_storage: bool,  // : 1;

    /// This holds the name of the operation.
    name: OperationName,

    /// This holds general named attributes for the operation.
    attributes: DictionaryAttribute
}

impl Operation {
    fn new(
        location: Location,
        name: OperationName,
        num_outputs: usize,
        num_successors: usize,
        num_regions: usize,
        attributes: DictionaryAttribute,
        has_operand_storage: bool
    ) -> Self {
        
    }
    
    /// The name of an operation is the key identifier for it.
    pub fn name(&self) -> &OperationName {
        &self.name
    }

    /**
    If this operation has a registered operation description, return it.
    Otherwise return std::nullopt.
    */
    pub fn registered_info(&self) -> *mut RegisteredOperationName {
        self.name().registered_info()
    }

    /// Returns true if this operation has a registered operation description,
    /// otherwise false.
    pub fn is_registered(&self) -> bool {
        self.name().is_registered()
    }

    /// Returns the operation block that contains this operation.
    pub fn block(&self) -> *mut Block {
        self.block
    }

    /// Return the context this operation is associated with.
    pub fn context(&self) -> *mut MLIRContext {
        self.location.context()
    }

    /**
    Return the dialect this operation is associated with, or nullptr if the associated dialect is not loaded.
    */
    pub fn dialect(&self) -> *mut Dialect {
        self.name().dialect()
    }

    /// The source location the operation was defined or derived from.
    pub fn location(&self) -> Location {
        self.location
    }

    /// Set the source location the operation was defined or derived from.
    pub fn set_location(&mut self, location: Location) {
        self.location = location;
    }

    /// Returns the region to which the instruction belongs. Returns nullptr if the instruction is unlinked.
    pub fn parent_region(&self) -> *mut Region {
        match self.block {
            None => None,
            Some(block) => block.parent()
        }
    }

    /// Returns the closest surrounding operation that contains this operation or nullptr if this is a top-level operation.
    pub fn parent(&self) -> *mut Operation {
        match self.block {
            None => None,
            Some(block) => block.parent_operation()
        }
    }

    /// Return the closest surrounding parent operation that is of type `OpTy`.
    pub fn parent_of_type<OpTy>(&self) -> OpTy {
        let operation = self;
        while let Some(operation) = operation.parent() {
            let parent = dyn_cast::<OpTy>(operation);
            if parent {
                return parent;
            }
        }
        OpTy::new()
    }

    /// Returns the closest surrounding parent operation with trait `Trait`.
    // template <typename T> class Trait>
    pub fn parent_with_trait<Trait>(&self) -> *mut Operation {
        let operation = self;
        while let Some(operation) = operation.parent() {
            if operation.has_trait::<Trait>() {
                return Some(operation);
            }
        }
        None
    }

    /// Return true if this operation is a proper ancestor of the `other` operation.
    pub fn is_proper_ancestor(&self, other: &Operation) -> bool {
        while let Some(other) = other.parent() {
            if self == other {
                return true;
            }
        }
        return false;
    }

    /// Return true if this operation is an ancestor of the `other` operation. An operation is considered as its own ancestor, use `is_proper_ancestor` to avoid this.
    pub fn is_ancestor(&self, other: &Operation) -> bool {
        self == other || self.is_proper_ancestor(other)
    }

    /// Replace any uses of 'from' with 'to' within this operation.
    pub fn replace_uses_of_with(&self, from: Value, to: Value) {

    }

    /// Replace all uses of outputs of this operation with the provided `values`.
    pub fn replace_all_uses_with<ValuesT>(&self, values: &&ValuesT) {
        self.outputs().replace_all_uses_with(std::forward::<ValuesT>(values));
    }

    /// Destroys this operation and its subclass data.
    pub fn destroy(&self) {

    }

    // TODO Rename input_uses
    /// This drops all input uses from this operation, which is an essential step in breaking cyclic dependences between references when they are to be deleted.
    pub fn drop_all_references(&self) {
        for input in self.inputs() {
            input.drop();
        }
        for region in self.regions() {
            region.drop_all_references();
        }
        for dest in self.block_operands() {
            dest.drop();
        }
    }

    // TODO Rename output_uses
    /// Drop uses of all values outputted by this operation or its nested regions.
    pub fn drop_all_defined_value_uses(&self) {
        self.drop_all_uses();
        for region in self.regions() {
            for block in region {
                block.drop_all_defined_value_uses();
            }
        }  
    }

    /// Unlink this operation from its current block and insert it right before `operation` which may be in the same or another block in the same function.
    pub fn move_before(&self, operation: *mut Operation) {
        self.move_before_(operation.block(), operation.iterator());
    }

    /// Unlink this operation from its current block and insert it right before
    /// `iterator` in the specified block.
    pub fn move_before_(&self, block: *mut Block, iterator: IList<Operation>::iterator) {
        block.operations().splice(iterator, self.block().operations(),
                                self.iterator());
    }

    /// Unlink this operation from its current block and insert it right after `operation` which may be in the same or another block in the same function.
    pub fn move_after(&self, operation: *mut Operation) {
        self.move_after_(operation.block(), operation.iterator());
    }

    /// Unlink this operation from its current block and insert it right after `iterator` in the specified block.
    pub fn move_after_(&self, block: *mut Block, iterator: IList<Operation>::iterator) {
        assert!(iterator != block.end(), "Cannot move after end of block");
        self.move_before_(block, iterator.next());
    }

    /// Given an operation `other` that is within the same parent block, return whether the current operation is before `other` in the operation list of the parent block.
    ///
    /// NOTE: This function has an average complexity of O(1), but worst case may take O(N) where N is the number of operations within the parent block.
    pub fn is_before_in_block(&self, other: *mut Operation) -> bool {
        assert!(
            !self.block.is_null(),
            "Operations without parent blocks have no order.");
        assert!(
            !other.is_null() && other.block == self.block,
            "Expected other operation to have the same parent block.");
        // If the order of the block is already invalid, directly recompute the
        // parent.
        if !self.block.is_op_order_valid() {
            self.block.recompute_op_order();
        } else {
            // Update the order either operation if necessary.
            self.update_order_if_necessary();
            other.update_order_if_necessary();
        }
      
        self.order_index < other.order_index
    }

    /* Inputs/Operands */

    /// Insert the given inputs into the input list at the given `index`.
    pub fn insert_inputs(&mut self, index: usize, inputs: ValueRange) {
        if likely(self.has_operand_storage) {
            self.set_inputs(index, 0, inputs);
        }
        assert!(inputs.is_empty(), "Inserting inputs without an operand storage");
    }

    pub fn num_inputs(&self) -> usize {
        if self.has_operand_storage {
            self.operand_storage().len()
        } else {
            0
        }
    }

    pub fn input(&self, index: usize) -> Value {
        self.op_operand(index).get()
    }


    pub fn set_input(&self, index: usize, value: Value) {
        self.input(index).set(value);
    }

    /// Removes the input at position `index`.
    pub fn remove_input(&self, index: usize) {
        self.remove_inputs(index, 1);
    }

    /// Removes the inputs starting at position `index` and ending at position
    /// `index + length`.
    pub fn remove_inputs(&self, index: usize, length: usize /* 1 */) {
        self.operand_storage().remove_inputs(index, length);
    }

    pub fn inputs(&self) -> &mut [Operand] {
        if core::intrinsics::likely(self.has_operand_storage) {
            self.operand_storage().inputs()
        } else {
            &mut []
        }
    }
    
    pub fn op_operand(&self, index: usize) -> &Operand {
        self.operand_storage().inputs()[index]
    }

    pub fn input_types(&self) -> TypeRange {
        self.inputs().types()
    }

    /* Outputs/Results */

    /// Return the number of outputs held by this operation.
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// Get the `index`th output of this operation.
    /// Returns a pointer to the use list for the given result, which may be either inline or out-of-line.
    pub fn output(&self, index: usize) -> Output {
        let max_inline_results = Output::max_inline_results();
        if index < max_inline_results {
            return self.inline_output(index);
        }
        return self.out_of_line_output(index - max_inline_results);
    }

    pub fn outputs(&self) -> OutputRange {
        if self.num_outputs == 0 {
            OutputRange::new(None, 0)
        } else {
            OutputRange::new(self.inline_output(0), self.num_outputs)
        }
    }

    pub fn output_types(&self) -> TypeRange<OutputRange> {  // TODO Creat the type
        self.outputs().types()
    }

    /* Attributes */

    // Operations may optionally carry a list of attributes that associate
    // constants to names.  Attributes may be dynamically added and removed over
    // the lifetime of an operation.

    /// Return all of the attributes on this operation.
    pub fn attributes(&self) -> &[NamedAttribute] {
        self.attributes.value()
    }

    /// Return all of the attributes on this operation as a DictionaryAttribute.
    pub fn attr_dictionary(&self) -> DictionaryAttribute {
        self.attributes
    }

    /* Blocks */

    /// Returns the number of regions held by this operation.
    pub fn num_regions(&self) -> usize {
        self.num_regions
    }

    /// Returns the regions held by this operation.
    pub fn regions(&self) -> &mut [Region] {
        // Check the count first, as computing the trailing objects can be slow.
        if self.num_regions == 0 {
            return &mut [];
        }
        
        let regions = TrailingObjects::<Region>::new();
        regions as &mut [Region]  // ; self.num_regions
    }

    /// Returns the region held by this operation at position `index`.
    pub fn region(&self, index: usize) -> &Region {
        assert!(index < self.num_regions, "Invalid region index.");
        &self.regions()[index]
    }

    /* Successors */

    pub fn block_operands(&self) -> &mut [BlockOperand] {
        TrailingObjects::<BlockOperand>::new().into()
    }

    pub fn has_successors(&self) -> bool {
        self.num_succs != 0
    }
    pub fn num_successors(&self) -> usize {
        self.num_succs
    }

    pub fn successor(&self, index: usize) -> *mut Block {
        assert!(index < self.num_successors());
        self.block_operands()[index].get()
    }

    pub fn set_successor(&mut self, block: *mut Block, index: usize) {
        assert(index < getNumSuccessors());
        self.block_operands()[index].set(block);
    }

    /* Accessors for various properties of operations */

    /// Attempt to fold this operation with the specified constant input values
    /// - the elements in `inputs` will correspond directly to the inputs of the operation, but may be null if non-constant. If folding is successful, this fills in the `outputs` vector. If not, `outputs` is unspecified.
    pub fn fold(
        &self,
        inputs: &[dyn Attribute],
        outputs: &SmallVector<[FoldResult]>
    ) -> LogicalResult {
        // If we have a registered operation definition matching this one, use it to try to constant fold the operation.
        let info = self.registered_info();
        if !info.is_null()
        && info.fold_hook(self, inputs, outputs).is_ok()
        {
            return Ok(());
        }

        // Otherwise, fall back on the dialect hook to handle it.
        let dialect = self.dialect();
        if dialect.is_null() {
            return Err(());
        }

        let interface = dyn_cast<DialectFoldInterface>(dialect);
        if !interface {
            return Err(());
        }

        interface.fold(self, inputs, outputs)
    }

    /// Returns true if the operation was registered with a particular trait, e.g. has_trait<OperandsAreSignlessIntegerLike>().
    // template <template <typename T> class Trait>
    pub fn has_trait<Trait>(&self) -> bool {
        self.name.has_trait::<Trait>()
    }

    /// Returns true if the operation *might* have the provided trait. This
    /// means that either the operation is unregistered, or it was registered with
    /// the provide trait.
    // template <template <typename T> class Trait>
    pub fn might_have_trait<Trait>(&self) -> bool {
        self.name.might_have_trait::<Trait>()
    }

    /* Operation Walkers */

    /* Uses */

    /// Drop all uses of outputs of this operation.
    pub fn drop_all_uses(&self) {
        for output in self.outputs() {
            output.drop_all_uses();
        }
    }

    /// Returns a range of all uses, which is useful for iterating over all uses.
    pub fn uses(&self) -> UseRange {
        self.outputs().uses()
    }


    /// Returns true if this operation has exactly one use.
    pub fn has_one_use(&self) -> bool {
        has_single_element(self.uses())
    }

    /// Returns true if this operation has no uses.
    pub fn use_empty(&self) -> bool {
        self.outputs().use_empty()
    }

    /// Returns true if the outputs of this operation are used outside of the given block.
    pub fn is_used_outside_of_block(&self, block: *mut Block) -> bool {
        self.outputs().any(
            |output: Output| output.is_used_outside_of_block(block))
    }

    /* Users */
    
    /// Returns a range of all users.
    pub fn users(&self) -> ValueUserIterator<UseIterator, Operand> {
        ValueUserIterator
    }

    /* Other */

    /// Emit an error with the op name prefixed, like `'dim' op ` which is convenient for verifiers.
    pub fn emit_op_error(&self, message: &Twine) -> Diagnostic {
        let diagnostic = self.emit_error("");
        let message = format!("{}'{}' op {}", diagnostic.message(), self.name(), message);
        diagnostic.set_message(message);
        diagnostic
    }

    /// Emit an error about fatal conditions with this operation, reporting up to any diagnostic handlers that may be listening.
    pub fn emit_error<T: Into<String>>(&self, message: T) -> Diagnostic {
        let diagnostic = Diagnostic::spanned(self.location, Level::Error, message);
        if self.context().should_print_op_on_diagnostic() {
            let message = format!(
                "See current operation: {}",
                append_op(
                    &self,
                    PrintingFlags::new().print_generic_op_form(),
                    Level::Error
                )
            );
            diagnostic.span_note(self.location, message);
        }
        diagnostic
    }

    /// Emit a warning about this operation, reporting up to any diagnostic handlers that may be listening.
    pub fn emit_warning(&self, message: &Twine) -> Diagnostic {
    }

    /// Emit a remark about this operation, reporting up to any diagnostic handlers that may be listening.
    pub fn emit_remark(&self, message: &Twine) -> Diagnostic {
    }

    /* Ordering */

    // --
    // 
    // --

    /// Returns the operand storage object.
    fn operand_storage(&self) -> &OperandStorage {
        assert!(
            self.has_operand_storage,
            "Expected operation to have operand storage"
        );
        *TrailingObjects::<OperandStorage>::new()
    }

    /// Returns a pointer to the use list for the given out-of-line output.
    fn out_of_line_output(&self, index: usize) -> *mut OutOfLineOutput {
        // Out-of-line outputs are stored in reverse order after (before in memory) the inline outputs.
        return reinterpret_cast::<*mut OutOfLineOutput>(
            self.inline_output(Output::max_inline_results() - 1)
            )
            - (index + 1);  // ++output_number
    }

    /// Returns a pointer to the use list for the given inline output.
    fn inline_output(&self, index: usize) -> *mut InlineOutput {
        // Inline outputs are stored in reverse order before the operation in
        // memory.
        reinterpret_cast::<*mut InlineOutput>(self) - (index + 1)  // ++output_number
    }

    /// Provide a 'get_parent' method for ilist_node_with_parent methods.
    /// We mark it as a const function because ilist_node_with_parent specifically requires a 'get_parent() const' method. Once ilist_node removes this constraint, we should drop the const to fit the rest of the MLIR const model.
    const fn parent(&self) -> *mut Block {
        self.block
    }
}

/// Class encompassing various options related to cloning an operation. Users of this class should pass it to Operation's 'clone' methods.
/// Current options include:
///
/// - Whether cloning should recursively traverse into the regions of the operation or not.
/// - Whether cloning should also clone the inputs of the operation.
pub struct CloneOptions {
    /// Whether regions should be cloned.
    clone_regions_flag: bool, // : 1;
    /// Whether inputs should be cloned.
    clone_operands_flag: bool  // : 1;
}

/// A utility iterator that filters out non-dialect attributes.
pub struct DialectAttrIterator {

}
