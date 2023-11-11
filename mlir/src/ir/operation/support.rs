/*!
This file defines a number of support types that Operation and related classes build on top of.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OperationSupport.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/OperationSupport.cpp>
*/

use alloc::alloc::Global;
use core::{
    any::TypeId,
    cell::Cell,
    cmp::PartialEq,
    mem::{drop, size_of},
    slice::from_raw_parts_mut
};

use crate::{
    mlir::{
        execution_engine::c_runner_utils::next_power_of2,
        ir::{
            attribute::{Attribute, NamedAttribute},
            block::{
                Block,
                support::BlockRange
            },
            builtins::attributes::StringAttribute,
            dialect::Dialect,
            location::Location,
            mlir_context::MLIRContext,
            operation::{
                Operation,
                definition::FoldResult
            },
            region::Region,
            type_range,
            r#type::Type,
            value::{Operand, Value},
            value_range::ValueRange
        },
        support::{
            interface_support::InterfaceMap,
            logical_result::LogicalResult
        },
    },
    llvm::{
        adt::{
            bitmask_enum,
            bit_vector::BitVector,
            pointer_int_pair::PointerIntPair,
            pointer_union,
            small_vector::SmallVector
        },
        support::{
            pointer_like_type_traits,
            trailing_objects
        }
    }
};

type GetCanonicalisationPatternsFn
    = fn(&RewritePatternSet, *mut MLIRContext);  // unique_function
type FoldHookFn
    = fn(*mut Operation, &[Attribute], &SmallVector<FoldResult>)
        -> LogicalResult; // unique_function< const>;
type HasTraitFn = fn(TypeId) -> bool; // unique_function
type ParseAssemblyFn
    = fn(&OpAsmParser, &OperationState) -> LogicalResult;  // unique_function
// Note: RegisteredOperationName is passed as reference here as the derived
// class is defined below.
type PopulateDefaultAttrsFn
    = fn(&RegisteredOperationName, &NamedAttrList);  // unique_function
type PrintAssemblyFn
    = fn(*mut Operation, &OpAsmPrinter, &str);  // unique_function
type VerifyInvariantsFn
    = fn(*mut Operation) -> LogicalResult;  // unique_function
type VerifyRegionInvariantsFn
    = fn(*mut Operation) -> LogicalResult;  // unique_function

pub struct OperationName {
    /// The name of the operation.
    name: StringAttribute,
    /// This is the dialect that this operation belongs to.
    dialect: &'static dyn Dialect,

    /// The unique identifier of the derived Op class.
    type_id: TypeId,

    /// A map of interfaces that were registered to this operation.
    interface_map: InterfaceMap,

    /// Internal callback hooks provided by the op implementation.
    fold_hook_fn: FoldHookFn,
    get_canonicalisation_patterns_fn: GetCanonicalisationPatternsFn,
    has_trait_fn: HasTraitFn,
    parse_assembly_fn: ParseAssemblyFn,
    populate_default_attrs_fn: PopulateDefaultAttrsFn,
    print_assembly_fn: PrintAssemblyFn,
    verify_invariants_fn: VerifyInvariantsFn,
    verify_region_invariants_fn: VerifyRegionInvariantsFn,

    /**
    A list of attribute names registered to this operation in StringAttribute form. This allows for operation classes to use StringAttribute for attribute lookup/creation/etc., as opposed to raw strings.
    */
    attribute_names: &'static [StringAttribute]
}

/**
This is a 'type erased' representation of a registered operation. This should only be used by things like the AsmPrinter and other things that need to be parameterized by generic operation hooks. Most user code should use the concrete operation types.
*/
pub struct RegisteredOperationName {

}

// ----------------------------------------------------------------------
// Attribute Dictionary-Like Interface
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// NamedAttrList
// ----------------------------------------------------------------------

/**
NamedAttrList is array of NamedAttributes that tracks whether it is sorted and does some basic work to remain sorted.
*/
pub struct NamedAttrList {
    // These are marked mutable as they may be modified (e.g., sorted)
    attributes: Cell<SmallVector<NamedAttribute>>,  // 4
    /**
    Pair with cached DictionaryAttr and status of whether attributes is sorted.
    Note: just because sorted does not mean a DictionaryAttr has been created but the case where there is a DictionaryAttr but attributes isn't sorted should not occur.
    */
    dictionary_sorted: Cell<PointerIntPair<dyn Attribute, 1, bool>>
}

// ----------------------------------------------------------------------
// OperationState
// ----------------------------------------------------------------------

/**
This represents an operation in an abstracted form, suitable for use with the builder APIs. This object is a large and heavy weight object meant to be used as a temporary object on the stack. It is generally unwise to put this in a collection.
*/
pub struct OperationState {
    location: Location,
    name: OperationName,
    inputs: SmallVector<[Value; 4]>,
    /// Types of the results of this operation.
    types: SmallVector<[dyn Type; 4]>,
    attributes: NamedAttrList,
    /// Successors of this operation and their respective operands.
    successors: SmallVector<[*mut Block; 1]>>,
    /// Regions that the operation will hold.
    regions: SmallVector<[Region; 1]>
}

impl OperationState {
    pub fn add_inputs(&self, inputs: ValueRange) {
        self.inputs.append(input);
    }

    pub fn add_types(&self, types: &[impl Type]) {
        self.types.append(types);
    }

    /// Add an attribute with the specified name.
    pub fn add_attribute(&self, name: &str, attribute: impl Attribute) {
        self.add_attribute(StringAttribute::get(self.context(), name), attribute);
    }

    /// Add an attribute with the specified name.
    pub fn add_attribute(&self, name: StringAttribute, attribute: impl Attribute) {
        self.attributes.append(name, attribute);
    }

    /// Add an array of named attributes.
    pub fn add_attributes(&self, attributes: &[NamedAttribute]) {
        self.attributes.append(attributes);
    }

    pub fn add_successors(&self, successor: &Block) {
        self.successors.push(successor);
    }
    pub fn add_successors(&self, new_successors: BlockRange);

    /**
    Create a region that should be attached to the operation. These regions can be filled in immediately without waiting for Operation to be created.  When it is, the region bodies will be transferred.
    */
    pub fn add_region(&self) -> *mut Region {

    }

    /**
    Take a region that should be attached to the Operation.  The body of the region will be transferred when the Operation is constructed.  If the region is null, a new empty region will be attached to the Operation.
    */
    pub fn add_region(&self, region: &&std::unique_ptr<Region>) {
        
    }

    /**
    Take ownership of a set of regions that should be attached to the Operation.
    */
    pub fn add_regions(&self, regions: &mut [Region]);

    /// Get the context held by this operation state.
    pub const fn context(&self) -> Option<MLIRContext> {
        self.location.context()
    }
}

// ----------------------------------------------------------------------
// OperandStorage
// ----------------------------------------------------------------------

/**
This class handles the management of operation operands. Operands are stored either in a trailing array, or a dynamically resizable vector.
*/
#[repr(align(8))]
pub struct InputStorage {
    /// The total capacity number of operands that the storage can hold.
    capacity: usize,  // : 31;
    /**
    A flag indicating if the operand storage was dynamically allocated, as opposed to inlined into the owning operation.
    */
    is_storage_dynamic: usize,  // : 1;
    /// The number of operands within the storage.
    len: usize,
    /// A pointer to the operand storage.
    input_storage: *mut Input
}

impl InputStorage {
    /**
    Replace the operands contained in the storage with the ones provided in 'values'.
    */
    pub fn set_inputs(
        &self, owner: &Operation, values: ValueRange) {
        let mut storage_operands = self.resize(owner, values.len());
        for i in 0..values.len() {
            storage_operands[i].set(values[i]);
        }
    }

    /**
    Replace the operands beginning at 'start' and ending at 'start' + 'length' with the ones provided in 'operands'. 'operands' may be smaller or larger than the range pointed to by 'start'+'length'.
    */
    pub fn set_inputs(
        &mut self,
        owner: &Operation,
        start: usize,
        length: usize,
        inputs: ValueRange
    ) {
        // If the new size is the same, we can update inplace.
        let new_size = inputs.len();
        if new_size == length {
            let mut storage_operands = self.inputs();
            for  i in 0..length {
                storage_operands[start + i] = inputs[i];
            }
            return;
        }
        /*
        If the new size is greater, remove the extra operands and set the rest inplace.
        */
        if new_size < length {
            self.remove_inputs(start + operands.len(), length - new_size);
            self.set_operands(owner, start, new_size, operands);
            return;
        }
        // Otherwise, the new size is greater so we need to grow the storage.
        let storage_operands
            = self.resize(owner, self.len() + (new_size - length));

        // Shift operands to the right to make space for the new operands.
        let rotate_size = storage_operands.len() - (start + length);
        let rbegin = storage_operands.rbegin();
        std::rotate(rbegin, rbegin.skip(new_size - length), rbegin + rotate_size);

        // Update the operands inplace.
        for i in 0..operands.len() {
            storage_operands[start + i].set(operands[i]);
        }
    }

    /// Erase the operands held by the storage within the given range.
    pub fn remove_inputs(&self, start: usize, length: usize) {
        let inputs = self.inputs();
        assert!((start + length) <= inputs.len());
        self.len -= length;

        // Shift all operands down if the operand to remove is not at the end.
        if start != self.len {
            let index_it = inputs.iter().skip(start);
            std::rotate(index_it, index_it.skip(length), inputs.end());
        }
        for i in 0..length {
            inputs[self.len + i].drop();
        }
    }

    // /**
    // Erase the operands held by the storage that have their corresponding bit set in `erase_indices`.
    // */
    // pub fn remove_inputs(&mut self, erase_indices: &BitVector) {
    //     let operands = self.operands();
    //     assert!(erase_indices.len() == operands.len());

    //     // Check that at least one operand is erased.
    //     let first_erased_indice = erase_indices.find_first();
    //     if first_erased_indice == -1 {
    //         return;
    //     }

    //     // Shift all of the removed operands to the end, and destroy them.
    //     self.len = first_erased_indice as usize;
    //     for i in (first_erased_indice + 1)..(operands.len()) {
    //         if !erase_indices.test(i) {
    //             operands[self.len] = operands[i];  // move
    //             self.len += 1;
    //         }
    //     }
    //     for operand in operands.drop_front(self.len) {
    //         drop(operand);
    //     }
    // }

    /// Get the operation operands held by the storage.
    pub fn inputs(&self) -> &mut [Input] {
        self.input_storage.as_mut_slice()
    }

    /// Return the number of operands held in the storage.
    pub fn len(&self) -> usize {
        self.len
    }

    /**
    Resize the storage to the given size. Returns the array containing the new operands.
    */
    fn resize(&mut self, owner: &Operation, len: usize) -> &mut [Input]
    {
        /*
        If the number of inputs is less than or equal to the current amount, we can just update in place.
        */
        let mut orig_inputs = self.inputs();
        if len <= self.len {
            /*
            If the number of new size is less than the current, remove any extra operands.
            */
            for i in len..self.len {
                orig_inputs[i].drop();
            }
            self.len = len;
            return orig_inputs.take_mut(..len);
        }

        // If the new size is within the original inline capacity, grow inplace.
        if len <= self.capacity {
            let mut op_begin = orig_inputs.iter_mut().next();
            while self.len != len {
                op_begin[self.len] = Input::new(owner);
                self.len += 1;
            }
            return unsafe { from_raw_parts_mut(op_begin, len) };
        }

        // Otherwise, we need to allocate a new storage.
        let new_capacity = next_power_of2(self.capacity + 2).max(len);
        let new_input_storage =
            reinterpret_cast::<*mut Input>(Global.allocate(Layout::array::<Operand>(new_capacity)?)?);

        // Move the current operands to the new storage.
        let mut new_inputs = from_raw_parts_mut(new_input_storage, len);
        std::uninitialized_move(orig_inputs,
                                new_inputs.begin());

        // Destroy the original inputs.
        for input in orig_inputs {
            input.drop();
        }

        // Initialise any new operands.
        while self.len != len {
            new_inputs[self.len] = Input::new(owner);
            self.len += 1;
        }

        // If the current storage is dynamic, free it.
        if self.is_storage_dynamic {
            free(self.input_storage);
        }

        // Update the storage representation to use the new dynamic storage.
        self.input_storage = new_input_storage;
        self.capacity = new_capacity;
        self.is_storage_dynamic = true;
        new_inputs
    }
}

/**
Set of flags used to control the behaviour of the various IR print methods (e.g. Operation::Print).
*/
pub struct PrintingFlags {
    /**
    Elide large elements attributes if the number of elements is larger than the upper limit.
    */
    elements_attr_element_limit: Option<i64>,

    /// Print debug information.
    print_debug_info_flag: bool, // : 1;
    print_debug_info_pretty_form_flag: bool,  // : 1;

    /// Print operations in the generic form.
    print_generic_op_form_flag: bool,  // : 1;

    /// Skip operation verification.
    assume_verified_flag: bool,  // : 1;

    /// Print operations with numberings local to the current operation.
    print_local_scope: bool, // : 1;

    /// Print users of values.
    print_value_users_flag: bool  // : 1;
}

impl PrintingFlags {
    // /**
    // Enables the elision of large elements attributes by printing a lexically
    // valid but otherwise meaningless form instead of the element data. The
    // `largeElementLimit` is used to configure what is considered to be a
    // "large" ElementsAttr by providing an upper limit to the number of
    // elements.
    // */
    // pub fnelide_large_elements_attrs(int64_t largeElementLimit = 16) -> &Self;

    // /**
    // Enable or disable printing of debug information (based on `enable`). If
    // 'prettyForm' is set to true, debug information is printed in a more
    // readable 'pretty' form. Note: The IR generated with 'prettyForm' is not
    // parsable.
    // */
    // pub fn enable_debug_info(bool enable = true, bool prettyForm = false) -> &Self;

    // /// Always print operations in the generic form.
    // pub fn print_generic_op_form() -> &Self;

    // /// Do not verify the operation when using custom operation printers.
    // pub fn assume_verified() -> &Self;

    // /**
    // Use local scope when printing the operation. This allows for using the
    // printer in a more localized and thread-safe setting, but may not
    // necessarily be identical to what the IR will look like when dumping
    // the full module.
    // */
    // pub fn use_local_scope() -> &Self;

    // /// Print users of values as comments.
    // pub fn print_value_users() -> &Self;

    // /// Return if the given ElementsAttr should be elided.
    // pub const fn should_elide_elements_attr(attr: ElementsAttr) -> bool;

    // /// Return the size limit for printing large ElementsAttr.
    // pub const fn get_large_elements_attr_limit() -> Option<i64>;

    // /// Return if debug information should be printed.
    // pub const fn should_print_debug_info() -> bool;

    // /// Return if debug information should be printed in the pretty form.
    // pub const fn should_print_debug_info_pretty_form() -> bool;

    // /// Return if operations should be printed in the generic form.
    // pub const fn should_print_generic_op_form() -> bool;

    // /// Return if operation verification should be skipped.
    // pub const fn should_assume_verified() -> bool;

    // /// Return if the printer should use local scope when dumping the IR.
    // pub const fn should_use_local_scope() -> bool;

    // /// Return if the printer should print users of values.
    // pub const fn should_print_value_users() -> bool;
}

/**
This class provides utilities for computing if two operations are equivalent.
*/
pub struct OperationEquivalence {
}

/**
A unique fingerprint for a specific operation, and all of it's internal operations.
*/
pub struct OperationFingerPrint {
    hash: [u8; 20]
}
