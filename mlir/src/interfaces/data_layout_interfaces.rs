/*!
# Data Layout Interface Decls

Defines the interfaces for the data layout specification, operations to which they can be attached, types subject to data layout and dialects containing data layout entries.

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/DataLayoutInterfaces.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/DataLayoutInterfaces.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Interfaces/DataLayoutInterfaces.cpp>
*/

use core::cell::Cell;

use crate::{
    mlir::ir::{
        builtins::{
            dialect,
            operations,
            types
        },
        dialect::interface,
        operation::Operation,
        r#type::Type
    },
    llvm::{
        adt::{
            dense_map::DenseMap,
            type_switch
        },
        support::{
            alignment::align_to,
            math_extras
        }
    }
};

/**
Attribute interface describing an entry in a data layout specification.

A data layout specification entry is a key-value pair. Its key is either a type, when the entry is related to a type or a class of types, or an identifier, when it is not. `DataLayoutEntryKey` is an alias allowing one to use both key types. Its value is an arbitrary attribute that is interpreted either by the type for type keys or by the dialect containing the identifier for identifier keys. The interface provides a hook that can be used by specific implementations to delegate the verification of attribute fitness for a particular key to the relevant type or dialect.
*/
pub trait DataLayoutEntryInterface {
    
}

/**
Attribute interface describing a data layout specification.

A data layout specification is seen as a sequence of entries, each of which is an attribute implementing the data layout entry interface. It assumes a contiguous underlying storage for entries. The interface provides a hook for implementations to verify the well-formedness of the specification, with a default implementation that verifies the absence of entries with duplicate keys and the well-formedness of each individual entry before dispatching to the type or dialect the entry is associated with.

Data layout specifications may need to be combined in case they appear on nested operations subject to layout, or to ensure the validity of layout modification. Concrete specification attributes must implement the corresponding hook.
*/
pub trait DataLayoutSpecInterface {
}

/**
Interface for operations that can have a data layout specification attached.

The `DataLayout` object, which can be used for data layout queries, can be constructed for such operations. The absence of a data layout specification must be handled without failing.

Concrete operations must implement the hook returning the data layout specification. They may optionally override the methods used in data layout queries, default implementations of which provide predefined answers for built-in types and dispatch to the type interface for all other types. These methods must be idempotent, that is return the same result on repeated queries with the same parameters. They are declared static and therefore have no access to the operation or its attributes. Instead, they receive a list of data layout entries relevant to the request. The entries are known to have passed the spec and entry verifier.
*/
pub trait DataLayoutOpInterface {
    
}

/**
Interface for types subject to data layout.

Types willing to be supported by the data layout subsystem should implement this interface by providing implementations of functions querying their size, required and preferred alignment. Each of these functions accepts as arguments a data layout object that can be used to perform recursive queries in the same scope, and a list of data layout entries relevant to this type.
Specifically, the entries are those that have as key _any instance_ of the same type class as the current type. For example, if IntegerType had implemented this interface, it would have received the entries with keys i1, i2, i8, etc. regardless of the bitwidth of this type. This mechanism allows types to 'interpolate' the results in a type-specific way instead of listing all possible types in the specification.

The list of entries may be empty, in which case the type must provide a reasonable default value. The entries in the list are known to have passed the spec and the entry verifiers, as well as the type-specified verifier if provided.

In case of nested layout specs or spec changes, the type can override a hook indicating whether the outer (old) and the inner (new) spec are compatible.
*/
pub trait DataLayoutTypeInterface {
    
}

/**
An interface to be implemented by dialects that can have identifiers in the data layout specification entries. Provides hooks for verifying the entry validity and combining two entries.
*/
pub struct DataLayoutDialectInterface {

}

/**
The main mechanism for performing data layout queries. Instances of this class can be created for an operation implementing `DataLayoutOpInterface`.
Upon construction, a layout spec combining that of the given operation with all its ancestors will be computed and used to handle further requests. For efficiency, results to all requests will be cached in this object.
Therefore, if the data layout spec for the scoping operation, or any of the enclosing operations, changes, the cache is no longer valid. The user is responsible creating a new DataLayout object after any spec change. In debug mode, the cache validity is being checked in every request.
*/
pub struct DataLayout {
    /// Operation defining the scope of requests.
    scope: *mut Operation,

    /// Caches for individual requests.
    sizes: Cell<DenseMap<dyn Type, usize>>,
    bit_sizes: Cell<DenseMap<dyn Type, usize>>,
    abi_alignments: Cell<DenseMap<dyn Type, usize>>,
    preferred_alignments: Cell<DenseMap<dyn Type, usize>>
}

impl DataLayout {
    /// Returns the layout of the closest parent operation carrying layout info.
    pub fn closest(operation: *mut Operation) -> DataLayout {
    }

    /// Returns the size of the given type in the current scope.
    pub const fn type_size(&self, r#type: dyn Type) -> usize {
    }

    /// Returns the size in bits of the given type in the current scope.
    pub const fn type_size_in_bits(&self, r#type: dyn Type) -> usize {
    }

    /// Returns the required alignment of the given type in the current scope.
    pub const fn type_abi_alignment(&self, r#type: dyn Type) -> usize {
    }

    /// Returns the preferred of the given type in the current scope.
    pub const fn type_preferred_alignment(&self, r#type: dyn Type) -> usize {
    }
}

/**
Reports that the given type is missing the data layout information and exits.
*/
// [[noreturn]]
pub fn report_missing_data_layout(r#type: Type) {
    let mut message = String::new();
    let os = llvm::raw_string_ostream(message);
    os << "Neither the scoping op nor the type class provide data layout information for {}"
       << r#type;
    llvm::report_fatal_error(Twine(os.str()));
}
  
/**
Returns the bitwidth of the index type if specified in the param list.
Assumes 64-bit index otherwise.
*/
pub fn index_bitwidth(params: [dyn DataLayoutEntryInterface]) -> usize {
    if params.is_empty() {
        return 64;
    }
    let attr = params.front().value().cast<IntegerAttr>();
    return attr.value().get_z_ext_value();
}

/**
Default handler for the type size request. Computes results for built-in types and dispatches to the DataLayoutTypeInterface for other types.
*/
pub fn default_type_size(
    r#type: dyn Type,
    data_layout: &DataLayout,
    params: [dyn DataLayoutEntryInterface])
-> usize {
}

/**
Default handler for the type size in bits request. Computes results for built-in types and dispatches to the DataLayoutTypeInterface for other types.
*/
pub fn default_type_size_in_bits(
    r#type: dyn Type,
    data_layout: &DataLayout,
    params: [dyn DataLayoutEntryInterface])
-> usize {
    if let Some(r#type) = r#type.downcast_ref::<IntegerType, FloatType>() {
        return r#type.int_or_float_bit_width();
    }
    if let Some(ctype) = r#type.downcast_ref::<ComplexType>() {
        let et = ctype.element_type();
        let inner_alignment
            = default_preferred_alignment(et, data_layout, params) * 8;
        let inner_size = default_type_size_in_bits(et, data_layout, params);

        // Include padding required to align the imaginary value in the complex
        // type.
        return align_to(inner_size, inner_alignment) + inner_size;
    }

    // Index is an integer of some bitwidth.
    if let Some(r#type) = r#type.downcast_ref::<IndexType>() {
        return data_layout.type_size_in_bits(
            IntegerType::new(r#type.context(), index_bitwidth(params)));
    }

    /*
    Sizes of vector types are rounded up to those of types with closest power-of-two number of elements in the innermost dimension. We also assume there is no bit-packing at the moment element sizes are taken in bytes and multiplied with 8 bits.

    TODO: make this extensible.
    */
    if let Some(vector_type) = r#type.downcast_ref::<VectorType>() {
        return vector_type.num_elements() / vector_type.shape().back() *
           llvm::PowerOf2Ceil(vector_type.shape().back()) *
           data_layout.type_size(vector_type.element_type()) * 8;
    }
    if let Some(type_interface) = r#type.downcast_ref::<DataLayoutTypeInterface>() {
        return type_interface.type_size_in_bits(data_layout, params);
    }
    report_missing_data_layout(r#type);
}

/**
Default handler for the required alignemnt request. Computes results for built-in types and dispatches to the DataLayoutTypeInterface for other types.
*/
pub fn default_abi_alignment(
    r#type: dyn Type,
    data_layout: &DataLayout,
    params: [dyn DataLayoutEntryInterface])
-> usize {
}

/**
Default handler for the preferred alignemnt request. Computes results for built-in types and dispatches to the DataLayoutTypeInterface for other types.
*/
pub fn default_preferred_alignment(
    r#type: dyn Type,
    data_layout: &DataLayout,
    params: [dyn DataLayoutEntryInterface])
-> usize {
}

/**
Given a list of data layout entries, returns a new list containing the entries with keys having the given type ID, i.e. belonging to the same type class.
*/
pub fn filter_entries_for_type(
    entries: [dyn DataLayoutEntryInterface],
    type_id: TypeId)
-> DataLayoutEntryList {
}

/**
Given a list of data layout entries, returns the entry that has the given identifier as key, if such an entry exists in the list.
*/
pub fn filter_entry_for_identifier(
    entries: [dyn DataLayoutEntryInterface],
    id: StringAttr)
-> DataLayoutEntryInterface
{

}

/**
Verifies that the operation implementing the data layout interface, or a module operation, is valid. This calls the verifier of the spec attribute and checks if the layout is compatible with specs attached to the enclosing operations.
*/
pub fn verify_data_layout_op(operation: *mut Operation) -> LogicalResult;

/**
Verifies that a data layout spec is valid. This dispatches to individual entry verifiers, and then to the verifiers implemented by the relevant type and dialect interfaces for type and identifier keys respectively.
*/
pub fn verify_data_layout_spec(
    spec: DataLayoutSpecInterface,
    location: Location) -> LogicalResult
{

}
