//! IR Dialect Description 
//!
//! This file defines the 'dialect' abstraction.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Dialect.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/Dialect.cpp>

pub mod base;
pub mod implementation;
pub mod interface;
pub mod registry;

use core::{
    any::{Any, TypeId},
    default::Default
};
use regex::Regex;
use llvm::{
    adt::{
        dense_map::DenseMap,
        map_vector,
        twine
    },
    support::{
        debug,
        managed_static,
    }
};
use crate::{
    ir::{
        attribute::{Attribute, NamedAttribute},
        builders::Builder,
        builtins::{
            attributes::StringAttribute,
            dialect,
            types::OpaqueType
        },
        diagnostics,
        dialect::{
            interface::DialectInterface,
            implementation::{DialectAsmParser, DialectAsmPrinter}
        },
        extensible_dialect,
        location::Location,
        mlir_context::MLIRContext,
        operation::{
            Operation,
            implementation::OpAsmPrinter,
            support::{OperationName, OperationState}
        },
        pattern_match::RewritePatternSet,
        r#type::Type
    },
    support::{
        interface_support::Interface,
        logical_result::LogicalResult
    }
};

/// Type for a callback provided by the dialect to parse a custom operation.
/// This is used for the dialect to provide an alternative way to parse custom operations, including unregistered ones.
type ParseOpHook = fn(&OpAsmParser, &OperationState) -> LogicalResult;
//    function_ref

/// Dialects are groups of MLIR operations, types and attributes, as well as behaviour associated with the entire group. For example, hooks into other systems for constant folding, interfaces, default named types for asm printing, etc.
///
/// Instances of the dialect object are loaded in a specific MLIRContext.
pub trait Dialect: Any {
    /// The name(space) of this dialect.
    const NAME: &'static str;

    /// This is the context that owns this Dialect object.
    const CONTEXT: *mut MLIRContext;

    /// Flag that specifies whether this dialect supports unregistered operations, i.e. operations prefixed with the dialect namespace but not registered with addOperation.
    const allows_unknown_operations: bool;

    /// Flag that specifies whether this dialect allows unregistered types, i.e. types prefixed with the dialect namespace but not registered with addType.
    /// These types are represented with OpaqueType.
    const allows_unknown_types: bool;

    /// A collection of registered dialect interfaces.
    const registered_interfaces: DenseMap<TypeId, Box<dyn DialectInterface>>;

    /// Utility function that returns if the given string is a valid dialect namespace
    fn is_valid_namespace(string: &str) -> bool {
        let dialect_name_regex = Regex::new("^[a-zA-Z_][a-zA-Z_0-9\\$]*$");
        dialect_name_regex.matchs(string)
    }

    fn context(&self) -> *mut MLIRContext {
        Self::CONTEXT
    }

    fn name(&self) -> &str {
        Self::NAME
    }

    // /// Returns true if this dialect allows for unregistered operations, i.e. operations prefixed with the dialect namespace but not registered with addOperation.
    // fn allows_unknown_operations(&self) -> bool {
    //     self.allows_unknown_operations
    // }

    // /// Return true if this dialect allows for unregistered types, i.e., types prefixed with the dialect namespace but not registered with addType. These are represented with OpaqueType.
    // fn allows_unknown_types(&self) -> bool {
    //     self.allows_unknown_types
    // }

    /// Register dialect-wide canonicalisation patterns. This method should only be used to register canonicalisation patterns that do not conceptually belong to any single operation in the dialect. (In that case, use the op's canonicaliser.) E.g., canonicalisation patterns for op interfaces should be registered here.
    fn canonicalisation_patterns(&self, results: &RewritePatternSet);

    /// Registered hook to materialize a single constant operation from a given attribute value with the desired resultant type. This method should use the provided builder to create the operation without changing the insertion position. The generated operation is expected to be constant like, i.e. single result, zero operands, non side-effecting, etc. On success, this hook should return the value generated to represent the constant value. Otherwise, it should return null on failure.
    fn materialise_constant(
        &self,
        builder: &Builder,
        value: Attribute,
        r#type: Type,
        location: Location
    ) -> *mut Operation;

    /* Parsing Hooks */

    /// Parse an attribute registered to this dialect. If 'type' is non-null, it refers to the expected type of the attribute.
    fn parse_attribute(&self, parser: &impl DialectAsmParser, r#type: Type)
    -> impl Attribute;

    /// Print an attribute registered to this dialect. Note: The type of the attribute need not be printed by this method as it is always printed by the caller.
    fn print_attribute(
        &self, attribute: Attribute, printer: &DialectAsmPrinter);

    // /// Parse a type registered to this dialect.
    // // virtual
    // fn parse_type(&self, parser: &DialectAsmParser) -> impl Type {
    //     // If this dialect allows unknown types, then represent this with OpaqueType.
    //     if self.allows_unknown_types() {
    //         let ns = StringAttribute::get(self.context(), self.namespace());
    //         return OpaqueType::get(ns, parser.get_full_symbol_spec());
    //     }

    //     parser.emit_error(parser.get_name_loc())
    //         << "dialect '" << self.namespace() << "' provides no type parsing hook";
    //     return Type();
    // }

    /// Print a type registered to this dialect.
    fn print_type(&self, r#type: Type, printer: &DialectAsmPrinter);

    /// Return the hook to parse an operation registered to this dialect, if any.
    /// By default this will lookup for registered operations and return the `parse()` method registered on the RegisteredOperationName. Dialects can override this behaviour and handle unregistered operations as well.
    fn parse_operation_hook(&self, op_name: &str) -> Option<ParseOpHook>;

    /// Print an operation registered to this dialect. This hook is invoked for registered operation which don't override the `print()` method to define their own custom assembly.
    fn operation_printer(
        &self,
        operation: *mut Operation
    ) -> fn(&Operation, &OpAsmPrinter);  // unique_function

    /* Verification Hooks */

    /// Verify an attribute from this dialect on the argument at `arg_index` for the region at `region_index` on the given operation. Returns `Err` if the verification failed, `Ok` otherwise. This hook may optionally be invoked from any operation containing a region.
    fn verify_region_arg_attribute(
        &self,
        operation: *mut Operation,
        region_index: usize,
        arg_index: usize,
        attribute: NamedAttribute
    ) -> LogicalResult;

    /// Verify an attribute from this dialect on the result at 'resultIndex' for the region at 'region_index' on the given operation. Returns failure if the verification failed, success otherwise. This hook may optionally be invoked from any operation containing a region.
    fn verify_region_result_attribute(
        &self,
        operation: *mut Operation,
        region_index: usize,
        result_index: usize,
        attribute: NamedAttribute
    ) -> LogicalResult;

    /// Verify an attribute from this dialect on the given operation. Returns failure if the verification failed, success otherwise.
    fn verify_operation_attribute(
        &self,
        operation: *mut Operation,
        attribute: NamedAttribute
    ) -> LogicalResult;

    /* Interfaces */

    /// Lookup an interface for the given ID if one is registered, otherwise nullptr.
    fn registered_interface(&self, interface_id: TypeId)
        -> Option<DialectInterface>
    {
        match self.registered_interfaces.find(interface_id) {
            None => None,
            Some(it) => it.second().get()
        }
    }

    // fn registered_interface<>(InterfaceT) -> InterfaceT * {
    //     return static_cast<InterfaceT *>(
    //         self.registered_interface(InterfaceT::get_interface_id()));
    // }

    // /// Lookup an op interface for the given ID if one is registered, otherwise nullptr.
    // fn registered_interface_for_op(
    //     &self,
    //     interface_id: TypeId,
    //     op_name: OperationName
    // ) -> *mut ();

    // fn registered_interface_for_op<T: Interface>(
    //     &self,
    //     op_name: OperationName
    // ) -> *mut T::Concept
    // {
    //     return static_cast::<*mut T::Concept>(
    //         self.registered_interface_for_op(
    //             InterfaceT::get_interface_id(), op_name));
    // }

    // /// Register a dialect interface with this dialect instance.
    // fn add_interface(
    //     &self,
    //     interface: std::unique_ptr<DialectInterface>
    // ) {
    //     let it = self.registered_interfaces.try_emplace(
    //         interface.get_id(), interface /* move */);
    //     (void)it;
    //     LLVM_DEBUG({
    //         if (!it.second) {
    //         llvm::dbgs() << "[" DEBUG_TYPE
    //                         "] repeated interface registration for dialect "
    //                     << self.namespace();
    //         }
    //     });
    // }

    // /// Register a set of dialect interfaces with this dialect instance.
    // template <typename... Args>
    // pub fn add_interfaces() {
    //     (self.add_interface(std::make_unique<Args>(this)), ...);
    // }

    // template <typename InterfaceT, typename... Args>
    // pub fn add_interface(Args &&...args) -> &InterfaceT {
    //     InterfaceT *interface = new InterfaceT(this, std::forward<Args>(args)...);
    //     self.add_interface(std::unique_ptr<DialectInterface>(interface));
    //     return *interface;
    // }
}

// impl Default for Dialect {
//     fn default() -> Self {
//         Self {
//             allows_unknown_operations: false,
//             allows_unknown_types: false
//         }
//     }
// }
