//! MLIR Attribute Classes
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Attributes.h>
//! - lib
//!   - <https://github.com/llvm/llvm-project/blob/main/llvm/lib/IR/AttributeImpl.h>
//!   - <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/Attributes.cpp>

pub mod detail;
pub mod support;

use core::{
    any::{Any, TypeId},
    cmp::{PartialEq, PartialOrd},
    fmt::Display,
    hash::Hash
};
use llvm::{
    adt::folding_set::FoldingSet,
    support::pointer_like_type_traits
};
use crate::{
    ir::{
        builtins::attributes::StringAttribute,
        dialect::Dialect,
        mlir_context::MLIRContext
    },
    support::interface_support::{Interface, InterfaceMap},
};

pub type HasTraitFn = Box<dyn Fn(TypeId) -> bool>;  // unique_function;

/// Attributes are known-constant values of operations.
///
/// Instances of the Attribute class are references to immortal key-value pairs with immutable, uniqued keys owned by MLIRContext. As such, an Attribute is a thin wrapper around an underlying storage pointer. Attributes are usually passed by value.
pub trait Attribute: Any {
    /*
    ====================================================================
    AbstractAttribute
    ====================================================================

    Static information common to all instances of a registered Attribute.
    */

    // /// This is the dialect that this attribute is registered to.
    // const DIALECT: &'static Dialect;

    // /// This is a collection of the interfaces registered to this attribute.
    // const interface_map: InterfaceMap;

    // /// Function to check if the attribute has a particular trait.
    // const has_trait_fn: HasTraitFn;

    // /**
    // Returns an instance of the concept object for the given interface if it is registered to this attribute, None otherwise. This should not be used directly.
    // */
    // const fn interface<T: ?>(&self) -> Option<T::Concept> {
    //     self.interface_map.lookup::<T>()
    // }

    // /// Returns true if the attribute has the interface with the given ID.
    // const fn has_interface(&self, interface_id: TypeId) -> bool {
    //     self.interface_map.contains(interface_id)
    // }

    // // /// Returns true if the attribute has a particular trait.
    // // template <template <typename T> class Trait>
    // // pub const fn has_trait<Trait>(&self) -> bool {
    // //     self.has_trait_fn(TypeId::of::<Trait>());
    // // }

    // /// Returns true if the attribute has a particular trait.
    // const fn has_trait(&self, trait_id: TypeId) -> bool {
    //     self.has_trait_fn(trait_id)
    // }

    // /// Return the context this attribute belongs to.
    // const fn context(&self) -> Option<MLIRContext> {
    //     Self::DIALECT.context()
    // }
}

// Display + Hash + PartialEq

/// NamedAttribute represents a combination of a name and an Attribute value.
#[derive(PartialEq)]
pub struct NamedAttribute {
    /// The name of the attribute. This is represented as a StringAttribute, but type-erased to Attribute in the field.
    name: Box<dyn Attribute>,
    /// The value of the attribute.
    value: Box<dyn Attribute>
}

impl NamedAttribute {
    pub fn new(name: impl Attribute, value: impl Attribute) -> Self {
        assert!(name && value, "Expected valid attribute name and value");
        assert!(name.len() != 0, "Expected valid attribute name");
        Self { name, value }
    }

    /// Return the name of the attribute.
    pub const fn name(&self) -> StringAttribute {
        self.name.cast::<StringAttribute>()
    }

    /// Return the dialect of the name of this attribute, if the name is prefixed by a dialect namespace. For example, `llvm.fast_math` would return the LLVM dialect (if it is loaded). Returns nullptr if the dialect isn't loaded, or if the name is not prefixed by a dialect namespace.
    pub const fn name_dialect(&self) -> Option<Dialect> {
        self.name().referenced_dialect()
    }

    /// Return the value of the attribute.
    pub const fn value(&self) -> impl Attribute {
        self.value
    }

    /// Set the name of this attribute.
    pub fn set_name(&mut self, name: StringAttribute) {
        assert!(self.name, "Expected valid attribute name");
        self.name = name;
    }

    /// Set the value of this attribute.
    pub fn set_value(&mut self, value: impl Attribute) {
        assert!(self.value, "Expected valid attribute value");
        self.value = value;
    }
}

/// This class represents the base of an attribute interface. See the definition of `detail::Interface` for requirements on the `Traits` type.
pub trait AttributeInterface: Interface {

}

// ----------------------------------------------------------------------
// Core AttributeTrait
// ----------------------------------------------------------------------

// /// Allow LLVM to steal the low bits of Attributes.
// struct PointerLikeTypeTraits<Attribute> {
//     static inline void *getAsVoidPointer(mlir::Attribute attr) {
//         return const_cast<void *>(attr.getAsOpaquePointer());
//     }
//     static inline mlir::Attribute getFromVoidPointer(void *ptr) {
//         return mlir::Attribute::getFromOpaquePointer(ptr);
//     }
//     static constexpr int NumLowBitsAvailable
//         = PointerLikeTypeTraits<AttributeStorage *>::NumLowBitsAvailable;
// }
