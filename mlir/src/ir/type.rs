//! MLIR Type Classes
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Types.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/Types.cpp>

pub mod detail;
pub mod support;

use core::{
    any::{Any, TypeId},
    cmp::PartialEq,
    fmt::Display,
    hash::Hash,
    ptr::Pointee
};

use crate::{
    mlir::{
        ir::{
            builtins::types,
            dialect::Dialect,
            mlir_context::MLIRContext,
            r#type::support::TypeUniquer,
        },
        support::interface_support::{Interface, InterfaceMap},
    },
    llvm::{
        adt::dense_map_info,
        support::pointer_like_type_traits
    }
};

pub type HasTraitFn = Box<dyn Fn(TypeId) -> bool>;  // unique_function

/**
Instances of the Type class are uniqued, have an immutable identifier and an optional mutable component. They wrap a pointer to the storage object owned by MLIRContext. Therefore, instances of Type are passed around by value.

Some types are 'primitives' meaning they do not have any parameters, for example the `Index` type. Parametric types have additional information that differentiates the types of the same class, for example the Integer type has bitwidth, making i8 and i16 belong to the same kind by be different instances of the `IntegerType`. Type parameters are part of the unique immutable key. The mutable component of the type can be modified after the type is created, but cannot affect the identity of the type.

Types are constructed and uniqued via the `TypeUniquer` class.

Derived type classes are expected to implement several required implementation hooks:

- Optional:
   - `static LogicalResult verify(
                               function_ref<InFlightDiagnostic()> emitError,
                               Args... args)`
    - This method is invoked when calling the `TypeBase::get/getChecked`
       methods to ensure that the arguments passed in are valid to construct
       a type instance with.
    - This method is expected to return failure if a type cannot be constructed with `args`, success otherwise.
    - `args` must correspond with the arguments passed into the `TypeBase::get` call.


Type storage objects inherit from `TypeStorage` and contain the following:

- The dialect that defined the type.
- Any parameters of the type.
- An optional mutable component.

For non-parametric types, a convenience `DefaultTypeStorage` is provided. Parametric storage types must derive `TypeStorage` and respect the following:

- Define a type alias, KeyTy, to a type that uniquely identifies the instance of the type.
  - The key type must be constructible from the values passed into the `TypeUniquer::get` call.
  - If the `KeyTy` does not have an `DenseMapInfo` specialisation, the storage class must define a hashing method: `static unsigned hashKey(const KeyTy &)`

- Provide a method, `bool operator==(const KeyTy &) const`, to compare the storage instance against an instance of the key type.

- Provide a static construction method: `DerivedStorage *construct(TypeStorageAllocator &, const KeyTy &key)` that builds a unique instance of the derived storage. The arguments to this function are an allocator to store any uniqued data within the context and the key type for this storage.

- If they have a mutable component, this component must not be a part of the key.
*/
pub trait Type: Any {
    /*
    ====================================================================
    Type
    ====================================================================
    */
}

// Display + Hash + PartialEq

/*
====================================================================
AbstractType
====================================================================

Static information common to all instances of a registered Type.
*/

impl dyn Type {
    /// This is the dialect that this type is registered to.
    const DIALECT: &'static Dialect;

    /// Return the MLIRContext in which this type was uniqued.
    fn context(&self) -> *mut MLIRContext {
        Self::DIALECT.context()
    }
}

pub trait HasInterfaces {
    /// This is a collection of the interfaces registered to this type.
    const INTERFACE_MAP: InterfaceMap;

    // /**
    // Returns an instance of the concept object for the given interface if it is registered to this type, None otherwise. This should not be used directly.
    // */
    // const fn interface<T: ?>(&self) -> Option<T::Concept> {
    //     self.interface_map.lookup::<T>()
    // }

    /// Returns true if the type has the interface with the given id.
    fn has_interface(&self, interface_id: TypeId) -> bool {
        self.interface_map.contains(interface_id)
    }
}

pub trait HasTraits {
    // /// Returns true if the type has a particular trait.
    // template <template <typename T> class Trait>
    // const fn has_trait<Trait>(&self) -> bool {
    //     self.has_trait_fn(TypeId::of::<Trait>());
    // }

    /// Returns true if the type has a particular trait.
    fn has_trait(&self, trait_id: TypeId) -> bool {
        self.has_trait_fn(trait_id)
    }
}

// /// Utility class for implementing types.
// pub type TypeBase<ConcreteType, BaseType, StorageType>
//     = StorageUserBase<ConcreteType, BaseType, StorageType, TypeUniquer>;

/// This class represents the base of a type interface. See the definition  of `Interface` for requirements on the `Traits` type.
// template <typename ConcreteType, typename Traits>
pub trait TypeInterface {

}

/*
Type Utils
*/

// /// We align TypeStorage by 8, so allow LLVM to steal the low bits.
// struct PointerLikeTypeTraits<Type> {
//     const NumLowBitsAvailable = 3;
// }
