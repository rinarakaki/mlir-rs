/*!
This file contains traits, interfaces, and utilities for defining and querying the side effects of an operation.

This file contains a set of interfaces that can be used to define information about what effects are applied by an operation.

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/SideEffectInterfaces.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/SideEffectInterfaces.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Interfaces/SideEffectInterfaces.cpp>
*/

use std::any::TypeId;

use crate::{
    mlir::{
        interfaces::side_effect_interface_base::EffectOpInterfaceBase,
        ir::{
            attribute::Attribute,
            operation::{
                base::SymbolRefAttr,
                definition
            },
            value::Value
        },
    },
    llvm::adt::{
        pointer_union::PointerUnion,
        small_ptr_set
    }
};

/// This class represents a base class for a specific effect type.
pub struct Effect {
    /// The id of the derived effect class.
    id: TypeId
}

/// This class represents a specific resource that an effect applies to. This
/// class represents an abstract interface for a given resource.
pub struct Resource {
    /// The id of the derived resource class.
    id: TypeId
}

/**
This class represents a specific instance of an effect. It contains the effect being applied, a resource that corresponds to where the effect is applied, and an optional symbol reference or value(either operand, result, or region entry argument) that the effect is applied to, and an optional parameters attribute further specifying the details of the effect.
*/
pub struct EffectInstance<EffectT, P: Attribute> {
    /// The specific effect being applied.
    effect: Option<EffectT>,

    /// The resource that the given value resides in.
    resource: Option<Resource>,

    /// The Symbol or Value that the effect applies to. This is optionally null.
    value: PointerUnion<SymbolRefAttr, Value>,

    /**
    Additional parameters of the effect instance. An attribute is used for type-safe structured storage and context-based uniquing. Concrete effects can use this at their convenience. This is optionally null.
    */
    parameters: P
}

/// This enum is returned from the `getSpeculatability` method in the `ConditionallySpeculatable` op interface.
pub enum Speculatability {
    /// The Operation in question cannot be speculatively executed.  This could be because it may invoke undefined behaviour or have other side effects.
    NotSpeculatable,

    /// The Operation in question can be speculatively executed.  It does not have any side effects or undefined behaviour.
    Speculatable,

    /// The Operation in question can be speculatively executed if all the operations in all attached regions can also be speculatively executed.
    RecursivelySpeculatable,
}

/**
This def represents the definition for the memory effects interface. Users should generally not use this directly, and should instead use `MemoryEffects`.
An interface used to query information about the memory effects applied by an operation.
*/
pub trait MemoryEffectsOpInterface: EffectOpInterfaceBase {
}
