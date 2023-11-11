//! Attr and Type SubElements
//!
//! This file contains interfaces and utilities for querying the sub elements of an attribute or type.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/SubElementInterfaces.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/PatternMatch.cpp>

use crate::{
    mlir::ir::{
        attribute::Attribute,
        operation,
        r#type::Type,
        visitors::WalkResult
    },
    llvm::adt::{
        dense_map::DenseMap,
        dense_set
    }
};

/// A replacement mapping function, which returns either std::nullopt (to signal the element wasn't handled), or a pair of the replacement element and a WalkResult.
type ReplaceFnResult<T> = Option<(T, WalkResult)>;
type ReplaceFn<T> = dyn Fn(T) -> ReplaceFnResult<T>;

/// This class provides a utility for replacing attributes/types, and their sub elements. Multiple replacement functions may be registered.
pub struct AttrTypeReplacer {
    /// The set of replacement functions that map sub elements.
    attribute_replacement_fns: Vec<ReplaceFn<dyn Attribute>>,
    type_replacement_fns: Vec<ReplaceFn<dyn Type>>,

    /// The set of cached mappings for attributes/types.
    attrtibute_map: DenseMap<dyn Attribute, dyn Attribute>,
    type_map: DenseMap<dyn Type, dyn Type>
}

/// This class is used by AttrTypeSubElementHandler instances to walking sub attributes and types.
pub struct AttrTypeSubElementWalker {
    walk_attrs_fn: fn(dyn Attribute),
    walk_types_fn: fn(dyn Type)
}

/// This class is used by AttrTypeSubElementHandler instances to process sub element replacements.
pub struct AttrTypeSubElementReplacements<'a, T> {
    /// The current set of replacements.
    replacements: &'a [T]
}

type AttrSubElementReplacements<'a>
    = AttrTypeSubElementReplacements<'a, Attribute>;
type TypeSubElementReplacements<'a>
    = AttrTypeSubElementReplacements<'a, dyn Type>;

/// This class provides support for interacting with the SubElementInterfaces for different types of parameters. An implementation of this class should be provided for any parameter class that may contain an attribute or type. There are two main methods of this class that need to be implemented:
///
/// - walk
///
///   This method should traverse into any sub elements of the parameter using the provided walker, or by invoking handlers for sub-types.
///
/// - replace
///
///   This method should extract any necessary sub elements using the provided replacer, or by invoking handlers for sub-types. The new post-replacement parameter value should be returned.
pub struct AttrTypeSubElementHandler<T, Enable = ()> {
    _unimplemented: std::marker::PhantomData<T>,
    _unimplemented1: std::marker::PhantomData<Enable>
}
