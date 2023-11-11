/*!
# Builtin Attribute Interfaces

This file contains the definition of the ElementsAttr interface.

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BuiltinAttributeInterfaces.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BuiltinAttributeInterfaces.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/BuiltinAttributeInterfaces.cpp>
*/

use crate::{
    mlir::{
        ir::{
            affine_map,
            attribute,
            builtins::{
                attribute_interfaces,
                type_interfaces,
                types
            },
            diagnostics,
            r#type
        },
        support::logical_result
    },
    llvm::{
        adt::{
            any,
            sequence
        },
        support::raw_ostream
    }
};

/**
This class provides support for indexing into the element range of an `ElementsAttr`. It is used to opaquely wrap either a contiguous range, via `ElementsAttrIndexer::contiguous`, or a non-contiguous range, via `ElementsAttrIndexer::nonContiguous`, A contiguous range is an array-like range, where all of the elements are layed out sequentially in memory. A non-contiguous range implies no contiguity, and elements may even be materialised when indexing, such as the case for a `mapped_range`.
*/
pub struct ElementsAttrIndexer {
    /// A boolean indicating if this range is contiguous or not.
    is_contiguous: bool,
    /// A boolean indicating if this range is a splat.
    is_splat: bool,
}

/// The underlying range state.
enum State {
    ConState(ContiguousState),
    NonConState(NonContiguousState)
}

/// This class contains all of the state necessary to index a contiguous range.
struct ContiguousState {
}

/// This class contains all of the state necessary to index a non-contiguous range.
struct NonContiguousState {
}

/// This class implements a generic iterator for ElementsAttr.
pub struct ElementsAttrIterator<T> {
    _unimplemented: std::marker::PhantomData<T>
}
