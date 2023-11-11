//! TypeRange
//!
//! This file defines the TypeRange and ValueTypeRange classes.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/TypeRange.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/TypeRange.cpp>

use crate::{
    mlir::ir::{
        operation,
        r#type,
        value,
        value_range
    },
    llvm::adt::{
        iterator_range::IteratorRange,
        pointer_union::PointerUnion,
        sequence,
        stl_extras::MappedIterator
    }
};

/// This class provides an abstraction over the various different ranges of value types. In many cases, this prevents the need to explicitly materialise a SmallVector/std::vector. This class should be used in places that are not suitable for a more derived type (e.g. ArrayRef) or a template range parameter.
pub trait TypeRange: MappedIterator {
}


/// This class implements iteration on the types of a given range of values.
pub trait ValueTypeIterator<ValueIterator>: MappedIterator {
}

/// This class implements iteration on the types of a given range of values.
pub trait ValueTypeRange<ValueRange>: IteratorRange {
}

/// Enable TypeRange to be introspected for sub-elements.
pub trait AttrTypeSubElementHandler<TypeRange> {
}
