//! Indexed Value-Iterators Range Classes
//!
//! This file defines the ValueRange related classes.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/ValueRange.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/ValueRange.cpp>

use crate::{
    mlir::ir::{
        attribute::NamedAttribute,
        builtins::attributes,
        operation::Operation,
        type_range::TypeRange,
        r#type,
        value
    },
    llvm::adt::{
        pointer_union,
        sequence,
        small_vector::SmallVector
    }
};

/// This class implements the operand iterators for the Operation class.
pub struct InputRange {

}

/// This class represents a contiguous range of operand ranges, e.g. from a `VariadicOfVariadic` operand group.
pub struct InputRangeRange {

}

/// This class provides a mutable adaptor for a range of operands. It allows for setting, inserting, and erasing operands from the given range.
pub struct MutableInputRange {
    /// The owning operation of this range.
    owner: *mut Operation,

    /// The start index of the operand range within the owner operand list, and the length starting from `start`.
    start: usize,
    length: usize,

    /// Optional set of input segments that should be updated when mutating the length of this range.
    input_segments: SmallVector<[InputSegment; 1]>
}

/// A pair of a named attribute corresponding to an operand segment attribute, and the index within that attribute. The attribute should correspond to a dense i32 array attr.
type InputSegment = (usize, NamedAttribute);

/// This class represents a contiguous range of mutable input ranges, e.g. from a `VariadicOfVariadic` operand group.
pub struct MutableInputRangeRange {

}

/// This class implements the result iterators for the Operation class.
pub struct OutputRange {

}

/// This class implements a use iterator for a range of operation results.
/// This iterates over all uses of all results within the given result range.
pub struct UseIterator {

}


/// This class provides an abstraction over the different types of ranges over `Value`s. In many cases, this prevents the need to explicitly materialise a SmallVector/std::vector. This class should be used in places that are not suitable for a more derived type (e.g. ArrayRef) or a template range parameter.
pub struct ValueRange {

}
