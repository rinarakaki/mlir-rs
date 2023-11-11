/*!
# Enums for the SparseTensor dialect

Typedefs and enums shared between MLIR code for manipulating the IR, and the lightweight runtime support library for sparse tensor manipulations.  That is, all the enums are used to define the API of the runtime library and hence are also needed when generating calls into the runtime library.  Moveover, the `DimLevelType` enum is also used as the internal IR encoding of dimension level types, to avoid code duplication (e.g., for the predicates).

This file also defines x-macros <https://en.wikipedia.org/wiki/X_Macro> so that we can generate variations of the public functions for each supported primary- and/or overhead-type.

Because this file defines a library which is a dependency of the runtime library itself, this file must not depend on any MLIR internals (e.g., operators, attributes, ArrayRefs, etc) lest the runtime library inherit those dependencies.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SparseTensor/IR/Enums.h>
*/

use std::cmp::PartialEq;

/**
This type is used in the public API at all places where MLIR expects values with the built-in type 'index'. For now, we simply assume that type is 64-bit, but targets with different 'index' bit widths should link with an alternatively built runtime support library.
*/
// TODO: support such targets?
pub type IndexType = u64;

/// Encoding of overhead types (both pointer overhead and indices
/// overhead), for "overloading" @newSparseTensor.
#[repr(align(32))]
pub enum OverheadType {
    Index = 0,
    U64 = 1,
    U32 = 2,
    U16 = 3,
    U8 = 4
}

/// Encoding of the elemental type, for "overloading" @newSparseTensor.
#[repr(align(32))]
pub enum PrimaryType {
    F64 = 1,
    F32 = 2,
    F16 = 3,
    BF16 = 4,
    I64 = 5,
    I32 = 6,
    I16 = 7,
    I8 = 8,
    C64 = 9,
    C32 = 10
}

/// The actions performed by @newSparseTensor.
#[repr(align(32))]
pub enum Action {
    Empty = 0,
    // newSparseTensor no longer handles `kFromFile=1`, so we leave this
    // number reserved to help catch any code that still needs updating.
    FromCOO = 2,
    SparseToSparse = 3,
    EmptyCOO = 4,
    ToCOO = 5,
    ToIterator = 6,
}

/**
This enum defines all the sparse representations supportable by the SparseTensor dialect.  We use a lightweight encoding to encode both the 'format' per se (dense, compressed, singleton) as well as the "properties" (ordered, unique).  The encoding is chosen for performance of the runtime library, and thus may change in future versions; consequently, client code should use the predicate functions defined below, rather than relying on knowledge about the particular binary encoding.

The `Undef` 'format' is a special value used internally for cases where we need to store an undefined or indeterminate `DimLevelType`.
It should not be used externally, since it does not indicate an actual/representable format.
*/
#[derive(PartialEq)]
#[repr(align(8))]
pub enum DimLevelType {
    Undef = 0,           // 0b000_00
    Dense = 4,           // 0b001_00
    Compressed = 8,      // 0b010_00
    CompressedNu = 9,    // 0b010_01
    CompressedNo = 10,   // 0b010_10
    CompressedNuNo = 11, // 0b010_11
    Singleton = 16,      // 0b100_00
    SingletonNu = 17,    // 0b100_01
    SingletonNo = 18,    // 0b100_10
    SingletonNuNo = 19,  // 0b100_11
}

/// This enum defines all the storage formats supported by the sparse compiler,
/// without the level properties.
#[repr(align(8))]
pub enum LevelFormat {
    Dense = 4,      // 0b001_00
    Compressed = 8, // 0b010_00
    Singleton = 16, // 0b100_00
}
