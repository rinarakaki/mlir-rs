/*!
This file is part of the lightweight runtime support library for sparse tensor manipulations. The functionality of the support library is meant to simplify benchmarking, testing, and debugging MLIR code operating on sparse tensors.  However, the provided functionality is **not** part of core MLIR itself.

This file contains definitions for the following classes:

- `SparseTensorStorageBase`
- `SparseTensorStorage<P, I, V>`
- `SparseTensorEnumeratorBase<V>`
- `SparseTensorEnumerator<P, I, V>`
- `SparseTensorNNZ`

Ideally we would split the storage classes and enumerator classes into separate files, to improve legibility.  But alas: because these are template-classes, they must therefore provide *definitions* in the header; and those definitions cause circular dependencies that make it impossible to split the file up along the desired lines. (We could split the base classes from the derived classes, but that doesn't particularly help improve legibility.)

This file contains method definitions for `SparseTensorStorageBase`.
In particular we want to ensure that the default implementations of the 'partial method specialisation' trick aren't inline (since there's no benefit). Though this also helps ensure that we avoid weak-vtables: <https://llvm.org/docs/CodingStandards.html#provide-a-virtual-method-anchor-for-classes-in-headers>

This file is part of the lightweight runtime support library for sparse tensor manipulations.  The functionality of the support library is meant to simplify benchmarking, testing, and debugging MLIR code operating on sparse tensors.  However, the provided functionality is **not** part of core MLIR itself.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/ExecutionEngine/SparseTensor/Storage.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/ExecutionEngine/SparseTensor/Storage.cpp>
*/

use crate::mlir::{
    dialect::sparse_tensor::ir::enums::DimLevelType,
    execution_engine::{
        float16bits,
        sparse_tensor::{
            arithmetic_utils,
            attributes,
            coo,
            error_handling
        }
    }
};

/**
Abstract base class for `SparseTensorStorage<P, I, V>`. This class takes responsibility for all the `<P, I, V>`-independent aspects of the tensor (e.g., shape, sparsity, permutation). In addition, we use function overloading to implement 'partial' method specialisation, which the C-API relies on to catch type errors arising from our use of opaque pointers.

Because this class forms a bridge between the denotational semantics of 'tensors' and the operational semantics of how we store and compute with them, it also distinguishes between two different coordinate spaces (and their associated rank, shape, sizes, etc).
Denotationally, we have the *dimensions* of the tensor represented by this object. Operationally, we have the *levels* of the storage representation itself. We use this 'dimension' vs 'level' terminology throughout, since alternative terminology like 'tensor-dimension', 'original-dimension', 'storage-dimension', etc, is both more verbose and prone to introduce confusion whenever the qualifiers are dropped.
Where necessary, we use 'axis' as the generic term.

The *size* of an axis is the cardinality of possible coordinate/index values along that axis (regardless of which coordinates have stored element values).  As such, each size must be non-zero since if any axis has size-zero then the whole tensor would have trivial storage (since there are no possible coordinates). Thus we use the plural term *sizes* for a collection of non-zero cardinalities, and use this term whenever referring to run-time cardinalities.  Whereas we use the term *shape* for a collection of compile-time cardinalities, where zero is used to indicate cardinalities which are dynamic (i.e., unknown/unspecified at compile-time). At run-time, these dynamic cardinalities will be inferred from or checked against sizes otherwise specified. Thus, dynamic cardinalities always have an 'immutable but unknown' value; so the term 'dynamic' should not be taken to indicate run-time mutability.
*/
/*
TODO: we'd like to factor out a class akin to `PermutationRef` for capturing known-valid sizes to avoid redundant validity assertions.
But calling that class `SizesRef` would be a terrible name (and `ValidSizesRef` isn't much better). Whereas, calling it `ShapeRef` would be a lot nicer, but then that conflicts with the terminology introduced above.  So we need to come up with some new terminology for distinguishing things, which allows a reasonable class name too.
*/
pub struct SparseTensorStorageBase {
    dim_sizes: Vec<u64>,  // const 
    level_sizes: Vec<u64>,  // const 
    level_types: Vec<DimLevelType>,  // const
    level2dim: Vec<u64>  // const
}

/**
A memory-resident sparse tensor using a storage scheme based on per-dimension sparse/dense annotations.  This data structure provides a bufferized form of a sparse tensor type. In contrast to generating setup methods for each differently annotated sparse tensor, this method provides a convenient 'one-size-fits-all' solution that simply takes an input tensor and annotations to implement all required setup in a general manner.
*/
pub struct SparseTensorStorage<P, I, V> {
    pointers: Vec<Vec<P>>,
    indices: Vec<Vec<I>>,
    values: Vec<V>,
    /// Cursor for lexicographic insertion.
    level_cursor: Vec<u64>
}

/**
A (higher-order) function object for enumerating the elements of some `SparseTensorStorage` under a permutation.  That is, the `forallElements` method encapsulates the loop-nest for enumerating the elements of the source tensor (in whatever order is best for the source tensor), and applies a permutation to the coordinates/indices before handing each element to the callback.  A single enumerator object can be freely reused for several calls to `forallElements`, just so long as each call is sequential with respect to one another.

N.B., this class stores a reference to the `SparseTensorStorageBase` passed to the constructor; thus, objects of this class must not outlive the sparse tensor they depend on.

Design Note: The reason we define this class instead of simply using `SparseTensorEnumerator<P,I,V>` is because we need to hide/generalize the `<P,I>` template parameters from MLIR client code (to simplify the type parameters used for direct sparse-to-sparse conversion).  And the reason we define the `SparseTensorEnumerator<P,I,V>` subclasses rather than simply using this class, is to avoid the cost of virtual-method dispatch within the loop-nest.
*/
// MLIR_SPARSETENSOR_GSL_POINTER
#[must_use]
pub struct SparseTensorEnumeratorBase<'a, V> {
    source: &'a SparseTensorStorageBase,
    // in target order.
    target_sizes: Vec<u64>,
    // source-levels -> target-dims/levels.
    level2target: Vec<u64>,
    // in target order.
    target_cursor: Vec<u64>
}

/**
Statistics regarding the number of nonzero subtensors in a source tensor, for direct sparse=>sparse conversion a la <https://arxiv.org/abs/2001.02609>.

N.B., this class stores references to the parameters passed to the constructor; thus, objects of this class must not outlive those parameters.

This class does not have the 'dimension' vs 'level' distinction, but since it is used for initialising the levels of a `SparseTensorStorage` object, we use the 'level' name throughout for the sake of consistency.
*/
// MLIR_SPARSETENSOR_GSL_POINTER
#[must_use]
pub struct SparseTensorNNZ<'a> {
    // All of these are in the target storage-order.
    level_sizes: &'a Vec<u64>,
    level_types: &'a Vec<DimLevelType>,
    nnz: Vec<Vec<u64>>
}
