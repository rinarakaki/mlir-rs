/*!
# Coordinate-scheme sparse tensor representation

This file is part of the lightweight runtime support library for sparse tensor manipulations.  The functionality of the support library is meant to simplify benchmarking, testing, and debugging MLIR code operating on sparse tensors.  However, the provided functionality is **not** part of core MLIR itself.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/ExecutionEngine/SparseTensor/COO.h>
*/

use std::convert::From;

/**
An element of a sparse tensor in coordinate-scheme representation (i.e., a pair of indices and value).  For example, a rank-1 vector element would look like

```text
({i}, a[i])
```

and a rank-5 tensor element would look like

```text
({i,j,k,l,m}, a[i,j,k,l,m])
```

The indices are represented as a (non-owning) pointer into a shared pool of indices, rather than being stored directly in this object.
This significantly improves performance because it: (1) reduces the per-element memory footprint, and (2) centralises the memory management for indices. The only downside is that the indices themselves cannot be retrieved without knowing the rank of the tensor to which this element belongs (and that rank is not stored in this object).
*/
pub struct Element<V> {
    // Pointer into shared index pool
    indices: Option<u64>,
    value: V
}

/**
A memory-resident sparse tensor in coordinate-scheme representation (a collection of `Element`s).  This data structure is used as an intermediate representation; e.g., for reading sparse tensors from external formats into memory, or for certain conversions between different `SparseTensorStorage` formats.

This class provides all the typedefs required by the "Container" concept (<https://en.cppreference.com/w/cpp/named_req/Container>); however, beware that it cannot fully implement that concept since it cannot have a default ctor (because the `dim_sizes` field is const).
Thus these typedefs are provided for familiarity reasons, rather than as a proper implementation of the concept.
*/
pub struct SparseTensorCOO<V> {
    // per-dimension sizes
    dim_sizes: Vec<u64>,  // const
    // all COO elements
    elements: Vec<Element<V>>,
    // shared index pool
    indices: Vec<u64>,
    is_sorted: bool
}

/**
Constructs a new coordinate-scheme sparse tensor with the given sizes and initial storage capacity.

Asserts:
- `dim_sizes` has nonzero size.
- the elements of `dim_sizes` are non-zero.
*/
impl<V> From<(&Vec<u64>, u64)> for SparseTensorCOO<V> {
    fn from(value: (&Vec<u64>, u64)) -> Self {
        
    }
}

// TODO: make a class for capturing known-valid sizes (a la PermutationRef),
// so that `SparseTensorStorage::toCOO` can avoid redoing these assertions.
// Also so that we can enforce the asserts *before* copying into `dim_sizes`.
//
/**
Constructs a new coordinate-scheme sparse tensor with the given sizes and initial storage capacity.

Precondition: `dim_sizes` must be valid for `dim_rank`.

Asserts:
- `dim_rank` is nonzero.
- the elements of `dim_sizes` are non-zero.
*/
impl<V> From<(u64, Option<u64>, u64)> for SparseTensorCOO<V> {
    fn from(value: (u64, Option<u64>, u64)) -> Self {
        
    }
}
