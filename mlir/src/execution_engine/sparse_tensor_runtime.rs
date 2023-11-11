/*!
This header file provides the enums and functions which comprise the public API of the `ExecutionEngine/SparseTensorRuntime.cpp` runtime upport library for the SparseTensor dialect.

This file implements a light-weight runtime support library for manipulating sparse tensors from MLIR.  More specifically, it provides C-API wrappers so that MLIR-generated code can call into the C++ runtime support library.  The functionality provided in this library is meant to simplify benchmarking, testing, and debugging of MLIR code operating on sparse tensors.  However, the provided functionality is **not** part of core MLIR itself.

The following memory-resident sparse storage schemes are supported:

(a) A coordinate scheme for temporarily storing and lexicographically
    sorting a sparse tensor by index (SparseTensorCOO).

(b) A 'one-size-fits-all' sparse tensor storage scheme defined by
    per-dimension sparse/dense annnotations together with a dimension
    ordering used by MLIR compiler-generated code (SparseTensorStorage).

The following external formats are supported:

(1) Matrix Market Exchange (MME): *.mtx
    <https://math.nist.gov/MatrixMarket/formats.html>

(2) Formidable Repository of Open Sparse Tensors and Tools (FROSTT): *.tns
    <http://frostt.io/tensors/file-formats.html>

Two public APIs are supported:

(I) Methods operating on MLIR buffers (memrefs) to interact with sparse
    tensors. These methods should be used exclusively by MLIR
    compiler-generated code.

(II) Methods that accept C-style data structures to interact with sparse
     tensors. These methods can be used by any external runtime that wants
     to interact with MLIR compiler-generated code.

In both cases (I) and (II), the SparseTensorStorage format is externally
only visible as an opaque pointer.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/ExecutionEngine/SparseTensorRuntime.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/ExecutionEngine/SparseTensorRuntime.cpp>
*/

use crate::mlir::{
    dialect::sparse_tensor::ir::enums::DimLevelType,
    execution_engine::{
        c_runner_utils::StridedMemRefType,
        float16bits,
        sparse_tensor::{
            arithmetic_utils,
            coo::SparseTensorCOO,
            error_handling,
            file,
            permutation_ref::PermutationRef,
            storage::SparseTensorStorage
        }
    }
};

/*
Public functions which operate on MLIR buffers (memrefs) to interact
with sparse tensors (which are only visible as opaque pointers externally).
Because these functions deal with memrefs, they should only be used
by MLIR compiler-generated code (or code similarly guaranteed to remain
in sync with MLIR; e.g., internal development tools like benchmarks).

Where appropriate, we use macros to generate all variations of these
functions for each supported primary- and overhead-type.
*/

/**
Wrapper class to avoid memory leakage issues. The `SparseTensorCOO<V>` class provides a standard C++ iterator interface, where the iterator is implemented as per `Vec`'s iterator.  However, for MLIR's usage we need to have an iterator which also holds onto the underlying `SparseTensorCOO<V>` so that it can be freed whenever the iterator is freed.
*/
/*
We name this `SparseTensorIterator` rather than `SparseTensorCOOIterator` for future-proofing, since the use of `SparseTensorCOO` is an implementation detail that we eventually want to change (e.g., to use `SparseTensorEnumerator` directly, rather than constructing the intermediate `SparseTensorCOO` at all).
*/
pub struct SparseTensorIterator<V> {

}

/*
TODO: When using this library from MLIR, the `to_mlir_sparse_tensor`/
`IMPL_CONVERTTOMLIRSPARSETENSOR` and `from_mlir_sparse_tensor`/
`IMPL_CONVERTFROMMLIRSPARSETENSOR` constructs will be codegened away;
therefore, these functions are only used by PyTACO, one place in the
Python integration tests, and possibly by out-of-tree projects.
This is notable because neither function can be easily generalized
to handle non-permutations.  In particular, while we could adjust
the functions to take all the arguments they'd need, that would just
push the problem into client code.  So if we want to generalize these
functions to support non-permutations, we'll need to figure out how
to do so without putting undue burden on clients.
*/

/**
Initialises sparse tensor from an external COO-flavored format.
The `rank` argument is both dimension-rank and level-rank, and the `dim2lvl` argument must be a permutation.
Used by `IMPL_CONVERTTOMLIRSPARSETENSOR`.
*/
// TODO: generalise beyond 64-bit indices.
pub fn to_mlir_sparse_tensor<V>(
    rank: u64,
    nse: u64,
    dim_sizes: Option<u64>,
    values: &[V],
    dim_indices: Option<u64>,
    dim2lvl: Option<u64>,
    lvl_types: &[DimLevelType]
) -> Option<SparseTensorStorage<u64, u64, V>>
{
// #ifndef NDEBUG
//     // Verify that the sparsity values are supported.
//     // TODO: update this check to match what we actually support.
//     for i in 0..rank { 
//         if 
//             lvl_types[i] != DimLevelType::Dense
//             && lvl_types[i] != DimLevelType::Compressed
//         {
//         MLIR_SPARSETENSOR_FATAL(
//             "Unsupported level type: {}\n", lvl_types[i] as u8);
//         }
//     }
// #endif
    // Verify that `dim2lvl` is a permutation of `[0..(rank-1)]`.
    // NOTE: The construction of `lvl_sizes` and `lvl2dim` don't generalize
    // to arbitrary `dim2lvl` mappings.  Whereas constructing `lvl_ind` from
    // `dim_ind` does (though the details would have to be updated, just
    // like for `IMPL_ADDELT`).
    let d2l = PermutationRef::new(rank, dim2lvl);
    // Convert external format to internal COO.
    let lvl_sizes = d2l.push_forward(rank, dim_sizes);
    let lvl_coo = SparseTensorCOO::<V>::new(lvl_sizes, nse);
    let lvl_ind = Vec::with_capacity(rank);
    let dim_ind = dim_indices;
    for i in 0..nse {
        d2l.push_forward(rank, dim_ind, lvl_ind.data());
        lvl_coo.add(lvl_ind, values[i]);
        dim_ind += rank;
    }
    // Return sparse tensor storage format as opaque pointer.
    let lvl2dim = d2l.inverse();
    let tensor = SparseTensorStorage::<u64, u64, V>::new_from_coo(
        rank, dim_sizes, rank, lvl_types, lvl2dim.data(), *lvl_coo);
    tensor
}

/**
Converts a sparse tensor to an external COO-flavoured format.
Used by `IMPL_CONVERTFROMMLIRSPARSETENSOR`.
*/
/*
TODO: Currently, values are copied from SparseTensorStorage to SparseTensorCOO, then to the output.  We may want to reduce the number of copies.

TODO: generalise beyond 64-bit indices, no dim ordering, all dimensions
compressed
*/
pub const fn from_mlir_sparse_tensor<V>(
    tensor: SparseTensorStorage<u64, u64, V>,
    p_rank: Option<u64>,
    p_nse: Option<u64>,
    p_shape: &mut [u64; N],
    p_values: &mut [V; N],
    p_indices: &mut [u64; N]
) {
    let dim_rank = tensor.dim_rank();
    let dim_sizes = tensor.dim_sizes();
    let identity_perm = (0..dim_rank).collect::<Vec<u64>>();
    let coo =
        tensor.to_coo(dim_rank, dim_sizes.data(), dim_rank, identity_perm.data());

    let elements = coo.elements();
    let nse = elements.size();

    let coo_sizes = coo.dim_sizes();
    assert!(coo_sizes.size() == dim_rank, "Rank mismatch");
    let shape = new [u64; dim_rank];
    std::memcpy(
        shape as *mut (),
        coo_sizes.data() as *const (),
        size_of(u64) * dim_rank
    );

    let mut values = new [V; nse];
    let mut indices = new [u64; dim_rank * nse];

    let mut base = 0;
    for i in 0..nse {
        values[i] = elements[i].value;
        for d in 0..dim_rank {
            indices[base + d] = elements[i].indices[d];
        }
        base += dim_rank;
    }

    drop(coo);
    *p_rank = dim_rank;
    *p_nse = nse;
    *p_shape = shape;
    *p_values = values;
    *p_indices = indices;
}

/**
Initialises the memref with the provided size and data pointer.  This is designed for functions which want to "return" a memref that aliases into memory owned by some other object (e.g., `SparseTensorStorage`), without doing any actual copying.  (The "return" is in scarequotes because the `_mlir_ciface_` calling convention migrates any returned memrefs into an out-parameter passed before all the other function parameters.)

We make this a function rather than a macro mainly for type safety reasons.  This function does not modify the data pointer, but it cannot be marked `const` because it is stored into the (necessarily) non-`const` memref.  This function is templated over the `DataSizeT` to work around signedness warnings due to many data types having varying signedness across different platforms.  The templating allows this function to ensure that it does the right thing and never introduces errors due to implicit conversions.
*/
#[inline]
pub const fn alias_into_memref<DataSizeT, T>(
    size: DataSizeT,
    data: *mut T,
    r#ref: &StridedMemRefType<T, 1>
) {
    r#ref.base_ptr = r#ref.data = data;
    r#ref.offset = 0;
    r#ref.sizes[0] = check_overflow_cast::<i64>(size);
    r#ref.strides[0] = 1;
}

/*
Public functions which accept only C-style data structures to interact
with sparse tensors (which are only visible as opaque pointers externally).
*/
