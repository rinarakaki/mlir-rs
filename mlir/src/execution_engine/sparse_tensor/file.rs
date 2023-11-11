/*!
# Parsing sparse tensors from files

This file implements parsing and printing of files in one of the following external formats:

(1) Matrix Market Exchange (MME): *.mtx
    <https://math.nist.gov/MatrixMarket/formats.html>

(2) Formidable Repository of Open Sparse Tensors and Tools (FROSTT): *.tns
    <http://frostt.io/tensors/file-formats.html>

This file is part of the lightweight runtime support library for sparse tensor manipulations. The functionality of the support library is meant to simplify benchmarking, testing, and debugging MLIR code operating on sparse tensors.  However, the provided functionality is **not** part of core MLIR itself.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/ExecutionEngine/SparseTensor/File.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/ExecutionEngine/SparseTensor/File.cpp>
*/

use std::{
    default::Default,
    fs::File
};

use crate::mlir::{
    dialect::sparse_tensor::ir::enums::{DimLevelType, PrimaryType},
    execution_engine::sparse_tensor::{
        coo::SparseTensorCOO,
        permutation_ref::PermutationRef,
        storage::SparseTensorStorage
    }
};

// /**
// Returns an element-value of non-complex type.  If `IsPattern` is true, then returns an arbitrary value.  If `IsPattern` is false, then reads the value from the current line buffer beginning at `line_ptr`.
// */
// #[inline]
// pub fn read_coo_value<V, const IS_PATTERN: bool>(
//     line_ptr: char **
// ) -> std::enable_if_t<!is_complex<V>::value, V> {
//     /*
//     The external formats always store these numerical values with the type
//     double, but we cast these values to the sparse tensor object type.
//     For a pattern tensor, we arbitrarily pick the value 1 for all entries.
//     */
//     if constexpr (IsPattern) {
//         return 1.0;
//     }
        
//     return strtod(*line_ptr, line_ptr);
// }


// /**
// Returns an element-value of complex type.  If `IsPattern` is true,
// then returns an arbitrary value.  If `IsPattern` is false, then reads
// the value from the current line buffer beginning at `line_ptr`.
// */
// #[inline]
// pub fn read_coo_value<V, const IS_PATTERN: bool>(
//     line_ptr: char **
// ) -> std::enable_if_t<is_complex<V>::value, V> {
//     /*
//     Read two values to make a complex. The external formats always store numerical values with the type double, but we cast these values to the sparse tensor object type. For a pattern tensor, we arbitrarily pick the value 1 for all entries.
//     */
//     if constexpr (IsPattern) {
//         return V(1.0, 1.0);
//     }
    
//     let re = strtod(*line_ptr, line_ptr);
//     let im = strtod(*line_ptr, line_ptr);
//     // Avoiding brace-notation since that forbids narrowing to `float`.
//     return V(re, im);
// }

/**
Returns an element-value.  If `is_pattern` is true, then returns an arbitrary value. If `is_pattern` is false, then reads the value from the current line buffer beginning at `line_ptr`.
*/
#[inline]
pub fn read_coo_value<V>(line_ptr: *mut char, is_pattern: bool) -> V {
    if is_pattern {
        return read_coo_value::<V, true>(line_ptr);
    }
    return read_coo_value::<V, false>(line_ptr);
}

/*
TODO: benchmark whether to keep various methods inline vs moving them off to the cpp file.
*/

/*
TODO: consider distinguishing separate classes for before vs after reading the header; so as to statically avoid the need to `assert(is_valid())`.
*/

/**
This class abstracts over the information stored in file headers, as well as providing the buffers and methods for parsing those headers.
*/
pub struct SparseTensorReader {
    filename: &'static str,
    file: Option<File>, // = nullptr;
    value_kind: ValueKind,  // = ValueKind::Invalid;
    is_symmetric: bool, // = false;
    idata: [u64; 512],
    line: [char; 1025]
}

impl SparseTensorReader {
    /// Opens the file for reading.
    pub fn open_file(&self) {

    }

    /// Closes the file.
    pub fn close_file(&self) {

    }

    /// Reads and parses the file's header.
    pub fn read_header(&self) {
        
    }

    pub const fn value_kind(&self) -> ValueKind {
        self.value_kind
    }

    /// Checks if a header has been successfully read.
    pub const fn is_valid(&self) -> bool {
        self.value_kind != ValueKind::Invalid
    }

    /**
    Checks if the file's ValueKind can be converted into the given tensor PrimaryType.  Is only valid after parsing the header.
    */
    pub const fn can_read_as(&self, val_ty: PrimaryType) -> bool {

    }

    /**
    Gets the MME "pattern" property setting.  Is only valid after parsing the header.
    */
    pub const fn is_pattern(&self) -> bool {
        assert!(self.is_valid(), "Attempt to is_pattern() before read_header()");
        self.value_kind == ValueKind::Pattern;
    }

    /**
    Gets the MME "symmetric" property setting.  Is only valid after parsing the header.
    */
    pub const fn is_symmetric(&self) -> bool {
        assert!(self.is_valid(), "Attempt to is_symmetric() before read_header()");
        self.is_symmetric
    }

    /// Gets the rank of the tensor.  Is only valid after parsing the header.
    pub const fn rank(&self) -> u64 {
        assert!(self.is_valid(), "Attempt to self.rank() before read_header()");
        self.idata[0]
    }
    
    /// Gets the number of non-zeros.  Is only valid after parsing the header.
    pub const fn nnz(&self) -> u64 {
        assert!(self.is_valid(), "Attempt to self.nnz() before read_header()");
        self.idata[1]
    }

    /**
    Gets the dimension-sizes array.  The pointer itself is always valid; however, the values stored therein are only valid after parsing the header.
    */
    pub const fn dim_sizes(&self) -> Option<u64> {
        self.idata + 2
    }

    /**
    Safely gets the size of the given dimension. Is only valid after parsing the header.
    */
    pub const fn dim_size(&self, d: u64) -> u64 {
        assert!(d < self.rank(), "Dimension out of bounds");
        self.idata[2 + d]
    }

    // /**
    // Asserts the shape subsumes the actual dimension sizes.  Is only valid after parsing the header.
    // */
    // pub const fn assert_matches_shape(&self, rank: u64, shape: Option<u64>);

    /**
    Reads a sparse tensor element from the next line in the input file and returns the value of the element. Stores the coordinates of the element to the `indices` array.
    */
    pub fn read_coo_element<V>(&self, rank: u64, indices: Option<u64>) -> V {
        assert!(rank == self.rank(), "rank mismatch");
        let line_ptr = self.read_coo_indices(indices);
        read_coo_value::<V>(&line_ptr, self.is_pattern())
    }

    // /**
    // Allocates a new COO object for `lvl_sizes`, initialises it by reading all the elements from the file and applying `dim2lvl` to their indices, and then closes the file.

    // Preconditions:

    // - `lvl_sizes` must be valid for `lvl_rank`.
    // - `dim2lvl` must be valid for `self.rank()`.
    // - `dim2lvl` maps indices valid for `self.dim_sizes()` to indices valid for `lvl_sizes`.
    // -= the file's actual value type can be read as `V`.

    // Asserts:

    // - `is_valid()`
    // - `dim2lvl` is a permutation, and therefore also `lvl_rank == self.rank()`.
    // (This requirement will be lifted once we functionalize `dim2lvl`.)
    // */
    // /*
    // NOTE: This method is factored out of `read_sparse_tensor` primarily to
    // reduce code bloat (since the bulk of the code doesn't care about the
    // `<P,I>` type template parameters).  But we leave it public since it's
    // perfectly reasonable for clients to use.
    // */
    // pub fn read_coo<V>(
    //     &self,
    //     lvl_rank: u64,
    //     lvl_sizes: Option<u64>,
    //     dim2lvl: Option<u64>
    // ) -> Option<SparseTensorCOO<V>>;

    /**
    Allocates a new sparse-tensor storage object with the given encoding, initialises it by reading all the elements from the file, and then closes the file.  Preconditions/assertions are as per `read_coo` and `SparseTensorStorage::new_from_coo`.
    */
    pub fn read_sparse_tensor<P, I, V>(
        &self,
        lvl_rank: u64,
        lvl_sizes: Option<u64>,
        lvl_types: Option<DimLevelType>,
        lvl2dim: Option<u64>,
        dim2lvl: Option<u64>
    ) -> Option<SparseTensorStorage<P, I, V>>{
        let lvl_coo = self.read_coo::<V>(lvl_rank, lvl_sizes, dim2lvl);
        let tensor = SparseTensorStorage::<P, I, V>::new_from_coo(
            self.rank(), self.dim_sizes(), lvl_rank, lvl_types, lvl2dim, *lvl_coo);
        tensor
    }

    // /**
    // Attempts to read a line from the file.  Is private because there's no reason for client code to call it.
    // */
    // fn read_line(&self);

    // /**
    // Reads the next line of the input file and parses the coordinates into the `indices` argument.  Returns the position in the `line` buffer where the element's value should be parsed from.  This method has been factored out from `read_coo_element` to minimise code bloat for the generated library.

    // Precondition: `indices` is valid for `getRank()`.
    // */
    // fn read_coo_indices(&self, indices: u64 *) -> char *;

    // /**
    // The internal implementation of `read_coo`. We template over `IsPattern` and `IsSymmetric` in order to perform LICM without needing to duplicate the source code.
    // */
    // /*
    // TODO: We currently take the `dim2lvl` argument as a `PermutationRef` since that's what `read_coo` creates. Once we update `read_coo` to functionalise the mapping, then this helper will just take that same function.
    // */
    // fn read_coo_loop<V, const IS_PATTERN: bool, const IS_SYMMETRIC: bool>(
    //     &self,
    //     lvl_rank: u64,
    //     dim2lvl: PermutationRef,
    //     lvl_coo: SparseTensorCOO<V> *
    // );

    // /// Reads the MME header of a general sparse matrix of type real.
    // fn read_mme_header(&self);

    // /**
    // Reads the 'extended' FROSTT header. Although not part of the
    // documented format, we assume that the file starts with optional
    // comments followed by two lines that define the rank, the number of
    // nonzeros, and the dimensions sizes (one per rank) of the sparse tensor.
    // */
    // fn read_ext_frostt_header(&self);
}

#[derive(Default)]
#[repr(align(8))]
pub enum ValueKind {
    // The value before calling `read_header`.
    #[default]
    Invalid = 0,
    // Values that can be set by `read_mme_header`.
    Pattern = 1,
    Real = 2,
    Integer = 3,
    Complex = 4,
    // The value set by `read_ext_frostt_header`.
    Undefined = 5
}
