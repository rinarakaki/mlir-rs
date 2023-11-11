/*!
# Vector Interfaces

This file implements the operation interfaces for vector operations.

Defines the interface for operations on vectors.

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/VectorInterfaces.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/VectorInterfaces.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Interfaces/VectorInterfaces.cpp>
*/

use crate::mlir::ir::{
    affine_map::AffineMap,
    builtins::{
        attributes::ArrayAttribute,
        types::VectorType,
        type_interfaces::ShapedType
    },
    value::Value,
    value_range::ValueRange
};

/**
Encodes properties of an operation on vectors that can be unrolled.
*/
pub trait VectorUnrollOpInterface: OpInterface {
    /**
    Return the shape ratio of unrolling to the target vector shape `target_shape`. Return `None` if the operation cannot be unrolled to the target vector shape.
    */
    fn shape_for_unroll(&self) -> Option<SmallVector<[i64; 4]>> {
        assert!(self.num_outputs() == 1);
        let vt = self.result().get_type().dyn_cast::<VectorType>();
        if !vt {
            return None;
        }
        SmallVector<[i64; 4]>::new(vt.shape())
    }
}

/**
Encodes properties of a transfer read or write operation.
*/
pub trait VectorTransferOpInterface: OpInterface {
    /// Return the `in_bounds` attribute name.
    fn in_bounds_attr_str_name() -> &'static str {
        "in_bounds"
    }

    /// Return the `permutation_map` attribute name.
    fn permutation_map_attr_str_name() -> &'static str {
        "permutation_map"
    }

    /// Return `true` if dimension `dim` is in-bounds. Return `false` otherwise.
    fn is_dim_in_bounds(&self, dim: usize) -> bool {
        self.is_broadcast_dim(dim)
        || (self.in_bounds()
            && self.in_bounds().cast::<ArrayAttribute>()[dim]
                                .cast::<BoolAttribute>().value())
    }

    /// Return the memref or ranked tensor operand.
    fn source(&self) -> Value;

    /// Return the vector operand or result.
    fn vector(&self) -> Value;

    /// Return the indices operands.
    fn indices(&self) -> ValueRange;

    /// Return the permutation map.
    fn permutation_map(&self) -> AffineMap;

    /// Returns true if the specified dimension is a broadcast.
    fn is_broadcast_dim(&self, index: usize) -> bool {
        let expr = self.permutation_map().result(index);
        return expr.isa::<AffineConstantExpr>() &&
               expr.dyn_cast::<AffineConstantExpr>().value() == 0;
    }

    /**
    Returns true if at least one of the dimensions in the permutation map is a broadcast.
    */
    fn has_broadcast_dim(&self) -> bool {
        for i in 0..self.transfer_rank() {
            if self.is_broadcast_dim(i) {
                return true;
            }
        }
        false
    }

    /// Return the `in_bounds` boolean ArrayAttribute.
    fn in_bounds(&self) -> Option<ArrayAttribute>;

    /// Return the ShapedType.
    fn shaped_type(&self) -> ShapedType {
        self.source().r#type().cast::<ShapedType>()
    }

    /// Return the VectorType.
    fn vector_type(&self) -> VectorType {
        self.vector().r#type().dyn_cast::<VectorType>()
    }

    /**
    Return the mask operand if the op has a mask. Otherwise, return a empty value.
    */
    fn mask(&self) -> Value;

    /**
    Return the mask type if the op has a mask. Otherwise, return an empty VectorType.
    */
    fn mask_type(&self) -> VectorType {
        if self.mask() { self.mask().r#type() } else { VectorType() }
    }

    /**
    Return the number of dimensions that participate in the permutation map.
    */
    fn transfer_rank(&self) -> usize {
        self.permutation_map().num_outputs()
    }

    /**
    Return the number of leading shaped dimensions that do not participate in the permutation map.
    */
    fn leading_shaped_rank(&self) -> usize {
        self.shaped_type().rank() - self.transfer_rank()
    }

    /**
    Returns true if at least one of the dimensions may be out-of-bounds.
    */
    fn has_out_of_bounds_dim(&self) -> bool {
        for index in 0..self.transfer_rank() {
            if !self.is_dim_in_bounds(index) {
                return true;
            }
        }
        false
    }

    /**
    Helper function to account for the fact that `permutation_map` results and `op.indices` sizes may not match and may not be aligned. The first `leading_shaped_rank()` indices may just be indexed and not transferred from/into the vector.

    For example:

    ```mlit
    vector.transfer %0[%i, %j, %k, %c0]
        : memref<?x?x?x?xf32>, vector<2x4xf32>
    ```
    
    with `permutation_map = (d0, d1, d2, d3) -> (d2, d3)`.
    Provide a zip function to coiterate on 2 running indices: `result_index` and
    `indices_index` which accounts for this misalignment.
    */
    fn zip_result_and_indexing(&self, f: fn(i64, i64)) {
        let mut result_index = 0;
        let mut indices_index = self.leading_shaped_rank();
        let e_result = self.transfer_rank();
        while result_index < e_result {
            result_index += 1;
            indices_index += 1;
            f(result_index, indices_index);
        }
    }

    /**
    Return an upper-bound shape accessed by the transfer op within the tensor/memref operand.

    # Example

    ```mlir
    vector.transfer %w0[%i, %j]
        { permutation_map = affine_map<(d0, d1) -> (d1, d0, 0)>}
        : tensor<?x?xf32>, vector<4x2x6xf32>
    ```

    returns a shape `[2, 4]`.
    */
    fn transfer_chunk_accessed(&self) -> SmallVector<[i64]> {
        let mut dim_sizes
            = SmallVector::<[i64]>(self.permutation_map().num_dims(), 0);
        for (dim, size)
        in self.permutation_map().outputs().zip(self.vector_type().shape())
        {
            // Skip broadcast.
            if dim.isa<AffineConstantExpr>() {
                continue;
            }
            dim_sizes[dim.cast<AffineDimExpr>().index()] = size;
        }
        dim_sizes
    }
}
