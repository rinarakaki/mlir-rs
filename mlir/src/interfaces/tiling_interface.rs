/*!
# Interface for Tiling Operations

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/TilingInterface.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/TilingInterface.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Interfaces/TilingInterface.cpp>
*/

use crate::{
    mlir::{ir::{
        builder::Builder,
        location::Location,
        operation::Operation
    }, support::logical_result::LogicalResult},
    llvm::adt::small_vector::SmallVector
};

/**
Interface for allowing operations to expose information needed to tile them (similar to LinalgOp, but without having access to indexing maps)
*/
pub trait TilingInterface: OperationInterface {
    /// Returns a list of iterator types that describe the number of loops.
    fn loop_iterator_types(&self) -> SmallVector<[IteratorType]> {
        Default::default()
    }
 
    /**
    Returns a list of ranges that describe the loop bounds and step for the loops of the operation.
    */
    fn iteration_domain(&self, builder: &Builder) -> SmallVector<[Range]> {
        Default::default() 
    }

    /**
    Method to generate the tiled implementation of an operation.

    The iteration space of the operation is returned by `iteration_domain`. The caller provides the information of the tile within this iteration space whose implementation the caller needs.
    - `offsets` provides the offset of the tile in the coordinate system of the original iteration space, i.e., if an iteration space dimension had non-zero offset, it must be included in the offset provided here (as opposed to zero-based offset 'relative' to the iteration space).
    - `sizes` provides the size of the tile.

    The method returns the operation that is the tiled implementation.
    */
    fn tiled_implementation(
        &self,
        builder: &Builder,
        offsets: &[FoldResult],
        sizes: &[FoldResult]
    ) -> SmallVector<[* mut Operation]> {
        SmallVector::new()
    }

    /**
    Method to return the position of the result tile computed by the tiled operation.

    Specifies what tile of the result of the original tensor is computed by the tiled implementation. Expects the same `offsets` and `sizes` as used to obtain the tiled implementation of the operation.
    */
    fn output_tile_position(
        &self,
        builder: &Builder,
        result_number: usize,
        offsets: &[FoldResult],
        sizes: &[FoldResult],
        result_offsets: &SmallVector<[FoldResult]>,
        result_sizes: &SmallVector<[FoldResult]>
    ) -> LogicalResult {
        Err(())
    }

    /**
    Method to generate the code that produces a tile of the result.

    Generates the IR that computes the tile of a result of the operation. The `offsets` and `sizes` describe the tile of the output required. This is different from `tiled_implementation` which generates the tiled implementation of the operation given a tile of the iteration space. This method generates a tiled implementation of the operation based on the tile of the result required. This method enables fusion by using tile and fuse. The method returns failure if the operation can't be tiled to generate the result tile. In practical terms this implies it cannot be tiled and fused with its consumers.

    - `offsets` provides the offset of the tile in the coordinate system of the original iteration space, i.e., if an iteration space dimension had non-zero offset, it must be included in the offset provided here (as opposed to zero-based offset 'relative' to the iteration space).
    - `sizes` provides the size of the tile.
    */
    fn generate_output_tile_value(
        &self,
        builder: &Builder,
        result_number: usize,
        offsets: &[FoldResult],
        sizes: &[FoldResult]
    ) -> Result<Value, ()> {
        Err(())
    }

    /**
    Generates the scalar implementation of the operation.

    Given the list `ivs` that represent points in the iteration space (as specified by `iteration_domain()`) returns the scalar operations that represent the computation at that point in the iteration space.
    This method is typically used as the "exit path", i.e. once all transformations are done, this method can be used to lower to scalar code that can then be lowered to LLVM or SPIR-V dialects.
    */
    fn generate_scalar_implementation(
        &self,
        builder: &Builder,
        location: Location,
        ivs: ValueRange
    ) -> LogicalResult {
        Err(())
    }
}

/**
Interface for allowing operations to expose information needed to tile reductions using partial reduction followed by merge. This is complementary to TilingInterface to tile reductions.
*/
pub trait PartialReductionInterface: OperationInterface {
    /**
    Method to generate a tensor initalized with the identity value of the operation reduction. The tensor shape is equal to operation result shape with new dimension for each non zero tile size.
    */
    fn generate_initial_tensor_for_partial_reduction(
        &self,
        builder: &Builder,
        location: Location,
        sizes: &[FoldResult],
        reduction_dim: &[int]
    )
    -> Result<*mut Operation, ()> {
        Err(())
    }

    /**
    Method to generate a tiled version of the operation where the tiled reduction dimension are converted to parallel dimensions with a size less or equal to the tile size. This is meant to be used with `merge_reductions` method which will combine the partial reductions.
    */
    fn tile_to_partial_reduction(
        &self,
        builder: &Builder,
        location: Location,
        init: ValueRange,
        offsets: &[FoldResult],
        sizes: &[FoldResult],
        reduction_dims: &[int]
    ) -> *mut Operation {
        null()
    }

    /**
    Method to merge partial reductions for an operation that has been tiled along the reduction dimensions. This will only apply the reduction the operation.
    */
    fn merge_reductions(
        &self,
        builder: &Builder,
        location: Location,
        partial_reduce: ValueRange,
        reduction_dim: &[int]
    ) -> *mut Operation {
        null()
    }
}
