/*!
# Tensor Dialect

- include <https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Dialect/Tensor/IR/Tensor.h>
*/

use crate::{
    mlir::{
        dialect::{
            bufferisation::ir::bufferisation,
            tensor::ir::tensor::operations::{Cast, ExtractSlice},
            utils::reshape_ops_utils
        },
        interfaces::{
            cast_interfaces,
            control_flow_interfaces,
            destination_style_op_interface,
            infer_type_op_interface,
            parallel_combining_op_interface,
            shaped_op_interfaces,
            side_effect_interfaces,
            tiling_interface,
            view_like_interface
        },
        ir::{
            builders::Builder,
            builtins::types::RankedTensorType,
            dialect,
            location::Location,
            operation::{
                Operation,
                definition::FoldResult,
                implementation
            },
            pattern_match::RewritePatternSet,
            r#type::Type,
            value::{Output, Value}
        }
    },
    llvm::adt::small_vector::SmallVector
};

/**
Returns true if `target` is a ranked tensor type that preserves static information available in the `source` ranked tensor type.
*/
pub fn preserves_static_information(source: Type, target: Type) -> bool {
    
}

/**
Determines whether tensor::Cast casts to a more dynamic version of the source tensor. This is useful to fold a tensor.cast into a consuming op and implement canonicalisation patterns for ops in different dialects that may consume the results of tensor.cast operations. Such foldable tensor.cast operations are typically inserted as `extract_slice` ops and are canonicalised, to preserve the type compatibility of their uses.

Returns true when all conditions are met:

1. source and result are ranked tensors with same element type and rank.
2. the tensor type has more static information than the result

# Example

```mlir
%1 = tensor.cast %0 : tensor<8x16xf32> to tensor<?x?xf32>
%2 = consumer %1 ... : tensor<?x?xf32> ...
```

folds into:

```mlir
%2 = consumer %0 ... : tensor<8x16xf32> ...
```
*/
pub fn can_fold_into_consumer_op(cast_op: Cast) -> bool {

}

/**
Determines whether the tensor::Cast casts to a more static version of the source tensor. This is useful to fold into a producing op and implement canonicaliation patterns with the `tensor.cast` op as the root, but producer being from different dialects. Returns true when all conditions are met:

1. source and result and ranked tensors with same element type and rank.
2. the result type has more static information than the source.

# Example

```mlir
%1 = producer ... : tensor<?x?xf32>
%2 = tensor.cast %1 : tensor<?x?xf32> to tensor<8x16xf32>
```

can be canonicalised to :

```mlir
%2 = producer ... : tensor<8x16xf32>
```

Not all ops might be canonicalisable this way, but for those that can be, this method provides a check that it is worth doing the canonicalisation.
*/
pub fn can_fold_into_producer_op(cast_op: Cast) -> bool {

}

/**
Performs folding of any operand of `op` if it comes from a tensor::Cast that can be folded.
*/
pub fn fold_tensor_cast(op: Option<Operation>
) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

/// Return the dimensions of the given tensor value.
pub fn mixed_sizes(
    builder: &Builder,
    location: Location,
    value: Value 
) -> SmallVector<[FoldResult]> {

}

/**
Create a rank-reducing ExtractSlice @[0 .. 0] with strides [1 .. 1] and appropriate sizes (i.e. `tensor.getSizes()`) to reduce the rank of `tensor` to that of `targetType`.
*/
pub fn create_canonical_rank_reducing_extract_slice_op(
    builder: &Builder,
    location: Location,
    tensor: Value,
    target_type: RankedTensorType
) -> Value
{

}

/**
Create a rank-reducing InsertSliceOp @[0 .. 0] with strides [1 .. 1] and appropriate sizes (i.e. `dest.getSizes()`). The result is a new tensor with rank increased to that of `dest`, obtained by inserting `tensor` into `dest` at the canonical [0 .. 0] position.
*/
pub fn create_canonical_rank_reducing_insert_slice_op(
    builder: &Builder,
    location: Location,
    tensor: Value,
    dest: Value
) -> Value
{

}

/**
This is a helper function for DestinationStyleOpInterface. If there is a destination operand for the given Output, return that operand. Otherwise, return an empty tensor (`tensor.empty`) with the shape of the Output.
Dynamic dimensions are queried via ReifyRankedShapedTypeOpInterface.
*/
pub fn get_or_create_destination(
    builder: &Builder,
    location: Location,
    op_result: Output
) -> Result<Value, Box<dyn std::error::Error>> {
    Err(())
}

/// This is a helper function for DestinationStyleOpInterface. Get or create
/// destinations for every tensor Output of the given op.
pub fn get_or_create_destinations(
    builder: &Builder,
    location: Location,
    op: Option<Operation>,
    result: &SmallVector<[Value]>
) -> Result<(), Box<dyn std::error::Error>> {
    Err(())
}

/// Function to control the folding of constant and extract slice
type ControlConstantExtractSliceFusionFn
    = dyn Fn(ExtractSlice) -> bool;  // std::function

/// Patterns to fold the extract slice op with its constant operand
pub fn populate_fold_constant_extract_slice_patterns(
    patterns: &RewritePatternSet,
    control_fn: &ControlConstantExtractSliceFusionFn
) {
    // default
    // control_fn = |op: ExtractSlice | {
    //     // Disable by default because the folding can generate a large
    //     // constant tensor, which would affect the compile time and storage.
    //     return false;
    //   }
}
