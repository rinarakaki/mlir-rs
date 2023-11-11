/*!
# Tensor Operation Definitions

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Tensor/IR/TensorOps.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Tensor/IR/TensorOps.cpp>
*/

use crate::mlir::{ir::{
    builtins::types::RankedTensorType,
    operation::{base::AnyRankedTensor, asm_interface::OpAsmOpInterface}
}, interfaces::cast_interfaces::CastOpInterface};

pub trait TensorOp {}

// // Base class for ops with static/dynamic offset, sizes and strides
// // attributes/arguments.
// pub struct Tensor_OpWithOffsetSizesAndStrides<string mnemonic,
//                                          list<Trait> traits = []>
//     : Tensor_Op<mnemonic, traits> {
//   code extraBaseClassDeclaration = [{
//     /// Returns the dynamic sizes for this subview operation if specified.
//     ::mlir::Operation::operand_range getDynamicSizes() { return getSizes(); }

//     /// Return the list of Range (i.e. offset, size, stride). Each
//     /// Range entry contains either the dynamic value or a ConstantIndexOp
//     /// constructed with `b` at location `loc`.
//     ::mlir::SmallVector<::mlir::Range, 8> getOrCreateRanges(
//         ::mlir::OpBuilder &b, ::mlir::Location loc) {
//       return ::mlir::getOrCreateRanges(*this, b, loc);
//     }
//   }];
// }

// ----------------------------------------------------------------------
// Cast
// ----------------------------------------------------------------------

/**
Tensor cast operation.

Convert a tensor from one type to an equivalent type without changing any data elements. The source and destination types must both be tensor types with the same element type. If both are ranked, then the rank should be the same and static dimensions should match. The operation is invalid if converting to a mismatching constant dimension.

# Examples

Convert from unknown rank to rank 2 with unknown dimension sizes:

```mlir
%2 = tensor.cast %1 : tensor<*xf32> to tensor<?x?xf32>
```

Convert to a type with more known dimensions:

```mlir
%3 = tensor.cast %2 : tensor<?x?xf32> to tensor<4x?xf32>
```

Discard static dimension and rank information:

```mlir
%4 = tensor.cast %3 : tensor<4x?xf32> to tensor<?x?xf32>
%5 = tensor.cast %4 : tensor<?x?xf32> to tensor<*xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$source attr-dict `:` type($source) `to` type($dest)"
)]
pub struct Cast<T, const I: usize, const O: usize> {
    #[input]
    input: Tensor<T, N>,  // renamed from `source`
    #[output]
    output: Tensor<T, O>  // renamed from `result`
}

impl Canonicalise for Cast {

}

impl CastOpInterface for Cast {

}

impl OpAsmOpInterface for Case {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        
    }
}

// ----------------------------------------------------------------------
// Dim
// ----------------------------------------------------------------------

/**
Dimension index operation.

The `tensor.dim` operation takes a tensor and a dimension operand of type `index`. It returns the size of the requested dimension of the given tensor. If the dimension index is out of bounds, the behaviour is undefined.

The specified tensor type is that of the first operand.

# Examples

Always returns 4, can be constant folded:

```mlir
%c0 = arith.constant 0 : index
%x = tensor.dim %A, %c0 : tensor<4x?xf32>
```

Returns the dynamic dimension of %A:

```mlir
%c1 = arith.constant 1 : index
%y = tensor.dim %A, %c1 : memref<4x?xf32>
```

Equivalent generic form:

```mlir
%x = "tensor.dim"(%A, %c0) : (memref<4x?xf32>, index) -> index
%y = "tensor.dim"(%A, %c1) : (memref<4x?xf32>, index) -> index
```
*/
#[mlir(
    traits = [, NoMemoryEffect],
    assembly_format = "attr-dict $source `,` $index `:` type($source)"
)]
pub struct Dim<T, const N: usize> {
    #[input]
    input: Tensor<T, N>,  // renamed from `source`
    #[input]
    index: usize,
    #[output]
    output: usize  // renamed from `result`
}

// let builders = [
//     Builder<(ins "Value":$source, "i64":$index)>
// ];

impl Dim {
    /// Helper function to get the index as a simple integer if it is constant.
    pub fn constant_index() -> Option<i64>;
}

impl Verify for Dim {

}

impl Fold for Dim {

}

impl Canonicalise for Dim {

}

impl OpAsmOpInterface for Dim {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        
    }
}

impl ConditionallySpeculatable for Dim {
// /// Interface method for ConditionallySpeculatable.
    // Speculation::Speculatability getSpeculatability();
}

impl ShapedDimOpInterface for Dim {
    /// Returns the source tensor.
    fn shaped_value(&self) -> Value {
        self.input
    }

    /// Returns the dimension.
    fn dimension(&self) -> FoldResult {
        self.index
    }
}

// ----------------------------------------------------------------------
// Empty
// ----------------------------------------------------------------------

/**
Empty tensor operation.

`tensor.empty` is an operation that defines a tensor of a particular shape.
The shape could be dynamic or static. The contents of the tensor are unspecified and the only purpose of the op result is to materialise the specified shape in IR and make it available to other transformations.

`tensor.empty` is useful in transformations that expect destination style ops. I.e., ops that implement `DestinationStyleOpInterface`. Ops that are not in destination style can be made compatible with such transformations with a `tensor.empty` destination.

Note: This op can be lowered to a `bufferization.alloc_tensor`, at which point it turns into an explicit buffer allocation.
*/
#[mlir(
    traits = [Pure],
    assembly_format = "`(`$dynamic_sizes`)` attr-dict `:` type($output)"
)]
pub struct Empty<T, const D: usize, const O: usize> {
    #[input]
    dynamic_sizes: [usize; D],
    #[output]
    output: Tensor<T, O>  // AnyRankedTensor  renamed from `result`
}

    // let builders = [
    //     // Build with fully static sizes.
    //     OpBuilder<(ins "&[i64]":$staticShape, "Type":$elementType,
    //                 CArg<"Attribute", "{}">:$encoding)>,

    //     // Build with mixed static/dynamic sizes.
    //     OpBuilder<(ins "&[i64]":$staticShape, "Type":$elementType,
    //                 "ValueRange":$dynamic_sizes,
    //                 CArg<"Attribute", "{}">:$encoding)>,

    //     // Build with mixed static/dynamic sizes.
    //     OpBuilder<(ins "&[FoldResult]":$sizes, "Type":$elementType,
    //                 CArg<"Attribute", "{}">:$encoding)>
    // ];

impl Empty {
    // pub fn r#type(&self) -> RankedTensorType {
    //     return get_result().get_type().cast<RankedTensorType>();
    // }

    // /// Return both static and dynamic sizes as a list of `FoldResult`.
    // pub fn mixed_sizes(&self) -> SmallVector<[FoldResult]> {}

    // /// Return the Value of the dynamic size of the tensor at dimension `index`.
    // /// Asserts that the shape is dynamic at that `index`.
    // pub fn dynamic_size(&self, index: usize) -> Value {}
}

impl Verify for Empty {

}

impl Canonicalise for Empty {

}

impl ReifyRankedShapedTypeOpInterface for Empty {

}

// ----------------------------------------------------------------------
// Extract
// ----------------------------------------------------------------------

/**
Element extraction operation.

The `tensor.extract` op reads a ranked tensor and returns one element as specified by the given indices. The result of the op is a value with the same type as the elements of the tensor. The arity of indices must match the rank of the accessed value. All indices should all be of `index` type.

# Examples

```mlir
%4 = tensor.extract %t[%1, %2] : tensor<4x4xi32>
```

```mlir
%5 = tensor.extract %rt[%1, %2] : tensor<?x?xi32>
```
*/
#[mlir(
    traits = [Pure]
    assembly_format = "$tensor `[` $indices `]` attr-dict `:` type($tensor)"
)]
pub struct Extract<T, const N: usize> {
    #[input]
    tensor: RankedTensor<T, N>,
    #[input]
    indices: [usize; N],
    #[output]
    output: T  // renamed from `result`
}

// let builders = [
//     OpBuilder<(ins "Value":$tensor, CArg<"ValueRange", "{}">:$indices), [{
//     auto resType = tensor.get_type().cast<ShapedType>().getElementType();
//     build($_builder, $_state, resType, tensor, indices);
//     }]>];


impl Verify for Extract {

}

impl Fold for Extract {

}

impl Canonicalise for Extract {

}

impl OpAsmOpInterface for Extract {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        set_name(self.output, "extracted");
    }
}

// ----------------------------------------------------------------------
// ExtractSlice
// ----------------------------------------------------------------------

/**
Extract slice operation.

The `extract_slice` operation extract a tensor from another tensor as specified by the operation's offsets, sizes and strides arguments.

The extract_slice operation supports the following arguments:

- source: the "base" tensor from which to extract a slice.
- offsets: tensor-rank number of offsets into the "base" tensor from which
            to extract the slice.
- sizes: tensor-rank number of sizes which specify the sizes of the result
          tensor type.
- strides: tensor-rank number of strides specifying subsampling in each
            dimension.

The representation based on offsets, sizes and strides support a partially-static specification via attributes specified through the `static_offsets`, `static_sizes` and `static_strides` arguments. A special sentinel value ShapedType::kDynamic and
ShapedType::kDynamic encodes that the corresponding entry has a dynamic value.

After buffer allocation, the "extract_slice" op is expected to lower into a memref.subview op.

An extract_slice operation may additionally reduce the rank of the resulting tensor by removing dimensions that are statically known to be of size 1.
This rank-reduction behaviour is not required by the op semantics: this flexibility allows to progressively drop unit dimensions while lowering between different flavors of ops on that operate on tensors.

## Verification vs Inference in the rank-reduced case

Note that there may be multiple ways to infer a resulting rank-reduced type.
  e.g. 1x6x1 could potentially rank-reduce to either 1x6 or 6x1 2-D shapes.

To disambiguate, the inference helpers `infer_canonical_rank_reduced_result_type` only drop the first unit dimensions, in order:
  e.g. 1x6x1 rank-reduced to 2-D will infer the 6x1 2-D shape, but not 1x6.

Verification however has access to result type and does not need to infer.
The verifier calls `isRankReducedType(get_source(), get_result())` to determine whether the result type is rank-reduced from the source type.
This computes a so-called rank-reduction mask, consisting of dropped unit dims, to map the rank-reduced type to the source type by dropping ones:

  e.g. 1x6 is a rank-reduced version of 1x6x1 by mask {2}
        6x1 is a rank-reduced version of 1x6x1 by mask {0}
        1x2x1x4 is a rank-reduced version of 1x1x2x1x1x4x1 by mask {1, 4, 6}
          (remaining common 1 dimensions are matched eagerly)

# Examples

Rank-reducing extract_slice:

```mlir
%1 = tensor.extract_slice %0[0, 0, 0][1, 16, 4][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<16x4xf32>
```

```mlir
%3 = tensor.extract_slice %2[%o0, 4, %o2][1, %sz1, 1][1, %st1, 1]
    : tensor<8x16x4xf32> to tensor<1x?xf32>
```
*/
#[mlir(
    // Tensor_OpWithOffsetSizesAndStrides
    traits = [AttrSizedOperandSegments, Pure],
    assembly_format = "$source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)"
)]
pub struct ExtractSlice<T, const N: usize> {
    #[input]
    source: RankedTensor<T, N>,
    #[input]
    offsets: [usize; _],
    #[input]
    sizes: [usize; _],
    #[input]
    strides: [usize; _],
    #[attribute]
    static_offsets: DenseI64ArrayAttr,
    #[attribute]
    static_sizes: DenseI64ArrayAttr,
    #[attribute]
    static_strides: DenseI64ArrayAttr,
    #[output]
    output: RankedTensor<_, _>  // renamed from `result`
}

// let builders = [
//     // Build an ExtractSlice with mixed static and dynamic entries and
//     // inferred result type.
//     OpBuilder<(ins "Value":$source, "&[FoldResult]":$offsets,
//     "&[FoldResult]":$sizes, "&[FoldResult]":$strides,
//     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
//     // Build an ExtractSlice with mixed static and dynamic entries and custom
//     // result type. If the type passed is nullptr, it is inferred.
//     OpBuilder<(ins "RankedTensorType":$resultType, "Value":$source,
//     "&[FoldResult]":$offsets, "&[FoldResult]":$sizes,
//     "&[FoldResult]":$strides,
//     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
//     // Build an ExtractSlice with dynamic entries and custom result type. If
//     // the type passed is nullptr, it is inferred.
//     OpBuilder<(ins "Value":$source, "ValueRange":$offsets,
//     "ValueRange":$sizes, "ValueRange":$strides,
//     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
//     // Build an ExtractSlice with dynamic entries and inferred result type.
//     OpBuilder<(ins "RankedTensorType":$resultType, "Value":$source,
//     "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
//     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
//     // Build an ExtractSlice with mixed static and dynamic entries packed in
//     // a Range vector.
//     OpBuilder<(ins "Value":$source, "ArrayRef<Range>":$ranges,
//     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
// ];

impl ExtractSlice {
    // /// Returns the type of the base tensor operand.
    // pub fn source_type(&self) -> RankedTensorType {
    //     return get_source().get_type().cast<RankedTensorType>();
    // }

    // /// The result of an extract_slice is always a tensor.
    // RankedTensorType get_type() {
    // return get_result().get_type().cast<RankedTensorType>();
    // }

    // /// Compute the rank-reduction mask that can be applied to map the source
    // /// tensor type to the result tensor type by dropping unit dims.
    // Option<llvm::SmallDenseSet<usize>>
    // compute_rank_reduction_mask() {
    // return ::mlir::compute_rank_reduction_mask(get_source_type().getShape(),
    //                                         get_type().getShape());
    // };

    // /**
    //  An extract_slice result type can be inferred, when it is not
    // rank-reduced, from the source type and the static representation of
    // offsets, sizes and strides. Special sentinels encode the dynamic case.
    // */
    // static RankedTensorType infer_result_type(
    //     ShapedType sourceShapedTensorType,
    //     &[i64] staticOffsets,
    //     &[i64] staticSizes,
    //     &[i64] staticStrides);
    // static RankedTensorType infer_result_type(
    //     ShapedType sourceShapedTensorType,
    //     &[FoldResult] staticOffsets,
    //     &[FoldResult] staticSizes,
    //     &[FoldResult] staticStrides);

    // /**
    //  If the rank is reduced (i.e. the desiredResultRank is smaller than the
    // number of sizes), drop as many size 1 as needed to produce an inferred type
    // with the desired rank.

    // Note that there may be multiple ways to compute this rank-reduced type:
    // e.g. 1x6x1 can rank-reduce to either 1x6 or 6x1 2-D tensors.

    // To disambiguate, this function always drops the first 1 sizes occurrences.
    // */
    // static RankedTensorType infer_canonical_rank_reduced_result_type(
    //     usize resultRank,
    //     RankedTensorType sourceRankedTensorType,
    //     &[i64] staticOffsets,
    //     &[i64] staticSizes,
    //     &[i64] staticStrides);
    // static RankedTensorType infer_canonical_rank_reduced_result_type(
    //     usize resultRank,
    //     RankedTensorType sourceRankedTensorType,
    //     &[FoldResult] staticOffsets,
    //     &[FoldResult] staticSizes,
    //     &[FoldResult] staticStrides);

    // /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    // /// and `static_strides` attributes.
    // pub fn get_array_attr_max_ranks() -> [usize; 3] {
    //     usize rank = get_source_type().get_rank();
    //     [rank, rank, rank]
    // }

    // /// Return the number of leading operands before the `offsets`, `sizes` and
    // /// and `strides` operands.
    // static usize get_offset_size_and_stride_start_operand_index() { return 1; }

    // /// Return the dimensions of the source that are dropped in the
    // /// result when the result is rank-reduced.
    // SmallBitVector get_dropped_dims();

    // /**
    //  Given a `value`, asserted to be of RankedTensorType, build an
    // ExtractSlice that results in a rank-reducing extract to the desired
    // tensor shape and return the new value created.
    // If the shape of `value` is already the `desiredShape`, just return
    // `value`.
    // If the shape of `value` cannot be rank-reduced to `desiredShape`, fail.
    // */
    // static Result<Value> rank_reduce_if_needed(
    // OpBuilder &b, Location loc, Value value, &[i64] desiredShape);
}

impl Verify for ExtractSlice {

}

impl Fold for ExtractSlice {

}

impl Canonicalise for ExtractSlice {

}

impl OpAsmOpInterface for ExtractSlice {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        set_name(self.output, "extracted_slice");
    }
}

impl ReifyRankedShapedTypeOpInterface for ExtractSlice {

}

impl OffsetSizeAndStrideOpInterface for ExtractSlice {

}

// ----------------------------------------------------------------------
// FromElements
// ----------------------------------------------------------------------

/**
`tensor.from_elements` operation.

Create an N-D tensor from a range of same-type elements. The number of provided `elements` should equal to the number of the elements in the result type. The `elements` correspond to a flattened tensor.

# Example

```mlir
tensor.from_elements %a, %b, %c, %d, %e, %f :  tensor<2x3xindex>
```

will result in a tensor

```text
[[%a, %b, %c]
  [%d, %e, %f]]
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$elements attr-dict `:` type($output)"
)]
pub struct FromElements<T, const N: usize> {
    #[input]
    elements: [T; N],
    #[output]
    output: AnyStaticShapeTensor<T>  // renamed from `result`
}

// let skipDefaultBuilders = 1;
// let builders = [
//     OpBuilder<(ins "Type":$resultType, "ValueRange":$elements)>,
//     // Special case builder for when `elements` has size >=1.
//     OpBuilder<(ins "ValueRange":$elements)>
// ];

impl Fold for FromElements {

}

impl Canonicalise for FromElements {

}

impl OpAsmOpInterface for FromElements {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        
    }
}

// ----------------------------------------------------------------------
// Gather
// ----------------------------------------------------------------------

/**
Gather a subset of a tensor at specified indices.

The `gather` operation extracts a subset of the elements from a `source` tensor at the given indices.

In its most general form, the tensor of indices specifies all the coordinates of every element to extract (i.e. COO format, without the payload).
The indices are expected to be confined to coordinate values that fit the range of the `source` tensor, otherwise the behaviour is undefined.

The leading dimensions of the index tensor give the result tensor its leading dimensions. The trailing dimensions of the result tensor are obtained from the source tensor by omitting the dimensions specified in `dims` (rank-reducing semantics) or setting them to `1` (rank-preserving semantics) (see examples).
The trailing dimension of the index tensor contains the coordinates and is expected to have its size equal to the number of dimensions being gathered.
This convention allows an idiomatic specification and lowering of "gathering multiple N-D slices from the source tensor".

Note: in the examples below, we separate out the indexing part of the tensor type by a whitespace for readability purposes.

# Examples

For each 1x2 triple of coordinates in %indices, extract the element (i.e. 0-D subset) at the coordinates triple in %source:

NOTE: result type may be further rank-reduced to `tensor<1x2x f32>`.

```mlir
%out = tensor.gather %source[%indices] dims([0, 1, 2])
    : (tensor<4x4x4xf32>, tensor<1x2x 3xindex>) -> tensor<1x2x 1x1x1xf32>
```

A slice variant is provided to allow specifying whole slices of the source tensor.


For each 5x6 singleton of coordinates in %indices, extract the 2-D slice `%source[*, %indices[...]:%indices[...] + 1, *]` with the indices corresponding to the `dims` attribute specified by %indices:

NOTE: result type may be further rank-reduced to `tensor<6x7x 3x5xf32>`.

```mlir
%out = tensor.gather %source[%indices] dims([1])
    : (tensor<3x4x5xf32>, tensor<6x7x 1xindex>) -> tensor<6x7x 3x1x5xf32>
```

The dimensions specified in the dims attribute are ones for which the result tensor has size `1`. I.e. if the source type is `axbxcxd` and the coordinates are [1, 3], then the shape suffix is `ax1xcx1`. Gather also allows rank-reducing semantics where the shape `ax1xcx1` can be further simplified to `axc`.

The elemental type of the indices tensor can be any integer type. In the absence of target-specific or problem specific information the default type one should use is `index`.

This operation does not support unranked tensors.

An optional `unique` unit attribute may be specified to indicate that the coordinates in `indices` are statically guaranteed to be unique at runtime. Incorrectly setting the `unique` attribute when the coordinates are not truly unique is undefined behaviour.

Only full slices are meant to be supported by this op, if one desires partial slices (e.g. strided windows) one should compose this op with other tensor ops (e.g. tensor.extract_slice). This is to avoid a slippery slope of complexity that would make the op unusable in practice.

At the tensor-level, the index tensor is specified in an AoS form (i.e. coordinate tuple is the most minor). It is the responsibility of further lowerings and bufferiation to implement various concrete layouts.

NOTE: As currently specified, the operation must lower to an abstraction that performs copies to the output tensor. This is because the buffer type system is currently not rich enough to allow multiple non-contiguous views in the same type. This is visible more clearly in a notional buffer version of the op:

`memref<?x4x1xf32>` is a contiguous buffer of ?x4x1 elements. gather from random source slices must copy to the contiguous output:

```mlir
%out = memref.gather %source[%indices] dims([1])
    : (memref<4x4xf32>, memref<?x1xindex>) -> memref<?x4x1xf32>
```

Nested buffer support would allow gather to directly index into the source buffer (i.e. represent a jagged view into the source):

```mlir
%out = memref.gather %source[%indices] dims([1])
    : (memref<4x4xf32>, memref<?x 1xindex>) -> memref<? x memref<4x1xf32>>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$source `[` $indices `]`
    `dims` `(` $dims `)`
    (`unique` $unique^)?
    attr-dict
    `:` functional-type(operands, results)"
)]
pub struct Gather<T, const N: usize, const D: usize> {
    #[input]
    source: Tensor<T, N>,
    #[input]
    indices: Tensor<AnySignlessIntegerOrIndex, _>,
    #[attribute]
    dims: [u64; D],  // DenseI64ArrayAttr
    #[attribute]
    unique: UnitAttribute,
    #[output]
    output: Tensor<_, _>  // renamed from `result`
}

impl Gather {
    // // TODO: InferTypeOpInterface once enough confidence is built with
    // // tensor<tensor> and its lowering to memref<memref>.
    // static RankedTensorType infer_result_type(
    //     RankedTensorType source_type,
    //     RankedTensorType indicesType,
    //     &[i64] gatherDims,
    //     bool rankReduced);

    // RankedTensorType get_indices_type() {
    //     return get_indices().get_type().cast<RankedTensorType>();
    // }

    // RankedTensorType get_source_type() {
    //     return get_source().get_type().cast<RankedTensorType>();
    // }

    // RankedTensorType get_result_type() {
    //     return get_result().get_type().cast<RankedTensorType>();
    // }
}

impl Verify for Gather {

}

impl OpAsmOpInterface for Gather {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        set_name(self.output, "gather");
    }
}

// ----------------------------------------------------------------------
// Generate
// ----------------------------------------------------------------------

/**
Creates a dynamically sized tensor from elements.

This operation creates a dynamically sized tensor with elements of any type.
It expects one index operand per dynamic extent of the result tensor.

The body region defines the tensor's elements. It takes index operands as its region arguments that span the index space. The element at the given position is yielded with the `yield` operation (see `YieldOp`). There is no defined ordering to the invocations of the body. It is conceptually a "parallel map" operation.

# Examples

```mlir
%tnsr = tensor.generate %m, %n {
    ^bb0(%i : index, %j : index, %k : index):
        ...
        yield %elem : f32
} : tensor<?x3x?f32>
```
*/
// #[derive(Canonicalise, Verify, RegionVeriry)]
pub struct Generate {
    // let arguments = (dynamicExtents: Variadic<Index>);
    // let results = (result: AnyRankedTensor);
    // let regions = (region body: SizedRegion<1>);
    // let assembly_format = "$dynamicExtents $body attr-dict `:` type($result)";

    // let builders = [
    //     // Build op and populate its body per callback function.
    //     OpBuilder<(ins "Type":$resultTy, "ValueRange":$dynamicExtents,
    //     "function_ref<void(OpBuilder &, Location, ValueRange)>")>,
    // ];
}

// ----------------------------------------------------------------------
// Insert
// ----------------------------------------------------------------------

/**
`tensor.insert` Element insertion operation.

`tensor.insert` op inserts a scalar into a ranked tensor `dest` as specified by the operation's indices.

It returns a copy of `dest` with the indexed position updated to the value of `scalar`.

The arity of `indices `must match the rank of the tensor `dest`. All indices should be of `index` type.

# Examples

```mlir
%4 = tensor.insert %t into %dest[%1, %2] : tensor<4x4xi32>
```

```mlir
%5 = tensor.insert %rt into %dest[%1, %2] : tensor<?x?xi32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$scalar `into` $dest `[` $indices `]` attr-dict `:` type($dest)"
)]
pub struct Insert<T, const N: usize> {
    scalar: T,
    dest: Tensor<T, _>,
    #[input]
    indices: [usize; _],
    #[output]
    output: Tensor<T, _>  // renamed from `result`
}

// let builders = [
//     OpBuilder<(ins "Value":$scalar, "Value":$dest,
//     CArg<"ValueRange", "{}">:$indices), [{
//     auto resType = dest.get_type();
//     build($_builder, $_state, resType, scalar, dest, indices);
//     }]>];


impl Insert {
//     pub fn get_dps_inits_position_range() -> (i64, i64) {
//         (1, 2)  // `dest` operand
//     }
}

impl Verify for Insert {

}

impl Fold for Insert {

}

impl OpAsmOpInterface for Insert {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        
    }
}

impl DestinationStyleOpInterface for Insert {

}

// ----------------------------------------------------------------------
// InsertSlice
// ----------------------------------------------------------------------

/**
insert_slice operation.

The `insert_slice` operation insert a tensor `source` into another tensor `dest` as specified by the operation's offsets, sizes and strides arguments.

It returns a copy of `dest` with the proper slice updated with the value of `source`.

The insert_slice operation supports the following arguments:

- source: the tensor that is inserted.
- dest: the tensor into which the source tensor is inserted.
- offsets: tensor-rank number of offsets into the `dest` tensor into which
           the slice is inserted.
- sizes: tensor-rank number of sizes which specify the sizes of the source
         tensor type.
- strides: tensor-rank number of strides that specify subsampling in each
           dimension.

The representation based on offsets, sizes and strides support a
partially-static specification via attributes specified through the
`static_offsets`, `static_sizes` and `static_strides` arguments. A special
sentinel value ShapedType::kDynamic and
ShapedType::kDynamic encodes that the corresponding entry has
a dynamic value.

After buffer allocation, the "insert_slice" op is expected to lower into a
memref.subview op.

An insert_slice operation may additionally specify insertion into a tensor of higher rank than the source tensor, along dimensions that are statically known to be of size 1.
This rank-altering behaviour is not required by the op semantics: this flexibility allows to progressively drop unit dimensions while lowering between different flavors of ops on that operate on tensors.
The rank-altering behaviour of tensor.insert_slice matches the rank-reducing behaviour of tensor.extract_slice.

## Verification in the rank-reduced case

The same verification discussion and mechanisms apply as for ExtractSlice. Unlike ExtractSlice however, there is no need for a specific inference.

# Examples

Rank-altering insert_slice:

```mlir
%1 = tensor.insert_slice %t into %0[0, 0, 0][1, 16, 4][1, 1, 1]
    : tensor<16x4xf32> into tensor<8x16x4xf32>
```

```mlir
%3 = tensor.insert_slice %tt into %2[%o0, 4, %o2][1, %sz1, 1][1, %st1, 1]
    : tensor<1x?xf32> into tensor<8x16x4xf32>
```
*/
// #[derive()]
pub struct InsertSlice {
    // let arguments = (
    //     source: AnyRankedTensor,
    //     dest: AnyRankedTensor,
    //     offsets: Variadic<Index>,
    //     sizes: Variadic<Index>,
    //     strides: Variadic<Index>,
    //     static_offsets: DenseI64ArrayAttr,
    //     static_sizes: DenseI64ArrayAttr,
    //     static_strides: DenseI64ArrayAttr
    // );
    // let results = (outs result: AnyRankedTensor);

    // let assembly_format = [{
    //     $source `into` $dest ``
    //     custom<DynamicIndexList>($offsets, $static_offsets)
    //     custom<DynamicIndexList>($sizes, $static_sizes)
    //     custom<DynamicIndexList>($strides, $static_strides)
    //     attr-dict `:` type($source) `into` type($dest)
    // }];

    // let builders = [
    //     // Build a InsertSliceOp with mixed static and dynamic entries.
    //     OpBuilder<(ins "Value":$source, "Value":$dest,
    //     "&[FoldResult]":$offsets, "&[FoldResult]":$sizes,
    //     "&[FoldResult]":$strides,
    //     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    //     // Build a InsertSliceOp with dynamic entries.
    //     OpBuilder<(ins "Value":$source, "Value":$dest,
    //     "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
    //     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    //     // Build an InsertSliceOp with mixed static and dynamic entries packed in
    //     // a Range vector.
    //     OpBuilder<(ins "Value":$source, "Value":$dest,
    //     "ArrayRef<Range>":$ranges,
    //     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
    // ];
}

impl InsertSlice {
    // /// Returns the type of the base tensor operand.
    // pub fn source_type(&self) -> RankedTensorType {
    //     return get_source().get_type().cast<RankedTensorType>();
    // }

    // /// The result of a insert_slice is always a tensor.
    // pub fn get_type() -> RankedTensorType {
    //     return get_result().get_type().cast<RankedTensorType>();
    // }

    // /// The `dest` type is the same as the result type.
    // RankedTensorType get_dest_type() {
    //     return get_type();
    // }

    // /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    // /// and `static_strides` attributes.
    // pub fn get_array_attr_max_ranks() -> [usize; 3]{
    //     usize rank = get_type().get_rank();
    //     [rank, rank, rank]
    // }

    // /// Return the number of leading operands before the `offsets`, `sizes` and
    // /// and `strides` operands.
    // static usize get_offset_size_and_stride_start_operand_index() { return 2; }

    // (i64, i64) get_dps_inits_position_range() {
    // return {1, 2};  // `dest` operand
    // }
}

Canonicalise, Fold, Verify

// ----------------------------------------------------------------------
// Rank
// ----------------------------------------------------------------------

/**
`tensor.rank` operation.

`tensor.rank` operation takes a tensor operand and returns its rank.

# Examples

```mlir
%0 = tensor.rank %arg0 : tensor<*xf32>
```

```mlir
%1 = tensor.rank %arg1 : tensor<?x?xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$tensor attr-dict `:` type($tensor)"
)]
pub struct Rank<T, const N: usize> {
    #[input]
    tensor: Tensor<T, N>,
    #[output]
    output: usize
}

impl Fold for Rank {

}

impl OpAsmOpInterface for Rank {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        
    }
}

// ----------------------------------------------------------------------
// Reshape
// ----------------------------------------------------------------------

/**
`tensor.reshape` operation.

`tensor.reshape` operation converts a tensor from one type to an equivalent type with a provided shape. The source and destination types are compatible if both have the same element type, same number of elements. The following combinations are possible:

# Examples

a. Source type is ranked or unranked. Shape argument has static size. Result type is ranked.

Reshape statically-shaped tensor:

```mlir
%dst = tensor.reshape %src(%shape)
    : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>
%dst0 = tensor.reshape %src(%shape0)
    : (tensor<4x1xf32>, tensor<2xi32>) -> tensor<2x2xf32>
```

Flatten unranked tensor:

```mlir
%dst = tensor.reshape %src(%shape)
    : (tensor<*xf32>, tensor<1xi32>) -> tensor<?xf32>
```

b. Source type is ranked or unranked. Shape argument has dynamic size. Result type is unranked.

Reshape dynamically-shaped 1D tensor:

```mlir
%dst = tensor.reshape %src(%shape)
    : (tensor<?xf32>, tensor<?xi32>) -> tensor<*xf32>
```

Reshape unranked tensor:

```mlir
%dst = tensor.reshape %src(%shape)
    : (tensor<*xf32>, tensor<?xi32>) -> tensor<*xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$source `(` $shape `)` attr-dict `:` functional-type(operands, results)"
)]
pub struct Reshape<T, U: Into<usize>, const N: usize> {
    #[input]
    source: Tensor<T, N>,
    #[input]
    shape: Tensor<U, 1>,  // [AnySignlessInteger, Index]
    #[output]
    output: Tensor<T, _>  // renamed from `result`
}

// let builders = [OpBuilder<
//     (ins "Tensor":$resultType, "Value":$operand, "Value":$shape), [{
//     $_state.addOperands(operand);
//     $_state.addOperands(shape);
//     $_state.addTypes(resultType);
//     }]>];


impl Reshape {
    // pub fn get_result_type() -> Tensor {
    //     get_result().get_type().cast<Tensor>();
    // }
}

impl Verify for Reshape {

}

impl OpAsmOpInterface for Reshape {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        set_name(self.output, "reshape");
    }
}

// ----------------------------------------------------------------------
// ExpandShape / CollapseShape
// ----------------------------------------------------------------------

// #[derive(Canonicalise, Fold, Verify)]
pub trait ReassociativeReshape { // TensorOp
    // static &'static str get_reassociation_attr_str_name() { return "reassociation"; }
    // SmallVector<[AffineMap; 4]> get_reassociation_maps();
    // SmallVector<[ReassociationExprs; 4]> get_reassociation_exprs();
    // SmallVector<[ReassociationIndices; 4]> get_reassociation_indices() {
    //   SmallVector<[ReassociationIndices; 4]> reassociation_indices;
    //   for (auto attr : get_reassociation())
    //     reassociation_indices.push_back(llvm::to_vector<2>(
    //         llvm::map_range(attr.cast<ArrayAttr>(), [&](Attribute indexAttr) {
    //           return indexAttr.cast<IntegerAttr>().getInt();
    //         })));
    //   return reassociation_indices;
    // };
    // RankedTensorType get_src_type() {
    //   return getSrc().get_type().cast<RankedTensorType>();
    // }
    // RankedTensorType get_result_type() {
    //   return get_result().get_type().cast<RankedTensorType>();
    // }

//   let assembly_format = [{
//     $src $reassociation attr-dict `:` type($src) `into` type($result)
//   }];
}

/**
Operation to produce a tensor with a higher rank.

The `tensor.expand_shape` op produces a new tensor with a higher rank whose sizes are a reassociation of the original `src`.

A reassociation is defined as a continuous grouping of dimensions and is represented with an array of DenseI64ArrayAttr attribute.

The verification rule is that the reassociation maps are applied to the result tensor with the higher rank to obtain the operand tensor with the smaller rank.

The operand tensor type of a reshape can be zero-ranked if the result tensor type is statically shaped with all dimensions being unit extent. In such cases the reassociation map is empty.

# Examples

Dimension expansion `i -> (i', j')` and `(k) -> (k')`:

```mlir
%b = tensor.expand_shape %a [[0, 1], [2]]
    : tensor<?x?xf32> into tensor<?x?x?xf32>
```
*/
// #[derive(Verify)]
pub struct TExpandShape {
    // let builders = [
    //     // Builders using ReassociationIndices.
    //     OpBuilder<(ins "Type":$resultType, "Value":$src,
    //     "ArrayRef<ReassociationIndices>":$reassociation,
    //     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    //     [{
    //     build($_builder, $_state, resultType, src, attrs);
    //     $_state.addAttribute("reassociation",
    //         getReassociationIndicesAttribute($_builder, reassociation));
    //     }]>,
    //     OpBuilder<(ins "Type":$resultType, "Value":$src,
    //     "ArrayRef<ReassociationExprs>":$reassociation,
    //     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    //     [{
    //     auto reassociationMaps =
    //         convertReassociationMapsToIndices($_builder, reassociation);
    //     build($_builder, $_state, resultType, src, reassociationMaps, attrs);
    //     }]>
    // ];
}

impl TExpandShape {
    // pub fn get_corresponding_source_dim(i64 resultDim) -> i64;
}

impl ReassociativeReshape for TExpandShape {

}

/**
Operation to produce a tensor with a smaller rank.

The `tensor.collapse_shape` op produces a new tensor with a smaller rank whose sizes are a reassociation of the original `src`.

A reassociation is defined as a continuous grouping of dimensions and is represented with an array of DenseI64ArrayAttr attribute.

The verification rule is that the reassociation maps are applied to the operand tensor with the higher rank to obtain the result tensor with the smaller rank.

The result tensor type of a reshape can be zero-ranked if the operand tensor type is statically shaped with all dimensions being unit extent. In such case the reassociation map is empty.

# Examples

```mlir
// Dimension collapse (i, j) -> i' and k -> k'
%b = tensor.collapse_shape %a [[0, 1], [2]]
    : tensor<?x?x?xf32> into tensor<?x?xf32>
```
*/
// #[derive(Verify)]
pub struct CollapseShape {
    // let builders = [
    //     // Builders for a contracting reshape whose result type is computed from
    //     // `src` and `reassociation`.
    //     OpBuilder<(ins "Value":$src,
    //     "ArrayRef<ReassociationIndices>":$reassociation,
    //     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    //     OpBuilder<(ins "Value":$src,
    //     "ArrayRef<ReassociationExprs>":$reassociation,
    //     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    //     [{
    //     auto reassociationMaps =
    //         convertReassociationMapsToIndices($_builder, reassociation);
    //     build($_builder, $_state, src, reassociationMaps, attrs);
    //     }]>,

    //     // Builders for a reshape whose result type is passed explicitly.
    //     OpBuilder<(ins "Type":$resultType, "Value":$src,
    //     "ArrayRef<ReassociationIndices>":$reassociation,
    //     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    //     [{
    //     build($_builder, $_state, resultType, src, attrs);
    //     $_state.addAttribute("reassociation",
    //         getReassociationIndicesAttribute($_builder, reassociation));
    //     }]>,
    //     OpBuilder<(ins "Type":$resultType, "Value":$src,
    //     "ArrayRef<ReassociationExprs>":$reassociation,
    //     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    //     [{
    //     auto reassociationMaps =
    //         convertReassociationMapsToIndices($_builder, reassociation);
    //     build($_builder, $_state, resultType, src, reassociationMaps, attrs);
    //     }]>
    // ];
}

impl ReassociativeReshape for CollapseShape {

}

// ----------------------------------------------------------------------
// Pad
// ----------------------------------------------------------------------

/**
`tensor.pad` operation.

`tensor.pad` is an operation that pads the `source` tensor with given `low` and `high` padding config.

The PadOp operation supports the following arguments:

- source: the "base" tensor on which to pad.
- low: A list contains the padding along the start of each
        dimension, i.e `low`.
- high: A list contains the padding along the end of each
        dimension, i.e. `high`.
- nofold: indicates that the operation should not be folded when source and
          result types are equal.

The result tensor dimensions are `low` + `dim` + `high` along that dimension. The number of elements of `low` and `high` must match the rank of the input tensor. They can be either a constant or a dynamic value.

The region of the `tensor.pad` operation returns the value to use for the padding. The arguments of the region represent the index of the source being accessed. There should be as many arguments as the rank of the `source` tensor. The value `yield`-ed by the region is used as the value of the view at the given position.

If `nofold` is set, the padding operation will not be folded away even if the source type and the padded type have the same static shape. This can be used, e.g., for packing or promotion to faster memory.

Example 1:

```mlir
%pad_value = ... : f32
%0 = tensor.pad %0 low[1, 2] high[2, 3] {
^bb0(%arg0 : index, %arg1 : index):
tensor.yield %pad_value : f32
} : tensor<?x?xf32> to tensor<?x?xf32>
```

Example 2:

```mlir
%pad_value = ... : f32
%0 = tensor.pad %arg0 low[2, %arg1, 3, 3] high[3, 3, %arg1, 2] {
^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
    tensor.yield %pad_value : f32
} : tensor<1x2x2x?xf32> to tensor<6x?x?x?xf32>
```

Example 3:

```mlir
%pad_value = ... : f32
%0 = tensor.pad %arg0 low[0, 0] high[%ub0, %ub1] {
^bb0(%arg1: index, %arg2: index):
tensor.yield %pad_value : f32
} : tensor<2x3xf32> to tensor<?x?xf32>
```

Example 4:

```mlir
// Force a padded value to be always exist with `nofold`.
%pad_value = ... : f32
%0 = tensor.pad %arg0 nofold low[0, 0] high[0, 0] {
^bb0(%arg1: index, %arg2: index):
tensor.yield %pad_value : f32
} : tensor<2x3xf32> to tensor<2x3xf32>
```
*/
// #[derive()]
pub struct Pad {
    // let arguments = (
    //     source: AnyTensor,
    //     low: Variadic<Index>:,
    //     high: Variadic<Index>:,
    //     static_low: DenseI64ArrayAttr,
    //     static_high: DenseI64ArrayAttr,
    //     nofold: UnitAttribute
    // );

    // let regions = (region SizedRegion<1>:$region);

    // let results = (outs result: AnyTensor);

    // // TODO: Remove custom<InferType> when AllTypesMatch supports opt. operands.
    // let assembly_format = [{
    //     $source
    //     (`nofold` $nofold^)?
    //     `low` `` custom<DynamicIndexList>($low, $static_low)
    //     `high` `` custom<DynamicIndexList>($high, $static_high)
    //     $region attr-dict `:` type($source) `to` type($result)
    // }];

    // let builders = [
    //     // Build a PadOp with mixed static and dynamic entries.
    //     OpBuilder<(ins "Value":$source, "&[i64]":$static_low,
    //     "&[i64]":$static_high, "ValueRange":$low, "ValueRange":$high,
    //     CArg<"bool", "false">:$nofold,
    //     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    //     // Build a PadOp with all dynamic entries.
    //     OpBuilder<(ins "Value":$source, "ValueRange":$low, "ValueRange":$high,
    //     CArg<"bool", "false">:$nofold,
    //     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    //     // Build a PadOp with mixed static and dynamic entries and custom
    //     // result type. If the type passed is nullptr, it is inferred.
    //     OpBuilder<(ins "Type":$resultType, "Value":$source,
    //     "&[FoldResult]":$low, "&[FoldResult]":$high,
    //     CArg<"bool", "false">:$nofold,
    //     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    //     // Build a PadOp with constant padding,  mixed static and dynamic entries
    //     // and custom result type. If the type passed is nullptr, it is inferred.
    //     OpBuilder<(ins "Type":$resultType, "Value":$source,
    //     "&[FoldResult]":$low, "&[FoldResult]":$high,
    //     "Value":$constantPadValue, CArg<"bool", "false">:$nofold,
    //     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
    // ];
}

impl Pad {
    // static &'static str get_static_low_attr_str_name() {
    //     "static_low"
    // }

    // static &'static str get_static_high_attr_str_name() {
    //     "static_high"
    // }

    // RankedTensorType get_source_type() {
    //     return get_source().get_type().cast<RankedTensorType>();
    // }
    // RankedTensorType get_result_type() {
    //     return get_result().get_type().cast<RankedTensorType>();
    // }

    // // Infer the shape of the result tensor given the type of the source tensor
    // // and paddings. Known result dimensions that cannot necessarily be inferred
    // // from low/high padding sizes can be optionally specified. Those will be
    // // considered when computing the result type.
    // static RankedTensorType infer_result_type(
    //                             RankedTensorType source_type,
    //                             &[i64] static_low,
    //                             &[i64] static_high,
    //                             &[i64] result_shape = {});

    // // Return the pad value if it is a constant. Return null value otherwise.
    // Value get_constant_padding_value();

    // // Return a vector of all the static or dynamic values (low/high padding) of
    // // the op.
    // inline SmallVector<[FoldResult]> get_mixed_pad_impl(&[i64] static_attrs,
    //                                                 ValueRange values) {
    // Builder builder(*this);
    // SmallVector<[FoldResult]> res;
    // usize num_dynamic = 0;
    // usize count = static_attrs.size();
    // for (usize idx = 0; idx < count; ++idx) {
    //     if (ShapedType::is_dynamic(static_attrs[idx]))
    //     res.push_back(values[num_dynamic++]);
    //     else
    //     res.push_back(builder.get_i64_integer_attr(static_attrs[idx]));
    // }
    // return res;
    // }
    // SmallVector<[FoldResult]> get_mixed_low_pad() {
    //     return get_mixed_pad_impl(get_static_low(), get_low());
    // }
    // SmallVector<[FoldResult]> get_mixed_high_pad() {
    //     return get_mixed_pad_impl(get_static_high(), get_high());
    // }
    // // Return true if low padding is guaranteed to be 0.
    // bool has_zero_low_pad() {
    // return llvm::all_of(get_mixed_low_pad(), [](FoldResult ofr) {
    //     return get_constant_int_value(ofr) == static_cast<i64>(0);
    // });
    // }
    // // Return true if high padding is guaranteed to be 0.
    // bool has_zero_high_pad() {
    // return llvm::all_of(get_mixed_high_pad(), [](FoldResult ofr) {
    //     return get_constant_int_value(ofr) == static_cast<i64>(0);
    // });
    // }
    // /// Return the dimensions with a non-zero low or high padding.
    // SmallBitVector get_padded_dims();
}

impl Verify for Pad {

}

impl Fold for Pad {
    
}

impl RegionVerify for Pad {
    
}

impl Canonicalise for Pad {
    
}

impl TensorOp for Pad {

}

// ----------------------------------------------------------------------
// ParallelInsertSliceOp
// ----------------------------------------------------------------------

/**
Specify the tensor slice update of a single thread of a parent ParallelCombiningOpInterface op.

The `parallel_insert_slice` yields a subset tensor value to its parent
ParallelCombiningOpInterface. These subset tensor values are aggregated to in some unspecified order into a full tensor value returned by the parent parallel iterating op.
The `parallel_insert_slice` is one such op allowed in the ParallelCombiningOpInterface op.

Conflicting writes result in undefined semantics, in that the indices written to by multiple parallel updates might contain data from any of the updates, or even a malformed bit pattern.

If an index is updated exactly once, the value contained at that index in the resulting tensor will be equal to the value at a corresponding index of a slice that was used for the updated. If an index is not updated at all, its value will be equal to the one in the original tensor.

This op does not create a new value, which allows maintaining a clean separation between the subset and full tensor.

Note that we cannot mark this operation as pure (Pures), even though it has no side effects, because it will get DCEd during canonicalisation.

The parallel_insert_slice operation supports the following arguments:

- source: the tensor that is inserted.
- dest: the tensor into which the source tensor is inserted.
- offsets: tensor-rank number of offsets into the `dest` tensor into which
           the slice is inserted.
- sizes: tensor-rank number of sizes which specify the sizes of the source
         tensor type.
- strides: tensor-rank number of strides that specify subsampling in each
           dimension.

The representation based on offsets, sizes and strides support a partially-static specification via attributes specified through the `static_offsets`, `static_sizes` and `static_strides` arguments. A special sentinel value ShapedType::kDynamic and ShapedType::kDynamic encodes that the corresponding entry has a dynamic value.

After buffer allocation, the "parallel_insert_slice" op is expected to lower into a memref.subview op.

A parallel_insert_slice operation may additionally specify insertion into a tensor of higher rank than the source tensor, along dimensions that are statically known to be of size 1.
This rank-altering behaviour is not required by the op semantics: this flexibility allows to progressively drop unit dimensions while lowering between different flavors of ops on that operate on tensors.
The rank-altering behaviour of tensor.parallel_insert_slice matches the rank-reducing behaviour of tensor.insert_slice and tensor.extract_slice.

## Verification in the rank-reduced case

The same verification discussion and mechanisms apply as for ExtractSlice.
Unlike ExtractSlice however, there is no need for a specific inference.
*/
// #[derive(Canonicalise, Verify)]
// TODO: Implement PerformConcurrentlyOpInterface.
pub struct ParallelInsertSlice {
    // inputs = (
    //     source: AnyRankedTensor,
    //     dest: AnyRankedTensor,
    //     offsets: Variadic<Index>,
    //     sizes: Variadic<Index>,
    //     strides: Variadic<Index>,
    //     static_offsets: DenseI64ArrayAttr,
    //     static_sizes: DenseI64ArrayAttr,
    //     static_strides: DenseI64ArrayAttr
    // );

    // let assembly_format = [{
    //     $source `into` $dest ``
    //     custom<DynamicIndexList>($offsets, $static_offsets)
    //     custom<DynamicIndexList>($sizes, $static_sizes)
    //     custom<DynamicIndexList>($strides, $static_strides)
    //     attr-dict `:` type($source) `into` type($dest)
    // }];

    // let builders = [
    //     // Build a ParallelInsertSliceOp with mixed static and dynamic entries.
    //     OpBuilder<(
    //         "Value":$source, "Value":$dest,
    //         "&[FoldResult]":$offsets,
    //         "&[FoldResult]":$sizes,
    //         "&[FoldResult]":$strides,
    //         CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    //     // Build a ParallelInsertSliceOp with mixed static and dynamic entries
    //     // packed into a Range vector.
    //     OpBuilder<(ins "Value":$source, "Value":$dest,
    //     "ArrayRef<Range>":$ranges,
    //     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    //     // Build a ParallelInsertSliceOp with dynamic entries.
    //     OpBuilder<(ins "Value":$source, "Value":$dest,
    //     "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
    //     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
    // ];
}

impl ParallelInsertSlice {
    // Type yielded_type() { return get_dest().get_type(); }

    // RankedTensorType get_source_type() {
    //     return get_source().get_type().cast<RankedTensorType>();
    // }

    // RankedTensorType get_dest_type() {
    //     return get_dest().get_type().cast<RankedTensorType>();
    // }

    // ParallelCombiningOpInterface get_parallel_combining_parent() {
    //   return dyn_cast<ParallelCombiningOpInterface>(
    //       getOperation()->getParentOp());
    // }

    // /// Return the expected rank of each of the `static_offsets`, `static_sizes`
    // /// and `static_strides` attributes.
    // pub fn get_array_attr_max_ranks() -> [usize; 3] {
    //     let rank = get_dest_type().get_rank();
    //     [rank, rank, rank]
    // }

    /**
    Return the number of leading operands before `offsets`, `sizes` and `strides` operands.
    */
    pub fn offset_size_and_stride_start_operand_index() -> usize {
        1
    }

    // /// Return the Output of the enclosing ForeachThreadOp that is
    // /// corresponding to this ParallelInsertSliceOp.
    // Output get_tied_op_result();
}

impl TensorOp for ParallelInsertSlice {

}


// ----------------------------------------------------------------------
// Scatter
// ----------------------------------------------------------------------

/**
Scatter a tensor into a destination tensor at specified indices.

The `scatter` operation inserts a `source` tensor into a `dest` tensor at the given indices.

In its most general form, the tensor of indices specifies all the coordinates of every element to insert (i.e. COO format, without the payload). The indices are expected to be confined to coordinate values that fit the range of the `dest` tensor, otherwise the behaviour is undefined.

The leading dimensions of the index tensor must match that of the dest tensor. The trailing dimensions of the dest tensor must match those of the source tensor by omitting the dimensions specified in `dims` (rank-reducing semantics) or setting them to `1` (rank-preserving semantics) (see examples). This convention allows an idiomatic specification and lowering of "scattering multiple N-D slices into the dest tensor". The result type must match the type of the dest tensor.

NOTE: in the examples below, we separate out the indexing part of the tensor type by a whitespace for readability purposes.

# Examples

For each 1x2 triple of coordinates in `%indices`, insert the element (i.e. 0-D subset) at the coordinates triple in `%dest`:

NOTE: source type may be further rank-reduced to `tensor<1x2xf32>`.

```mlir
%out = tensor.scatter %source into %dest[%indices]
    dims([0, 1, 2]) unique
    : (tensor<1x2x1x1x1xf32>, tensor<4x4x4xf32>, tensor<1x2x3xindex>)
    -> tensor<4x4x4xf32>
```

A slice variant is provided to allow specifying insertion of whole tensor
slices into the `dest` tensor.

For each 3 singleton of coordinates in `%indices`, insert the 2-D slice into `%dest[*, %indices[...]:%indices[...] + 1, *]` with the indices corresponding to the `dims` attribute specified by `%indices`:

```mlir
%out = tensor.scatter %source into %dest[%indices] dims([1]) unique
    : (tensor<3x4x1x6xf32>, tensor<4x5x6xf32>, tensor<3x1xindex>)
    -> tensor<4x5x6xf32>
```

The dimensions specified in the `dims` attribute are ones for which the source tensor has size `1`. I.e. if the dest type is `axbxcxd` and the coordinates are [1, 3], then the source type suffix is `ax1xcx1`. Sactter also allows rank-reducing semantics where the shape `ax1xcx1` can be further simplified to `axc`.

The elemental type of the indices tensor can be any integer type. In the absence of target-specific or problem specific information the default type one should use is `index`.

This operation does not support unranked tensors.

A `unique` unit attribute must be be specified to indicate that the coordinates are statically guaranteed to be unique at runtime. If coordinates are not truly unique at runtime, the behaviour is undefined.

Only full slices are meant to be supported by this op, if one desires partial slices (e.g. strided windows) one should compose this op with other tensor ops (e.g. tensor.insert_slice). This is to avoid a slippery slope of complexity that would make the op unusable in practice.

At the tensor-level, the index tensor is specified in an AoS form (i.e. coordinate tuple is the most minor). It is the responsibility of further lowerings and bufferiation to implement various concrete layouts.

NOTE: As currently specified, the operation must lower to an abstraction that performs copies to the output tensor. This is because the buffer type system is currently not rich enough to allow multiple non-contiguous views in the same type. This is visible more clearly in a notional buffer version of the op:

`memref<?x`4xf32>` is a contiguous buffer of ?x4 elements, scatter into random dest slices must copy to the contiguous dest:

```mlir
some_side_effecting_op_writing_into %source, ... : memref<3x4xf32>
memref.scatter %source into %dest[%indices] dims([1]) unique
    : (memref<3x4xf32>, memref<?x4xf32>, memref<?x1xindex>)
```

Nested buffer support in the producing op would allow writing directly into the dest buffer:

```mlir
%v = some_nested_buffer_view_op %dest[%indices] dims([1]) unique
    : memref<?xmemref<4xf32>>
some_side_effecting_op_writing_into %v, ...: memref<? x memref<4xf32>>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$source `into` $dest `[` $indices `]`
    `dims` `(` $dims `)`
    (`unique` $unique^)?
    attr-dict
    `:` functional-type(operands, results)"
)]
pub struct Scatter<T, const N: usize, const D: usize> {
    #[input]
    source: Tensor<T, N>,
    #[input]
    dest: Tensor<_, D>,
    #[input]
    indices: Tensor<AnySignlessIntegerOrIndex, _>,
    #[attribute]
    dims: [u64; D],
    #[output]
    output: Tensor<_, _>  // renamed from `result`
}

impl Verify for Scatter {

}

impl OpAsmOpInterface for Scatter {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        set_name(self.output, "scatter");
    }
}

// ----------------------------------------------------------------------
// Splat
// ----------------------------------------------------------------------

/**
Tensor splat or broadcast operation.

Broadcast the operand to all elements of the result tensor. The operand is required to be of integer/index/float type, and the result tensor must be statically shaped.

# Examples

```mlir
%s = arith.constant 10.1 : f32
%t = tensor.splat %s : tensor<8x16xf32>
```

TODO: This operation is easy to extend to broadcast to dynamically shaped tensors:

Broadcasts %s to a 2-d dynamically shaped tensor, with %m, %n binding to the sizes of the two dynamic dimensions:

```mlir
%m = "foo"() : () -> (index)
%n = "bar"() : () -> (index)
%t = tensor.splat %s [%m, %n] : tensor<?x?xf32>
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$input attr-dict `:` type($aggregate)"
)]
// AnyTypeOf<[AnySignlessInteger, Index, AnyFloat]>
pub struct Splat<T> {
    /// integer/index/float type
    #[input]
    input: T,
    #[output]
    output: AnyStaticShapeTensor<T>  // aggregate
}

// let builders = [
//     OpBuilder<(ins "Value":$element, "Type":$aggregateType),
//     [{ build($_builder, $_state, aggregateType, element); }]>];

impl TensorOp for Splat {

}

impl OpAsmOpInterface for Splat {
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn) {
        
    }
}

// ----------------------------------------------------------------------
// Pack
// ----------------------------------------------------------------------

pub trait Relayout {  // TensorOp
    // i64 get_source_rank() { return get_source().get_type().get_rank(); };
    // i64 get_dest_rank() { return get_dest().get_type().get_rank(); };
    // RankedTensorType get_source_type() {
    //   return get_source().get_type().cast<RankedTensorType>(); };
    // RankedTensorType get_dest_type() {
    //   return get_dest().get_type().cast<RankedTensorType>(); };

    // /// Return position for init operand. Init operand is `dest`.
    // (i64, i64) get_dps_inits_position_range() {
    //   return {1, 2}; // `dest` operand
    // }

    // /// Interface method for ConditionallySpeculatable.
    // Speculation::Speculatability getSpeculatability();

    // /**
    // Return a mapping from positions `inner_dims_pos` to their
    // tile factors.
    // */
    // DenseMap<i64, FoldResult> get_dim_and_tile_mapping();

    // /// Return the tile sizes as FoldResult.
    // SmallVector<[FoldResult]> get_mixed_tiles();

    // /**
    // Return the tile sizes as `i64`. If a tile size is dynamic
    // a sentinel `kDynamic` is introduced at that position in
    // the returned vector.
    // */
    // SmallVector<[i64]> get_static_tiles();

    //   let hasVerifier = 1;
}

/**
Tensor pack operation.

The pack operation converts an input tensor to a higher-dimensional tensor with a tiled and packed layout. The mandatory `inner_dims_pos` attribute specifies a permutation for the original dimensions, while `inner_tiles` is the tiling factor for each dimension. The optional attribute `outer_dims_perm` specifies the order for the tiled data dimension, while the attribute `padding_value` specifies a padding value at the boundary on non-perfectly divisible dimensions. Padding is optional:

- If absent, it is UB if the tile does not perfectly divide the dimension.
- If present, it will pad along high dimensions (high-padding) to make the
  tile complete.

Example NC_to_NCnc:

```mlir
tensor.pack %source inner_dims_pos = [0, 1]
  inner_tiles = [8, 32] into %dest : tensor<128x256xf32> -> tensor<16x8x8x32xf32>
```
Example CK to KCck

```mlir
tensor.pack %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
  inner_tiles = [8, 32] into %dest : tensor<128x256xf32> -> tensor<8x16x8x32xf32>
```

In all cases, dimension at position 0 in the input tensor (128) is tiled with a factor of 8, while dimension at position 1 (256) is tiled with a factor of 32. In the second example, the outer data dimensions are interchanged according to `outer_dims_perm`.

Example NC_to_NCnc with padding:

```mlir
tensor.pack %arg padding_value(%pad : f32) inner_dims_pos = [0, 1]
  inner_tiles = [8, 2] into %arg1 : tensor<13x15xf32> -> tensor<2x8x8x2xf32>
```
*/
pub struct Pack {
    // let arguments = (
    //     source: AnyRankedTensor,
    //     dest: AnyRankedTensor,
    //     padding_value: Option<AnyType>,
    //     outer_dims_perm: DenseI64ArrayAttr,
    //     inner_dims_pos: DenseI64ArrayAttr,
    //     inner_tiles: Variadic<Index>,
    //     static_inner_tiles: DenseI64ArrayAttr
    // );

    // let assembly_format = [{
    //     $source
    //     (`padding_value` `(` $padding_value^ `:` type($padding_value) `)`)?
    //     (`outer_dims_perm` `=` $outer_dims_perm^)?
    //     `inner_dims_pos` `=` $inner_dims_pos
    //     `inner_tiles` `=`
    //     custom<DynamicIndexList>($inner_tiles, $static_inner_tiles)
    //     `into` $dest attr-dict `:` type($source) `->` type($dest)
    // }];

//   let builders = [
//     OpBuilder<(ins "Value":$source, "Value":$dest,
//       "&[i64]":$innerDimsPos,
//       "&[FoldResult]":$innerTiles,
//       CArg<"Optional<Value>", "std::nullopt">:$paddingValue,
//       CArg<"&[i64]", "{}">:$outerDimsPerm)>
//   ];
}

// type PackInput = (
//     AnyRankedTensor,
//     AnyRankedTensor,
//     Option<AnyType>,
//     DenseI64ArrayAttr,
//     DenseI64ArrayAttr,
//     Variadic<Index>,
//     DenseI64ArrayAttr
// );

// impl FnOnce<PackInput> for Pack {
//     type Output = AnyRankedTensor;

//     extern "rust-call" fn call_once(self, args: PackInput) -> Self::Output {
//         self.call(args)
//     }
// }

// impl FnMut<PackInput> for Pack {
//     extern "rust-call" fn call_mut(&mut self, args: PackInput) -> Self::Output {
//         self.call(args)
//     }
// }

// impl Fn<PackInput> for Pack {
//     extern "rust-call" fn call(&self, args: PackInput) -> Self::Output {
        
//     }
// }

impl Pack {
    // /**
    // Method to get the `ShapedType` of the result based on the inner tiles,
    // position of the inner tiles (innerDimsPos)  and interchange vector of
    // outer loops (outerDimsPerm).
    // */
    // static ShapedType inferPackedType(ShapedType source_type,
    //     &[i64] innerTileSizes, &[i64] innerDimsPos,
    //     &[i64] outerDimsPerm = {});

    // static Value createDestinationTensor(OpBuilder &b, Location loc,
    //     Value source, &[FoldResult] innerTileSizes,
    //     &[i64] innerDimsPos, &[i64] outerDimsPerm);
}

// ----------------------------------------------------------------------
// UnPackOp
// ----------------------------------------------------------------------

/**
Tensor unpack operation.

The unpack operation converts a tensor with a tiled and packed layout to a lower-dimensional tensor. Similar to `pack`,  the mandatory attributes `inner_dims_pos` specifies a permutation for the inner data dimensions, while `inner_tiles` is the tiling factor. The attribute `outer_dims_perm` has the exact behaviour as the one described in `pack`. In `unpack`, it is UB if the tile does not perfectly divide the dimension.

Example NCnc_to_NC:

```mlir
tensor.unpack %source inner_dims_pos = [0, 1]
  inner_tiles = [8, 32] into %dest : tensor<16x8x8x32xf32> -> tensor<128x256xf32>
```

Example CK to KCck:

```mlir
tensor.unapck %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
  inner_tiles = [8, 32] into %dest : tensor<8x16x8x32xf32> -> tensor<128x256xf32>
```
*/
pub struct Unpack {
    // inputs = (
    //     source: AnyRankedTensor,
    //     dest: AnyRankedTensor,
    //     outer_dims_perm: DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">,
    //     inner_dims_pos: DenseI64ArrayAttr,
    //     inner_tiles: Variadic<Index>,
    //     static_inner_tiles: DenseI64ArrayAttr
    // );

    // let assembly_format = [{
    //     $source
    //     (`outer_dims_perm` `=` $outer_dims_perm^)?
    //     `inner_dims_pos` `=` $inner_dims_pos
    //     `inner_tiles` `=`
    //     custom<DynamicIndexList>($inner_tiles, $static_inner_tiles)
    //     `into` $dest attr-dict `:` type($source) `->` type($dest)
    // }];

    // let builders = [
    //     OpBuilder<(
    //         ins "Value":$source, 
    //         "Value":$dest,
    //         "&[i64]":$inner_dims_pos,
    //     "&[FoldResult]":$inner_tiles,
    //     CArg<"&[i64]", "{}">:$outer_dims_perm)>
    // ];
}

impl Relayout for Unpack {

}

// ----------------------------------------------------------------------
// Yield
// ----------------------------------------------------------------------

/**
Yield a value from a region.

This operation is used to yield a single value from a within a region. It is used to create dynamically sized tensors (see `tensor.generate` and `tensor.pad` ops).
*/
pub struct Yield {
    // inputs = (value: AnyType);

    // let assembly_format = "$value attr-dict `:` type($value)";

    // // Dummy builder to appease code in templated ensureTerminator that
    // // GenerateOp's auto-generated parser calls.
    // let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

impl TensorOp for Yield {
    
}
