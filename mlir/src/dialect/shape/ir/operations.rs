/*!
# Shape Operations Definition 

This is the operation definition file for Shape dialect operations.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Shape/IR/ShapeOps.td>
*/

use core::ptr::null;

use crate::mlir::ir::{
    operation::definition::FoldResult,
    function::interfaces::FunctionOpInterface
};

/**
`shape.add` operation whoch adds sizes and indices.

Adds two sizes or indices. If either operand is an error it will be propagated to the result. The operands can be of type `size` or `index`. If at least one of the operands can hold an error, i.e. if it is of type `size`, the result must be of type `size`. If error propagation is not possible because both operands are of type `index` then the result may be of type `size` or `index`.
*/
#[mlir(
    traits = [Commutative, Pure],
    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
)]
pub struct Add {
    lhs: SizeOrIndexType,
    rhs: SizeOrIndexType,
    #[output]
    result: SizeOrIndexType
}

impl Add {
    /**
    Returns when two result types are compatible for this op; method used by InferTypeOpInterface
    */
    static bool is_compatible_return_types(TypeRange l, TypeRange r);
}

impl Fold for Add {
    fn fold(&self, adaptor: FoldAdaptor) -> FoldResult {
        // add(x, 0) -> x
        if match_pattern(self.rhs, m_Zero()) {
            return self.lhs;
        }
      
        const_fold_binary_op::<IntegerAttribute>(
            adaptor.inputs(),
            |a: APInt, b: &APInt| std::move(a) + b)
    }
}

impl Verify for Add {
    fn verify(&self) -> LogicalResult {
        verify_size_or_index_op(self)
    }
}

impl InferTypeOpInterface for Add {

}

/**
`shape.broadcast` operation which returns the broadcasted output shape of two or more inputs.

Returns the broadcasted shape for input shapes or extent tensors. The rest of this description is simplified for the 2 input case but can be extended to more inputs. Both operands can be of type `shape.shape` or `tensor<?xindex>`. The result is of type `shape.shape` and, if both operands are tensors, may be of type `tensor<?xindex>`.

If the two operand shapes are of different rank the smaller one is padded with 1's from the left. The resulting broadcasted shape is then defined as

```text
result[i] = lhs[i] if lhs[i] == rhs[i]
          = lhs[i] if rhs[i] == 1
          = rhs[i] if lhs[i] == 1.
```

In case the resulting shape is undefined, i.e. if corresponding extents are different from each other but none is 1, the result is an error shape.
Likewise error values are propagated if any of the operands holds an error value. If the result type is an extent tensor (and can therefore not hold the error value) the behaviour may be undefined. The optional string attribute can be used to describe the error case.
*/
#[mlir(
    traits = [Commutative, Pure],
    assembly_format = "$shapes attr-dict `:` type($shapes) `->` type($result)"
)]
pub struct Broadcast {
    shapes: [ShapeOrExtentTensorType],
    error: OptionalAttr<StringAttribute>,
    #[output]
    result: ShapeOrExtentTensorType

//   let builders = [OpBuilder<(ins Value:$shape)>];


//   let builders = [OpBuilder<(ins Type:$result,
//                                 Value:$lhs, Value:$rhs,
//                                 "/*optional*/ ::mlir::StringAttr":$error), [{
//       build($_builder, $_state, result, ::llvm::ArrayRef({lhs, rhs}),
//         error);
//     }]>
//   ];
}

impl Fold for Broadcast {
    fn fold(&self, adaptor: FoldAdaptor) -> FoldResult {
        if self.shapes.len() == 1 {
            /*
            Otherwise, we need a cast which would be a canonicalisation, not folding.
            */
            if self.shapes[0].r#type != r#type {
                return null();
            }
            return self.shapes[0];
        }
      
        // TODO: Support folding with more than 2 input shapes
        if self.shapes.len() > 2 {
            return null();
        }
      
        if !self.shapes[0] || !self.shapes[1] {
            return null();
        }
        let lhs_shape = llvm::to_vector<6>(
            self.shapes[0].cast<DenseIntElementsAttribute>().values::<i64>());
        let rhs_shape = llvm::to_vector<6>(
            self.shapes[1].cast<DenseIntElementsAttribute>().values::<i64>());
        let mut result_shape = SmallVector<[i64; 6]>::new();
      
        // If the shapes are not compatible, we can't fold it.
        // TODO: Fold to an "error".
        if !get_broadcasted_shape(lhs_shape, rhs_shape, &mut result_shape) {
            return null();
        }
      
        let builder = Builder::new(self.context());
        builder.index_tensor_attribute(result_shape)
    }
}

impl Verify for Broadcast {
    fn verify(&self) -> LogicalResult {
         verify_shape_or_extent_tensor_op(self)
    }
}

impl Canonicalise for Broadcast {
    fn canonicalisation_patterns(
        &self,
        patterns: &RewritePatternSet,
        context: *mut MLIRContext
    ) {
        patterns.add::<
            BroadcastConcretizeResultTypePattern,
            BroadcastFoldConstantOperandsPattern,
            BroadcastForwardSingleOperandPattern,
            CanonicaliseCastExtentTensorOperandsPattern<BroadcastOp>,
            RemoveDuplicateOperandsPattern<BroadcastOp>,
            RemoveEmptyShapeOperandsPattern<BroadcastOp>
        >(context);
    }
}

/**
`shape.const_shape` which creates a constant shape or extent tensor.

Creates a constant shape or extent tensor. The individual extents are given as the `shape` attribute. The number of these values equals the shape's rank.

```mlir
%0 = shape.const_shape [] : !shape.shape
%1 = shape.const_shape [1, 2, 3] : !shape.shape
%2 = shape.const_shape [4, 5, 6] : tensor<3xindex>
```
*/
#[mlir(
    traits = [ConstantLike, Pure]
)]
pub struct ConstShape {
    shape: IndexElementsAttribute,
    #[output]
    result: ShapeOrExtentTensorType

//   let hasCustomAssemblyFormat = 1;
}

impl ConstShape {
    // InferTypeOpInterface:
    static bool is_compatible_return_types(TypeRange l, TypeRange r);
}

impl Fold for ConstShape {
    fn fold(&self, _: FoldAdaptor) -> FoldResult {
        get_shape_attr()
    }
}

impl Canonicalise for ConstShape {
    fn canonicalisation_patterns(
        &self,
        patterns: &RewritePatternSet,
        context: *mut MLIRContext
    ) {
        patterns.add<TensorCastConstShape>(context);
    }
}

impl AssemblyFormat for ConstShape {

}

impl InferTypeOpInterface for ConstShape {

}

/**
Creates a constant of type `shape.size`.

Creates a `shape.size` type representing the constant size given by `value`.

```mlir
%x = shape.const_size 10
```
*/
#[mlir(
    traits = [
        ConstantLike, Pure,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["asm_output_names"]>
    ],
    assembly_format = "$value attr-dict"
)]
pub struct ConstSize {
    value: IndexAttribute,
    #[output]
    result: SizeType

//   let builders = [OpBuilder<(ins i64:$value)>];
}

impl Fold for ConstSize {

}

impl OpAsmOpInterface for ConstSize {

}

/**
Division of sizes and indices.

Divides two sizes or indices. If either operand is an error it will be propagated to the result. The operands can be of type `size` or `index`.
If at least one of the operands can hold an error, i.e. if it is of type `size`, the result must be of type `size`. If error propagation is not possible because both operands are of type `index` then the result may be of type  `size` or `index`. If both operands and result are of type `index`, their runtime values could be negative. The result is rounded toward negative infinity, i.e. floor(lhs / rhs), such that

```tet
div(lhs, rhs) * rhs + mod(lhs, rhs) = lhs
```

always holds. If any of the values is of type `size`, the behaviour for negative value is undefined.
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
)]
pub struct Div {
    lhs: SizeOrIndexType,
    rhs: SizeOrIndexType,
    #[output]
    result: SizeOrIndexType
}

impl Div {
    // Returns when two result types are compatible for this op; method used by
    // InferTypeOpInterface
    static bool is_compatible_return_types(TypeRange l, TypeRange r);
}

impl Fold for Div {
    fn fold(&self, adaptor: FoldAdaptor) -> FoldResult {
        let lhs = adaptor.lhs.dyn_cast_or_null<IntegerAttribute>();
        if !lhs {
            return null();
        }
        let rhs = adaptor.rhs.dyn_cast_or_null<IntegerAttribute>();
        if !rhs {
            return null();
        }
      
        // Division in APInt does not follow floor(lhs, rhs) when the result is
        // negative. Rather, APInt rounds toward zero.
        let quotient: APInt;
        let remainder: APInt;
        APInt::sdivrem(lhs.value, rhs.value, quotient, remainder);
        if quotient.is_negative() && !remainder.is_null_value() {
            quotient -= 1;
        }
      
        let index_type = IndexType::new(self.context());
        IntegerAttribute::new(index_type, quotient)
    }
}

impl Verify for Div {
    fn verify(&self) -> LogicalResult {
        verify_size_or_index_op(self)
    }
}

impl InferTypeOpInterface for Div {

}

/**
Returns whether the input shapes or extent tensors are equal.

Takes one or more shape or extent tensor operands and determines whether they are equal. When extent tensors are compared to shapes they are regarded as their equivalent non-error shapes. Error shapes can be tested for equality like any other shape value, meaning that the error value is equal to itself.
*/
#[mlir(
    traits = [Pure, Commutative],
    assembly_format = "$shapes attr-dict `:` type($shapes)"
)]
pub struct ShapeEq {
    shapes: [ShapeOrExtentTensorType],
    #[output]
    result: I1

//   // Convenience builder alias for the binary version.
//   let builders = [
//   OpBuilder<(ins Value:$lhs, Value:$rhs),
//     [{ build($_builder, $_state, ::llvm::ArrayRef({lhs, rhs})); }]>,
//   ];
}

impl Fold for ShapeEq {
    fn fold(&self) -> FoldResult {
        let all_same = true;
        if !self.shapes.is_empty() && !self.shapes[0] {
            return {};
        }
        for operand in self.shapes.drop_front() {
            if !operand {
                return {};
            }
            all_same = all_same && operand == self.shapes[0];
        }
        BoolAttribute::new(self.context(), all_same)
    }
}

/**
`shape.from_extents` operation.

Creates a shape from multiple SSA values representing the extents of the shape.

# Examples

Rank 2 shape:

```mlir
%s0 = shape.from_extents %a, %b
```

Rank 0 shape:

```mlir
%s1 = shape.from_extents
```
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$extents attr-dict `:` type($extents)"
)]
pub struct FromExtents {
    extents: Variadic<SizeOrIndexType>,
    #[output]
    shape: ShapeType
}

impl Fold for FromExtents {
    fn fold(&self) -> FoldResult{
        if self.extents.any(|attribute| !attribute) {
            return null();
        }
        let mut extents = SmallVector<[i64; 6]>::new();
        for attr in self.extents {
            extents.push(attr.cast<IntegerAttribute>().get_int());
        }
        let builder = Builder::new(self.context());
        builder.index_tensor_attr(extents)
    }
}

/**
`shape.from_extent_tensor` operation which creates a shape from a tensor of extents.

Creates a shape from a 1D integral tensor of extents. The rank of the resulting shape equals the number of elements in the tensor, and the extents match the values of the elements.
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$input attr-dict `:` type($input)"
)]
pub struct FromExtentTensor {
    input: 1DTensorOf<[Index]>,
    #[output]
    result: ShapeType
}

/**
`shape.is_broadcastable` operation which determines if 2+ shapes can be successfully broadcasted.

Given multiple input shapes or extent tensors, return a predicate specifying if they are broadcastable. This broadcastable follows the same logic as what `shape.broadcast` documents.

Concretely, shape.is_broadcastable returning true implies that `shape.broadcast` will not give an error, and shape.cstr_broadcastable will not result in an assertion failure. Similarly, false implies an error or assertion failure.

# Examples

```mlir
%true = shape.is_broadcastable [2, 2], [3, 1, 2]
%false = shape.is_broadcastable [2, 2], [3, 2]
```
*/
#[mlir(
    traits = [Commutative],
    assembly_format = "$shapes attr-dict `:` type($shapes)"
)]
pub struct IsBroadcastable {
    shapes: [ShapeOrExtentTensorType],
    #[output]
    result: I1

//   let builders = [
//   OpBuilder<(ins Value:$lhs, Value:$rhs),
//     [{ build($_builder, $_state, ::llvm::ArrayRef({lhs, rhs})); }]>,
//   ];

}

impl Fold for IsBroadcastable {

}

impl Canonicalise for IsBroadcastable {

}

/**
`shape.rank` operation which gets the rank of a shape.

Returns the rank of the shape or extent tensor, i.e. the number of extents.
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$shape attr-dict `:` type($shape) `->` type($rank)"
)]
pub struct Rank {
    shape: ShapeOrExtentTensorType,
    #[output]
    rank: SizeOrIndexType
}

impl Rank {
    /**
    Returns when two result types are compatible for this op; method used by
    InferTypeOpInterface
    */
    static bool is_compatible_return_types(TypeRange l, TypeRange r);
}

impl Fold for Rank {
    fn fold(&self, adaptor: FoldAdaptor) -> FoldResult {
        let shape = self.shape.dyn_cast_or_null<DenseIntElementsAttribute>();
        if !shape {
            return {};
        }
        let rank = shape.num_elements();
        let builder = Builder::new(self.context());
        builder.get_index_attribute(rank)
    }
}

impl Verify for Rank {
    fn verify(&self) -> LogicalResult {
        verify_size_or_index_op(self)
    }
}

impl Canonicalise for Rank {
    fn canonicalisation_patterns(
        &self,
        patterns: &RewritePatternSet,
        context: *mut MLIRContext
    ) {
        patterns.add::<RankShapeOfCanonicalisationPattern>(context);
    }
}

impl InferTypeOpInterface for Rank {
    fn infer_return_types(
        &self,
        context: *mut MLIRContext,
        location: Option<Location>,
        operands: ValueRange,
        attributes: DictionaryAttribute,
        regions: RegionRange,
        inferred_return_types: &SmallVectorImpl<Type>
    ) -> LogicalResult {
        if operands[0].r#type.isa<ShapeType>() {
            inferred_return_types.assign({SizeType::new(context)});
        } else {
            inferred_return_types.assign({IndexType::new(context)});
        }
        Ok(())
    }
}

/**
`shape.to_extent_tensor` operation which creates a dimension tensor from a shape.

Converts a shape to a 1D integral tensor of extents. The number of elements in the tensor equals the rank of the shape, and the elements equal the extents of the shape.

If the shape represents an error, this op's behaviour is undefined.
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$input attr-dict `:` type($input) `->` type($result)"
)]
pub struct ToExtentTensor {
    input: ShapeOrExtentTensorType,
    #[output]
    result: IndexTensor
}

impl Fold for ToExtentTensor {
    fn fold(&self, adaptor: FoldAdaptor) -> FoldResult {
        if !adaptor.get_input() {
            return FoldResult();
        }
        let builder = Builder::(self.context());
        let shape = llvm::to_vector<6>(
            adaptor.get_input().cast<DenseIntElementsAttribute>().getValues<i64>());
        let r#type = RankedTensorType::new(
            {static_cast<i64>(shape.len())},
            builder.getIndexType());
        DenseIntElementsAttribute::new(r#type, shape)
    }
}

impl CastOpInterface for ToExtentTensor {

}

/**
`shape.dim` operation which gets the specified extent from the shape of a shaped input.

Gets the extent indexed by `dim` from the shape of the `value` operand. If the index is error or out-of-bound then it returns an invalid size if the return type carries error information else the behaviour is undefined.

This is a convenience op that performs the equivalent of getting the extent of a shape (e.g., `dim(x, i) == get_extent(shape_of(x), i)`).
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$value `,` $index attr-dict `:` type($value) `,` type($index) `->` type($extent)"
)]
pub struct Dim {
    value: AnyShaped,
    index: SizeOrIndexType,
    #[output]
    extent: SizeOrIndexType

//   let builders = [
//     // Builder that allows passing a constant dimension as a simple integer.
//     OpBuilder<(ins Value:$value, i64:$index)>
//   ];
}

impl Dim {
    /// Get the `index` value as integer if it is constant.
    Option<i64> get_constant_index();

    /// Returns when two result types are compatible for this op; method used
    /// by InferTypeOpInterface
    static bool is_compatible_return_types(TypeRange l, TypeRange r);
}

impl Fold for Dim {
    fn fold(&self, adaptor: FoldAdaptor) -> FoldResult {
        let val_type = self.value.r#type;
        let val_shaped_type = val_type.dyn_cast<ShapedType>();
        if !val_shaped_type || !val_shaped_type.has_rank() {
            return null();
        }
        let index = get_constant_index();
        if !index.has_value(){
            return null();
        }
        if index.value() >= val_shaped_type.get_rank() {
            return null();
        }
        let extent = val_shaped_type.getDimSize(*index);
        if ShapedType::is_dynamic(extent) {
            return null();
        }
        IntegerAttribute::new(IndexType::new(self.context()), extent)
    }
}

impl Verify for Dim {
    fn verify(&self) -> LogicalResult {
        let shaped_type = self.value.r#type.cast<ShapedType>();
        if !shaped_type.has_rank() {
            return Ok(());
        }
        if let index = get_constant_index() {
            if *index < 0 || *index >= shaped_type.get_rank() {
                return emitOpError("index is out of range");
            }
        }
        Ok(())
    }
}

impl InferTypeOpInterface for Dim {

}

/**
`shape.get_extent` operation which gets the specified extent from a shape or extent tensor.

Gets the extent indexed by `dim` from the `shape` operand. If the shape is an error then it returns an invalid size.
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$shape `,` $dim attr-dict `:` type($shape) `,` type($dim) `->` type($extent)"
)]
pub struct GetExtent {
    shape: ShapeOrExtentTensorType,
    dim: SizeOrIndexType,
    #[output]
    extent: SizeOrIndexType

//   let builders = [
//     // Builder that allows passing a constant dimension as a simple integer.
//     OpBuilder<(ins Value:$shape, i64:$dim)>
//   ];
}

impl GetExtent {
    /// Get the `dim` value as integer if it is constant.
    pub fn constant_dim() -> Option<i64> {

    }

    /**
    Returns when two result types are compatible for this op; method used by InferTypeOpInterface
    */
    static bool is_compatible_return_types(TypeRange l, TypeRange r);
}

impl Fold for GetExtent {
    fn fold(&self) -> FoldResult {
        let elements = self.shape.dyn_cast_or_null<DenseIntElementsAttribute>();
        if !elements {
            return null();
        }
        match self.constant_dim() {
            None => null(),
            Some(value) => {
                if value >= elements.num_elements() {
                    return null();
                }
                elements.values::<Attribute>()[value as u64]
            }
        }
    }
}

impl Verify for GetExtent {
    pub fn verify(&self) -> LogicalResult {
        verify_size_or_index_op(self)
    }
}

impl InferTypeOpInterface for GetExtent {
    fn infer_return_types(
        &self,
        context: *mut MLIRContext,
        location: Option<Location>,
        operands: ValueRange,
        attributes: DictionaryAttribute,
        regions: RegionRange,
        inferred_return_types: &SmallVectorImpl<Type>
    ) -> LogicalResult {
        if operands.is_empty() {
            return Err(());
        }
    
        let is_shape_type = |arg: Type| {
            if arg.isa<ShapeType>() {
                return true;
            }
            return is_extent_tensor_type(arg);
        };
    
        let types = operands.get_types();
        let acc = types[0];
        for t in drop_begin(types) {
            let l = acc;
            let r = t;
            if !l.isa<ShapeType, SizeType>() {
                std::swap(l, r);}
        
            // Handle sizes, propagate error type if present.
            if l.isa<SizeType>() {
                if r.isa::<SizeType, IndexType>() {
                    acc = l;
                } else {
                    return emitOptionalError(location, "requires all sizes or shapes");
                }
            } else if l.isa<IndexType>() {
                if (r.isa<IndexType>()){
                    acc = r;
                } else {
                    return emitOptionalError(location, "requires all sizes or shapes");
                }
            } else if l.isa<ShapeType>() {
                // Handle shapes, propagate error type if present.
                if is_shape_type(r) {
                    acc = l;
                } else {
                    return emitOptionalError(location, "requires all sizes or shapes");
                }
            } else if is_extent_tensor_type(l) {
                let rank1 = l.cast<RankedTensorType>().shape()[0];
                let rank2 = r.cast<RankedTensorType>().shape()[0];
                if ShapedType::is_dynamic(rank1) {
                    acc = l;
                } else if ShapedType::is_dynamic(rank2) {
                    acc = r;
                } else if rank1 != rank2 {
                    return emitOptionalError(location, "unequal shape cardinality");
                } else {
                    acc = l;
                }
            }
        }
        inferred_return_types.assign({acc});
        Ok(())
    }
}

/**
`shape.index_to_size` which converts a standard index to a shape size.

Converts a standard index to a `shape.size`. This operation and its inverse, `size_to_index`, facilitate index conversion between the standard and the shape dialect.

The behaviour is undefined for negative indices.
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$arg attr-dict"
)]
pub struct IndexToSize {
    arg: Index,
    #[output]
    result: SizeType
}

impl Fold for IndexToSize {
    fn fold(&self, adaptor: FoldAdaptor) -> FoldResult {
        /*
        Constant values of both types, `shape.size` and `index`, are represented as `IntegerAttribute`s which makes constant folding simple.
        */
        if arg = self.arg {
            return arg;
        }
        return {};
    }
}

impl Verify for IndexToSize {

}

impl Canonicalse for IndexToSize {
    fn canonicalisation_patterns(
        &self,
        patterns: &RewritePatternSet,
        context: *mut MLIRContext
    ) {
        patterns.add<SizeToIndexToSizeCanonicalization>(context);
    }
}

/**
`shape.max`: Elementwise maximum.

Computes the elementwise maximum of two sizes or shapes with equal ranks.
If either operand is an error, then an error will be propagated to the result. If the input types mismatch or the ranks do not match, then the result is an error.
*/
#[mlir(
    traits = [Commutative, Pure],
    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
)]
pub struct Max {
    lhs: ShapeOrSizeType,
    rhs: ShapeOrSizeType,
    // out
    result: ShapeOrSizeType
}

impl Max {
    // Returns when two result types are compatible for this op; method used by
    // InferTypeOpInterface
    static bool is_compatible_return_types(TypeRange l, TypeRange r);
}

impl Fold for Max {
    fn fold(&self) -> FoldResult{
        // If operands are equal, just propagate one.
        if self.lhs == self.rhs {
            return self.lhs;
        }
        null()
    }
}

impl InferTypeOpInterface for Max {
    fn infer_return_types(
        context: *mut MLIRContext,
        location: Option<Location>,
        operands: ValueRange,
        attributes: DictionaryAttribute,
        regions: RegionRange,
        inferred_return_types: &SmallVectorImpl<Type>
    ) -> LogicalResult {
        if operands[0].r#type == operands[1].r#type {
            inferred_return_types.assign({operands[0].r#type});
        } else {
            inferred_return_types.assign({SizeType::new(context)});
        }
        Ok(())
    }
}

/**
`shape.meet` returns the least general shape or size of its operands.

An operation that computes the least general shape or dim of input operands.
This effectively asserts that corresponding static dimensions are equal.
The behaviour is to match each element of the shape/size and propagate the most restrictive information, returning an invalid shape if there are contradictory requirements. E.g., using pseudo code

```text
shape.meet([*], [*]) -> [*]
shape.meet([*], [1, ?]) -> [1, ?]
shape.meet([1, 2], [1, ?]) -> [1, 2]
shape.meet([*], [1, 2]) -> [1, 2]
shape.meet([], []) -> []
shape.meet([], [*]) -> []
shape.meet([], [?, ?]) -> [invalid]
shape.meet([1, ?], [2, ?, ?]) -> [invalid]
```

`shape.meet` also allows specifying an optional error string, that may be used to return an error to the user upon mismatch of dimensions.

```mlir
%c = shape.meet %a, %b, error="<reason>"
    : !shape.shape, !shape.shape -> !shape.shape
```
*/
#[mlir(
    traits = [Commutative],
    assembly_format = "$arg0 `,` $arg1 (`,` `error` `=` $error^)? attr-dict `:`
    type($arg0) `,` type($arg1) `->` type($result)"
)]
pub struct Meet {
    arg0: AnyShapeOrSizeType,
    arg1: AnyShapeOrSizeType,
    error: OptionalAttr<StringAttribute>,
    #[output]
    result: AnyShapeOrSizeType
}

impl Meet {
    /**
    Returns when two result types are compatible for this op; method used by
    InferTypeOpInterface
    */
    static bool is_compatible_return_types(TypeRange l, TypeRange r);
}

impl InferTypeOpInterface for Meet {

}

/**
`shape.min` elementwise minimum.

Computes the elementwise minimum of two sizes or shapes with equal ranks.
If either operand is an error, then an error will be propagated to the result. If the input types mismatch or the ranks do not match, then the result is an error.
*/
#[mlir(
    traits = [Commutative, Pure],
    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
)]
pub struct Min {
    lhs: ShapeOrSizeType,
    rhs: ShapeOrSizeType,
    #[output]
    result: ShapeOrSizeType
}

impl Min {
    // Returns when two result types are compatible for this op; method used by
    // InferTypeOpInterface
    static bool is_compatible_return_types(TypeRange l, TypeRange r);
}

impl Fold for Min {
    fn fold(&self) -> FoldResult {
        // If operands are equal, just propagate one.
        if self.lhs == self.rhs {
            return self.lhs;
        }
        null()
    }
}

impl InferTypeOpInterface for Min {
    fn infer_return_types(
        context: *mut MLIRContext,
        location: Option<Location>,
        operands: ValueRange,
        attributes: DictionaryAttribute,
        regions: RegionRange,
        inferred_return_types: &SmallVectorImpl<Type>
    ) -> LogicalResult  {
        if operands[0].r#type == operands[1].r#type {
            inferred_return_types.assign({operands[0].r#type});
        } else {
            inferred_return_types.assign({SizeType::new(context)});
        }
        Ok(())
    }
}

/**
Multiplication of sizes and indices.

Multiplies two sizes or indices. If either operand is an error it will be propagated to the result. The operands can be of type `size` or `index`. If at least one of the operands can hold an error, i.e. if it is of type `size`, the result must be of type `size`. If error propagation is not possible because both operands are of type `index` then the result may be of type `size` or `index`.
*/
#[mlir(
    traits = [Commutative, Pure],
    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
)]
pub struct Mul {
    lhs: SizeOrIndexType,
    rhs: SizeOrIndexType,
    #[output]
    result: SizeOrIndexType
}

impl Mul {
    // Returns when two result types are compatible for this op; method used by
    // InferTypeOpInterface
    static bool is_compatible_return_types(TypeRange l, TypeRange r);
}

impl Fold for Mul {
    fn fold(&self) -> FoldResult {
        let lhs = self.lhs.dyn_cast_or_null<IntegerAttribute>();
        if !lhs {
            return null();
        }
        let rhs = self.rhs.dyn_cast_or_null<IntegerAttribute>();
        if !rhs {
            return null();
        }
        let folded = lhs.value * rhs.value;
        let index_type = IndexType::new(self.context());
        IntegerAttribute::new(index_type, folded)
    }
}

impl Verify for Mul {
    fn verify(&self) -> LogicalResult {
        verify_size_or_index_op(self)
    }
}

impl InferTypeOpInterface for Mul {
    fn infer_return_types(
        context: *mut MLIRContext,
        location: Option<Location>,
        operands: ValueRange,
        attributes: DictionaryAttribute,
        regions: RegionRange,
        inferred_return_types: &SmallVectorImpl<Type>
    ) -> LogicalResult {
        if operands[0].r#type.isa<SizeType>()
        || operands[1].r#type.isa<SizeType>()
        {
            inferred_return_types.assign({SizeType::new(context)});
        } else {
            inferred_return_types.assign({IndexType::new(context)});
        }
        Ok(())
    }
}

/**
`shape.num_elements` returns the number of elements for a given shape.

Returns the number of elements for a given shape which is the product of its extents. If the argument is of type `shape` then the result will be of type `size` and potential errors will be propagated. Otherwise, if the argument is and extent tensor `tensor<?xindex>` then the result will be of type `index`.
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$shape attr-dict `:` type($shape) `->` type($result)"
)]
pub struct NumElements {
    shape: ShapeOrExtentTensorType,
    #[output]
    result: SizeOrIndexType
}

impl NumElements {
    // Returns when two result types are compatible for this op; method used by
    // InferTypeOpInterface
    static bool is_compatible_return_types(TypeRange l, TypeRange r);
}

impl Fold for NumElements {

}

impl Verify for NumElements {
    
}

impl InferTypeOpInterface for NumElements {

}

/**
`shape.reduce` returns an expression reduced over a shape or extent tensor.

An operation that takes as input a shape or extent tensor, and a number of initial values. This operation has a region that is applied repeatedly for every extent of the input. Starting with the initial values, the individual extents are then aggregated as defined by the associated region.

Conceptually this op performs the following reduction:

```text
res[] = init;
for i in 0..shape.rank() {
    res = reduce(i, shape[i], res[0], ..., res[n]);
}
```

Where `reduce` represents the region attached and the result of the reduce op is the last computed output of the reduce region. As an example, the number of elements can be computed as follows:

```mlir
func.func @reduce(%shape : !shape.shape, %init : !shape.size) ->
    !shape.size {
    %num_elements = shape.reduce(%shape, %init) -> !shape.size  {
    ^bb0(%index: index, %dim: !shape.size, %acc: !shape.size):
        %updated_acc = "shape.mul"(%acc, %dim)
            : (!shape.size, !shape.size) -> !shape.size
        shape.yield %updated_acc : !shape.size
    }
    return %num_elements : !shape.size
}
```
*/
#[mlir(
    traits = [SingleBlockImplicitTerminator<"Yield">]
)]
pub struct Reduce {
    shape: ShapeOrExtentTensorType,
    init_vals: Variadic<AnyType>,
    #[output]
    result: Variadic<AnyType>,
    #[region]
    region: SizedRegion<1>

//   let builders = [OpBuilder<(ins Value:$shape, "ValueRange":$init_vals)>];
}

impl Verify for Reduce {
    fn verify(&self) -> LogicalResult {
        // Verify block arg types.
        let block = self.region[0];
      
        // The block takes index, extent, and aggregated values as arguments.
        let block_args_count = self.init_vals.size() + 2;
        if block.get_num_arguments() != block_args_count {
            return emitOpError() << "Reduce body is expected to have "
                               << block_args_count << " arguments";
        }
      
        // The first block argument is the index and must always be of type `index`.
        if (!block.get_argument(0).r#type.isa<IndexType>()) {
            return emitOpError(
                "Argument 0 of Reduce body is expected to be of IndexType");
        }
      
        /*
        The second block argument is the extent and must be of type `size` or `index`, depending on whether the reduce operation is applied to a shape or to an extent tensor.
        */
        let extent_type = block.get_argument(1).r#type;
        if self.shape.r#type.isa<ShapeType>() {
            if !extent_type.isa<SizeType>() {
                return emitOpError("Argument 1 of Reduce body is expected to be of SizeType if the Reduce operates on a ShapeType");
            }
        } else {
            if !extent_type.isa<IndexType>() {
                return emitOpError(
                    "Argument 1 of Reduce body is expected to be of IndexType if the Reduce operates on an extent tensor");
            }
        }
      
        for (index, value) in self.init_vals.enumerate() {
            if block.get_argument(index + 2).r#type != value.r#type {
                return emitOpError(
                    "Type mismatch between argument {} of Reduce body and initial value {}",
                    index + 2,
                    index
                );
            }
        }
        Ok(())
    }
}

impl AssemblyFormat for Reduce {
    fn parse(&self, parser: &OpAsmParser, resultL &OperationState)
    -> ParseResult {
        // Parse operands.
        let operands = SmallVector<[OpAsmParser::UnresolvedOperand; 3]>::new();
        let shape_or_extent_tensor_type;
        if (parser.parse_operand_list(operands, /*requiredOperandCount=*/-1,
                                    OpAsmParser::Delimiter::Paren)
        || parser.parse_colon_type(shape_or_extent_tensor_type)
        || parser.parse_optional_arrow_type_list(result.types)) {
            return Err(());
        }
      
        // Resolve operands.
        let init_vals = llvm::ArrayRef(operands).drop_front();
        if parser.resolve_operand(
            operands[0], shape_or_extent_tensor_type, result.operands)
        || parser.resolve_operands(
            init_vals, result.types, parser.get_name_loc(), result.operands)
        {
            return Err(());
        }
      
        // Parse the body.
        let body = result.add_region();
        if parser.parse_region(*body, /*args=*/{}, /*argTypes=*/{}) {
            return Err(());
        }
      
        // Parse attributes.
        if parser.parse_optional_attr_dict(result.attributes) {
            return Err(());
        }
      
        Ok(())
    }

    fn print(&self, p: &OpAsmPrinter) {
        p << (
            "({}, {}) : {}",
            self.shape,
            self.init_vals,
            self.shape.r#type
        );
        p.print_optional_arrow_type_list(get_result_types());
        p << ' ';
        p.print_region(self.region);
        p.print_optional_attr_dict(self.get_attrs());
    }
}

/**
Returns shape of a value or shaped type operand.

The operation takes a value or a shaped operand as an argument and it returns a shape or extent tensor.
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$arg attr-dict `:` type($arg) `->` type($result)"
)]
pub struct ShapeOf {
    arg: AnyTypeOf<[AnyShaped, ValueShapeType]>,
    #[output]
    result: ShapeOrExtentTensorType
}

impl Fold for ShapeOf {
    fn fold(&self) -> FoldResult {
        let r#type = get_operand().r#type.dyn_cast<ShapedType>();
        if !r#type || !r#type.has_static_shape() {
            return null();
        }
        let builder = Builder::new(self.context());
        builder.get_index_tensor_attr(r#type.shape)
    }
}

impl Verify for ShapeOf {
    fn verify(&self) -> LogicalResult {
        verify_shape_or_extent_tensor_op(self)
    }
}

impl Canonicalise for ShapeOf {
    fn canonicalisation_patterns(
        &self,
        patterns: &RewritePatternSet,
        context: *mut MLIRContext
    ) {
        patterns.add::<
            ShapeOfCastExtentTensor, ShapeOfWithTensor,
            ExtractFromShapeOfExtentTensor>(context);
    }

}

impl InferTypeOpInterface for ShapeOf {
    fn infer_return_types(
        context: *mut MLIRContext,
        location: Option<Location>,
        operands: ValueRange,
        attributes: DictionaryAttribute,
        regions: RegionRange,
        inferred_return_types: &SmallVectorImpl<Type>
    ) -> LogicalResult
    {
        if operands[0].r#type.isa<ValueShapeType>() {
            inferred_return_types.assign({ShapeType::get(context)});
        } else {
            let shaped_ty = operands[0].r#type.cast<ShapedType>();
            let rank =
                shaped_ty.has_rank() ? shaped_ty.get_rank() : ShapedType::kDynamic;
            let index_type = IndexType::new(context);
            let extent_tensor_ty = RankedTensorType::new({rank}, index_type);
            inferred_return_types.assign({extent_tensor_ty});
        }
        Ok(())
    }
}

impl ShapeOf {
    // Returns when two result types are compatible for this op; method used by
    // InferTypeOpInterface
    static bool is_compatible_return_types(TypeRange l, TypeRange r);
}

/**
`shape.value_of` returns value of a !shape.value_shape operand.

The operation takes !shape.value_shape, a.k.a. (value, shape) tuple as an argument, and returns its value. The behaviour is undefined for unknown and invalid arguments.
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$arg attr-dict `:` type($result)"
)]
pub struct ValueOf {
    arg: ValueShapeType,
    #[output]
    result: AnyShaped
}

/**
`shape.size_to_index` casts between index types of the shape and standard dialect.

Converts a `shape.size` to a standard index. This operation and its inverse, `index_to_size`, facilitate index conversion between the standard and the shape dialect. The behaviour is undefined for unknown and invalid arguments.
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$arg attr-dict `:` type($arg)"
)]
pub struct SizeToIndex {
    arg: SizeOrIndexType,
    #[output]
    result: Index
}

impl Fold for SizeToIndex {
    fn fold(&self, adaptor: FoldAdaptor) -> FoldResult {
        /*
        Constant values of both types, `shape.size` and `index`, are represented as `IntegerAttribute`s which makes constant folding simple.
        */
        if let Attribute arg = self.arg {
            return arg;
        }
        return FoldResult();
    }
}

impl Canonicalse for SizeToIndex {
    fn canonicalisation_patterns(
        patterns: &RewritePatternSet,
        context: *mut MLIRContext
    ) {
        patterns.add<IndexToSizeToIndexCanonicalization>(context);
    }
}

impl CastOpInterface for SizeToIndex {

}

/**
`shape.value_as_shape` returns value as a shape.

The operations takes a ValueShape and returns a Shape corresponding to the value. If the input value cannot be shape (e.g., not a 1D tensor of integral value representing sizes) then this propagages the error shape.
E.g.,

```mlir
// The following
%0 = arith.constant dense<[1, 2]> : tensor<2xi32>
%shape = shape.value_as_shape %0 : tensor<2xi32> -> !shape.shape
// is equivalent to
%shape' = shape.const_shape [1, 2] : !shape.shape
```

This operation is the complement of `shape_of` wrt ValueShape values.
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$arg attr-dict `:` type($arg) `->` type($result)"
)]
pub struct ValueAsShape {
    arg: AnyTypeOf<[1DTensorOf<[AnyInteger, Index]>, ValueShapeType]>,
    #[output]
    result: ShapeOrExtentTensorType
}

/**
`shape.with_shape` returns ValueShape with given shape.

Returns ValueShape with the shape updated to match the shape operand. That is a new ValueShape tuple is created with value equal to `operand`'s value and shape equal to `shape`. If the ValueShape and given `shape` are non-conformant, then the returned ValueShape will represent an error of this mismatch. Similarly if either inputs are in an error state, then an error is propagated.

# Usage

```mlir
%0 = shape.with_shape %1, %2 : tensor<...>, !shape.shape
```

This is used, for example, where one combines shape function calculations and/or call one shape function from another. E.g.,

```mlir
func.func @shape_foobah(
    %a: !shape.value_shape,
    %b: !shape.value_shape,
    %c: !shape.value_shape
) -> !shape.shape {
    %0 = call @shape_foo(%a, %b)
        : (!shape.value_shape, !shape.value_shape) -> !shape.shape
    %1 = shape.with_shape %b, %0 : !shape.value_shape, !shape.shape
    %2 = call @shape_bah(%c, %1)
        : (!shape.value_shape, !shape.value_shape) -> !shape.shape
    return %2 : !shape.shape
}
```

This operation need not be a refinement of the shape. In non-error cases the input ValueShape's value and shape are conformant and so too for the output, but the result may be less specified than `operand`'s shape as `shape` is merely used to construct the new ValueShape. If join behaviour is desired then a join op should be used.
*/
#[mlir(
    traits = [Pure],
    assembly_format = "operands attr-dict `:` type($operand) `,` type($shape)"
)]
pub struct WithShape {
    operand: AnyTypeOf<[AnyShaped, ValueShapeType]>,
    shape: ShapeOrExtentTensorType,
    #[output]
    result: ValueShapeType
}

/**
Returns the value to parent op.

*/
#[mlir(
    traits = [
        HasParent<"Reduce, FunctionLibrary">, Pure, ReturnLike, Terminator
    ],
    assembly_format = "attr-dict ($operands^ `:` type($operands))?"
)]
pub struct Yield {
    operands: [AnyType]

//   let builders = [OpBuilder<(ins),
//     [{ build($_builder, $_state, std::nullopt); }]>
//   ];
}

impl Verify for Yield {
    fn verify(&self) -> LogicalResult {
        let parent_op = self.parent();
        let outputs = parent_op.outputs();
        let inputs = inputs();
      
        if parent_op.num_outputs() != num_inputs() {
            return emitOpError(
                "Number of inputs does not match number of outputs of its parent"
            );
        }
        for (output, input) in outputs.zip(inputs) {
            if output.r#type != input.r#type {
                return emitOpError(
                    "Types mismatch between yield op and its parent");
            }
        }
        Ok(())
    }
}

// TODO: Add Ops: if_static, if_ranked

// For testing usage.
/**
Prints the input shape or size.

Prints the input dim or shape and passes through input.

NOTE: This is intended for testing and debugging only.
*/
#[mlir]
pub struct DebugPrint {
    input: ShapeOrSizeType,
    #[output]
    output: ShapeOrSizeType
}

/**
`shape.split_at` splits a shape at a given index.

Splits a shape at a given dimension `index`, returning two shapes. If `index` is negative, it is treated as indexing from the back of the shape.
This negative-handling behaviour is important when handling unranked shapes, where the positive index is not necessarily knowable due to a dynamic number of leading dimensions. If the result is in extent tensor form out of bounds indices result in undefined behaviour.

# Examples

- split_at([4, 5, 6], index=0) -> [], [4, 5, 6]
- split_at([4, 5, 6], index=1) -> [4], [5, 6]
- split_at([4, 5, 6], index=2) -> [4, 5], [6]
- split_at([4, 5, 6], index=3) -> [4, 5, 6], []
- split_at([4, 5, 6], index=4) -> error
- split_at([4, 5, 6], index=-1) -> [4, 5], [6]
- split_at([4, 5, 6], index=-2) -> [4], [5, 6]
- split_at([4, 5, 6], index=-3) -> [], [4, 5, 6]
- split_at([4, 5, 6], index=-4) -> error

# Requires

- `index` is in the range [-rank(operand), rank(operand)]
*/
#[mlir(
    traits = [Pure]
)]
pub struct SplitAt {
    operand: ShapeOrExtentTensorType,
    index: SizeOrIndexType,
    #[output]
    head: ShapeOrExtentTensorType,
    #[output]
    tail: ShapeOrExtentTensorType
}

impl Fold for SplitAt {
    fn fold(&self, results: &SmallVectorImpl<FoldResult>) -> LogicalResult {
        if !self.operand || !self.index {
            return Err(());
        }
        let shape_vec = llvm::to_vector<6>(
            self.operand.cast<DenseIntElementsAttribute>().values::<i64>());
        let shape = llvm::ArrayRef(shape_vec);
        let mut split_point = self.index.cast<IntegerAttribute>().getInt();
        // Verify that the split point is in the correct range.
        // TODO: Constant fold to an "error".
        let rank = shape.len();
        if -rank > split_point || split_point > rank {
            return Err(());
        }
        if split_point < 0 {
            split_point += shape.len();
        }
        let builder = Builder::new(self.operand.context());
        results.push(builder.get_index_tensor_attr(shape.take_front(split_point)));
        results.push(builder.get_index_tensor_attr(shape.drop_front(split_point)));
        Ok(())
    }
}

/**
`shape.concat` concatenates two shapes.

Creates a shape whose dimensions consist of first the dimensions from `lhs` followed by the dimensions of `rhs`.

# Examples

- concat([2, 3], [4, 5]) -> [2, 3, 4, 5]
- concat([], []) -> []
- concat([], [4, 5, 6]) -> [4, 5, 6]
*/
#[mlir(
    traits = [Pure],
    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
)]
pub struct Concat {
    lhs: ShapeOrExtentTensorType,
    rhs: ShapeOrExtentTensorType,
    #[output]
    result: ShapeOrExtentTensorType
}

impl Fold for Concat {
    fn fold(&self) -> FoldResult {
        if !self.lhs || !self.rhs {
            return null();
        }
        let lhs_shape = llvm::to_vector<6>(
            self.lhs.cast<DenseIntElementsAttribute>().values::<i64>());
        let rhs_shape = llvm::to_vector<6>(
            self.rhs.cast<DenseIntElementsAttribute>().values::<i64>());
        let mut result_shape = SmallVector<[i64; 6]>::new();
        result_shape.append(lhs_shape);
        result_shape.append(rhs_shape);
        let builder = Builder::new(self.context());
        build(result_shape)
    }
}

// ----------------------------------------------------------------------
// Shape constraint related ops.
// ----------------------------------------------------------------------

// TODO: Move the code below and witnesses to a different file.

/**
`shape.any` return any combination of the input shapes.

This operation takes multiple input shapes or extent tensors and returns some combination of their dimensions. This can be best seen with examples below.

The result is undefined, but still side-effect free, in cases where the inputs have differing ranks or differ in extents of shared dimensions.

# Examples

```mlir
%s0 = shape.any [2, ?], [?, 3]  // [2, 3]
%s1 = shape.any [?, ?], [1, 2]  // [1, 2]
```
*/
#[mlir(
    traits = [Commutative,  Pure],
    assembly_format = "$inputs attr-dict `:` type($inputs) `->` type($result)"
)]
pub struct Any {
    inputs: [ShapeOrExtentTensorType],
    #[output]
    result: ShapeOrExtentTensorType
}

/*
TODO: Canonicalisation should be implemented for shapes that can be determined through mixtures of the known dimensions of the inputs.
*/
impl Fold for Any {
    fn fold(&self) -> FoldResult {
        // Only the last operand is checked because Any is commutative.
        if let Some(input) = self.inputs.last() {
            return input;
        }
        null()
    }
}

/**
`shape.assuming_all` return a logical AND of all witnesses.

Used to simplify constraints as any single failing precondition is enough to prevent execution.

'assuming' operations represent an execution order restriction to the compiler, information for dependent code to rely on (by assuming), and nothing else. They should not exist after a program is fully lowered and ready to execute.

# Example

```mlir
%w0 = shape.cstr_broadcastable [2, 2], [3, 1, 2]  // Passing
%w1 = shape.cstr_broadcastable [2, 2], [3, 2]  // Failure
%w2 = shape.cstr_eq [1, 2], [1, 2], [1, 2]  // Passing
%wf = shape.assuming_all %w0, %w1  // Failure
%wt = shape.assuming_all %w0, %w2  // Passing
```
*/
#[mlir(
    traits = [Commutative, Pure],
    assembly_format = "$inputs attr-dict"
)]
pub struct AssumingAll {
    inputs: Variadic<WitnessType>,
    #[output]
    result: WitnessType
}

impl Fold for AssumingAll {
    fn fold(&self) -> FoldResult {
        /*
        Iterate in reverse to first handle all constant operands. They are guaranteed to be the tail of the inputs because this is commutative.
        */
        for index in (0..self.inputs.len()).rev() {
            let a = self.inputs[index];
            // Cannot fold if any inputs are not constant;
            if !a {
                return null();
            }
        
            /*
            We do not need to keep statically known values after handling them in this method.
            */
            get_operation().erase_input(index);
        
            // Always false if any input is statically known false
            if !a.cast<BoolAttribute>().value() {
                return a;
            }
        }
        // If this is reached, all inputs were statically known passing.
        BoolAttribute::new(self.context(), true)
    }
}

impl Verify for AssumingAll {
    fn verify(&self) -> LogicalResult {
        // Ensure that AssumingAllOp contains at least one operand
        if get_num_operands() == 0 {
            return emitOpError("No operands specified");
        }
        Ok(())
    }
}

impl Canonicalise for AssumingAll {
    fn canonicalisation_patterns(
        &self,
        patterns: &RewritePatternSet,
        context: *mut MLIRContext
    ) {
        patterns.add::<
            MergeAssumingAllOps, AssumingAllOneOp,
            AssumingAllOfCstrBroadcastable, AssumingAllToCstrEqCanonicalization,
            RemoveDuplicateOperandsPattern<AssumingAllOp>
        >(context);
    }
}

/**
Execute the region.

Executes the region assuming all witnesses are true.

'assuming' operations represent an execution order restriction to the compiler, information for dependent code to rely on (by assuming), and nothing else. They should not exist after a program is fully lowered and ready to execute.
*/
#[mlir(
    traits =  [
        SingleBlockImplicitTerminator<"AssumingYield">,
        DeclareOpInterfaceMethods<RegionBranchOpInterface>,
        RecursiveMemoryEffects
    ]
)]
pub struct Assuming {
    witness: WitnessType,
    #[output]
    results: Variadic<AnyType>,
    #[region]
    do_region: SizedRegion<1>

//   let builders = [
//     OpBuilder<(ins Value:$witness,
//         CArg<"function_ref<SmallVector<Value, 2>(OpBuilder &, Location)>">)>
//   ];
}

impl Assuming {
    /**
    Inline the region into the region containing the Assuming and delete the Assuming.

    This does no checks on the inputs to the Assuming.
    */
    static void inline_region_into_parent(Assuming &op,
        PatternRewriter &rewriter);
}

impl Canonicalise for Assuming {

}

impl AssemblyFormat for Assuming {

}

/**
`shape.assuming_yield` operation.

This yield operation represents a return operation within the `shape.assuming` operation region. The operation takes variable number of operands and produces no results. The operand number and types must match the number and types of parent `shape.assuming` results.
*/
#[mlir(
    traits = [Pure, ReturnLike, Terminator, HasParent<"Assuming">],
    assembly_format = "attr-dict ($operands^ `:` type($operands))?"
)]
pub struct AssumingYield {
    operands: Variadic<AnyType>

//   let builders = [
//     OpBuilder<(ins), [{ /* nothing to do */ }]>,
//   ];
}

/**
`shape.cstr_broadcastable` determines if 2+ shapes can be successfully broadcasted.

Given input shapes or extent tensors, return a witness specifying if they are broadcastable. This broadcastable follows the same logic as what `shape.broadcast` documents.

`cstr` operations represent runtime assertions.

# Examples

```mlir
%w0 = shape.cstr_broadcastable [2, 2], [3, 1, 2]  // Passing
%w1 = shape.cstr_broadcastable [2, 2], [3, 2]  // Failure
```
*/
#[mlir(
    traits = [Commutative],
    assembly_format = "$shapes attr-dict `:` type($shapes)"
)]
pub struct CstrBroadcastable {
    shapes: [ShapeOrExtentTensorType],
    #[output]
    result: WitnessType

//   let builders = [
//   OpBuilder<(ins Value:$lhs, Value:$rhs),
//     [{ build($_builder, $_state, ::llvm::ArrayRef({lhs, rhs})); }]>,
//   ];
}

impl Fold for CstrBroadcastable {
    fn fold(&self, adaptor: FoldAdaptor) -> FoldResult {
        // No broadcasting is needed if all operands but one are scalar.
        if has_at_most_single_non_scalar(self.shapes) {
            return BoolAttribute::new(self.context(), true);
        }
      
        if {
            let extents = SmallVector::<[SmallVector<[i64; 6]>; 6]>::new();
            for operand in self.shapes {
                if !operand {
                    return false;
                }
                extents.push(llvm::to_vector<6>(
                    operand.cast<DenseIntElementsAttribute>().values::<i64>()));
            }
            statically_known_broadcastable(extents)
        } {
            return BoolAttribute::new(self.context(), true);
        }
      
        /*
        Lastly, see if folding can be completed based on what constraints are known on the input shapes.
        */
        if {
              let extents = SmallVector::<[SmallVector<[i64; 6]>; 6]>::new();
              for shape_value in shapes() {
                    extents.emplace_back();
                    if get_shape_vec(shape_value, extents.last()).is_err() {
                        return false;
                    }
              }
              statically_known_broadcastable(extents)
        } {
            return BoolAttribute::new(self.context(), true);
        }
      
        /*
        Because a failing witness result here represents an eventual assertion failure, we do not replace it with a constant witness.
        */
        null()
    }
}

impl Verify for CstrBroadcastable {
    fn verify(&self) -> LogicalResult {
        // Ensure that CstrBroadcastable contains at least two operands
        if get_num_inputs() < 2 {
            return emitOpError("Required at least 2 input shapes");
        }
        Ok(())
    }
}

impl Canonicalise for CstrBroadcastable {
    fn canonicalisation_patterns(
        &self,
        patterns: &RewritePatternSet,
        context: *mut MLIRContext
    ) {
        /*
        Canonicalisation patterns have overlap with the considerations during folding in case additional shape information is inferred at some point that does not result in folding.
        */
        patterns.add::<
            CanonicaliseCastExtentTensorOperandsPattern<CstrBroadcastable>,
            CstrBroadcastableEqOps,
            RemoveDuplicateOperandsPattern<CstrBroadcastable>,
            RemoveEmptyShapeOperandsPattern<CstrBroadcastable>>(context);
    }
}

/**
`shape.cstr_eq` determines if all input shapes are equal.

Given 1 or more input shapes, determine if all shapes are the exact same.

'cstr' operations represent runtime assertions.

#Example

```mlir
%w0 = shape.cstr_eq [1, 2], [1, 2], [1, 2]  // Passing
%w1 = shape.cstr_eq [2, 2], [1, 2]  // Failure
```
*/
#[mlir(
    traits = [Commutative],
    assembly_format = "$shapes attr-dict `:` type($shapes)"
)]
pub struct CstrEq {
    shapes: [ShapeOrExtentTensorType],
    #[output]
    result: WitnessType
}

impl Fold for SttrEq {
    fn fold(&self, adaptor: FoldAdaptor) -> FoldResult {
        if self.shapes.all(|attribute|
              attribute && attribute == self.shapes[0])
        {
            return BoolAttribute::new(self.context(), true);
        }
      
        /*
        Because a failing witness result here represents an eventual assertion failure, we do not try to replace it with a constant witness. Similarly, we cannot if there are any non-const inputs.
        */
        null()
    }
}

impl Canonicalise for CstrEq {
    fn canonicalisation_patterns(
        patterns: &RewritePatternSet,
        context: *mut MLIRContext
    ) {
        // If inputs are equal, return passing witness
        patterns.add::<CstrEqEqOps>(context);
    }
}

/**
`shape.const_witness` operation which returns a statically known witness value.

This operation represents a statically known witness result. This can be often used to canonicalize/fold constraint and assuming code that will always pass.

```mlir
%0 = shape.const_shape [1, 2, 3]
%1 = shape.const_shape [1, 2, 3]
%w0 = shape.cstr_eq(%0, %1)  // Can be folded to "const_witness true"
%w1 = shape.const_witness true
%w2 = shape.assuming_all(%w0, %w2)  // Can be folded to "const_witness true"
```
*/
#[mlir(
    traits = [ConstantLike, Pure],
    assembly_format = "$passing attr-dict"
)]
pub struct ConstWitness {
    passing: BoolAttribute,
    #[output]
    result: WitnessType
}

impl Fold for ConstWitness {
    fn fold(&self) -> FoldResult {
        self.passing_attr()
    }
}

/**
`shape.cstr_require` represents a runtime assertion that an i1 is `true`.

Represents a runtime assertion that an i1 is true. It returns a `!shape.witness` to order this assertion.

For simplicity, prefer using other cstr_* ops if they are available for a given constraint.

# Examples

```mlir
%bool = ...
%w0 = shape.cstr_require %bool, "msg"  // Passing if `%bool` is true.
```

Since this op can be used to express many different possible assertions (depending on whatever computation calculated `pred`), the `msg` should clarify the nature of the assertion for users.
*/
#[mlir(
    assembly_format = "$pred `,` $msg attr-dict"
)]
pub struct CstrRequire {
    pred: I1,
    msg: StringAttribute,
    #[output]
    result: WitnessType
}

impl Fold for CstrRequire {
    fn fold(&self, adaptor: FoldAdaptor) -> FoldResult {
        self.pred
    }
}

// ----------------------------------------------------------------------
// Shape collection ops.
// ----------------------------------------------------------------------

/**
Represents shape functions and corresponding ops.

Represents a list of shape functions and the ops whose shape transfer functions they represent.

# Example

```mlir
shape.function_library {
    func @same_result_shape(%arg: !shape.value_shape) -> !shape.shape {
        %0 = shape_of %arg : !shape.value_shape -> !shape.shape
        return %0 : !shape.shape
    }
} mapping {
    std.atan = @same_result_shape
}
```
*/
#[mlir(
    traits = [
        AffineScope, IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol,
        NoTerminator, SingleBlock
    ]
)]
pub struct FunctionLibrary {
    sym_name: SymbolNameAttribute,
    sym_visibility: OptionalAttr<StringAttribute>,
    mapping: DictionaryAttribute,
    #[region]
    body: AnyRegion

//   let builders = [OpBuilder<(ins "&str":$name)>];
//   let skipDefaultBuilders = 1;
//   let hasCustomAssemblyFormat = 1;
}

impl FunctionLibrary {
    /// Returns an associated shape function for an operation if defined.
    pub fn shape_function(Operation *op) -> Func {

    }
}

impl OpAsmOpInterface for FunctionLibrary {
    // This will filter the `shape.` prefix in front of operations inside the
    // func body.
    static &str default_dialect() { return "shape"; }
}

/**
Shape function.

An operation with a name containing a single `SSACFG` region which represents a shape transfer function or helper function for shape transfer function.
*/
#[mlir(
    name = "func",
    traits = [
        AffineScope, AutomaticAllocationScope, IsolatedFromAbove, Symbol
    ]
)]
pub struct Func {
    sym_name: SymbolNameAttribute,
    function_type: TypeAttrOf<FunctionType>,
    arg_attrs: OptionalAttr<DictArrayAttr>,
    res_attrs: OptionalAttr<DictArrayAttr>,
    sym_visibility: OptionalAttr<StringAttribute>,
    #[region]
    body: AnyRegion

//   let builders = [OpBuilder<(ins
//     "&str":$name, "FunctionType":$type,
//     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
//     CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
//   >];
}

impl Func {
    static Func create(Location location, &str name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static Func create(Location location, &str name, FunctionType type,
                         Operation::dialect_attr_range attrs);
    static Func create(Location location, &str name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs);

    // ------------------------------------------------------------------
    // SymbolOpInterface Methods
    // ------------------------------------------------------------------

    bool isDeclaration() { return isExternal(); }
}

impl AssemblyFormat for Func {

}

impl CallableOpInterface for Func {
    /**
    Returns the region on the current operation that is callable. This may return null in the case of an external callable object, e.g. an external function.
    */
    fn get_callable_region() -> *mut Region {
        isExternal() ? null() : &getBody()
    }

    /**
    Returns the results types that the callable region produces when executed.
    */
    fn get_callable_results() -> &[Type] {
        get_function_type().outputs()
    }
}

impl FunctionOpInterface for Func {
    /// Returns the argument types of this function.
    &[Type] get_argument_types() { return get_function_type().inputs(); }

    /// Returns the result types of this function.
    &[Type] get_result_types() { return get_function_type().outputs(); }
}

impl OpAsmOpInterface for Func {
    /**
    This will filter the `shape.` prefix in front of operations inside the func body.
    */
    static &str default_dialect() { return "shape"; }
}

/**
Shape function return operation.

The `shape.return` operation represents a return operation within a function.  The operation takes variable number of operands and produces no results.
*/
#[mlir(
    name = "return",
    traits = [Pure, HasParent<"Func">, ReturnLike, Terminator],
    assembly_format = "attr-dict ($operands^ `:` type($operands))?"
)]
pub struct Return {
    operands: Variadic<AnyType>

  // TODO: Tighten verification.
}
