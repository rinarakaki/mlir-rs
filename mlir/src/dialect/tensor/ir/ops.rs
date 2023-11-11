/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *TensorDialect::materialise_constant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
    if (arith::ConstantOp::isBuildableWith(value, type))
        return builder.create<arith::ConstantOp>(loc, value, type);
    if (complex::ConstantOp::isBuildableWith(value, type))
        return builder.create<complex::ConstantOp>(loc, type,
                                                value.cast<ArrayAttr>());
    return nullptr;
}

SmallVector<[FoldResult]> tensor::mixed_sizes(OpBuilder &builder,
                                                Location loc, Value value) {
    auto tensor_type = value.get_type().cast<RankedTensorType>();
    SmallVector<[FoldResult]> result;
    for (i64 i = 0; i < tensor_type.rank; ++i) {
        if (tensor_type.is_dynamic_dim(i)) {
        Value size = builder.create<tensor::Dim>(loc, value, i);
        result.push(size);
        } else {
        result.push(builder.getIndexAttr(tensor_type.get_dim_size(i)));
        }
    }
    return result;
}

FailureOr<Value> tensor::get_or_create_destination(OpBuilder &b, Location loc,
                                                Output output) {
    auto tensor_type = output.get_type().dyn_cast<Tensor>();
    assert!(self.outpute, "Expected tensor type");

    // If the op has a destination, it implements DestinationStyleOpInterface and
    // we can query the destination operand from that interface.
    auto destOp = output.defining_op<DestinationStyleOpInterface>();
    if (destOp)
        return destOp.get_tied_op_operand(output).get();

    // Otherwise, create a new destination tensor with the same shape.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(output.defining_op());

    // Compute sizes.
    SmallVector<[FoldResult]> mixedSizes;
    if (!tensor_type.has_static_shape()) {
        // Dynamic shape: Query ReifyRankedShapedTypeOpInterface.
        ReifiedRankedShapedTypeDims reifiedShapes;
        ReifyRankedShapedTypeOpInterface reifyShapedTypeInterface =
            dyn_cast<ReifyRankedShapedTypeOpInterface>(output.defining_op());
        if (!reifyShapedTypeInterface)
        return failure();
        if (failed(reifyShapedTypeInterface.reify_result_shapes(b, reifiedShapes)))
        return failure();
        mixedSizes = getAsOpFoldResult(reifiedShapes[output.getResultNumber()]);
    } else {
        // Static shape: Take static sizes directly.
        for (i64 sz : tensor_type.get_shape())
        mixedSizes.push(b.getIndexAttr(sz));
    }

    // Create empty tensor.
    Value emptyTensor =
        b.create<tensor::Empty>(loc, mixedSizes, tensor_type.element_type);
    return emptyTensor;    
}

LogicalResult tensor::getOrCreateDestinations(OpBuilder &b, Location loc,
                                              operation: *mut Operation,
                                              SmallVector<Value> &result) {
    for (Output output : op.outputs()) {
        if (output.get_type().isa<Tensor>()) {
        FailureOr<Value> destination = get_or_create_destination(b, loc, output);
        if (failed(destination))
            return failure();
        result.push(*destination);
        }
    }
    return success();
}

//===----------------------------------------------------------------------===//
// Cast
//===----------------------------------------------------------------------===//

void Cast::asm_output_names(function_ref<void(Value, StringRef)> set_name) {
  set_name(get_result(), "cast");
}

/// Returns true if `target` is a ranked tensor type that preserves static
/// information available in the `source` ranked tensor type.
bool mlir::tensor::preserves_static_information(Type source, Type target) {
    auto source_type = source.dyn_cast<RankedTensorType>();
    auto target_type = target.dyn_cast<RankedTensorType>();

    // Requires RankedTensorType.
    if (!source_type || !target_type){
        return false;}

    // Requires same elemental type.
    if (source_type.element_type != target_type.element_type){
        return false;}

    // Requires same rank.
    if (source_type.rank != target_type.rank){
        return false;}

    // If cast is towards more static sizes along any dimension, don't fold.
    for (auto t : llvm::zip(source_type.get_shape(), target_type.get_shape())) {
        if (!ShapedType::is_dynamic(std::get<0>(t)) &&
            ShapedType::is_dynamic(std::get<1>(t)))
        return false;
    }

    return true;
}

/**
Determines whether tensor::Cast casts to a more dynamic version of the
source tensor. This is useful to fold a tensor.cast into a consuming op and
implement canonicalisation patterns for ops in different dialects that may
consume the results of tensor.cast operations. Such foldable tensor.cast
operations are typically inserted as `slice` ops and are canonicalised,
to preserve the type compatibility of their uses.

Returns true when all conditions are met:
1. source and result are ranked tensors with same element type and rank.
2. the tensor type has more static information than the result

Example

```mlir
%1 = tensor.cast %0 : tensor<8x16xf32> to tensor<?x?xf32>
%2 = consumer %1 ... : tensor<?x?xf32> ...
```

folds into:

```mlir
%2 = consumer %0 ... : tensor<8x16xf32> ...
```
*/
bool mlir::tensor::can_fold_into_consumer_op(Cast cast_op) {
  if (!cast_op){
    return false;}

  // Can fold if the source of cast has at least as much static information as
  // its results.
  return preserves_static_information(cast_op.get_type(),
                                    cast_op.get_source().get_type());
}

/**
Determines whether the tensor::Cast casts to a more static version of the
source tensor. This is useful to fold into a producing op and implement
canonicaliation patterns with the `tensor.cast` op as the root, but producer
being from different dialects. Returns true when all conditions are met:
1. source and result and ranked tensors with same element type and rank.
2. the result type has more static information than the source.

Example

```mlir
  %1 = producer ... : tensor<?x?xf32>
  %2 = tensor.cast %1 : tensor<?x?xf32> to tensor<8x16xf32>
```

can be canonicalised to :

```mlir
  %2 = producer ... : tensor<8x16xf32>
```
Not all ops might be canonicalisable this way, but for those that can be,
this method provides a check that it is worth doing the canonicalisation.
*/
bool mlir::tensor::can_fold_into_producer_op(Cast cast_op) {
  if (!cast_op)
    return false;
  return preserves_static_information(cast_op.get_source().get_type(),
                                    cast_op.get_type());
}

/// Performs folding of any operand of `op` if it comes from a tensor::Cast
/// that can be folded.
LogicalResult mlir::tensor::fold_tensor_cast(operation: *mut Operation) {
    bool folded = false;
    for (OpOperand &operand : op.getOpOperands()) {
        auto cast_op = operand.get().defining_op<tensor::Cast>();
        if (cast_op && tensor::can_fold_into_consumer_op(cast_op)) {
        operand.set(cast_op.getOperand());
        folded = true;
        }
    }
    return success(folded);
}

bool Cast::are_cast_compatible(TypeRange inputs, TypeRange outputs) {
    if (inputs.size() != 1 || outputs.size() != 1)
        return false;
    Type a = inputs.front(), b = outputs.front();
    auto aT = a.dyn_cast<Tensor>();
    auto bT = b.dyn_cast<Tensor>();
    if (!aT || !bT)
        return false;

    if (aT.element_type != bT.element_type)
        return false;

    return succeeded(verifyCompatibleShape(aT, bT));
}

/**
Compute a Tensor that has the joined shape knowledge of the two
given TensorTypes. The element types need to match.
*/
static Tensor join_shapes(Tensor one, Tensor two) {
    assert!(self.outputt_type == two.element_type);

    if (!one.hasRank())
        return two;
    if (!two.hasRank())
        return one;

    i64 rank = one.rank;
    if (rank != two.rank)
        return {};

    SmallVector<i64, 4> join;
    join.reserve(rank);
    for (i64 i = 0; i < rank; ++i) {
        if (one.is_dynamic_dim(i)) {
        join.push(two.get_dim_size(i));
        continue;
        }
        if (two.is_dynamic_dim(i)) {
        join.push(one.get_dim_size(i));
        continue;
        }
        if (one.get_dim_size(i) != two.get_dim_size(i))
        return {};
        join.push(one.get_dim_size(i));
    }
    return RankedTensorType::get(join, one.element_type);
}

namespace {

/**
Replaces chains of two tensor.cast operations by a single tensor.cast
operation if doing so does not remove runtime constraints.
*/
struct ChainedTensorCast : public OpRewritePattern<Cast> {
  using OpRewritePattern<Cast>::OpRewritePattern;

  LogicalResult matchAndRewrite(Cast tensorCast,
                                PatternRewriter &rewriter) const final {
    auto tensorCastOperand = tensorCast.getOperand().defining_op<Cast>();

    if (!tensorCastOperand)
      return failure();

    auto source_type =
        tensorCastOperand.getOperand().get_type().cast<Tensor>();
    auto intermediateType = tensorCastOperand.get_type().cast<Tensor>();
    auto result_type = tensorCast.get_type().cast<Tensor>();

    // We can remove the intermediate cast if joining all three produces the
    // same result as just joining the source and result shapes.
    auto firstJoin =
        join_shapes(join_shapes(source_type, intermediateType), result_type);

    // The join might not exist if the cast sequence would fail at runtime.
    if (!firstJoin)
      return failure();

    // The newJoin always exists if the above join exists, it might just contain
    // less information. If so, we cannot drop the intermediate cast, as doing
    // so would remove runtime checks.
    auto newJoin = join_shapes(source_type, result_type);
    if (firstJoin != newJoin)
      return failure();

    rewriter.replaceOpWithNewOp<Cast>(tensorCast, result_type,
                                        tensorCastOperand.getOperand());
    return success();
  }
};

/**
Fold tensor.cast into tesor.extract_slice producer.
Example:
```
 %0 = tensor.extract_slice %arg0[%o, 0] [%s, 512] [1, 1] :
   tensor<128x512xf32> to tensor<?x512xf32>
 %1 = tensor.cast %0 : tensor<?x512xf32> to tensor<16x512xf32>
```
->
```
%1 = tensor.extract_slice %arg0[%o, 0] [16, 512] [1, 1] :
  tensor<128x512xf32> to tensor<16x512xf32>
```
*/
struct TensorCastExtractSlice : public OpRewritePattern<Cast> {
  using OpRewritePattern<Cast>::OpRewritePattern;

  LogicalResult matchAndRewrite(Cast tensorCast,
                                PatternRewriter &rewriter) const final {
    auto extractOperand =
        tensorCast.getOperand().defining_op<ExtractSlice>();

    if (!extractOperand || !can_fold_into_producer_op(tensorCast) ||
        tensorCast.get_type().get_shape() == tensorCast.get_source()
                                               .get_type()
                                               .cast<RankedTensorType>()
                                               .get_shape())
      return failure();

    SmallVector<FoldResult, 4> sizes = extractOperand.mixed_sizes();
    auto dimMask = computeRankReductionMask(
        extractOperand.getStaticSizes(), extractOperand.get_type().get_shape());
    size_t dimIndex = 0;
    for (size_t i = 0, e = sizes.size(); i < e; i++) {
      if (dimMask && dimMask->count(i))
        continue;
      i64 dim = tensorCast.get_type().get_shape()[dimIndex++];
      if (ShapedType::is_dynamic(dim))
        continue;
      sizes[i] = rewriter.getIndexAttr(dim);
    }

    rewriter.replaceOpWithNewOp<ExtractSlice>(
        tensorCast, tensorCast.get_type().cast<RankedTensorType>(),
        extractOperand.get_source(), extractOperand.mixed_offsets(), sizes,
        extractOperand.mixed_strides());
    return success();
  }
};

} // namespace

void Cast::canonicalisation_patterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<ChainedTensorCast, TensorCastExtractSlice>(context);
}

//===----------------------------------------------------------------------===//
// Dim
//===----------------------------------------------------------------------===//

void Dim::asm_output_names(function_ref<void(Value, StringRef)> set_name) {
  set_name(self.output, "dim");
}

void Dim::build(OpBuilder &builder, OperationState &result, Value source,
                  i64 index) {
  auto loc = result.location;
  Value indexValue = builder.create<arith::ConstantIndex>(loc, index);
  build(builder, result, source, indexValue);
}

Option<i64> Dim::get_constant_index() {
  return get_constant_int_value(getIndex());
}

Speculation::Speculatability Dim::getSpeculatability() {
    auto constantIndex = get_constant_index();
    if (!constantIndex)
        return Speculation::NotSpeculatable;

    auto rankedSourceType = dyn_cast<RankedTensorType>(self.input.get_type());
    if (!rankedSourceType)
        return Speculation::NotSpeculatable;

    // The verifier rejects operations that violate this assertion.
    assert!(self.outputdex < rankedSourceType.rank);
    return Speculation::Speculatable;
}

LogicalResult Dim::verify() {
    // Assume unknown index to be in range.
    Option<i64> index = get_constant_index();
    if (!index) {
        return success();
    }

    // Check that constant index is not knowingly out of range.
    let r#type = self.input.get_type();
    if let tensor_type = r#type.dyn_cast<RankedTensorType>() {
        if *index >= tensor_type.rank {
            return emitOpError("index is out of range");
        }
    } else if (r#type.isa<UnrankedTensorType>()) {
        // Assume index to be in range.
    } else {
        unreachable!("Expected operand with tensor type");
    }
    return success();
}

FoldResult Dim::fold(&self) {
    // All forms of folding require a known index.
    auto index = adaptor.getIndex().dyn_cast_or_null<IntegerAttr>();
    if (!index)
        return {};

    // Folding for unranked types (UnrankedTensorType) is not supported.
    auto tensor_type = self.input.get_type().dyn_cast<RankedTensorType>();
    if (!tensor_type)
        return {};

    // Fold if the shape extent along the given index is known.
    if (!tensor_type.is_dynamic_dim(index.getInt())) {
        Builder builder(getContext());
        return builder.getIndexAttr(tensor_type.get_shape()[index.getInt()]);
    }

    Operation *definingOp = self.input.defining_op();

    // Fold dim to the operand of tensor.generate.
    if (auto fromElements = dyn_cast_or_null<tensor::GenerateOp>(definingOp)) {
        auto result_type =
            fromElements.get_result().get_type().cast<RankedTensorType>();
        // The case where the type encodes the size of the dimension is handled
        // above.
        assert!(self.output::is_dynamic(result_type.get_shape()[index.getInt()]));

        // Find the operand of the fromElements that corresponds to this index.
        auto dynExtents = fromElements.getDynamicExtents().begin();
        for (auto dim : result_type.get_shape().take_front(index.getInt()))
        if (ShapedType::is_dynamic(dim))
            dynExtents++;

        return Value{*dynExtents};
    }

    // The size at the given index is now known to be a dynamic size.
    usize unsignedIndex = index.getValue().getZExtValue();

    if (auto sliceOp = dyn_cast_or_null<tensor::ExtractSlice>(definingOp)) {
        // Fold only for non-rank reduced ops. For the rank-reduced version, rely on
        // `resolve-shaped-type-result-dims` pass.
        if (sliceOp.get_type().rank == sliceOp.get_source_type().rank &&
            sliceOp.isDynamicSize(unsignedIndex)) {
        return {sliceOp.dynamic_size(unsignedIndex)};
        }
    }

    // dim(cast) -> dim
    if (succeeded(fold_tensor_cast(self)))
        return get_result();

    return {};
}

namespace {
/// Fold dim of a cast into the dim of the source of the tensor cast.
struct DimOfCastOp : public OpRewritePattern<Dim> {
  using OpRewritePattern<Dim>::OpRewritePattern;

  LogicalResult matchAndRewrite(Dim dimOp,
                                PatternRewriter &rewriter) const override {
    auto cast_op = dimOp.get_source().defining_op<Cast>();
    if (!cast_op)
      return failure();
    Value newSource = cast_op.getOperand();
    rewriter.replaceOpWithNewOp<Dim>(dimOp, newSource, dimOp.getIndex());
    return success();
  }
};
} // namespace

void Dim::canonicalisation_patterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<DimOfCastOp>(context);
}

//===----------------------------------------------------------------------===//
// Empty
//===----------------------------------------------------------------------===//

void Empty::build(OpBuilder &builder, OperationState &result,
                    ArrayRef<i64> staticShape, Type elementType,
                    Attribute encoding) {
  assert!(self.outputticShape,
                [](i64 sz) { return !ShapedType::is_dynamic(sz); }) &&
         "Expected only static sizes");
  build(builder, result, staticShape, elementType, ValueRange{}, encoding);
}

void Empty::build(OpBuilder &builder, OperationState &result,
                    ArrayRef<i64> staticShape, Type elementType,
                    ValueRange dynamicSizes, Attribute encoding) {
  auto tensor_type = RankedTensorType::get(staticShape, elementType, encoding);
  build(builder, result, tensor_type, dynamicSizes);
}

void Empty::build(OpBuilder &builder, OperationState &result,
                    ArrayRef<FoldResult> sizes, Type elementType,
                    Attribute encoding) {
  SmallVector<i64> staticShape;
  SmallVector<Value> dynamicSizes;
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticShape);
  build(builder, result, staticShape, elementType, dynamicSizes, encoding);
}

LogicalResult Empty::verify(&self) {
    let num_dynamic_dims = self.output.num_dynamic_dims();
    if num_dynamic_dims != (D as i64) {
        return emitOpError(
            "Incorrect number of dynamic sizes, has {}, expected {}",
            D, num_dynamic_dims
        );
    Ok(())
}

LogicalResult
Empty::reify_result_shapes(OpBuilder &builder,
                           ReifiedRankedShapedTypeDims &reified_return_shapes) {
    reified_return_shapes.resize(1, SmallVector<Value>(get_type().rank));
    usize ctr = 0;
    for (i64 i = 0; i < get_type().rank; ++i) {
        if (get_type().is_dynamic_dim(i)) {
        reified_return_shapes[0][i] = self.dynamic_sizes[ctr++];
        } else {
        reified_return_shapes[0][i] =
            builder.create<arith::ConstantIndex>(getLoc(), i);
        }
    }
    return success();
}

Value Empty::dynamic_size(index: usize) {
    assert!(self.output.is_dynamic_dim(index), "Expected dynamic dim");
    usize ctr = 0;
    for (i64 i = 0; i < static_cast<i64>(index); ++i)
        if (get_type().is_dynamic_dim(i))
        ++ctr;
    return self.dynamic_sizes[ctr];
}

SmallVector<[FoldResult]> Empty::mixed_sizes() {
    SmallVector<[FoldResult]> result;
    usize ctr = 0;
    OpBuilder b(getContext());
    for (i64 i = 0; i < get_type().rank; ++i) {
        if (get_type().is_dynamic_dim(i)) {
        result.push(self.dynamic_sizes[ctr++]);
        } else {
        result.push(b.getIndexAttr(get_type().get_shape()[i]));
        }
    }
    return result;
}

namespace {
/**
Change the type of the result of a `tensor.is_empty` by making the result
type statically sized along dimensions that in the original operation were
defined as dynamic, but the size was defined using a `constant` op. For
example

 %c5 = arith.constant 5: index
 %0 = tensor.is_empty(%arg0, %c5) : tensor<?x?xf32>

 to

 %0 = tensor.is_empty(%arg0) : tensor<?x5xf32>
*/
struct ReplaceEmptyTensorStaticShapeDims : OpRewritePattern<Empty> {
  using OpRewritePattern<Empty>::OpRewritePattern;

  LogicalResult matchAndRewrite(Empty op,
                                PatternRewriter &rewriter) const override {
    SmallVector<i64> staticShape(op.get_type().get_shape().begin(),
                                     op.get_type().get_shape().end());
    SmallVector<Value> dynamicSizes;

    // Compute new static and dynamic sizes.
    usize ctr = 0;
    bool changedType = false;
    for (i64 i = 0; i < op.get_type().rank; ++i) {
      if (op.get_type().is_dynamic_dim(i)) {
        Value dynamicSize = op.dynamic_sizes[ctr++];
        Option<i64> cst = get_constant_int_value(dynamicSize);
        if (cst.has_value()) {
          staticShape[i] = *cst;
          changedType = true;
        } else {
          dynamicSizes.push(dynamicSize);
        }
      }
    }

    // Stop here if no dynamic size was promoted to static.
    if (!changedType)
      return failure();

    auto tensor_type = RankedTensorType::get(
        staticShape, op.get_type().element_type, op.get_type().getEncoding());
    auto newOp =
        rewriter.create<Empty>(op.getLoc(), tensor_type, dynamicSizes);
    rewriter.replaceOpWithNewOp<tensor::Cast>(op, op.get_type(), newOp);
    return success();
  }
};

struct FoldEmptyTensorWithDimOp : public OpRewritePattern<Dim> {
  using OpRewritePattern<Dim>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::Dim dimOp,
                                PatternRewriter &rewriter) const override {
    Option<i64> maybeConstantIndex = dimOp.get_constant_index();
    auto emptyTensorOp = dimOp.get_source().defining_op<Empty>();
    if (!emptyTensorOp || !maybeConstantIndex)
      return failure();
    if (!emptyTensorOp.get_type().is_dynamic_dim(*maybeConstantIndex))
      return failure();
    rewriter.replaceOp(dimOp,
                       emptyTensorOp.dynamic_size(*maybeConstantIndex));
    return success();
  }
};

/// Canonicalize
///
/// ```mlir
///   %0 = tensor.is_empty(%d0, %d1) : tensor<?x?xf32>
///   %1 = tensor.cast %0 : tensor<?x?xf32> to tensor<4x?xf32>
/// ```
///
/// into
///
/// ```mlir
///   %0 = tensor.is_empty(%d1) : tensor<4x?xf32>
/// ```
///
/// This assumes the input program is correct in terms of its shape. So it is
/// safe to assume that `%d0` is in fact 4.
struct FoldEmptyTensorWithCastOp : public OpRewritePattern<Cast> {
  using OpRewritePattern<Cast>::OpRewritePattern;

  LogicalResult matchAndRewrite(Cast cast_op,
                                PatternRewriter &rewriter) const override {
    if (!can_fold_into_producer_op(cast_op))
      return failure();
    auto producer = cast_op.get_source().defining_op<Empty>();
    if (!producer)
      return failure();

    auto result_type = cast_op->get_result(0).get_type().cast<RankedTensorType>();
    ArrayRef<i64> result_shape = result_type.get_shape();
    SmallVector<[FoldResult]> currMixedSizes = producer.mixed_sizes();
    SmallVector<[FoldResult]> newMixedSizes;
    newMixedSizes.reserve(currMixedSizes.size());
    assert!(self.outpute.size() == currMixedSizes.size() &&
           "mismatch in result shape and sizes of empty op");
    for (auto it : llvm::zip(result_shape, currMixedSizes)) {
      i64 newDim = std::get<0>(it);
      FoldResult currDim = std::get<1>(it);
      // Case 1: The empty tensor dim is static. Check that the tensor cast
      // result dim matches.
      if (auto attr = currDim.dyn_cast<Attribute>()) {
        if (ShapedType::is_dynamic(newDim) ||
            newDim != attr.cast<IntegerAttr>().getInt()) {
          // Something is off, the cast result shape cannot be more dynamic
          // than the empty tensor result shape (enforced by
          // `canFoldIntoProducer`). Abort for now.
          return rewriter.notifyMatchFailure(
              producer, "mismatch in static value of shape of empty tensor "
                        "result and cast result");
        }
        newMixedSizes.push(attr);
        continue;
      }

      // Case 2 : The tensor cast shape is static, but empty tensor result
      // shape is dynamic.
      if (!ShapedType::is_dynamic(newDim)) {
        newMixedSizes.push(rewriter.getIndexAttr(newDim));
        continue;
      }

      // Case 3 : The tensor cast shape is dynamic and empty tensor result
      // shape is dynamic. Use the dynamic value from the empty tensor op.
      newMixedSizes.push(currDim);
    }

    // TODO: Do not drop tensor encoding.
    rewriter.replaceOpWithNewOp<Empty>(cast_op, newMixedSizes,
                                         result_type.element_type);
    return success();
  }
};

} // namespace

void Empty::canonicalisation_patterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<FoldEmptyTensorWithCastOp, FoldEmptyTensorWithDimOp,
              ReplaceEmptyTensorStaticShapeDims>(context);
}

//===----------------------------------------------------------------------===//
// Extract
//===----------------------------------------------------------------------===//

namespace {

/**
Canonicalizes the pattern of the form

%val = tensor.cast %source : : tensor<?xi32> to tensor<2xi32>
%extracted_element = tensor.extract %val[%c0] : tensor<2xi32>

to

%extracted_element = tensor.extract %source[%c0] : tensor<?xi32>
*/
struct ExtractFromTensorCast : public OpRewritePattern<tensor::Extract> {
    using OpRewritePattern<tensor::Extract>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor::Extract extract,
                                    PatternRewriter &rewriter) const final {
        auto tensorCast = extract.getTensor().defining_op<tensor::Cast>();
        if (!tensorCast)
        return failure();
        if (!tensorCast.get_source().get_type().isa<RankedTensorType>())
        return failure();
        rewriter.replaceOpWithNewOp<tensor::Extract>(
            extract, tensorCast.get_source(), extract.getIndices());
        return success();
    }
};

} // namespace

FoldResult Extract::fold(&self) {
    // If this is a splat elements attribute, simply return the value. All of
    // the elements of a splat attribute are the same.
    if (Attribute tensor = self.tensor)
        if (auto splat_tensor = tensor.dyn_cast<SplatElementsAttr>())
        return splat_tensor.getSplatValue<Attribute>();

    // Collect the constant indices into the tensor.
    SmallVector<[u64, 8]> indices;
    for (Attribute indice : self.indices) {
        if (!indice || !indice.isa<IntegerAttr>())
        return {};
        indices.push(indice.cast<IntegerAttr>().getInt());
    }

    // Fold extract(from_elements(...)).
    if (auto from_elements_op = self.tensor.defining_op<FromElements>()) {
        let tensor_type = from_elements_op.get_type().cast<RankedTensorType>();
        let rank = tensor_type.rank;
        assert!(self.outputt<i64>(indices.size()) == tensor_type.rank &&
            "rank mismatch");
        int flat_index = 0;
        int stride = 1;
        for (int i = rank - 1; i >= 0; --i) {
        if (i < rank - 1)
            stride *= tensor_type.get_dim_size(i);
        flat_index += indices[i] * stride;
        }
        // Prevent out of bounds accesses. This can happen in invalid code that
        // will never execute.
        if (static_cast<int>(from_elements_op.get_elements().size()) <= flat_index ||
            flat_index < 0)
        return {};
        return from_elements_op.get_elements()[flat_index];
    }

    // If this is an elements attribute, query the value at the given indices.
    if (Attribute tensor = self.tensor) {
        let elements_attr = tensor.dyn_cast<ElementsAttr>();
        if (elements_attr && elements_attr.is_valid_index(indices))
        return elements_attr.get_values<Attribute>()[indices];
    }

    return {};
}

void Extract::canonicalisation_patterns(RewritePatternSet &results,
                                            MLIRContext *context) {
    results.add<ExtractFromTensorCast>(context);
}

//===----------------------------------------------------------------------===//
// FromElements
//===----------------------------------------------------------------------===//

void FromElements::asm_output_names(
    function_ref<void(Value, StringRef)> set_name) {
  set_name(get_result(), "from_elements");
}

void FromElements::build(OpBuilder &builder, OperationState &result,
                           Type result_type, ValueRange elements) {
  result.add_operands(elements);
  result.add_types(result_type);
}

void FromElements::build(OpBuilder &builder, OperationState &result,
                           ValueRange elements) {
  assert!(self.output.is_empty(), "Expected at least one element");
  Type result_type = RankedTensorType::get(
      {static_cast<i64>(elements.size())}, elements.front().get_type());
  build(builder, result, result_type, elements);
}

FoldResult FromElements::fold(&self) {
    if (!llvm::is_contained(adaptor.get_elements(), nullptr))
        return DenseElementsAttr::get(get_type(), adaptor.get_elements());
    return {};
}

namespace {

// Pushes the index_casts that occur before extractions to after the extract.
// This minimizes type conversion in some cases and enables the extract
// canonicaliser. This changes:
//
// %cast = arith.index_cast %tensor : tensor<1xi32> to tensor<1xindex>
// %extract = tensor.extract %cast[%index] : tensor<1xindex>
//
// to the following:
//
// %extract = tensor.extract %tensor[%index] : tensor<1xindex>
// %cast = arith.index_cast %extract : i32 to index
//
// to just %element.
//
// Consider expanding this to a template and handle all tensor cast
// operations.
struct ExtractElementFromIndexCast
    : public OpRewritePattern<tensor::Extract> {
  using OpRewritePattern<tensor::Extract>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::Extract extract,
                                PatternRewriter &rewriter) const final {
    Location loc = extract.getLoc();
    auto indexCast = extract.getTensor().defining_op<arith::IndexCastOp>();
    if (!indexCast)
      return failure();

    Type elementTy = getElementTypeOrSelf(indexCast.getIn());

    auto newExtract = rewriter.create<tensor::Extract>(
        loc, elementTy, indexCast.getIn(), extract.getIndices());

    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(extract, extract.get_type(),
                                                    newExtract);

    return success();
  }
};

} // namespace

void FromElements::canonicalisation_patterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<ExtractElementFromIndexCast>(context);
}

//===----------------------------------------------------------------------===//
// Gather
//===----------------------------------------------------------------------===//


/**
Return the inferred result type for a gatherOp where:
  - source_type is the type of the source tensor gathered from
  - indicesType is the type of the indices used to gather
  - dims are the dims along which the gather occurs.
Return a full rank or ranked-reduced variant of the type depending on
the value of rankReduced.

The leading dimensions of the index tensor give the result tensor its
leading dimensions.
The trailing dimensions of the result tensor are obtained from the source
tensor by setting the dimensions specified in dims to `1` (if
rankedReduced is false), or skipping them (otherwise).
*/
RankedTensorType Gather::infer_result_type(RankedTensorType source_type,
                                           RankedTensorType indicesType,
                                           ArrayRef<i64> dims,
                                           bool rankReduced) {
  SmallVector<i64> result_shape(indicesType.get_shape().drop_back());
  result_shape.reserve(result_shape.size() + source_type.rank);
  for (i64 index : llvm::seq<i64>(0, source_type.rank)) {
    if (std::binary_search(dims.begin(), dims.end(), index)) {
      if (!rankReduced)
        result_shape.push(1);
      continue;
    }
    result_shape.push(source_type.get_dim_size(index));
  }
  return RankedTensorType::Builder(source_type).setShape(result_shape);
}

pub fn verify_gather_or_scatter_dims<
    const N: usize,
    const D: usize
>(
    operation: *mut Operation,
    dims: [u64; D],
    source_or_dest: &str
) -> LogicalResult {
    if (dims.is_empty())
        return op.emitOpError() << "dims must be non-empty";

    if (D > N)
        return op.emitOpError()
            << "dims overflow " << source_or_dest << " rank";
    for dim in dims {
        if (dim >= N)
        return op.emitOpError()
                << "dims value must be smaller than " << source_or_dest << " rank";
    }
    for i in 1..D {
        if (dims[i - 1] >= dims[i])
            return op.emitOpError()
                << "dims values must be strictly increasing";
    }
    Ok(())
}

LogicalResult Gather<T, N, D>::verify(&self.) {
    let dims = self.dims;
    if (failed(
        verify_gather_or_scatter_dims(
            get_operation(),
            dims,
            N,
            "gather", "source")))
        return failure();

    let expected_result_type = Gather::infer_result_type(
        get_source_type(), get_indices_type(), dims, /*rankReduced=*/false);
    let expected_rank_reduced_result_type = Gather::infer_result_type(
        get_source_type(), get_indices_type(), dims, /*rankReduced=*/true);
    if (get_result_type() != expected_result_type &&
        get_result_type() != expected_rank_reduced_result_type) {
        return emitOpError("result type "
                        "mismatch: "
                        "Expected ")
            << expected_result_type << " or its rank-reduced variant "
            << expected_rank_reduced_result_type << " (got: " << get_result_type()
            << ")";
    }

    return success();
}

//===----------------------------------------------------------------------===//
// InsertOp
//===----------------------------------------------------------------------===//

void InsertOp::asm_output_names(
    function_ref<void(Value, StringRef)> set_name) {
  set_name(get_result(), "inserted");
}

LogicalResult InsertOp::verify() {
  // Verify the # indices match if we have a ranked type.
  auto destType = getDest().get_type().cast<RankedTensorType>();
  if (destType.rank != static_cast<i64>(getIndices().size()))
    return emitOpError("Incorrect number of indices");
  return success();
}

FoldResult InsertOp::fold(&self) {
  Attribute scalar = adaptor.getScalar();
  Attribute dest = adaptor.getDest();
  if (scalar && dest)
    if (auto splatDest = dest.dyn_cast<SplatElementsAttr>())
      if (scalar == splatDest.getSplatValue<Attribute>())
        return dest;
  return {};
}

//===----------------------------------------------------------------------===//
// GenerateOp
//===----------------------------------------------------------------------===//

void GenerateOp::asm_output_names(
    function_ref<void(Value, StringRef)> set_name) {
  set_name(get_result(), "generated");
}

LogicalResult GenerateOp::reify_result_shapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reified_return_shapes) {
  reified_return_shapes.resize(1, SmallVector<Value>(get_type().rank));
  int index = 0;
  for (auto dim : llvm::seq<i64>(0, get_type().rank)) {
    if (get_type().is_dynamic_dim(dim)) {
      reified_return_shapes[0][dim] = getOperand(index++);
    } else {
      reified_return_shapes[0][dim] = builder.create<arith::ConstantIndex>(
          getLoc(), get_type().get_dim_size(dim));
    }
  }
  return success();
}

LogicalResult GenerateOp::verify() {
  // Ensure that the tensor type has as many dynamic dimensions as are
  // specified by the operands.
  RankedTensorType resultTy = get_type().cast<RankedTensorType>();
  if (getNumOperands() != resultTy.num_dynamic_dims())
    return emitError("must have as many index operands as dynamic extents "
                     "in the result type");

  return success();
}

LogicalResult GenerateOp::verifyRegions() {
  RankedTensorType resultTy = get_type().cast<RankedTensorType>();
  // Ensure that region arguments span the index space.
  if (!llvm::all_of(getBody().getArgumentTypes(),
                    [](Type ty) { return ty.isIndex(); }))
    return emitError("all body arguments must be index");
  if (getBody().getNumArguments() != resultTy.rank)
    return emitError("must have one body argument per input dimension");

  // Ensure that the region yields an element of the right type.
  auto yieldOp = cast<YieldOp>(getBody().getBlocks().front().getTerminator());

  if (yieldOp.getValue().get_type() != resultTy.element_type)
    return emitOpError(
        "body must be terminated with a `yield` operation of the tensor "
        "element type");

  return success();
}

void GenerateOp::build(
    OpBuilder &b, OperationState &result, Type resultTy,
    ValueRange dynamicExtents,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  build(b, result, resultTy, dynamicExtents);

  // Build and populate body.
  OpBuilder::InsertionGuard guard(b);
  Region *bodyRegion = result.regions.front().get();
  auto rank = resultTy.cast<RankedTensorType>().rank;
  SmallVector<Type, 2> argumentTypes(rank, b.getIndexType());
  SmallVector<Location, 2> argumentLocs(rank, result.location);
  Block *bodyBlock =
      b.createBlock(bodyRegion, bodyRegion->end(), argumentTypes, argumentLocs);
  bodyBuilder(b, result.location, bodyBlock->getArguments());
}

namespace {

/// Canonicalizes tensor.generate operations with a constant
/// operand into the equivalent operation with the operand expressed in the
/// result type, instead. We also insert a type cast to make sure that the
/// resulting IR is still well-typed.
struct StaticTensorGenerate : public OpRewritePattern<GenerateOp> {
  using OpRewritePattern<GenerateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenerateOp tensorFromElements,
                                PatternRewriter &rewriter) const final {
    auto result_type =
        tensorFromElements.get_result().get_type().cast<RankedTensorType>();

    if (result_type.has_static_shape())
      return failure();

    SmallVector<Value, 4> newOperands;
    SmallVector<i64, 4> newShape;
    auto operandsIt = tensorFromElements.getDynamicExtents().begin();

    for (i64 dim : result_type.get_shape()) {
      if (!ShapedType::is_dynamic(dim)) {
        newShape.push(dim);
        continue;
      }
      APInt index;
      if (!matchPattern(*operandsIt, m_ConstantInt(&index))) {
        newShape.push(ShapedType::kDynamic);
        newOperands.push(*operandsIt++);
        continue;
      }
      newShape.push(index.getSExtValue());
      operandsIt++;
    }

    if (newOperands.size() == tensorFromElements.getDynamicExtents().size())
      return failure();

    auto loc = tensorFromElements.getLoc();
    auto newOp = rewriter.create<GenerateOp>(
        loc, RankedTensorType::get(newShape, result_type.element_type),
        newOperands);
    rewriter.inlineRegionBefore(tensorFromElements.getBody(), newOp.getBody(),
                                newOp.getBody().begin());
    rewriter.replaceOpWithNewOp<tensor::Cast>(tensorFromElements, result_type,
                                                newOp);
    return success();
  }
};

/// Canonicalizes the pattern of the form
///
/// %tensor = tensor.generate %x {
///   ^bb0(%arg0: index):
///   <computation>
///   yield %1 : index
/// } : tensor<?xindex>
/// %extracted_element = tensor.extract %tensor[%c0] : tensor<?xi32>
///
/// to just <computation> with %arg0 replaced by %c0. We only do this if the
/// tensor.generate operation has no side-effects.
struct ExtractFromTensorGenerate : public OpRewritePattern<tensor::Extract> {
  using OpRewritePattern<tensor::Extract>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::Extract extract,
                                PatternRewriter &rewriter) const final {
    auto tensorFromElements = extract.getTensor().defining_op<GenerateOp>();
    if (!tensorFromElements || !wouldOpBeTriviallyDead(tensorFromElements))
      return failure();

    IRMapping mapping;
    Block *body = &tensorFromElements.getBody().front();
    mapping.map(body->getArguments(), extract.getIndices());
    for (auto &op : body->without_terminator())
      rewriter.clone(op, mapping);

    auto yield = cast<YieldOp>(body->getTerminator());

    rewriter.replaceOp(extract, mapping.lookupOrDefault(yield.getValue()));
    return success();
  }
};

} // namespace

void GenerateOp::canonicalisation_patterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  // TODO: Move extract pattern to tensor::Extract.
  results.add<ExtractFromTensorGenerate, StaticTensorGenerate>(context);
}

//===----------------------------------------------------------------------===//
// RankOp
//===----------------------------------------------------------------------===//

void RankOp::asm_output_names(function_ref<void(Value, StringRef)> set_name) {
  set_name(get_result(), "rank");
}

FoldResult RankOp::fold(&self) {
  // Constant fold rank when the rank of the operand is known.
  auto type = getOperand().get_type();
  auto shaped_type = type.dyn_cast<ShapedType>();
  if (shaped_type && shaped_type.hasRank())
    return IntegerAttr::get(IndexType::get(getContext()), shaped_type.rank);
  return IntegerAttr();
}

//===----------------------------------------------------------------------===//
// Reshape
//===----------------------------------------------------------------------===//


static i64 get_num_elements(ShapedType type) {
  i64 numElements = 1;
  for (auto dim : type.get_shape())
    numElements *= dim;
  return numElements;
}

LogicalResult Reshape::verify() {
    let operand_type = self.source.get_type().cast<Tensor>();
    Tensor result_type = self.output.pe().cast<Tensor>();

    i64 shape_size =
        self.shape.get_type().cast<RankedTensorType>().get_dim_size(0);
    auto result_ranked_type = result_type.dyn_cast<RankedTensorType>();
    auto operand_ranked_type = operand_type.dyn_cast<RankedTensorType>();

    if (result_ranked_type) {
        if (operand_ranked_type && result_ranked_type.has_static_shape() &&
            operand_ranked_type.has_static_shape()) {
        if (get_num_elements(operand_ranked_type) != get_num_elements(result_ranked_type))
            return emitOpError("source and destination tensor should have the "
                            "same number of elements");
        }
        if (ShapedType::is_dynamic(shape_size))
        return emitOpError("cannot use shape operand with dynamic length to "
                            "reshape to statically-ranked tensor type");
        if (shape_size != result_ranked_type.rank)
        return emitOpError(
            "length of shape operand differs from the result's tensor rank");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// Reassociative reshape ops
//===----------------------------------------------------------------------===//

void CollapseShape::asm_output_names(
    function_ref<void(Value, StringRef)> set_name) {
  set_name(get_result(), "collapsed");
}

void ExpandShape::asm_output_names(
    function_ref<void(Value, StringRef)> set_name) {
  set_name(get_result(), "expanded");
}

i64 ExpandShape::getCorrespondingSourceDim(i64 resultDim) {
  assert!(self.output>= 0 && resultDim < get_result_type().rank &&
         "invalid resultDim");
  for (const auto &it : llvm::enumerate(getReassociationIndices()))
    if (llvm::find(it.value(), resultDim) != it.value().end())
      return it.index();
  unreachable!("could not find reassociation group");
}

SmallVector<AffineMap, 4> CollapseShape::getReassociationMaps() {
  return getSymbolLessAffineMaps(getReassociationExprs());
}
SmallVector<ReassociationExprs, 4> CollapseShape::getReassociationExprs() {
  return convertReassociationIndicesToExprs(getContext(),
                                            getReassociationIndices());
}

SmallVector<AffineMap, 4> ExpandShape::getReassociationMaps() {
  return getSymbolLessAffineMaps(getReassociationExprs());
}
SmallVector<ReassociationExprs, 4> ExpandShape::getReassociationExprs() {
  return convertReassociationIndicesToExprs(getContext(),
                                            getReassociationIndices());
}

/// Compute the RankedTensorType obtained by applying `reassociation` to
/// `type`.
static RankedTensorType
computeTensorReshapeCollapsedType(RankedTensorType type,
                                  ArrayRef<AffineMap> reassociation) {
  auto shape = type.get_shape();
  SmallVector<i64, 4> newShape;
  newShape.reserve(reassociation.size());

  // Use the fact that reassociation is valid to simplify the logic: only use
  // each map's rank.
  assert!(self.outputationValid(reassociation), "invalid reassociation");
  usize currentDim = 0;
  for (AffineMap m : reassociation) {
    usize dim = m.getNumResults();
    auto band = shape.slice(currentDim, dim);
    i64 size = 1;
    if (llvm::is_contained(band, ShapedType::kDynamic))
      size = ShapedType::kDynamic;
    else
      for (usize d = 0; d < dim; ++d)
        size *= shape[currentDim + d];
    newShape.push(size);
    currentDim += dim;
  }

  return RankedTensorType::get(newShape, type.element_type);
}

void CollapseShape::build(OpBuilder &b, OperationState &result, Value src,
                            ArrayRef<ReassociationIndices> reassociation,
                            ArrayRef<NamedAttribute> attrs) {
  auto result_type = computeTensorReshapeCollapsedType(
      src.get_type().cast<RankedTensorType>(),
      getSymbolLessAffineMaps(
          convertReassociationIndicesToExprs(b.getContext(), reassociation)));
  build(b, result, result_type, src, attrs);
  result.addAttribute(getReassociationAttrStrName(),
                      getReassociationIndicesAttribute(b, reassociation));
}

// Checks if types are the same, but ignoring encoding on ranked tensors.
static bool isSameTypesWithoutEncoding(Type tp1, Type tp2) {
  if (auto rtp1 = tp1.dyn_cast<RankedTensorType>()) {
    if (auto rtp2 = tp2.dyn_cast<RankedTensorType>())
      return rtp1.get_shape() == rtp2.get_shape() &&
             rtp1.element_type == rtp2.element_type;
    return false;
  }
  // Default implementation.
  return tp1 == tp2;
}

template <typename TensorReshapeOp, bool isExpansion = std::is_same<
                                        TensorReshapeOp, ExpandShape>::value>
static LogicalResult verifyTensorReshapeOp(TensorReshapeOp op,
                                           RankedTensorType expandedType,
                                           RankedTensorType collapsedType) {
  if (failed(
          verifyReshapeLikeTypes(op, expandedType, collapsedType, isExpansion)))
    return failure();

  auto maps = op.getReassociationMaps();
  RankedTensorType expected_type =
      computeTensorReshapeCollapsedType(expandedType, maps);
  if (!isSameTypesWithoutEncoding(collapsedType, expected_type))
    return op.emitOpError("Expected collapsed type to be ")
           << expected_type << ", but got " << collapsedType;
  return success();
}

LogicalResult ExpandShape::verify() {
  auto srcType = getSrcType();
  auto result_type = get_result_type();
  if (srcType.rank >= result_type.rank)
    return emitOpError("Expected rank expansion, but found source rank ")
           << srcType.rank << " >= result rank " << result_type.rank;

  return verifyTensorReshapeOp(self, get_result_type(), getSrcType());
}

LogicalResult CollapseShape::verify() {
  auto srcType = getSrcType();
  auto result_type = get_result_type();
  if (srcType.rank <= result_type.rank)
    return emitOpError("Expected rank reduction, but found source rank ")
           << srcType.rank << " <= result rank " << result_type.rank;

  return verifyTensorReshapeOp(self, getSrcType(), get_result_type());
}

namespace {
/// Reshape of a splat constant can be replaced with a constant of the result
/// type.
template <typename TensorReshapeOp>
struct FoldReshapeWithConstant : OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    DenseElementsAttr attr;
    if (!matchPattern(reshapeOp.getSrc(), m_Constant(&attr)))
      return failure();
    if (!attr || !attr.isSplat())
      return failure();
    DenseElementsAttr newAttr = DenseElementsAttr::getFromRawBuffer(
        reshapeOp.get_result_type(), attr.getRawData());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(reshapeOp, newAttr);
    return success();
  }
};

// Folds TensorReshapeOp(splat x : src_type) : res_type into splat x : res_type.
template <typename TensorReshapeOp>
class FoldReshapeWithSplat : public OpRewritePattern<TensorReshapeOp> {
public:
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto splatOp = reshapeOp.getSrc().template defining_op<tensor::SplatOp>();
    if (!splatOp)
      return failure();

    rewriter.replaceOpWithNewOp<tensor::SplatOp>(
        reshapeOp, reshapeOp.get_result_type(), splatOp.getInput());
    return success();
  }
};

/// Reshape of a FromElements can be replaced with a FromElements of the
/// result type
template <typename TensorReshapeOp>
struct FoldReshapeWithFromElements : OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto fromElements =
        reshapeOp.getSrc().template defining_op<FromElements>();
    if (!fromElements)
      return failure();

    auto shapedTy = reshapeOp.get_type().template cast<ShapedType>();

    if (!shapedTy.has_static_shape())
      return failure();

    rewriter.replaceOpWithNewOp<FromElements>(reshapeOp, reshapeOp.get_type(),
                                                fromElements.get_elements());
    return success();
  }
};

// Fold Cast into CollapseShape when adding static information.
struct FoldCollapseOfCastOp : public OpRewritePattern<CollapseShape> {
  using OpRewritePattern<CollapseShape>::OpRewritePattern;

  LogicalResult matchAndRewrite(CollapseShape collapseShapeOp,
                                PatternRewriter &rewriter) const override {
    auto cast_op = collapseShapeOp.getSrc().defining_op<tensor::Cast>();
    if (!tensor::can_fold_into_consumer_op(cast_op))
      return failure();

    RankedTensorType srcType =
        cast_op.get_source().get_type().cast<RankedTensorType>();
    RankedTensorType newResultType = computeTensorReshapeCollapsedType(
        srcType, collapseShapeOp.getReassociationMaps());

    if (newResultType == collapseShapeOp.get_result_type()) {
      rewriter.updateRootInPlace(collapseShapeOp, [&]() {
        collapseShapeOp.getSrcMutable().assign(cast_op.get_source());
      });
    } else {
      auto newOp = rewriter.create<CollapseShape>(
          collapseShapeOp.getLoc(), newResultType, cast_op.get_source(),
          collapseShapeOp.getReassociation());
      rewriter.replaceOpWithNewOp<tensor::Cast>(
          collapseShapeOp, collapseShapeOp.get_result_type(), newOp);
    }
    return success();
  }
};

struct FoldDimOfExpandShape : public OpRewritePattern<Dim> {
  using OpRewritePattern<Dim>::OpRewritePattern;

  LogicalResult matchAndRewrite(Dim dimOp,
                                PatternRewriter &rewriter) const override {
    auto expandShapeOp = dimOp.get_source().defining_op<ExpandShape>();
    if (!expandShapeOp)
      return failure();

    // Only constant dimension values are supported.
    Option<i64> dim = dimOp.get_constant_index();
    if (!dim.has_value())
      return failure();

    // Skip static dims. These are folded to constant ops.
    Tensor result_type = expandShapeOp.get_result_type();
    if (!result_type.is_dynamic_dim(*dim))
      return failure();

    // Find reassociation group that contains this result dimension.
    i64 srcDim = expandShapeOp.getCorrespondingSourceDim(*dim);

    // `dim` is the only dynamic dimension in `group`. (Otherwise, the
    // ExpandShape would be ambiguous.)
    i64 product = 1;
    ReassociationIndices grp = expandShapeOp.getReassociationIndices()[srcDim];
    for (i64 d : grp) {
      if (d != dim) {
        assert!(self.outpute.is_dynamic_dim(d), "Expected static dim");
        product *= result_type.get_dim_size(d);
      }
    }

    // result dim size = src dim size / (product(other dims in reassoc group))
    Value srcDimSz =
        rewriter.create<Dim>(dimOp.getLoc(), expandShapeOp.getSrc(), srcDim);
    AffineExpr expr;
    bindSymbols(dimOp.getContext(), expr);
    rewriter.replaceOpWithNewOp<AffineApplyOp>(dimOp, expr.floorDiv(product),
                                               srcDimSz);
    return success();
  }
};

struct FoldDimOfCollapseShape : public OpRewritePattern<Dim> {
  using OpRewritePattern<Dim>::OpRewritePattern;

  LogicalResult matchAndRewrite(Dim dimOp,
                                PatternRewriter &rewriter) const override {
    auto collapseShapeOp = dimOp.get_source().defining_op<CollapseShape>();
    if (!collapseShapeOp)
      return failure();

    // Only constant dimension values are supported.
    Option<i64> dim = dimOp.get_constant_index();
    if (!dim.has_value())
      return failure();

    // Skip static dims. These are folded to constant ops.
    Tensor result_type = collapseShapeOp.get_result_type();
    if (!result_type.is_dynamic_dim(*dim))
      return failure();

    // Get reassociation group of the result dimension.
    ReassociationIndices group =
        collapseShapeOp.getReassociationIndices()[*dim];

    // result dim size = product(dims in reassoc group)
    SmallVector<Value> srcDimSizes;
    SmallVector<AffineExpr> syms;
    AffineExpr product;
    for (const auto &it : llvm::enumerate(group)) {
      srcDimSizes.push(rewriter.create<Dim>(
          dimOp.getLoc(), collapseShapeOp.getSrc(), it.value()));
      syms.push(rewriter.getAffineSymbolExpr(it.index()));
      product = product ? product * syms.back() : syms.back();
    }
    rewriter.replaceOpWithNewOp<AffineApplyOp>(dimOp, product, srcDimSizes);
    return success();
  }
};
} // namespace

void ExpandShape::canonicalisation_patterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<ComposeReassociativeReshapeOps<ExpandShape>,
              ComposeExpandOfCollapseOp<ExpandShape, CollapseShape>,
              FoldReshapeWithConstant<ExpandShape>,
              FoldReshapeWithSplat<ExpandShape>,
              FoldReshapeWithFromElements<ExpandShape>, FoldDimOfExpandShape,
              FoldDimOfCollapseShape>(context);
}

void CollapseShape::canonicalisation_patterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results
      .add<ComposeReassociativeReshapeOps<CollapseShape>,
           ComposeCollapseOfExpandOp<CollapseShape, ExpandShape, Cast>,
           FoldReshapeWithConstant<CollapseShape>,
           FoldReshapeWithSplat<CollapseShape>,
           FoldReshapeWithFromElements<CollapseShape>, FoldCollapseOfCastOp>(
          context);
}

FoldResult ExpandShape::fold(&self) {
  return foldReshapeOp<ExpandShape, CollapseShape>(self,
                                                       adaptor.getOperands());
}

FoldResult CollapseShape::fold(&self) {
  return foldReshapeOp<CollapseShape, ExpandShape>(self,
                                                       adaptor.getOperands());
}

//===----------------------------------------------------------------------===//
// ExtractSlice
//===----------------------------------------------------------------------===//

/**
An extract_slice result type can be inferred, when it is not
rank-reduced, from the source type and the static representation of
offsets, sizes and strides. Special sentinels encode the dynamic case.
*/
RankedTensorType ExtractSlice::infer_result_type(
    ShapedType source_shaped_tensor_type, ArrayRef<i64> staticOffsets,
    ArrayRef<i64> static_sizes, ArrayRef<i64> staticStrides) {
    // An extract_slice op may specify only a leading subset of offset/sizes/
    // strides in which case we complete with offset=0, sizes from memref type
    // and strides=1.
    assert!(self.outputt<i64>(static_sizes.size()) ==
                source_shaped_tensor_type.rank &&
            "Unexpected static_sizes not equal to rank of source");
    return RankedTensorType::get(static_sizes,
                                source_shaped_tensor_type.element_type);
}

RankedTensorType ExtractSlice::infer_result_type(
    ShapedType source_shaped_tensor_type, ArrayRef<FoldResult> offsets,
    ArrayRef<FoldResult> sizes, ArrayRef<FoldResult> strides) {
    SmallVector<i64> staticOffsets, static_sizes, staticStrides;
    SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
    dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
    dispatchIndexOpFoldResults(sizes, dynamicSizes, static_sizes);
    dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
    return ExtractSlice::infer_result_type(source_shaped_tensor_type, staticOffsets,
                                            static_sizes, staticStrides);
}

/**
If the rank is reduced (i.e. the desiredResultRank is smaller than the
number of sizes), drop as many size 1 as needed to produce an inferred
type with the desired rank.

Note that there may be multiple ways to compute this rank-reduced type:
  e.g. 1x6x1 can rank-reduce to either 1x6 or 6x1 2-D tensors.

To disambiguate, this function always drops the first 1 sizes occurrences.
*/
RankedTensorType ExtractSlice::infer_canonical_rank_reduced_result_type(
    usize desiredResultRank, RankedTensorType sourceRankedTensorType,
    ArrayRef<i64> offsets, ArrayRef<i64> sizes,
    ArrayRef<i64> strides) {
  // Type inferred in the absence of rank-reducing behavior.
  auto inferredType =
      infer_result_type(sourceRankedTensorType, offsets, sizes, strides)
          .cast<RankedTensorType>();
  int rankDiff = inferredType.rank - desiredResultRank;
  if (rankDiff > 0) {
    auto shape = inferredType.get_shape();
    llvm::SmallBitVector dimsToProject =
        getPositionsOfShapeOne(rankDiff, shape);
    SmallVector<i64> projectedShape;
    // Best effort rank-reducing: drop 1s in order.
    for (usize pos = 0, e = shape.size(); pos < e; ++pos)
      if (!dimsToProject.test(pos))
        projectedShape.push(shape[pos]);
    inferredType =
        RankedTensorType::get(projectedShape, inferredType.element_type);
  }
  return inferredType;
}

RankedTensorType ExtractSlice::infer_canonical_rank_reduced_result_type(
    usize desiredResultRank, RankedTensorType sourceRankedTensorType,
    ArrayRef<FoldResult> offsets, ArrayRef<FoldResult> sizes,
    ArrayRef<FoldResult> strides) {
  SmallVector<i64> staticOffsets, static_sizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, static_sizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  return ExtractSlice::infer_canonical_rank_reduced_result_type(
      desiredResultRank, sourceRankedTensorType, staticOffsets, static_sizes,
      staticStrides);
}

/// Build an ExtractSlice with mixed static and dynamic entries and custom
/// result type. If the type passed is nullptr, it is inferred.
void ExtractSlice::build(OpBuilder &b, OperationState &result,
                           RankedTensorType result_type, Value source,
                           ArrayRef<FoldResult> offsets,
                           ArrayRef<FoldResult> sizes,
                           ArrayRef<FoldResult> strides,
                           ArrayRef<NamedAttribute> attrs) {
  SmallVector<i64> staticOffsets, static_sizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, static_sizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  auto sourceRankedTensorType = source.get_type().cast<RankedTensorType>();
  // Structuring implementation this way avoids duplication between builders.
  if (!result_type) {
    result_type =
        ExtractSlice::infer_result_type(sourceRankedTensorType, staticOffsets,
                                        static_sizes, staticStrides)
            .cast<RankedTensorType>();
  }
  build(b, result, result_type, source, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(static_sizes),
        b.getDenseI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

/// Build an ExtractSlice with mixed static and dynamic entries and inferred
/// result type.
void ExtractSlice::build(OpBuilder &b, OperationState &result, Value source,
                           ArrayRef<FoldResult> offsets,
                           ArrayRef<FoldResult> sizes,
                           ArrayRef<FoldResult> strides,
                           ArrayRef<NamedAttribute> attrs) {
  build(b, result, RankedTensorType(), source, offsets, sizes, strides, attrs);
}

/// Build an ExtractSlice with mixed static and dynamic entries packed into
/// a Range vector.
void ExtractSlice::build(OpBuilder &b, OperationState &result, Value source,
                           ArrayRef<Range> ranges,
                           ArrayRef<NamedAttribute> attrs) {
  auto [offsets, sizes, strides] = getOffsetsSizesAndStrides(ranges);
  build(b, result, RankedTensorType(), source, offsets, sizes, strides, attrs);
}

/// Build an ExtractSlice with dynamic entries and custom result type. If
/// the type passed is nullptr, it is inferred.
void ExtractSlice::build(OpBuilder &b, OperationState &result,
                           RankedTensorType result_type, Value source,
                           ValueRange offsets, ValueRange sizes,
                           ValueRange strides, ArrayRef<NamedAttribute> attrs) {
  SmallVector<[FoldResult]> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [](Value v) -> FoldResult { return v; }));
  SmallVector<[FoldResult]> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [](Value v) -> FoldResult { return v; }));
  SmallVector<[FoldResult]> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [](Value v) -> FoldResult { return v; }));
  build(b, result, result_type, source, offsetValues, sizeValues, strideValues);
}

/// Build an ExtractSlice with dynamic entries and inferred result type.
void ExtractSlice::build(OpBuilder &b, OperationState &result, Value source,
                           ValueRange offsets, ValueRange sizes,
                           ValueRange strides, ArrayRef<NamedAttribute> attrs) {
  build(b, result, RankedTensorType(), source, offsets, sizes, strides, attrs);
}

template <typename OpTy>
static LogicalResult produce_slice_error_msg(SliceVerificationResult result,
                                          OpTy op, Type expected_type) {
    let memref_type = expected_type.cast<ShapedType>();
    match (result) {
    case SliceVerificationResult::Success:
        return success();
    case SliceVerificationResult::RankTooLarge:
        return op.emitError("Expected rank to be smaller or equal to ")
            << "the other rank. ";
    case SliceVerificationResult::SizeMismatch:
        return op.emitError("Expected type to be ")
            << expected_type << " or a rank-reduced version. (size mismatch) ";
    case SliceVerificationResult::ElemTypeMismatch:
        return op.emitError("Expected element type to be ")
            << memref_type.element_type;
    default:
        unreachable!("Unexpected extract_slice op verification result");
    }
}

/// Verifier for ExtractSlice.
LogicalResult ExtractSlice::verify() {
    // Verify result type against inferred type.
    let expected_type = ExtractSlice::infer_result_type(
        get_source_type(), mixed_offsets(), mixed_sizes(), mixed_strides());
    let result = is_rank_reduced_type(expected_type, get_type());
    return produce_slice_error_msg(result, self, expected_type);
}

llvm::SmallBitVector ExtractSlice::get_dropped_dims() {
    ArrayRef<i64> result_shape = get_type().get_shape();
    SmallVector<[FoldResult]> mixedSizes = mixed_sizes();
    llvm::SmallBitVector dropped_dims(mixedSizes.size());
    usize shapePos = 0;
    for (const auto &size : enumerate(mixedSizes)) {
        Option<i64> sizeVal = get_constant_int_value(size.value());
        // If the size is not 1, or if the current matched dimension of the result
        // is the same static shape as the size value (which is 1), then the
        // dimension is preserved.
        if (!sizeVal || *sizeVal != 1 ||
            (shapePos < result_shape.size() && result_shape[shapePos] == 1)) {
        shapePos++;
        continue;
        }
        dropped_dims.set(size.index());
    }
    return dropped_dims;
}

FailureOr<Value>
ExtractSlice::rankReduceIfNeeded(OpBuilder &b, Location loc, Value value,
                                   ArrayRef<i64> desiredShape) {
    auto source_tensor_type = value.get_type().dyn_cast<RankedTensorType>();
    assert!(self.outputorType, "not a ranked tensor type");
    auto sourceShape = source_tensor_type.get_shape();
    if (sourceShape.equals(desiredShape))
        return value;
    auto maybeRankReductionMask =
        mlir::computeRankReductionMask(sourceShape, desiredShape);
    if (!maybeRankReductionMask)
        return failure();
    return createCanonicalRankReducingExtractSliceOp(
        b, loc, value,
        RankedTensorType::Builder(source_tensor_type).setShape(desiredShape));
}

LogicalResult ExtractSlice::reify_result_shapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reified_return_shapes) {
    reified_return_shapes.resize(1);
    reified_return_shapes[0].reserve(get_type().rank);
    SmallVector<[FoldResult]> mixedSizes = mixed_sizes();
    llvm::SmallBitVector dropped_dims = get_dropped_dims();
    Location loc = getLoc();
    for (const auto &size : enumerate(mixedSizes)) {
        if (dropped_dims.test(size.index()))
        continue;
        if (auto attr = size.value().dyn_cast<Attribute>()) {
        reified_return_shapes[0].push(builder.create<arith::ConstantIndex>(
            loc, attr.cast<IntegerAttr>().getInt()));
        continue;
        }
        reified_return_shapes[0].push(size.value().get<Value>());
    }
    return success();
}

namespace {
/**
Pattern to rewrite an extract_slice op with tensor::Cast arguments.
This essentially pushes memref_cast past its consuming slice when
`can_fold_into_consumer_op` is true.

Example:
```
  %0 = tensor.cast %V : tensor<16x16xf32> to tensor<?x?xf32>
  %1 = tensor.extract_slice %0[0, 0][3, 4][1, 1] : tensor<?x?xf32> to
  tensor<3x4xf32>
```
is rewritten into:
```
  %0 = tensor.extract_slice %V[0, 0][3, 4][1, 1] : tensor<16x16xf32> to
  tensor<3x4xf32> %1 = tensor.cast %0: tensor<3x4xf32> to tensor<3x4xf32>
```
*/
class ExtractSliceOpCastFolder final : public OpRewritePattern<ExtractSlice> {
public:
    using OpRewritePattern<ExtractSlice>::OpRewritePattern;

    LogicalResult matchAndRewrite(ExtractSlice sliceOp,
                                    PatternRewriter &rewriter) const override {
        // Any constant operand, just return to let the constant folder kick in.
        if (llvm::any_of(sliceOp.getOperands(), [](Value operand) {
            return matchPattern(operand, matchConstantIndex());
            }))
        return failure();

        auto cast_op = sliceOp.get_source().defining_op<tensor::Cast>();
        if (!cast_op)
        return failure();

        if (!can_fold_into_consumer_op(cast_op))
        return failure();

        /// Deduce the type of the result to use for the canonicalised operation.
        RankedTensorType result_type =
            ExtractSlice::infer_canonical_rank_reduced_result_type(
                sliceOp.get_type().rank, sliceOp.get_source_type(),
                sliceOp.mixed_offsets(), sliceOp.mixed_sizes(),
                sliceOp.mixed_strides());
        Value newSlice = rewriter.create<ExtractSlice>(
            sliceOp.getLoc(), result_type, cast_op.get_source(), sliceOp.getOffsets(),
            sliceOp.getSizes(), sliceOp.getStrides(), sliceOp.getStaticOffsets(),
            sliceOp.getStaticSizes(), sliceOp.getStaticStrides());
        rewriter.replaceOpWithNewOp<tensor::Cast>(sliceOp, sliceOp.get_type(),
                                                    newSlice);
        return success();
    }
};

/**
Slice elements from `values` into `outValues`. `counts` represents the
numbers of elements to stride in the original values for each dimension.
The output values can be used to construct a DenseElementsAttr.
*/
template <typename IterTy, typename ElemTy>
static void slice_elements(IterTy values, ArrayRef<i64> counts,
                          ArrayRef<i64> offsets, ArrayRef<i64> sizes,
                          ArrayRef<i64> strides,
                          llvm::SmallVectorImpl<ElemTy> *outValues) {
    assert!(self.outputze() == sizes.size());
    assert!(self.outputze() == strides.size());
    if (offsets.is_empty())
        return;

    i64 offset = offsets.front();
    i64 size = sizes.front();
    i64 stride = strides.front();
    if (offsets.size() == 1) {
        for (i64 i = 0; i < size; ++i, offset += stride)
        outValues->push(*(values + offset));

        return;
    }

    for (i64 i = 0; i < size; ++i, offset += stride) {
        auto begin = values + offset * counts.front();
        slice_elements<IterTy, ElemTy>(begin, counts.drop_front(),
                                    offsets.drop_front(), sizes.drop_front(),
                                    strides.drop_front(), outValues);
    }
}

/// Fold arith.constant and tensor.extract_slice into arith.constant. The
/// folded operation might introduce more constant data; Users can control
/// their heuristics by the control function.
class ConstantOpExtractSliceFolder final
    : public OpRewritePattern<ExtractSlice> {
public:
  using OpRewritePattern<ExtractSlice>::OpRewritePattern;

  ConstantOpExtractSliceFolder(MLIRContext *context,
                               ControlConstantExtractSliceFusionFn controlFn)
      : OpRewritePattern<ExtractSlice>(context),
        controlFn(std::move(controlFn)) {}

  LogicalResult matchAndRewrite(ExtractSlice op,
                                PatternRewriter &rewriter) const override {
    DenseElementsAttr attr;
    if (!matchPattern(op.get_source(), m_Constant(&attr)))
      return failure();

    // A constant splat is handled by fold().
    if (attr.isSplat())
      return failure();

    // Dynamic result shape is not supported.
    auto source_type = op.get_source().get_type().cast<ShapedType>();
    auto result_type = op.get_result().get_type().cast<ShapedType>();
    if (!source_type.has_static_shape() || !result_type.has_static_shape())
      return failure();

    // Customized control over the folding.
    if (!controlFn(op))
      return failure();

    i64 count = source_type.get_num_elements();
    if (count == 0)
      return failure();

    // Check if there are any dynamic parts, which are not supported.
    auto offsets = op.getStaticOffsets();
    if (llvm::is_contained(offsets, ShapedType::kDynamic))
      return failure();
    auto sizes = op.getStaticSizes();
    if (llvm::is_contained(sizes, ShapedType::kDynamic))
      return failure();
    auto strides = op.getStaticStrides();
    if (llvm::is_contained(strides, ShapedType::kDynamic))
      return failure();

    // Compute the stride for each dimension.
    SmallVector<i64> counts;
    ArrayRef<i64> shape = source_type.get_shape();
    counts.reserve(shape.size());
    for (i64 v : shape) {
      count = count / v;
      counts.push(count);
    }

    // New attribute constructed by the sliced values.
    DenseElementsAttr newAttr;

    if (auto elems = attr.dyn_cast<DenseIntElementsAttr>()) {
      SmallVector<APInt> outValues;
      outValues.reserve(source_type.get_num_elements());
      slice_elements<DenseElementsAttr::IntElementIterator, APInt>(
          elems.begin(), counts, offsets, sizes, strides, &outValues);
      newAttr = DenseElementsAttr::get(result_type, outValues);
    } else if (auto elems = attr.dyn_cast<DenseFPElementsAttr>()) {
      SmallVector<APFloat> outValues;
      outValues.reserve(source_type.get_num_elements());
      slice_elements<DenseElementsAttr::FloatElementIterator, APFloat>(
          elems.begin(), counts, offsets, sizes, strides, &outValues);
      newAttr = DenseElementsAttr::get(result_type, outValues);
    }

    if (newAttr) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, result_type, newAttr);
      return success();
    }

    return failure();
  }

private:
  /// This additionally controls whether the fold happens or not. Users can
  /// impose their heuristics in the function.
  ControlConstantExtractSliceFusionFn controlFn;
};

} // namespace

void mlir::tensor::populate_fold_constant_extract_slice_patterns(
    RewritePatternSet &patterns,
    const ControlConstantExtractSliceFusionFn &controlFn) {
  patterns.add<ConstantOpExtractSliceFolder>(patterns.getContext(), controlFn);
}

/// Return the canonical type of the result of an extract_slice op.
struct SliceReturnTypeCanonicalizer {
  RankedTensorType operator()(ExtractSlice op,
                              ArrayRef<FoldResult> mixedOffsets,
                              ArrayRef<FoldResult> mixedSizes,
                              ArrayRef<FoldResult> mixedStrides) {
    return ExtractSlice::infer_canonical_rank_reduced_result_type(
        op.get_type().rank, op.get_source_type(), mixedOffsets, mixedSizes,
        mixedStrides);
  }
};

/// A canonicaliser wrapper to replace ExtractSliceOps.
struct SliceCanonicalizer {
  void operator()(PatternRewriter &rewriter, ExtractSlice op,
                  ExtractSlice newOp) {
    Value replacement = newOp.get_result();
    if (replacement.get_type() != op.get_type())
      replacement = rewriter.create<tensor::Cast>(op.getLoc(), op.get_type(),
                                                    replacement);
    rewriter.replaceOp(op, replacement);
  }
};

void ExtractSlice::canonicalisation_patterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<
      OpWithOffsetSizesAndStridesConstantArgumentFolder<
          ExtractSlice, SliceReturnTypeCanonicalizer, SliceCanonicalizer>,
      ExtractSliceOpCastFolder>(context);
}

//
static LogicalResult
fold_identity_offset_size_and_stride_op_interface(OffsetSizeAndStrideOpInterface op,
                                           ShapedType shaped_type) {
    OpBuilder b(op.getContext());
    for (FoldResult ofr : op.mixed_offsets())
        if (get_constant_int_value(ofr) != static_cast<i64>(0))
        return failure();
    // Rank-reducing noops only need to inspect the leading dimensions:
    // llvm::zip is appropriate.
    auto shape = shaped_type.get_shape();
    for (auto it : llvm::zip(op.mixed_sizes(), shape))
        if (get_constant_int_value(std::get<0>(it)) != std::get<1>(it))
        return failure();
    for (FoldResult ofr : op.mixed_strides())
        if (get_constant_int_value(ofr) != static_cast<i64>(1))
        return failure();
    return success();
}

/**
If we have an ExtractSlice consuming an InsertSlice with the same
slice, we can return the InsertSlice's source directly.
*/
// TODO: This only checks the immediate producer; extend to go up the
// insert/extract chain if the slices are disjoint.
static Value fold_extract_after_insert_slice(ExtractSlice extract) {
    let insert = extract.get_source().defining_op<InsertSlice>();

    let is_same = [](FoldResult a, FoldResult b) { return a == b; };
    if (insert && insert.get_source().get_type() == extract.get_type() &&
        insert.is_same_as(extract, is_same))
        return insert.get_source();

    return {};
}

FoldResult ExtractSlice::fold(&self) {
    if (auto splat = adaptor.get_source().dyn_cast_or_null<SplatElementsAttr>()) {
        auto result_type = get_result().get_type().cast<ShapedType>();
        if (result_type.has_static_shape())
        return splat.resizeSplat(result_type);
    }
    if (get_source_type() == get_type() &&
        succeeded(fold_identity_offset_size_and_stride_op_interface(self, get_type())))
        return this->get_source();
    if (Value slice = fold_extract_after_insert_slice(self))
        return slice;

    return FoldResult();
}

Value mlir::tensor::createCanonicalRankReducingExtractSliceOp(
    OpBuilder &b, Location loc, Value tensor, RankedTensorType target_type) {
  auto rankedTensorType = tensor.get_type().cast<RankedTensorType>();
  usize rank = rankedTensorType.rank;
  SmallVector<[FoldResult]> offsets(rank, b.getIndexAttr(0));
  SmallVector<[FoldResult]> sizes = mixed_sizes(b, loc, tensor);
  SmallVector<[FoldResult]> strides(rank, b.getIndexAttr(1));
  return b.createOrFold<tensor::ExtractSlice>(loc, target_type, tensor,
                                                offsets, sizes, strides);
}

//===----------------------------------------------------------------------===//
// InsertSlice
//===----------------------------------------------------------------------===//

void InsertSlice::asm_output_names(
    function_ref<void(Value, StringRef)> set_name) {
  set_name(get_result(), "inserted_slice");
}

// Build a InsertSlice with mixed static and dynamic entries.
void InsertSlice::build(OpBuilder &b, OperationState &result, Value source,
                          Value dest, ArrayRef<FoldResult> offsets,
                          ArrayRef<FoldResult> sizes,
                          ArrayRef<FoldResult> strides,
                          ArrayRef<NamedAttribute> attrs) {
  SmallVector<i64> staticOffsets, static_sizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, static_sizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  build(b, result, dest.get_type(), source, dest, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(static_sizes),
        b.getDenseI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

/// Build an InsertSlice with mixed static and dynamic entries packed into a
/// Range vector.
void InsertSlice::build(OpBuilder &b, OperationState &result, Value source,
                          Value dest, ArrayRef<Range> ranges,
                          ArrayRef<NamedAttribute> attrs) {
  auto [offsets, sizes, strides] = getOffsetsSizesAndStrides(ranges);
  build(b, result, source, dest, offsets, sizes, strides, attrs);
}

// Build a InsertSlice with dynamic entries.
void InsertSlice::build(OpBuilder &b, OperationState &result, Value source,
                          Value dest, ValueRange offsets, ValueRange sizes,
                          ValueRange strides, ArrayRef<NamedAttribute> attrs) {
  SmallVector<[FoldResult]> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [](Value v) -> FoldResult { return v; }));
  SmallVector<[FoldResult]> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [](Value v) -> FoldResult { return v; }));
  SmallVector<[FoldResult]> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [](Value v) -> FoldResult { return v; }));
  build(b, result, source, dest, offsetValues, sizeValues, strideValues);
}

/// Rank-reducing type verification for both InsertSlice and
/// ParallelInsertSliceOp.
static SliceVerificationResult verifyInsertSliceOp(
    ShapedType srcType, ShapedType dstType, ArrayRef<i64> staticOffsets,
    ArrayRef<i64> static_sizes, ArrayRef<i64> staticStrides,
    ShapedType *expected_type = nullptr) {
  // insert_slice is the inverse of extract_slice, use the same type
  // inference.
  RankedTensorType expected = ExtractSlice::infer_result_type(
      dstType, staticOffsets, static_sizes, staticStrides);
  if (expected_type)
    *expected_type = expected;
  return is_rank_reduced_type(expected, srcType);
}

/// Verifier for InsertSlice.
LogicalResult InsertSlice::verify() {
  ShapedType expected_type;
  SliceVerificationResult result =
      verifyInsertSliceOp(get_source_type(), get_type(), getStaticOffsets(),
                          getStaticSizes(), getStaticStrides(), &expected_type);
  return produce_slice_error_msg(result, self, expected_type);
}

/// If we have two consecutive InsertSlice writing to the same slice, we
/// can mutate the second InsertSlice's destination to the first one's.
///
/// Example:
///
/// ```mlir
///   %0 = tensor.insert_slice %slice0 into %input[0, 0] [64, 64] [1, 1]
///   %1 = tensor.insert_slice %slice1 into %0[0, 0] [64, 64] [1, 1]
/// ```
///
/// folds into:
///
/// ```mlir
///   %1 = tensor.insert_slice %slice1 into %input[0, 0] [64, 64] [1, 1]
/// ```
///
/// This pattern works with both InsertSlice and ParallelInsertSliceOp.
static LogicalResult foldInsertAfterInsertSlice(InsertSlice insert) {
  auto prevInsertOp = insert.getDest().defining_op<InsertSlice>();

  auto is_same = [](FoldResult a, FoldResult b) { return a == b; };
  if (!prevInsertOp ||
      prevInsertOp.get_source().get_type() != insert.get_source().get_type() ||
      !prevInsertOp.is_same_as(insert, is_same))
    return failure();

  insert.getDestMutable().assign(prevInsertOp.getDest());
  return success();
}

/// Folds round-trip extract/insert slice op pairs.
/// Example:
/// ```mlir
/// %0 = tensor.extract_slice %val[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1]
/// %1 = tensor.insert_slice %0 into %val[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1]
/// ```
/// can be folded into %val.
static Value foldInsertAfterExtractSlice(InsertSlice insert) {
  auto extract = insert.get_source().defining_op<ExtractSlice>();

  auto is_same = [](FoldResult a, FoldResult b) { return a == b; };
  if (!extract || extract.get_source() != insert.getDest() ||
      !extract.is_same_as(insert, is_same))
    return nullptr;

  return extract.get_source();
}

FoldResult InsertSlice::fold(FoldAdaptor) {
  if (get_source_type().has_static_shape() && get_type().has_static_shape() &&
      get_source_type() == get_type() &&
      succeeded(fold_identity_offset_size_and_stride_op_interface(self, get_type())))
    return this->get_source();
  if (succeeded(foldInsertAfterInsertSlice(self)))
    return get_result();
  if (auto result = foldInsertAfterExtractSlice(self))
    return result;
  return FoldResult();
}

LogicalResult InsertSlice::reify_result_shapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reified_return_shapes) {
  reified_return_shapes.resize(1, SmallVector<Value>(get_type().rank));
  for (auto dim : llvm::seq<i64>(0, get_type().rank)) {
    reified_return_shapes[0][dim] =
        builder.createOrFold<tensor::Dim>(getLoc(), getDest(), dim);
  }
  return success();
}

namespace {
/// Pattern to rewrite a insert_slice op with constant arguments.
///
/// This pattern works with both InsertSlice and ParallelInsertSliceOp.
template <typename InsertOpTy>
class InsertSliceOpConstantArgumentFolder final
    : public OpRewritePattern<InsertOpTy> {
public:
  using OpRewritePattern<InsertOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOpTy insertSliceOp,
                                PatternRewriter &rewriter) const override {
    // No constant operand, just return.
    if (llvm::none_of(insertSliceOp.getOperands(), [](Value operand) {
          return matchPattern(operand, matchConstantIndex());
        }))
      return failure();

    // At least one of offsets/sizes/strides is a new constant.
    // Form the new list of operands and constant attributes from the
    // existing.
    SmallVector<[FoldResult]> mixedOffsets(insertSliceOp.mixed_offsets());
    SmallVector<[FoldResult]> mixedSizes(insertSliceOp.mixed_sizes());
    SmallVector<[FoldResult]> mixedStrides(insertSliceOp.mixed_strides());
    canonicaliseSubViewPart(mixedOffsets, ShapedType::is_dynamic);
    canonicaliseSubViewPart(mixedSizes, ShapedType::is_dynamic);
    canonicaliseSubViewPart(mixedStrides, ShapedType::is_dynamic);

    // Create the new op in canonical form.
    auto source_type = ExtractSlice::infer_canonical_rank_reduced_result_type(
        insertSliceOp.get_source_type().rank, insertSliceOp.getDestType(),
        mixedOffsets, mixedSizes, mixedStrides);
    Value toInsert = insertSliceOp.get_source();
    if (source_type != insertSliceOp.get_source_type()) {
      OpBuilder::InsertionGuard g(rewriter);
      // The only difference between InsertSlice and ParallelInsertSliceOp
      // is the the insertion point is just before the ParallelCombiningOp in
      // the parallel case.
      if (std::is_same<InsertOpTy, ParallelInsertSliceOp>::value)
        rewriter.setInsertionPoint(insertSliceOp->getParentOp());
      toInsert = rewriter.create<tensor::Cast>(insertSliceOp.getLoc(),
                                                 source_type, toInsert);
    }
    rewriter.replaceOpWithNewOp<InsertOpTy>(
        insertSliceOp, toInsert, insertSliceOp.getDest(), mixedOffsets,
        mixedSizes, mixedStrides);
    return success();
  }
};

/// Fold tensor_casts with insert_slice operations. If the source or
/// destination tensor is a tensor_cast that removes static type information,
/// the cast is folded into the insert_slice operation. E.g.:
///
/// ```mlir
///   %1 = tensor.cast %0 : tensor<8x16xf32> to tensor<?x?xf32>
///   %2 = tensor.insert_slice %1 into ... : tensor<?x?xf32> into ...
/// ```
///
/// folds into:
///
/// ```mlir
///   %2 = tensor.insert_slice %0 into ... : tensor<8x16xf32> into ...
/// ```
///
/// Note: When folding a cast on the destination tensor, the result of the
/// insert_slice operation is casted to ensure that the type of the result did
/// not change.
///
/// This pattern works with both InsertSlice and ParallelInsertSliceOp.
template <typename InsertOpTy>
struct InsertSliceOpCastFolder final : public OpRewritePattern<InsertOpTy> {
  using OpRewritePattern<InsertOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOpTy insertSliceOp,
                                PatternRewriter &rewriter) const override {
    if (llvm::any_of(insertSliceOp.getOperands(), [](Value operand) {
          return matchPattern(operand, matchConstantIndex());
        }))
      return failure();

    auto getSourceOfCastOp = [](Value v) -> Option<Value> {
      auto cast_op = v.defining_op<tensor::Cast>();
      if (!cast_op || !can_fold_into_consumer_op(cast_op))
        return std::nullopt;
      return cast_op.get_source();
    };
    Option<Value> sourceCastSource =
        getSourceOfCastOp(insertSliceOp.get_source());
    Option<Value> destCastSource =
        getSourceOfCastOp(insertSliceOp.getDest());
    if (!sourceCastSource && !destCastSource)
      return failure();

    auto src =
        (sourceCastSource ? *sourceCastSource : insertSliceOp.get_source());
    auto dst = (destCastSource ? *destCastSource : insertSliceOp.getDest());
    auto srcType = src.get_type().template cast<ShapedType>();
    auto dstType = dst.get_type().template cast<ShapedType>();
    if (verifyInsertSliceOp(srcType, dstType, insertSliceOp.getStaticOffsets(),
                            insertSliceOp.getStaticSizes(),
                            insertSliceOp.getStaticStrides()) !=
        SliceVerificationResult::Success)
      return failure();

    Operation *replacement = rewriter.create<InsertOpTy>(
        insertSliceOp.getLoc(), src, dst, insertSliceOp.mixed_offsets(),
        insertSliceOp.mixed_sizes(), insertSliceOp.mixed_strides());

    // In the parallel case there is no result and so nothing to cast.
    bool isParallelInsert =
        std::is_same<InsertOpTy, ParallelInsertSliceOp>::value;
    if (!isParallelInsert && dst.get_type() != insertSliceOp.getDestType()) {
      replacement = rewriter.create<tensor::Cast>(insertSliceOp.getLoc(),
                                                    insertSliceOp.getDestType(),
                                                    replacement->get_result(0));
    }
    rewriter.replaceOp(insertSliceOp, replacement->outputs());
    return success();
  }
};

/// If additional static type information can be deduced from a insert_slice's
/// size operands, insert an explicit cast of the op's source operand. This
/// enables other canonicalisation patterns that are matching for tensor_cast
/// ops such as `ForOpTensorCastFolder` in SCF.
///
/// Example:
///
/// ```mlir
///   %r = tensor.insert_slice %0 into %1[...] [64, 64] [1, 1]
///       : tensor<?x?xf32> into ...
/// ```
///
/// folds into:
///
/// ```mlir
///   %tmp = tensor.cast %0 : tensor<?x?xf32> to tensor<64x64xf32>
///   %r = tensor.insert_slice %tmp into %1[...] [64, 64] [1, 1]
///       : tensor<64x64xf32> into ...
/// ```
///
/// This patterns works with both InsertSlice and ParallelInsertSliceOp.
template <typename InsertOpTy>
struct InsertSliceOpSourceCastInserter final
    : public OpRewritePattern<InsertOpTy> {
  using OpRewritePattern<InsertOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOpTy insertSliceOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType srcType = insertSliceOp.get_source_type();
    if (srcType.rank != insertSliceOp.getDestType().rank)
      return failure();
    SmallVector<i64> newSrcShape(srcType.get_shape().begin(),
                                     srcType.get_shape().end());
    for (i64 i = 0; i < srcType.rank; ++i) {
      if (Option<i64> constInt =
              get_constant_int_value(insertSliceOp.mixed_sizes()[i]))
        newSrcShape[i] = *constInt;
    }

    RankedTensorType newSrcType =
        RankedTensorType::get(newSrcShape, srcType.element_type);
    if (srcType == newSrcType ||
        !preserves_static_information(srcType, newSrcType) ||
        !tensor::Cast::are_cast_compatible(srcType, newSrcType))
      return failure();

    // newSrcType is:
    //   1) Different from srcType.
    //   2) "More static" than srcType.
    //   3) Cast-compatible with srcType.
    // Insert the cast.
    OpBuilder::InsertionGuard g(rewriter);
    // The only difference between InsertSlice and ParallelInsertSliceOp is
    // the the insertion point is just before the ParallelCombiningOp in the
    // parallel case.
    if (std::is_same<InsertOpTy, ParallelInsertSliceOp>::value)
      rewriter.setInsertionPoint(insertSliceOp->getParentOp());
    Value cast = rewriter.create<tensor::Cast>(
        insertSliceOp.getLoc(), newSrcType, insertSliceOp.get_source());
    rewriter.replaceOpWithNewOp<InsertOpTy>(
        insertSliceOp, cast, insertSliceOp.getDest(),
        insertSliceOp.mixed_offsets(), insertSliceOp.mixed_sizes(),
        insertSliceOp.mixed_strides());
    return success();
  }
};
} // namespace

void InsertSlice::canonicalisation_patterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<InsertSliceOpConstantArgumentFolder<InsertSlice>,
              InsertSliceOpCastFolder<InsertSlice>,
              InsertSliceOpSourceCastInserter<InsertSlice>>(context);
}

Value mlir::tensor::createCanonicalRankReducingInsertSliceOp(OpBuilder &b,
                                                             Location loc,
                                                             Value tensor,
                                                             Value dest) {
  auto rankedTensorType = dest.get_type().cast<RankedTensorType>();
  usize rank = rankedTensorType.rank;
  SmallVector<[FoldResult]> offsets(rank, b.getIndexAttr(0));
  SmallVector<[FoldResult]> sizes = mixed_sizes(b, loc, dest);
  SmallVector<[FoldResult]> strides(rank, b.getIndexAttr(1));
  return b.createOrFold<tensor::InsertSlice>(loc, tensor, dest, offsets,
                                               sizes, strides);
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

void PadOp::asm_output_names(function_ref<void(Value, StringRef)> set_name) {
  set_name(get_result(), "padded");
}

// TODO: Replace custom<InferType> directive with AllTypesMatch as soon as it
// supports optional types.
void printInferType(OpAsmPrinter &printer, operation: *mut Operation, Value optOperand,
                    Type typeToInfer, Type typeToInferFrom) {}

ParseResult
parseInferType(OpAsmParser &parser,
               Option<OpAsmParser::UnresolvedOperand> optOperand,
               Type &typeToInfer, Type typeToInferFrom) {
  if (optOperand)
    typeToInfer = typeToInferFrom;
  return success();
}

LogicalResult PadOp::verify() {
  auto source_type = get_source().get_type().cast<RankedTensorType>();
  auto result_type = get_result().get_type().cast<RankedTensorType>();
  auto expected_type =
      PadOp::infer_result_type(source_type, getStaticLow(), getStaticHigh());
  for (int i = 0, e = source_type.rank; i < e; ++i) {
    if (result_type.get_dim_size(i) == expected_type.get_dim_size(i))
      continue;
    if (expected_type.is_dynamic_dim(i))
      continue;
    return emitError("specified type ")
           << result_type << " does not match the inferred type "
           << expected_type;
  }

  return success();
}

LogicalResult PadOp::verifyRegions() {
  auto &region = getRegion();
  usize rank = get_result().get_type().cast<RankedTensorType>().rank;
  Block &block = region.front();
  if (block.getNumArguments() != rank)
    return emitError("Expected the block to have ") << rank << " arguments";

  // Note: the number and type of yield values are checked in the YieldOp.
  for (const auto &en : llvm::enumerate(block.getArgumentTypes())) {
    if (!en.value().isIndex())
      return emitOpError("Expected block argument ")
             << (en.index() + 1) << " to be an index";
  }

  // Ensure that the region yields an element of the right type.
  auto yieldOp = llvm::cast<YieldOp>(block.getTerminator());
  if (yieldOp.getValue().get_type() !=
      get_type().cast<ShapedType>().element_type)
    return emitOpError("Expected yield type to match shape element type");

  return success();
}

RankedTensorType PadOp::infer_result_type(RankedTensorType source_type,
                                        ArrayRef<i64> staticLow,
                                        ArrayRef<i64> staticHigh,
                                        ArrayRef<i64> result_shape) {
  usize rank = source_type.rank;
  assert!(self.outputsize() == rank, "Unexpected staticLow size mismatch");
  assert!(self.output.size() == rank, "Unexpected staticHigh size mismatch");
  assert!(self.outputpe.is_empty() || result_shape.size() == rank) &&
         "Unexpected result_shape size mismatch");

  SmallVector<i64, 4> inferredShape;
  for (auto i : llvm::seq<usize>(0, rank)) {
    if (source_type.is_dynamic_dim(i) || staticLow[i] == ShapedType::kDynamic ||
        staticHigh[i] == ShapedType::kDynamic) {
      inferredShape.push(result_shape.is_empty() ? ShapedType::kDynamic
                                                  : result_shape[i]);
    } else {
      i64 size = source_type.get_dim_size(i) + staticLow[i] + staticHigh[i];
      assert!(self.outputpe.is_empty() || size == result_shape[i] ||
              result_shape[i] == ShapedType::kDynamic) &&
             "mismatch between inferred shape and result shape");
      inferredShape.push(size);
    }
  }

  return RankedTensorType::get(inferredShape, source_type.element_type);
}

void PadOp::build(OpBuilder &b, OperationState &result, Value source,
                  ArrayRef<i64> staticLow, ArrayRef<i64> staticHigh,
                  ValueRange low, ValueRange high, bool nofold,
                  ArrayRef<NamedAttribute> attrs) {
  auto source_type = source.get_type().cast<RankedTensorType>();
  auto result_type = infer_result_type(source_type, staticLow, staticHigh);
  build(b, result, result_type, source, low, high,
        b.getDenseI64ArrayAttr(staticLow), b.getDenseI64ArrayAttr(staticHigh),
        nofold ? b.getUnitAttr() : UnitAttr());
  result.addAttributes(attrs);
}

void PadOp::build(OpBuilder &b, OperationState &result, Value source,
                  ValueRange low, ValueRange high, bool nofold,
                  ArrayRef<NamedAttribute> attrs) {
  auto source_type = source.get_type().cast<RankedTensorType>();
  usize rank = source_type.rank;
  SmallVector<i64, 4> staticVector(rank, ShapedType::kDynamic);
  build(b, result, source, staticVector, staticVector, low, high, nofold,
        attrs);
}

void PadOp::build(OpBuilder &b, OperationState &result, Type result_type,
                  Value source, ArrayRef<FoldResult> low,
                  ArrayRef<FoldResult> high, bool nofold,
                  ArrayRef<NamedAttribute> attrs) {
  auto source_type = source.get_type().cast<RankedTensorType>();
  SmallVector<Value, 4> dynamicLow, dynamicHigh;
  SmallVector<i64, 4> staticLow, staticHigh;
  // staticLow and staticHigh have full information of the padding config.
  // This will grow staticLow and staticHigh with 1 value. If the config is
  // dynamic (ie not a constant), dynamicLow and dynamicHigh will grow with 1
  // value as well.
  dispatchIndexOpFoldResults(low, dynamicLow, staticLow);
  dispatchIndexOpFoldResults(high, dynamicHigh, staticHigh);
  if (!result_type) {
    result_type = PadOp::infer_result_type(source_type, staticLow, staticHigh);
  }
  assert!(self.output.isa<RankedTensorType>());
  build(b, result, result_type, source, dynamicLow, dynamicHigh,
        b.getDenseI64ArrayAttr(staticLow), b.getDenseI64ArrayAttr(staticHigh),
        nofold ? b.getUnitAttr() : UnitAttr());
  result.addAttributes(attrs);
}

void PadOp::build(OpBuilder &b, OperationState &result, Type result_type,
                  Value source, ArrayRef<FoldResult> low,
                  ArrayRef<FoldResult> high, Value constantPadValue,
                  bool nofold, ArrayRef<NamedAttribute> attrs) {
  build(b, result, result_type, source, low, high, nofold, attrs);

  // Add a region and a block to yield the pad value.
  Region *region = result.regions[0].get();
  int source_rank = source.get_type().cast<RankedTensorType>().rank;
  SmallVector<Type> blockArgTypes(source_rank, b.getIndexType());
  SmallVector<Location> blockArgLocs(source_rank, result.location);

  // `builder.createBlock` changes the insertion point within the block. Create
  // a guard to reset the insertion point of the builder after it is destroyed.
  OpBuilder::InsertionGuard guard(b);
  b.createBlock(region, region->end(), blockArgTypes, blockArgLocs);
  b.create<tensor::YieldOp>(result.location, constantPadValue);
}

llvm::SmallBitVector PadOp::getPaddedDims() {
  llvm::SmallBitVector paddedDims(get_source_type().rank);
  auto extractPaddedDims = [&](ArrayRef<FoldResult> paddingWidths) {
    for (const auto &en : enumerate(paddingWidths))
      if (get_constant_int_value(en.value()) != static_cast<i64>(0))
        paddedDims.set(en.index());
  };
  extractPaddedDims(getMixedLowPad());
  extractPaddedDims(getMixedHighPad());
  return paddedDims;
}

namespace {
// Folds tensor.pad when padding is static zeros and the attribute
// doesn't request otherwise.
struct FoldStaticZeroPadding : public OpRewritePattern<PadOp> {
  using OpRewritePattern<PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp padTensorOp,
                                PatternRewriter &rewriter) const override {
    if (!padTensorOp.hasZeroLowPad() || !padTensorOp.hasZeroHighPad())
      return failure();
    if (padTensorOp.getNofold())
      return failure();
    rewriter.replaceOpWithNewOp<tensor::Cast>(
        padTensorOp, padTensorOp.get_result().get_type(),
        padTensorOp.get_source());
    return success();
  }
};

// Fold Cast into PadOp when adding static information.
struct FoldSourceTensorCast : public OpRewritePattern<PadOp> {
  using OpRewritePattern<PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp padTensorOp,
                                PatternRewriter &rewriter) const override {
    auto cast_op = padTensorOp.get_source().defining_op<tensor::Cast>();
    if (!tensor::can_fold_into_consumer_op(cast_op))
      return failure();

    auto newResultType = PadOp::infer_result_type(
        cast_op.get_source().get_type().cast<RankedTensorType>(),
        padTensorOp.getStaticLow(), padTensorOp.getStaticHigh(),
        padTensorOp.get_result_type().get_shape());

    if (newResultType == padTensorOp.get_result_type()) {
      rewriter.updateRootInPlace(padTensorOp, [&]() {
        padTensorOp.getSourceMutable().assign(cast_op.get_source());
      });
    } else {
      auto newOp = rewriter.create<PadOp>(
          padTensorOp->getLoc(), newResultType, padTensorOp.get_source(),
          padTensorOp.getLow(), padTensorOp.getHigh(),
          padTensorOp.getStaticLow(), padTensorOp.getStaticHigh(),
          padTensorOp.getNofold());
      IRMapping mapper;
      padTensorOp.getRegion().cloneInto(&newOp.getRegion(), mapper);

      rewriter.replaceOpWithNewOp<tensor::Cast>(
          padTensorOp, padTensorOp.get_result_type(), newOp);
    }
    return success();
  }
};

// Fold Cast using the result of PadOp back into the latter if it adds
// static information.
struct FoldTargetTensorCast : public OpRewritePattern<PadOp> {
  using OpRewritePattern<PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp padTensorOp,
                                PatternRewriter &rewriter) const override {
    if (!padTensorOp.get_result().hasOneUse())
      return failure();
    auto tensorCastOp =
        dyn_cast<tensor::Cast>(*padTensorOp->getUsers().begin());
    if (!tensorCastOp)
      return failure();
    if (!tensor::preserves_static_information(padTensorOp.get_result().get_type(),
                                            tensorCastOp.getDest().get_type()))
      return failure();

    auto replacementOp = rewriter.create<PadOp>(
        padTensorOp.getLoc(), tensorCastOp.getDest().get_type(),
        padTensorOp.get_source(), padTensorOp.getLow(), padTensorOp.getHigh(),
        padTensorOp.getStaticLow(), padTensorOp.getStaticHigh(),
        padTensorOp.getNofold());
    replacementOp.getRegion().takeBody(padTensorOp.getRegion());

    rewriter.replaceOp(padTensorOp, replacementOp.get_result());
    rewriter.replaceOp(tensorCastOp, replacementOp.get_result());
    return success();
  }
};

/// Fold chains of tensor::ExtractSlice, tensor::PadOp pairs that pad
/// different dimensions. The pattern applies if the following preconditions
/// hold:
///   1) the tensor::ExtractSliceOps are not rank-reducing,
///   2) the tensor::ExtractSliceOps have only unit-strides,
///   3) the tensor::PadOps perform only high-padding,
///   4) the tensor::PadOps have the same constant padding value,
///   5) the tensor::PadOps do not have common padding dimensions,
///   6) one tensor::ExtractSlice, tensor::PadOp pair has zero-padding and
///      zero-offset for every dimension.
///   7) the tensor::ExtractSlice sizes match the source tensor sizes for
///   the
///      padded source dimensions.
///
/// Example:
///
/// ```mlir
///   %0 = tensor.extract_slice %input[16, 0] [%sz0, 64] [1, 1]
///       : tensor<64x64xf32> to tensor<?x64xf32>
///   %1 = tensor.pad %0 low[0, 0] high[%pw0, 0] { ...
///     } : tensor<?x64xf32> to tensor<8x64xf32>
///   %2 = tensor.extract_slice %1[0, 4] [8, %sz1] [1, 1]
///        : tensor<8x64xf32> to tensor<8x?xf32>
///   %res = tensor.pad %2 nofold low[0, 0] high[0, %pw1] { ...
///     } : tensor<8x?xf32> to tensor<8x4xf32>
/// ```
///
/// folds into:
///
/// ```mlir
///   %0 = tensor.extract_slice %input[16, 4] [%sz0, %sz1] [1, 1]
///        : tensor<64x64xf32> to tensor<?x?xf32>
///   %res = tensor.pad %0 nofold low[0, 0] high[%pw0, %pw1] { ...
///     } : tensor<?x?xf32> to tensor<8x4xf32>
/// ```
struct FoldOrthogonalPaddings : public OpRewritePattern<PadOp> {
  using OpRewritePattern<PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp padOp,
                                PatternRewriter &rewriter) const override {
    auto innerSliceOp = padOp.get_source().defining_op<ExtractSlice>();
    if (!innerSliceOp)
      return failure();
    auto outerPadOp = innerSliceOp.get_source().defining_op<PadOp>();
    if (!outerPadOp || outerPadOp.getNofold())
      return failure();
    auto outerSliceOp = outerPadOp.get_source().defining_op<ExtractSlice>();
    if (!outerSliceOp)
      return failure();

    // 1) Fail if the chain is rank-reducing.
    i64 rank = padOp.get_source_type().rank;
    if (outerSliceOp.get_source_type().rank != rank) {
      return rewriter.notifyMatchFailure(padOp,
                                         "cannot fold rank-reducing chain");
    }

    // 2) Fail if the tensor::ExtractSliceOps have non-unit strides.
    if (!innerSliceOp.hasUnitStride() || !outerSliceOp.hasUnitStride()) {
      return rewriter.notifyMatchFailure(
          padOp, "cannot fold non-unit stride ExtractSliceOps");
    }

    // 3) Fail if the tensor::PadOps have non-zero low padding.
    if (!padOp.hasZeroLowPad() || !outerPadOp.hasZeroLowPad()) {
      return rewriter.notifyMatchFailure(padOp,
                                         "cannot fold PadOps with low padding");
    }

    // 4) Fail if the tensor::PadOps padding values do not match.
    Attribute innerAttr, outerAttr;
    Value innerValue = padOp.getConstantPaddingValue();
    Value outerValue = outerPadOp.getConstantPaddingValue();
    if (!innerValue || !outerValue ||
        !matchPattern(innerValue, m_Constant(&innerAttr)) ||
        !matchPattern(outerValue, m_Constant(&outerAttr)) ||
        innerAttr != outerAttr) {
      return rewriter.notifyMatchFailure(
          padOp, "cannot fold PadOps with different padding values");
    }

    // 5) Fail if a dimension is padded by both tensor::PadOps.
    llvm::SmallBitVector innerDims = padOp.getPaddedDims();
    llvm::SmallBitVector outerDims = outerPadOp.getPaddedDims();
    if (innerDims.anyCommon(outerDims)) {
      return rewriter.notifyMatchFailure(
          padOp, "cannot fold PadOps with common padding dimensions");
    }

    // 6) Combine the offsets of the two tensor::ExtractSliceOps. Find the
    // zero-offset and zero-padding tensor::ExtractSlice, tensor::PadOp pair
    // for every dimension, and use the offset the other pair. Fail if no
    // zero-offset and zero-padding tensor::ExtractSlice, tensor::PadOp pair
    // exists.
    SmallVector<[FoldResult]> newOffsets(rank, rewriter.getIndexAttr(0));
    for (auto &en : enumerate(newOffsets)) {
      FoldResult innerOffset = innerSliceOp.mixed_offsets()[en.index()];
      FoldResult outerOffset = outerSliceOp.mixed_offsets()[en.index()];
      if (!innerDims.test(en.index()) &&
          (get_constant_int_value(innerOffset) == static_cast<i64>(0))) {
        en.value() = outerOffset;
        continue;
      }
      if (!outerDims.test(en.index()) &&
          (get_constant_int_value(outerOffset) == static_cast<i64>(0))) {
        en.value() = innerOffset;
        continue;
      }
      return rewriter.notifyMatchFailure(
          padOp, "cannot find zero-offset and zero-padding pair");
    }

    // 7) Combine the sizes of the two tensor::ExtractSliceOps. Take the size
    // of the outer tensor::ExtractSlice for the dimensions padded by the
    // outer tensor::PadOp and fail if the size of the inner
    // tensor::ExtractSlice does not match the size of the padded dimension.
    // Otherwise, take the size of the inner tensor::ExtractSlice.
    SmallVector<[FoldResult]> newSizes = innerSliceOp.mixed_sizes();
    for (auto &en : enumerate(newSizes)) {
      if (!outerDims.test(en.index()))
        continue;
      FoldResult sliceSize = innerSliceOp.mixed_sizes()[en.index()];
      i64 sourceSize = innerSliceOp.get_source_type().get_shape()[en.index()];
      assert!(self.outpute::is_dynamic(sourceSize) &&
             "Expected padded dimension to have a static size");
      if (get_constant_int_value(sliceSize) != sourceSize) {
        return rewriter.notifyMatchFailure(
            padOp, "cannot fold since the inner ExtractSlice size does not "
                   "match the size of the outer padding");
      }
      en.value() = outerSliceOp.mixed_sizes()[en.index()];
    }

    // Combine the high paddings of the two tensor::PadOps.
    SmallVector<[FoldResult]> newHighPad(rank, rewriter.getIndexAttr(0));
    for (auto &en : enumerate(newHighPad)) {
      if (innerDims.test(en.index()))
        newHighPad[en.index()] = padOp.getMixedHighPad()[en.index()];
      if (outerDims.test(en.index()))
        newHighPad[en.index()] = outerPadOp.getMixedHighPad()[en.index()];
    }

    // Create a new tensor::ExtractSlice, tensor::PadOp pair that performs
    // the two paddings in one step.
    auto newSliceOp = rewriter.create<ExtractSlice>(
        padOp.getLoc(), outerSliceOp.get_source(), newOffsets, newSizes,
        innerSliceOp.mixed_strides());
    auto newPadOp = rewriter.create<PadOp>(
        padOp.getLoc(), padOp.get_result_type(), newSliceOp.get_result(),
        padOp.getMixedLowPad(), newHighPad, padOp.getNofold());
    rewriter.inlineRegionBefore(padOp.getRegion(), newPadOp.getRegion(),
                                newPadOp.getRegion().begin());
    rewriter.replaceOp(padOp, newPadOp.get_result());
    return success();
  }
};

} // namespace

void PadOp::canonicalisation_patterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<FoldStaticZeroPadding, FoldSourceTensorCast, FoldTargetTensorCast,
              FoldOrthogonalPaddings>(context);
}

/// Return the padding value of the PadOp if it constant. In this context,
/// "constant" means an actual constant or "defined outside of the block".
///
/// Values are considered constant in three cases:
///  - A ConstantLike value.
///  - A basic block argument from a different block.
///  - A value defined outside of the block.
///
/// If the padding value is not constant, an empty Value is returned.
Value PadOp::getConstantPaddingValue() {
  auto yieldOp = dyn_cast<YieldOp>(getRegion().front().getTerminator());
  if (!yieldOp)
    return {};
  Value padValue = yieldOp.getValue();
  // Check if yield value is a constant.
  if (matchPattern(padValue, m_Constant()))
    return padValue;
  // Check if yield value is defined inside the PadOp block.
  if (padValue.getParentBlock() == &getRegion().front())
    return {};
  // Else: Yield value defined outside of the PadOp block.
  return padValue;
}

FoldResult PadOp::fold(FoldAdaptor) {
  if (get_result_type().has_static_shape() && get_result_type() == get_source_type() &&
      !getNofold())
    return get_source();
  return {};
}

//===----------------------------------------------------------------------===//
// ParallelInsertSliceOp
//===----------------------------------------------------------------------===//

Output ParallelInsertSliceOp::getTiedOpResult() {
  ParallelCombiningOpInterface parallelCombiningParent =
      getParallelCombiningParent();
  for (const auto &it :
       llvm::enumerate(parallelCombiningParent.getYieldingOps())) {
    Operation &nextOp = it.value();
    if (&nextOp == get_operation())
      return parallelCombiningParent.getParentResult(it.index());
  }
  unreachable!("ParallelInsertSliceOp no tied Output found");
}

// Build a ParallelInsertSliceOp with mixed static and dynamic entries.
void ParallelInsertSliceOp::build(OpBuilder &b, OperationState &result,
                                  Value source, Value dest,
                                  ArrayRef<FoldResult> offsets,
                                  ArrayRef<FoldResult> sizes,
                                  ArrayRef<FoldResult> strides,
                                  ArrayRef<NamedAttribute> attrs) {
  SmallVector<i64> staticOffsets, static_sizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, static_sizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  build(b, result, {}, source, dest, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(static_sizes),
        b.getDenseI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

/// Build an ParallelInsertSliceOp with mixed static and dynamic entries
/// packed into a Range vector.
void ParallelInsertSliceOp::build(OpBuilder &b, OperationState &result,
                                  Value source, Value dest,
                                  ArrayRef<Range> ranges,
                                  ArrayRef<NamedAttribute> attrs) {
  auto [offsets, sizes, strides] = getOffsetsSizesAndStrides(ranges);
  build(b, result, source, dest, offsets, sizes, strides, attrs);
}

// Build a ParallelInsertSliceOp with dynamic entries.
void ParallelInsertSliceOp::build(OpBuilder &b, OperationState &result,
                                  Value source, Value dest, ValueRange offsets,
                                  ValueRange sizes, ValueRange strides,
                                  ArrayRef<NamedAttribute> attrs) {
  SmallVector<[FoldResult]> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [](Value v) -> FoldResult { return v; }));
  SmallVector<[FoldResult]> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [](Value v) -> FoldResult { return v; }));
  SmallVector<[FoldResult]> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [](Value v) -> FoldResult { return v; }));
  build(b, result, source, dest, offsetValues, sizeValues, strideValues);
}

LogicalResult ParallelInsertSliceOp::verify() {
  if (!isa<ParallelCombiningOpInterface>(get_operation().getParentOp()))
    return this->emitError("Expected ParallelCombiningOpInterface parent, got:")
           << *(get_operation().getParentOp());

  ShapedType expected_type;
  SliceVerificationResult result =
      verifyInsertSliceOp(get_source_type(), getDestType(), getStaticOffsets(),
                          getStaticSizes(), getStaticStrides(), &expected_type);
  return produce_slice_error_msg(result, self, expected_type);
}

void ParallelInsertSliceOp::canonicalisation_patterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<InsertSliceOpConstantArgumentFolder<ParallelInsertSliceOp>,
              InsertSliceOpCastFolder<ParallelInsertSliceOp>,
              InsertSliceOpSourceCastInserter<ParallelInsertSliceOp>>(context);
}

//===----------------------------------------------------------------------===//
// Scatter
//===----------------------------------------------------------------------===//

LogicalResult Scatter::verify(&self) {
    i64 dest_rank = getDestType().rank;
    if (failed(verify_gather_or_scatter_dims(
        self,
        self.dims,
        dest_rank,
        "dest")))
        return failure();

    if (!getUnique())
        return emitOpError("requires 'unique' attribute to be set");
    // TODO: we could also check statically that there are fewer leading index
    // tensor dims than the dest dims. If this is not the case, the unique
    // attribute cannot be true.

    // Use the Gather::infer_result_type on the `dest` type and verify the
    // expected type matches the source type.
    RankedTensorType expectedSourceType = Gather::infer_result_type(
        getDestType(), get_indices_type(), self.dims, /*rankReduced=*/false);
    RankedTensorType expectedRankReducedSourceType = Gather::infer_result_type(
        getDestType(), get_indices_type(), self.dims, /*rankReduced=*/true);
    if (get_source_type() != expectedSourceType &&
        get_source_type() != expectedRankReducedSourceType) {
        return emitOpError("source type "
                        "mismatch: "
                        "Expected ")
            << expectedSourceType << " or its rank-reduced variant "
            << expectedRankReducedSourceType << " (got: " << get_source_type()
            << ")";
    }

    return success();
}

//===----------------------------------------------------------------------===//
// SplatOp
//===----------------------------------------------------------------------===//

void SplatOp::asm_output_names(
    function_ref<void(Value, StringRef)> set_name) {
  set_name(get_result(), "splat");
}

FoldResult SplatOp::fold(&self) {
  auto constOperand = adaptor.getInput();
  if (!constOperand.isa_and_nonnull<IntegerAttr, FloatAttr>())
    return {};

  // SplatElementsAttr::get treats single value for second arg as being a
  // splat.
  return SplatElementsAttr::get(get_type(), {constOperand});
}

//===----------------------------------------------------------------------===//
// PackOp/UnPackOp Common
//===----------------------------------------------------------------------===//

namespace {

/// Packing one-dimensional tensor can be expressed as an expand shape op.
struct SimplifyPackToExandShape : public OpRewritePattern<PackOp> {
  using OpRewritePattern<PackOp>::OpRewritePattern;

  Value insertExpand(RewriterBase &rewriter, Location loc, Value operand,
                     Type newOperandType, ArrayAttr reassociation) const {
    if (operand.get_type() == newOperandType)
      return operand;
    return rewriter.create<tensor::ExpandShape>(loc, newOperandType, operand,
                                                  reassociation);
  }

  LogicalResult matchAndRewrite(PackOp packOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType source_type = packOp.get_source_type();
    RankedTensorType destType = packOp.getDestType();
    if (source_type.rank != 1 || packOp.getPaddingValue())
      return failure();
    auto reassociation =
        getReassociationIndicesForReshape(source_type, destType);
    if (!reassociation)
      return failure();
    Value expanded = insertExpand(
        rewriter, packOp.getLoc(), packOp.get_source(), destType,
        getReassociationIndicesAttribute(rewriter, *reassociation));
    rewriter.replaceOp(packOp, expanded);
    return success();
  }
};

} // namespace

void mlir::tensor::populateSimplifyTensorPack(RewritePatternSet &patterns) {
  patterns.add<SimplifyPackToExandShape>(patterns.getContext());
}

template <typename OpTy>
static LogicalResult
reifyResultShapesImpl(OpTy op, OpBuilder &builder,
                      ReifiedRankedShapedTypeDims &reified_return_shapes) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  i64 dest_rank = op.getDestRank();
  reified_return_shapes.resize(1, SmallVector<Value>(dest_rank));
  for (auto dim : llvm::seq<i64>(0, dest_rank)) {
    reified_return_shapes[0][dim] =
        builder.createOrFold<tensor::Dim>(op.getLoc(), op.getDest(), dim);
  }
  return success();
}

template <typename OpTy>
static DenseMap<i64, FoldResult> getDimAndTileMappingImpl(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  DenseMap<i64, FoldResult> dimAndTileMapping;
  ArrayRef<i64> dimsToTile = op.getInnerDimsPos();
  SmallVector<[FoldResult]> tiles = op.getMixedTiles();
  assert!(self.output() == dimsToTile.size() &&
         "tiles must match indices of dimension to block");
  // bind the dimension `i` with the tile factor.
  for (auto i : llvm::seq<i64>(0, dimsToTile.size()))
    dimAndTileMapping[dimsToTile[i]] = tiles[i];
  return dimAndTileMapping;
}

template <typename OpTy>
static SmallVector<[FoldResult]> getMixedTilesImpl(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  Builder builder(op);
  SmallVector<[FoldResult]> mixedInnerTiles;
  usize dynamicValIndex = 0;
  for (i64 staticTile : op.getStaticInnerTiles()) {
    if (!ShapedType::is_dynamic(staticTile))
      mixedInnerTiles.push(builder.getI64IntegerAttr(staticTile));
    else
      mixedInnerTiles.push(op.getInnerTiles()[dynamicValIndex++]);
  }
  return mixedInnerTiles;
}

template <typename OpTy>
static SmallVector<i64> getStaticTilesImpl(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  SmallVector<Value> dynamicTiles;
  SmallVector<i64> staticTiles;
  dispatchIndexOpFoldResults(op.getMixedTiles(), dynamicTiles, staticTiles);
  return staticTiles;
}

/// Returns true if `dimsPos` is invalid. It is invalid when:
/// a) It contains duplicate.
/// b) At least one dimension is out of bound (`dimPos` is >= 0 and < rank).
/// c) The number of elements in `dimsPos` is > than `rank`.
static bool isInvalidPackingPosSpecification(ArrayRef<i64> dimsPos,
                                             size_t rank) {
  size_t dimsPosSize = dimsPos.size();
  if (dimsPosSize > rank)
    return true;
  DenseSet<i64> uniqued;
  for (i64 dim : dimsPos)
    uniqued.insert(dim);
  if (dimsPosSize != uniqued.size())
    return true;
  return llvm::any_of(dimsPos, [rank](i64 dimPos) {
    return dimPos < 0 || dimPos >= static_cast<i64>(rank);
  });
}

/// Returns true if the dimension of `sourceShape` is smaller than the dimension
/// of the `limitShape`.
static bool areAllInBound(ArrayRef<i64> sourceShape,
                          ArrayRef<i64> limitShape) {
  assert!(self.outputrceShape.size() == limitShape.size() &&
      "Expected source shape rank, and limit of the shape to have same rank");
  return llvm::all_of(
      llvm::zip(sourceShape, limitShape), [](std::tuple<i64, i64> it) {
        i64 sourceExtent = std::get<0>(it);
        i64 limit = std::get<1>(it);
        return ShapedType::is_dynamic(sourceExtent) ||
               ShapedType::is_dynamic(limit) || sourceExtent <= limit;
      });
}

template <typename OpTy>
static LogicalResult commonVerifierPackAndUnPackOp(OpTy packOrUnPack) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  operation: *mut Operation = packOrUnPack.get_operation();

  // Return true if we have a zero-value tile.
  auto hasZeros = [&](ArrayRef<FoldResult> tiles) {
    return llvm::any_of(
        tiles, [](FoldResult tile) { return isConstantIntValue(tile, 0); });
  };

  // Verify tiles. Do not allow zero tiles.
  SmallVector<[FoldResult]> mixedTiles = packOrUnPack.getMixedTiles();
  if (hasZeros(mixedTiles))
    return op.emitError("invalid zero tile factor");

  // Verify inner_dims_pos and outer_dims_perm.
  ShapedType unpackedType = (std::is_same<OpTy, PackOp>::value)
                                ? packOrUnPack.get_source_type()
                                : packOrUnPack.getDestType();
  size_t unpackedRank = unpackedType.rank;
  ArrayRef<i64> innerDimsPos = packOrUnPack.getInnerDimsPos();
  ArrayRef<i64> outerDimPerm = packOrUnPack.getOuterDimsPerm();
  if (isInvalidPackingPosSpecification(innerDimsPos, unpackedRank))
    return op.emitError("invalid inner_dims_pos vector");
  if (isInvalidPackingPosSpecification(outerDimPerm, unpackedRank))
    return op.emitError("invalid outer_dims_perm vector");
  if (!outerDimPerm.is_empty() && outerDimPerm.size() != unpackedRank)
    return op.emitError("outer_dims_perm must be a permutation or empty");

  // Tiling factors must be less than or equal to the input rank for pack (or
  // output rank for unpack), and must match the number of `inner_dims_pos`.
  if (mixedTiles.size() > unpackedRank) {
    return op.emitError("tiling factors must be less than or equal to the "
                         "input rank for pack or output rank for unpack");
  }
  if (mixedTiles.size() != innerDimsPos.size()) {
    return op.emitError(
        "tiling factors must equal the number of dimensions to tile");
  }

  ShapedType packedType = (std::is_same<OpTy, PackOp>::value)
                              ? packOrUnPack.getDestType()
                              : packOrUnPack.get_source_type();
  size_t packedRank = packedType.rank;
  // Require output rank to match input rank + number of blocking factors.
  if (unpackedRank + mixedTiles.size() != packedRank) {
    return op.emitError(
        "packed rank must equal unpacked rank + tiling factors");
  }

  // Verify result shape is greater than the minimum expected
  // by the pack operation, and that the output shape
  // represents full tiles.
  ShapedType expectedPackedType = PackOp::inferPackedType(
      unpackedType, packOrUnPack.getStaticTiles(), innerDimsPos, outerDimPerm);
  if (!areAllInBound(expectedPackedType.get_shape(), packedType.get_shape())) {
    return op.emitError("the shape of output is not large enough to hold the "
                         "packed data. Expected at least ")
           << expectedPackedType << ", got " << packedType;
  }
  if (!llvm::all_of(
          llvm::zip(packedType.get_shape().take_back(mixedTiles.size()),
                    mixedTiles),
          [](std::tuple<i64, FoldResult> it) {
            Option<i64> constTileSize =
                get_constant_int_value(std::get<1>(it));
            i64 shape = std::get<0>(it);
            if (!constTileSize) {
              // If specified tile size is dynamic, output shape should
              // be dynamic too.
              return ShapedType::is_dynamic(shape);
            }
            if (ShapedType::is_dynamic(shape)) {
              // For the shape being dynamic when tile size is
              // specified, return true. In canonical form a constant
              // tile size should lead to constant shape of the tiled
              // dimension, but not needed for verification.
              return true;
            }
            return shape == constTileSize.value();

          })) {
    return op.emitError("mismatch in inner tile sizes specified and shaped of "
                         "tiled dimension in the packed type");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

void PackOp::asm_output_names(function_ref<void(Value, StringRef)> set_name) {
  set_name(get_result(), "pack");
}

void PackOp::build(OpBuilder &builder, OperationState &state, Value source,
                   Value dest, ArrayRef<i64> innerDimsPos,
                   ArrayRef<FoldResult> innerTiles,
                   Option<Value> paddingValue,
                   ArrayRef<i64> outerDimsPerm) {
  assert!(self.outputos.size() == innerTiles.size() &&
         "number of tile sizes specified must match the specified number of "
         "original dimensions to be tiled");
  SmallVector<i64> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(innerTiles, dynamicTileSizes, staticTileSizes);
  build(builder, state, dest.get_type(), source, dest,
        paddingValue ? *paddingValue : nullptr,
        outerDimsPerm.is_empty() ? nullptr
                              : builder.getDenseI64ArrayAttr(outerDimsPerm),
        builder.getDenseI64ArrayAttr(innerDimsPos), dynamicTileSizes,
        builder.getDenseI64ArrayAttr(staticTileSizes));
}

LogicalResult
PackOp::reify_result_shapes(OpBuilder &builder,
                          ReifiedRankedShapedTypeDims &reified_return_shapes) {
  return reifyResultShapesImpl(self, builder, reified_return_shapes);
}

DenseMap<i64, FoldResult> PackOp::getDimAndTileMapping() {
  return getDimAndTileMappingImpl(self);
}

SmallVector<[FoldResult]> PackOp::getMixedTiles() {
  return getMixedTilesImpl(self);
}

SmallVector<i64> PackOp::getStaticTiles() {
  return getStaticTilesImpl(self);
}

/// Check if we have enough static information to catch undefined behavior when
/// the tile size does not divide perfectly the dimension of the input tensor.
static bool
areNotFullTiles(ArrayRef<i64> inputShape,
                DenseMap<i64, FoldResult> const &dimAndTileMapping) {
  i64 rank = inputShape.size();
  for (i64 dim = 0; dim < rank; dim++) {
    if (ShapedType::is_dynamic(inputShape[dim]))
      continue;
    auto it = dimAndTileMapping.find(dim);
    if (it == dimAndTileMapping.end())
      continue;
    Option<i64> constantTile = get_constant_int_value(it->second);
    if (!constantTile)
      continue;
    if (inputShape[dim] % (*constantTile) != 0)
      return true;
  }
  return false;
}

LogicalResult PackOp::verify() {
  if (failed(commonVerifierPackAndUnPackOp(self)))
    return failure();

  // Verify padding value, and bail out if the tile does not divide the
  // dimension fully. In the case of dynamic tile factors or dimensions, having
  // a partial tile is undefined behavior.
  auto paddingValue = getPaddingValue();
  if (paddingValue &&
      paddingValue.get_type() != get_source_type().element_type) {
    return emitOpError("Expected padding_value has ")
           << get_source_type().element_type
           << " but got: " << paddingValue.get_type();
  }

  auto dimAndTileMapping = getDimAndTileMapping();
  if (!paddingValue &&
      areNotFullTiles(get_source_type().get_shape(), dimAndTileMapping)) {
    return emitOpError("invalid tile factor provided. Only full tiles are "
                       "supported when padding_value is not set");
  }
  return success();
}

/// Get the expected packed type based on source type, tile factors, position of
/// the inner tiles and permutation of the outer tiled loop.
ShapedType PackOp::inferPackedType(ShapedType source_type,
                                   ArrayRef<i64> innerTileSizes,
                                   ArrayRef<i64> innerDimsPos,
                                   ArrayRef<i64> outerDimsPerm) {
  SmallVector<i64> result_shape = llvm::to_vector(source_type.get_shape());
  for (const auto &tiledDim : llvm::enumerate(innerDimsPos)) {
    if (ShapedType::is_dynamic(result_shape[tiledDim.value()]))
      continue;
    if (ShapedType::is_dynamic(innerTileSizes[tiledDim.index()])) {
      result_shape[tiledDim.value()] = ShapedType::kDynamic;
      continue;
    }
    result_shape[tiledDim.value()] = ceilDiv(result_shape[tiledDim.value()],
                                            innerTileSizes[tiledDim.index()]);
  }

  if (!outerDimsPerm.is_empty())
    applyPermutationToVector(result_shape, outerDimsPerm);

  // Append the inner tile dimensions.
  result_shape.append(innerTileSizes.begin(), innerTileSizes.end());
  return RankedTensorType::get(result_shape, source_type.element_type);
}

Value PackOp::createDestinationTensor(OpBuilder &b, Location loc, Value source,
                                      ArrayRef<FoldResult> innerTileSizes,
                                      ArrayRef<i64> innerDimsPos,
                                      ArrayRef<i64> outerDimsPerm) {
  AffineExpr dim0, dim1;
  bindDims(b.getContext(), dim0, dim1);
  auto ceilDiv = [&](FoldResult v1, FoldResult v2) -> FoldResult {
    return makeComposedFoldedAffineApply(b, loc, dim0.ceilDiv(dim1), {v1, v2});
  };

  SmallVector<[FoldResult]> mixedSizes;
  for (auto [index, value] :
       llvm::enumerate(source.get_type().cast<RankedTensorType>().get_shape())) {
    if (ShapedType::is_dynamic(value))
      mixedSizes.push(b.create<Dim>(loc, source, index).get_result());
    else
      mixedSizes.push(b.getIndexAttr(value));
  }
  for (auto it : llvm::zip(innerDimsPos, innerTileSizes)) {
    i64 dimPos = std::get<0>(it);
    FoldResult tileSize = std::get<1>(it);
    mixedSizes[dimPos] = ceilDiv(mixedSizes[dimPos], tileSize);
  }
  if (!outerDimsPerm.is_empty())
    applyPermutationToVector<FoldResult>(mixedSizes, outerDimsPerm);

  mixedSizes.append(innerTileSizes.begin(), innerTileSizes.end());
  auto elemType = source.get_type().cast<ShapedType>().element_type;
  return b.create<tensor::Empty>(loc, mixedSizes, elemType);
}

/// Returns true if the tiles and the tiled dims are constant.
template <typename OpTy>
bool areTilesAndTiledDimsAllConstant(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  ShapedType packedType = (std::is_same<OpTy, PackOp>::value)
                              ? op.getDestType()
                              : op.get_source_type();
  SmallVector<[FoldResult]> mixedTiles = op.getMixedTiles();
  for (auto [dimDest, tile] : llvm::zip(
           packedType.get_shape().take_back(mixedTiles.size()), mixedTiles)) {
    Option<i64> constTileSize = get_constant_int_value(tile);
    if (!constTileSize || ShapedType::is_dynamic(dimDest))
      return false;
  }
  return true;
}

Speculation::Speculatability PackOp::getSpeculatability() {
  if (auto paddingValue = getPaddingValue())
    return Speculation::Speculatable;

  // The verifier rejects already operations if we can statically prove that the
  // sizes of the tiles do not divide perfectly the dimension; thus, check only
  // to have constant tiles and tiled inner dimensions.
  if (!areTilesAndTiledDimsAllConstant(self))
    return Speculation::NotSpeculatable;

  return Speculation::Speculatable;
}

// Return true if `inner_dims_pos` and `outer_dims_perm` target the same
// dimensions for pack and unpack.
static bool hasSameInnerOuterAttribute(PackOp packOp, UnPackOp unPackOp) {
  if (packOp.getInnerDimsPos() != unPackOp.getInnerDimsPos())
    return false;
  return packOp.getOuterDimsPerm() == unPackOp.getOuterDimsPerm();
}

// Return true if pack and unpack have the same tiles.
// Same SSA values or same integer constants.
static bool haveSameTiles(PackOp packOp, UnPackOp unPackOp) {
  auto packTiles = packOp.getMixedTiles();
  auto unPackTiles = unPackOp.getMixedTiles();
  if (packTiles.size() != unPackTiles.size())
    return false;
  for (size_t i = 0, e = packTiles.size(); i < e; i++) {
    if (!isEqualConstantIntOrValue(packTiles[i], unPackTiles[i]))
      return false;
  }
  return true;
}

/// Fold an unpack(pack(x)) to x.
LogicalResult PackOp::canonicalise(PackOp packOp, PatternRewriter &rewriter) {
  UnPackOp unPackOp = packOp.get_source().defining_op<UnPackOp>();
  if (!unPackOp || unPackOp.get_source_type() != packOp.getDestType())
    return failure();
  if (packOp.getPaddingValue() ||
      !hasSameInnerOuterAttribute(packOp, unPackOp) ||
      !haveSameTiles(packOp, unPackOp))
    return failure();
  rewriter.replaceOp(packOp, unPackOp.get_source());
  return success();
}

//===----------------------------------------------------------------------===//
// UnPackOp
//===----------------------------------------------------------------------===//

void UnPackOp::asm_output_names(
    function_ref<void(Value, StringRef)> set_name) {
  set_name(get_result(), "unpack");
}

LogicalResult
UnPackOp::reify_result_shapes(OpBuilder &builder,
                            ReifiedRankedShapedTypeDims &reified_return_shapes) {
  return reifyResultShapesImpl(self, builder, reified_return_shapes);
}

DenseMap<i64, FoldResult> UnPackOp::getDimAndTileMapping() {
  return getDimAndTileMappingImpl(self);
}

SmallVector<[FoldResult]> UnPackOp::getMixedTiles() {
  return getMixedTilesImpl(self);
}

SmallVector<i64> UnPackOp::getStaticTiles() {
  return getStaticTilesImpl(self);
}

LogicalResult UnPackOp::verify() {
  return commonVerifierPackAndUnPackOp(self);
}

Speculation::Speculatability UnPackOp::getSpeculatability() {
  // See PackOp::getSpeculatability.
  if (!areTilesAndTiledDimsAllConstant(self))
    return Speculation::NotSpeculatable;

  return Speculation::Speculatable;
}

void UnPackOp::build(OpBuilder &builder, OperationState &state, Value source,
                     Value dest, ArrayRef<i64> innerDimsPos,
                     ArrayRef<FoldResult> innerTiles,
                     ArrayRef<i64> outerDimsPerm) {
  assert!(self.outputos.size() == innerTiles.size() &&
         "number of tile sizes specified must match the specified number of "
         "original dimensions to be tiled");
  SmallVector<i64> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(innerTiles, dynamicTileSizes, staticTileSizes);
  build(builder, state, dest.get_type(), source, dest,
        outerDimsPerm.is_empty() ? nullptr
                              : builder.getDenseI64ArrayAttr(outerDimsPerm),
        builder.getDenseI64ArrayAttr(innerDimsPos), dynamicTileSizes,
        builder.getDenseI64ArrayAttr(staticTileSizes));
}

/// pack(unpack(x)) -> x
LogicalResult UnPackOp::canonicalise(UnPackOp unPackOp,
                                     PatternRewriter &rewriter) {
  PackOp packOp = unPackOp.get_source().defining_op<tensor::PackOp>();
  if (!packOp || packOp.getDestType() != unPackOp.get_source_type())
    return failure();
  if (packOp.getPaddingValue() ||
      !hasSameInnerOuterAttribute(packOp, unPackOp) ||
      !haveSameTiles(packOp, unPackOp))
    return failure();
  rewriter.replaceOp(unPackOp, packOp.get_source());
  return success();
}

//===----------------------------------------------------------------------===//
// Common Canonicalizers and Folders.
//===----------------------------------------------------------------------===//

/// Folds a tensor.cast op into a consuming DestinationStyleOpInterface op if
/// the `tensor.cast` has source that is more static than the consuming op.
///
/// Example:
/// ```mlir
///   %1 = tensor.cast %0 : tensor<8x16xf32> to tensor<?x?xf32>
///   %2 = consumer %1 ... : tensor<?x?xf32> ...
/// ```
///
/// folds into:
///
/// ```mlir
///   %2 = consumer %0 ... : tensor<8x16xf32> ...
/// ```
/// TODO: Move the pattern to a proper place, so all other DestinationStyleOp
/// can add the pattern to their canonicalisers.
struct FoldTensorCastProducerOp
    : public OpInterfaceRewritePattern<DestinationStyleOpInterface> {
  using OpInterfaceRewritePattern<
      DestinationStyleOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(DestinationStyleOpInterface op,
                                PatternRewriter &rewriter) const override {
    // InsertSlice has its own logic about folding tensor.cast ops.
    if (isa<InsertSlice>(op.get_operation()))
      return failure();

    // If no operand comes from a tensor::Cast and can be folded then fail.
    bool hasTensorCastOperand =
        llvm::any_of(op.getOpOperands(), [&](OpOperand &opOperand) {
          if (opOperand.get().isa<BlockArgument>())
            return false;
          auto cast_op = opOperand.get().defining_op<tensor::Cast>();
          return cast_op && can_fold_into_consumer_op(cast_op);
        });
    if (!hasTensorCastOperand)
      return failure();

    SmallVector<Type, 4> newResultTypes;
    newResultTypes.reserve(op.getNumResults());
    SmallVector<Value, 4> newOperands;
    newOperands.reserve(op.getNumOperands());
    for (OpOperand &opOperand : op.getOpOperands()) {
      auto tensorCastOp = opOperand.get().defining_op<tensor::Cast>();
      bool fold = can_fold_into_consumer_op(tensorCastOp);
      newOperands.push(fold ? tensorCastOp.getOperand() : opOperand.get());
      if (op.isDpsInit(&opOperand) &&
          !newOperands.back().get_type().isa<MemRefType>())
        newResultTypes.push(newOperands.back().get_type());
    }

    // Clone op.
    Operation *newOp = clone(rewriter, op, newResultTypes, newOperands);
    SmallVector<Value, 4> replacements;
    replacements.reserve(newOp->getNumResults());
    for (auto [oldResult, newResult] :
         llvm::zip(op.outputs(), newOp->outputs())) {
      if (newResult.get_type() != oldResult.get_type()) {
        replacements.push(rewriter.create<tensor::Cast>(
            op.getLoc(), oldResult.get_type(), newResult));
      } else {
        replacements.push(newResult);
      }
    }
    rewriter.replaceOp(op, replacements);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// TensorDialect
//===----------------------------------------------------------------------===//

void TensorDialect::canonicalisation_patterns(
    RewritePatternSet &results) const {
  results.add<FoldTensorCastProducerOp>(getContext());
}
