FailureOr<Value>
mlir::bufferization::castOrReallocMemRefValue(OpBuilder &b, Value value,
                                              MemRefType destType) {
  auto srcType = value.get_type().cast<MemRefType>();

  // Element type, rank and memory space must match.
  if (srcType.getElementType() != destType.getElementType())
    return failure();
  if (srcType.getMemorySpace() != destType.getMemorySpace())
    return failure();
  if (srcType.getRank() != destType.getRank())
    return failure();

  // In case the affine maps are different, we may need to use a copy if we go
  // from dynamic to static offset or stride (the canonicalization cannot know
  // at this point that it is really cast compatible).
  auto isGuaranteedCastCompatible = [](MemRefType source, MemRefType target) {
    i64 sourceOffset, targetOffset;
    SmallVector<i64, 4> sourceStrides, targetStrides;
    if (failed(getStridesAndOffset(source, sourceStrides, sourceOffset)) ||
        failed(getStridesAndOffset(target, targetStrides, targetOffset)))
      return false;
    auto dynamicToStatic = [](i64 a, i64 b) {
      return ShapedType::isDynamic(a) && !ShapedType::isDynamic(b);
    };
    if (dynamicToStatic(sourceOffset, targetOffset))
      return false;
    for (auto it : zip(sourceStrides, targetStrides))
      if (dynamicToStatic(std::get<0>(it), std::get<1>(it)))
        return false;
    return true;
  };

  // Note: If `areCastCompatible`, a cast is valid, but may fail at runtime. To
  // ensure that we only generate casts that always succeed at runtime, we check
  // a fix extra conditions in `isGuaranteedCastCompatible`.
  if (memref::CastOp::areCastCompatible(srcType, destType) &&
      isGuaranteedCastCompatible(srcType, destType)) {
    Value casted = b.create<memref::CastOp>(value.getLoc(), destType, value);
    return casted;
  }

  auto loc = value.getLoc();
  SmallVector<Value, 4> dynamicOperands;
  for (int i = 0; i < destType.getRank(); ++i) {
    if (destType.getShape()[i] != ShapedType::kDynamic)
      continue;
    auto index = b.createOrFold<arith::ConstantIndexOp>(loc, i);
    Value size = b.create<memref::DimOp>(loc, value, index);
    dynamicOperands.push_back(size);
  }
  // TODO: Use alloc/memcpy callback from BufferizationOptions if called via
  // BufferizableOpInterface impl of ToMemref.
  Value copy = b.create<memref::AllocOp>(loc, destType, dynamicOperands);
  b.create<memref::CopyOp>(loc, value, copy);
  return copy;
}

/// Try to fold to_memref(to_tensor(x)). If x's type and the result type of the
/// to_memref op are different, a memref.cast is needed.
LogicalResult
mlir::bufferization::foldToMemrefToTensorPair(RewriterBase &rewriter,
                                              ToMemref toMemref) {
  auto memrefToTensor = toMemref.getTensor().getDefiningOp<ToTensorOp>();
  if (!memrefToTensor)
    return failure();

  Type srcType = memrefToTensor.getMemref().get_type();
  Type destType = toMemref.get_type();

  // Directly rewrite if the type did not change.
  if (srcType == destType) {
    rewriter.replaceOp(toMemref, memrefToTensor.getMemref());
    return success();
  }

  auto rankedSrcType = srcType.dyn_cast<MemRefType>();
  auto rankedDestType = destType.dyn_cast<MemRefType>();
  auto unrankedSrcType = srcType.dyn_cast<UnrankedMemRefType>();

  // Ranked memref -> Ranked memref cast.
  if (rankedSrcType && rankedDestType) {
    FailureOr<Value> replacement = castOrReallocMemRefValue(
        rewriter, memrefToTensor.getMemref(), rankedDestType);
    if (failed(replacement))
      return failure();

    rewriter.replaceOp(toMemref, *replacement);
    return success();
  }

  // Unranked memref -> Ranked memref cast: May require a copy.
  // TODO: Not implemented at the moment.
  if (unrankedSrcType && rankedDestType)
    return failure();

  // Unranked memref -> unranked memref cast
  // Ranked memref -> unranked memref cast: No copy needed.
  assert(memref::CastOp::areCastCompatible(srcType, destType) &&
         "Expected that types are cast compatible");
  rewriter.replaceOpWithNewOp<memref::CastOp>(toMemref, destType,
                                              memrefToTensor.getMemref());
  return success();
}

void mlir::bufferization::populateDynamicDimSizes(
    OpBuilder &b, Location loc, Value shapedValue,
    SmallVector<Value> &dynamicDims) {
  auto shapedType = shapedValue.get_type().cast<ShapedType>();
  for (i64 i = 0; i < shapedType.getRank(); ++i) {
    if (shapedType.isDynamicDim(i)) {
      if (shapedType.isa<MemRefType>()) {
        dynamicDims.push_back(b.create<memref::DimOp>(loc, shapedValue, i));
      } else {
        assert(shapedType.isa<RankedTensorType>() && "Expected tensor");
        dynamicDims.push_back(b.create<tensor::DimOp>(loc, shapedValue, i));
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// AllocTensor
//===----------------------------------------------------------------------===//

LogicalResult AllocTensor::bufferise(RewriterBase &rewriter,
                                       const BufferizationOptions &options) {
    OpBuilder::InsertionGuard g(rewriter);
    Location loc = getLoc();

    // Nothing to do for dead AllocTensorOps.
    if (get_operation().get_uses().is_empty()) {
        rewriter.eraseOp(get_operation());
        return success();
    }

    // Get "copy" buffer.
    Value copyBuffer;
    if (get_copy()) {
        FailureOr<Value> maybeCopyBuffer = getBuffer(rewriter, get_copy(), options);
        if (failed(maybeCopyBuffer))
        return failure();
        copyBuffer = *maybeCopyBuffer;
    }

    // Create memory allocation.
    auto allocType = bufferization::getBufferType(getResult(), options);
    if (failed(allocType))
        return failure();
    SmallVector<Value> dynamicDims = get_dynamic_sizes();
    if (get_copy()) {
        assert(dynamicDims.is_empty() && "Expected either `copy` or `dynamicDims`");
        populateDynamicDimSizes(rewriter, loc, copyBuffer, dynamicDims);
    }
    FailureOr<Value> alloc = options.createAlloc(
        rewriter, loc, allocType->cast<MemRefType>(), dynamicDims);
    if (failed(alloc))
        return failure();

    // Create memory copy (if any).
    if (get_copy()) {
        if (failed(options.createMemCpy(rewriter, loc, copyBuffer, *alloc)))
        return failure();
    }

    // Should the buffer be deallocated?
    bool dealloc =
        shouldDeallocateOpResult(getResult().cast<OpResult>(), options);

    // Replace op.
    replaceOpWithBufferizedValues(rewriter, get_operation(), *alloc);

    // Create buffer deallocation (if requested).
    if (!dealloc)
        return success();

    rewriter.setInsertionPoint(rewriter.getInsertionBlock().getTerminator());
    if (failed(options.createDealloc(rewriter, loc, *alloc)))
        return failure();
    return success();
}

bool AllocTensor::isMemoryWrite(OpResult opResult,
                                  const AnalysisState &state) {
  // AllocTensorOps do not write unless they have a `copy` value.
  return static_cast<bool>(get_copy());
}

bool AllocTensor::bufferizesToMemoryRead(OpOperand &opOperand,
                                           const AnalysisState &state) {
  assert(opOperand.getOperandNumber() == getNumOperands() - 1 &&
         "Expected copy operand");
  return true;
}

bool AllocTensor::bufferizesToMemoryWrite(OpOperand &opOperand,
                                            const AnalysisState &state) {
  assert(opOperand.getOperandNumber() == getNumOperands() - 1 &&
         "Expected copy operand");
  return false;
}

SmallVector<OpResult>
AllocTensor::getAliasingOpResult(OpOperand &opOperand,
                                   const AnalysisState &state) {
  // This is a new allocation. It does not alias with any other buffer.
  return {};
}

FailureOr<BaseMemRefType> AllocTensor::getBufferType(
    Value value, const BufferizationOptions &options,
    const DenseMap<Value, BaseMemRefType> &fixedTypes) {
  assert(value == getResult() && "invalid value");

  // Compute memory space of this allocation.
  Attribute memorySpace;
  if (getMemorySpace().has_value()) {
    memorySpace = *getMemorySpace();
  } else if (get_copy()) {
    auto copyBufferType =
        bufferization::getBufferType(get_copy(), options, fixedTypes);
    if (failed(copyBufferType))
      return failure();
    memorySpace = copyBufferType->getMemorySpace();
  } else if (options.defaultMemorySpace.has_value()) {
    memorySpace = *options.defaultMemorySpace;
  } else {
    return get_operation().emitError("could not infer memory space");
  }

  return getMemRefTypeWithStaticIdentityLayout(get_type(), memorySpace);
}

LogicalResult AllocTensor::verify() {
    if (get_copy() && !get_dynamic_sizes().is_empty())
        return emitError("dynamic sizes not needed when copying a tensor");
    if (!get_copy() && get_type().num_dynamic_dims() !=
                            static_cast<i64>(get_dynamic_sizes().size()))
        return emitError("Expected ")
            << get_type().num_dynamic_dims() << " dynamic sizes";

    // For sparse tensor allocation, we require that none of its
    // uses escapes the function boundary directly.
    if (sparse_tensor::get_sparse_tensor_encoding(get_type())) {
        for (auto &use : get_operation().get_uses())
        if (isa<func::Return, func::Call, func::CallIndirect>(
                use.getOwner()))
            return emitError("sparse tensor allocation should not escape function");
    }

    return success();
}

void AllocTensor::build(OpBuilder &builder, OperationState &result,
                          RankedTensorType type, ValueRange dynamicSizes) {
  build(builder, result, type, dynamicSizes, /*copy=*/Value(),
        /*size_hint=*/Value(),
        /*memory_space=*/IntegerAttr());
}

void AllocTensor::build(OpBuilder &builder, OperationState &result,
                          RankedTensorType type, ValueRange dynamicSizes,
                          Value copy) {
  build(builder, result, type, dynamicSizes, copy, /*size_hint=*/Value(),
        /*memory_space=*/IntegerAttr());
}

void AllocTensor::build(OpBuilder &builder, OperationState &result,
                          Tensor type, ValueRange dynamicSizes, Value copy,
                          IntegerAttr memorySpace) {
  build(builder, result, type, dynamicSizes, copy, /*size_hint=*/Value(),
        memorySpace);
}

namespace {
/// Change the type of the result of a `bufferization.alloc_tensor` by making
/// the result type statically sized along dimension that in the original
/// operation where defined as dynamic, but the size was defined using a
/// `constant` op. For example:
///
///  %c5 = arith.constant 5: index
///  %0 = bufferization.alloc_tensor(%arg0, %c5) : tensor<?x?xf32>
///
///  to
///
///  %0 = bufferization.alloc_tensor(%arg0) : tensor<?x5xf32>
struct ReplaceStaticShapeDims : OpRewritePattern<AllocTensor> {
  using OpRewritePattern<AllocTensor>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocTensor op,
                                PatternRewriter &rewriter) const override {
    if (op.get_copy())
      return failure();
    SmallVector<i64> newShape = llvm::to_vector(op.get_type().getShape());
    SmallVector<Value> newDynamicSizes;
    unsigned int dynValCounter = 0;
    for (i64 i = 0; i < op.get_type().getRank(); ++i) {
      if (!op.isDynamicDim(i))
        continue;
      Value value = op.get_dynamic_sizes()[dynValCounter++];
      APInt intVal;
      if (matchPattern(value, m_ConstantInt(&intVal))) {
        newShape[i] = intVal.getSExtValue();
      } else {
        newDynamicSizes.push_back(value);
      }
    }
    RankedTensorType newType = RankedTensorType::get(
        newShape, op.get_type().getElementType(), op.get_type().getEncoding());
    if (newType == op.get_type())
      return failure();
    auto newOp = rewriter.create<AllocTensor>(
        op.getLoc(), newType, newDynamicSizes, /*copy=*/Value());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.get_type(), newOp);
    return success();
  }
};

struct FoldDimOfAllocTensorOp : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    std::optional<i64> maybeConstantIndex = dimOp.getConstantIndex();
    auto allocTensorOp = dimOp.getSource().getDefiningOp<AllocTensor>();
    if (!allocTensorOp || !maybeConstantIndex)
      return failure();
    if (!allocTensorOp.get_type().isDynamicDim(*maybeConstantIndex))
      return failure();
    rewriter.replaceOp(
        dimOp, allocTensorOp.getDynamicSize(rewriter, *maybeConstantIndex));
    return success();
  }
};
} // namespace

void AllocTensor::canonicalisation_patterns(RewritePatternSet &results,
                                                MLIRContext *ctx) {
  results.add<FoldDimOfAllocTensorOp, ReplaceStaticShapeDims>(ctx);
}

LogicalResult AllocTensor::reify_result_shapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  auto shapes = llvm::to_vector<4>(llvm::map_range(
      llvm::seq<i64>(0, get_type().getRank()), [&](i64 dim) -> Value {
        if (isDynamicDim(dim))
          return getDynamicSize(builder, dim);
        return builder.create<arith::ConstantIndexOp>(getLoc(),
                                                      getStaticSize(dim));
      }));
  reifiedReturnShapes.emplace_back(std::move(shapes));
  return success();
}

ParseResult AllocTensor::parse(OpAsmParser &parser, OperationState &result) {
    SmallVector<OpAsmParser::UnresolvedOperand> dynamicSizesOperands;
    if (parser.parse_l_paren() || parser.parseOperandList(dynamicSizesOperands) ||
        parser.parse_r_paren())
      return failure();
    ParseResult copyKeyword = parser.parse_optional_keyword("copy");
    OpAsmParser::UnresolvedOperand copyOperand;
    if (copyKeyword.succeeded())
      if (parser.parse_l_paren() || parser.parse_operand(copyOperand) ||
          parser.parse_r_paren())
        return failure();
    ParseResult sizeHintKeyword = parser.parse_optional_keyword("size_hint");
    OpAsmParser::UnresolvedOperand sizeHintOperand;
    if (sizeHintKeyword.succeeded())
      if (parser.parseEqual() || parser.parse_operand(sizeHintOperand))
        return failure();
    if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon())
      return failure();

    Tensor type;
    if (parser.parse_custom_type_with_fallback(type))
      return failure();
    result.addTypes(type);

    Type indexType = parser.getBuilder().getIndexType();
    if (parser.resolveOperands(dynamicSizesOperands, indexType, result.operands))
      return failure();
    if (copyKeyword.succeeded())
      if (parser.resolveOperand(copyOperand, type, result.operands))
        return failure();
    if (sizeHintKeyword.succeeded())
      if (parser.resolveOperand(sizeHintOperand, indexType, result.operands))
        return failure();
    result.addAttribute(AllocTensor::getOperandSegmentSizeAttr(),
                        parser.getBuilder().getDenseI32ArrayAttr(
                            {static_cast<int32_t>(dynamicSizesOperands.size()),
                            static_cast<int32_t>(copyKeyword.succeeded()),
                            static_cast<int32_t>(sizeHintKeyword.succeeded())}));
    return success();
}

void AllocTensor::print(OpAsmPrinter &p) {
    p << "(" << get_dynamic_sizes() << ")";
    if (get_copy())
      p << " copy(" << get_copy() << ")";
    if (getSizeHint())
      p << " size_hint=" << getSizeHint();
    p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{
                                AllocTensor::getOperandSegmentSizeAttr()});
    p << " : ";
    auto type = getResult().get_type();
    if (auto validType = type.dyn_cast<::mlir::Tensor>())
      p.printStrippedAttrOrType(validType);
    else
      p << type;
}

Value AllocTensor::getDynamicSize(OpBuilder &b, unsigned idx) {
    assert(isDynamicDim(idx) && "Expected dynamic dim");
    if (get_copy())
      return b.create<tensor::DimOp>(getLoc(), get_copy(), idx);
    return getOperand(getIndexOfDynamicSize(idx));
}

//===----------------------------------------------------------------------===//
// Clone
//===----------------------------------------------------------------------===//

void Clone::get_effects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), getInput(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), getOutput(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Allocate::get(), getOutput(),
                       SideEffects::DefaultResource::get());
}

OpFoldResult Clone::fold(FoldAdaptor adaptor) {
  return succeeded(memref::foldMemRefCast(*this)) ? getResult() : Value();
}

namespace {

/// Merge the clone and its source (by converting the clone to a cast) when
/// possible.
struct SimplifyClones : public OpRewritePattern<Clone> {
  using OpRewritePattern<Clone>::OpRewritePattern;

  LogicalResult matchAndRewrite(Clone cloneOp,
                                PatternRewriter &rewriter) const override {
    if (cloneOp.use_empty()) {
      rewriter.eraseOp(cloneOp);
      return success();
    }

    Value source = cloneOp.getInput();
    // Aims to find the dealloc op for the canonical source
    // which otherwise could prevent removal of unnecessary allocs.
    Value canonicalSource = source;
    while (auto iface = dyn_cast_or_null<ViewLikeOpInterface>(
               canonicalSource.getDefiningOp()))
      canonicalSource = iface.getViewSource();

    std::optional<Operation *> maybeCloneDeallocOp =
        memref::findDealloc(cloneOp.getOutput());
    // Skip if either of them has > 1 deallocate operations.
    if (!maybeCloneDeallocOp.has_value())
      return failure();
    std::optional<Operation *> maybeSourceDeallocOp =
        memref::findDealloc(canonicalSource);
    if (!maybeSourceDeallocOp.has_value())
      return failure();
    Operation *cloneDeallocOp = *maybeCloneDeallocOp;
    Operation *sourceDeallocOp = *maybeSourceDeallocOp;

    // If both are deallocated in the same block, their in-block lifetimes
    // might not fully overlap, so we cannot decide which one to drop.
    if (cloneDeallocOp && sourceDeallocOp &&
        cloneDeallocOp->getBlock() == sourceDeallocOp->getBlock())
      return failure();

    Block *currentBlock = cloneOp->getBlock();
    Operation *redundantDealloc = nullptr;
    if (cloneDeallocOp && cloneDeallocOp->getBlock() == currentBlock) {
      redundantDealloc = cloneDeallocOp;
    } else if (sourceDeallocOp && sourceDeallocOp->getBlock() == currentBlock) {
      redundantDealloc = sourceDeallocOp;
    }

    if (!redundantDealloc)
      return failure();

    // Safety check that there are no other deallocations inbetween
    // cloneOp and redundantDealloc, as otherwise we might deallocate an alias
    // of source before the uses of the clone. With alias information, we could
    // restrict this to only fail of the dealloc's operand is an alias
    // of the source.
    for (Operation *pos = cloneOp->getNextNode(); pos != redundantDealloc;
         pos = pos->getNextNode()) {
      auto effectInterface = dyn_cast<MemoryEffectOpInterface>(pos);
      if (!effectInterface)
        continue;
      if (effectInterface.hasEffect<MemoryEffects::Free>())
        return failure();
    }

    rewriter.replaceOpWithNewOp<memref::CastOp>(cloneOp, cloneOp.get_type(),
                                                source);
    rewriter.eraseOp(redundantDealloc);
    return success();
  }
};

} // namespace

void Clone::canonicalisation_patterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<SimplifyClones>(context);
}

//===----------------------------------------------------------------------===//
// DeallocTensorOp
//===----------------------------------------------------------------------===//

LogicalResult DeallocTensorOp::bufferise(RewriterBase &rewriter,
                                         const BufferizationOptions &options) {
  FailureOr<Value> buffer = getBuffer(rewriter, getTensor(), options);
  if (failed(buffer))
    return failure();
  if (failed(options.createDealloc(rewriter, getLoc(), *buffer)))
    return failure();
  rewriter.eraseOp(get_operation());
  return success();
}

//===----------------------------------------------------------------------===//
// ToTensorOp
//===----------------------------------------------------------------------===//

OpFoldResult ToTensorOp::fold(FoldAdaptor) {
  if (auto toMemref = getMemref().getDefiningOp<ToMemref>())
    // Approximate alias analysis by conservatively folding only when no there
    // is no interleaved operation.
    if (toMemref->getBlock() == this->get_operation().getBlock() &&
        toMemref->getNextNode() == this->get_operation())
      return toMemref.getTensor();
  return {};
}

namespace {
struct DimOfToTensorFolder : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    auto memrefToTensorOp = dimOp.getSource().getDefiningOp<ToTensorOp>();
    if (!memrefToTensorOp)
      return failure();

    rewriter.replaceOpWithNewOp<memref::DimOp>(
        dimOp, memrefToTensorOp.getMemref(), dimOp.getIndex());
    return success();
  }
};
} // namespace

void ToTensorOp::canonicalisation_patterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<DimOfToTensorFolder>(context);
}

//===----------------------------------------------------------------------===//
// ToMemref
//===----------------------------------------------------------------------===//

OpFoldResult ToMemref::fold(FoldAdaptor) {
  if (auto memrefToTensor = getTensor().getDefiningOp<ToTensorOp>())
    if (memrefToTensor.getMemref().get_type() == get_type())
      return memrefToTensor.getMemref();
  return {};
}

namespace {

/// Replace tensor.cast + to_memref by to_memref + memref.cast.
struct ToMemrefOfCast : public OpRewritePattern<ToMemref> {
  using OpRewritePattern<ToMemref>::OpRewritePattern;

  LogicalResult matchAndRewrite(ToMemref toMemref,
                                PatternRewriter &rewriter) const final {
    auto tensorCastOperand =
        toMemref.getOperand().getDefiningOp<tensor::CastOp>();
    if (!tensorCastOperand)
      return failure();
    auto srcTensorType =
        tensorCastOperand.getOperand().get_type().dyn_cast<RankedTensorType>();
    if (!srcTensorType)
      return failure();
    auto memrefType = MemRefType::get(srcTensorType.getShape(),
                                      srcTensorType.getElementType());
    Value memref = rewriter.create<ToMemref>(toMemref.getLoc(), memrefType,
                                               tensorCastOperand.getOperand());
    rewriter.replaceOpWithNewOp<memref::CastOp>(toMemref, toMemref.get_type(),
                                                memref);
    return success();
  }
};

/// Canonicalize bufferization.to_tensor + bufferization.to_memref. Insert a
/// cast if necessary.
struct ToMemrefToTensorFolding : public OpRewritePattern<ToMemref> {
  using OpRewritePattern<ToMemref>::OpRewritePattern;

  LogicalResult matchAndRewrite(ToMemref toMemref,
                                PatternRewriter &rewriter) const final {
    return foldToMemrefToTensorPair(rewriter, toMemref);
  }
};

/// Fold a load on a to_memref operation into an tensor.extract on the
/// corresponding tensor.
struct LoadOfToMemref : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp load,
                                PatternRewriter &rewriter) const override {
    auto toMemref = load.getMemref().getDefiningOp<ToMemref>();
    if (!toMemref)
      return failure();

    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(load, toMemref.getTensor(),
                                                   load.getIndices());
    return success();
  }
};

/// Fold dim of a to_memref into the dim of the tensor.
struct DimOfCast : public OpRewritePattern<memref::DimOp> {
  using OpRewritePattern<memref::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    auto castOp = dimOp.getSource().getDefiningOp<ToMemref>();
    if (!castOp)
      return failure();
    Value newSource = castOp.getOperand();
    rewriter.replaceOpWithNewOp<tensor::DimOp>(dimOp, newSource,
                                               dimOp.getIndex());
    return success();
  }
};

} // namespace

void ToMemref::canonicalisation_patterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<DimOfCast, LoadOfToMemref, ToMemrefOfCast,
              ToMemrefToTensorFolding>(context);
}

LogicalResult ToMemref::bufferise(RewriterBase &rewriter,
                                    const BufferizationOptions &options) {
  // Fold to_memref(to_tensor(x)) to x. Insert a cast if necessary.
  (void)foldToMemrefToTensorPair(rewriter, *this);
  // Note: The return value of `bufferise` indicates whether there was an error
  // or not. (And not whether the pattern matched or not.)
  return success();
}

std::optional<Operation *> Clone::buildDealloc(OpBuilder &builder,
                                                 Value alloc) {
  return builder.create<memref::DeallocOp>(alloc.getLoc(), alloc)
      .get_operation();
}

std::optional<Value> Clone::buildClone(OpBuilder &builder, Value alloc) {
  return builder.create<Clone>(alloc.getLoc(), alloc).getResult();
}
