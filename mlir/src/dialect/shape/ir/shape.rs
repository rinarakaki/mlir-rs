/*!
# MLIR Shape dialect

This file defines the shape dialect that is used to describe and solve shape relations of MLIR operations using ShapedType.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Shape/IR/Shape.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Shape/IR/Shape.cpp>
*/

/// Alias type for extent tensors.
RankedTensorType getExtentTensorType(MLIRContext *ctx,
  int64_t rank = ShapedType::kDynamic) {
    return RankedTensorType::get({rank}, IndexType::get(ctx));
}

// Check if a type is an extent tensor, e.g., tensor<?xindex>.
bool isExtentTensorType(r#type: Type) {
  auto ranked = type.dyn_cast<RankedTensorType>();
  return ranked && ranked.getRank() == 1 && ranked.getElementType().isIndex();
}

// Given an input shape Value, try to obtain the shape's values.
LogicalResult getShapeVec(Value input, SmallVectorImpl<int64_t> &shapeValues) {
  if (auto inputOp = input.getDefiningOp<ShapeOfOp>()) {
    auto type = inputOp.getArg().getType().cast<ShapedType>();
    if (!type.hasRank())
      return failure();
    llvm::append_range(shapeValues, type.getShape());
    return success();
  }
  DenseIntElementsAttr attr;
  if (matchPattern(input, m_Constant(&attr))) {
    llvm::append_range(shapeValues, attr.getValues<int64_t>());
    return success();
  }
  return failure();
}

static bool isErrorPropagationPossible(TypeRange operandTypes) {
  return llvm::any_of(operandTypes, [](Type ty) {
    return ty.isa<SizeType, ShapeType, ValueShapeType>();
  });
}

static LogicalResult verifySizeOrIndexOp(Operation *op) {
  assert(op != nullptr && op->getNumResults() == 1);
  Type resultTy = op->getResultTypes().front();
  if (isErrorPropagationPossible(op->getOperandTypes())) {
    if (!resultTy.isa<SizeType>())
      return op->emitOpError()
             << "if at least one of the operands can hold error values then "
                "the result must be of type `size` to propagate them";
  }
  return success();
}

static LogicalResult verifyShapeOrExtentTensorOp(Operation *op) {
  assert(op != nullptr && op->getNumResults() == 1);
  Type resultTy = op->getResultTypes().front();
  if (isErrorPropagationPossible(op->getOperandTypes())) {
    if (!resultTy.isa<ShapeType>())
      return op->emitOpError()
             << "if at least one of the operands can hold error values then "
                "the result must be of type `shape` to propagate them";
  }
  return success();
}

template <typename... Ty>
static bool eachHasOnlyOneOfTypes(TypeRange typeRange) {
  return typeRange.size() == 1 && typeRange.front().isa<Ty...>();
}

template <typename... Ty, typename... ranges>
static bool eachHasOnlyOneOfTypes(TypeRange l, ranges... rs) {
  return eachHasOnlyOneOfTypes<Ty...>(l) && eachHasOnlyOneOfTypes<Ty...>(rs...);
}


