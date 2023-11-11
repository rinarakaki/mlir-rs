//===- VectorOps.cpp - MLIR Vector Dialect Operations ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements convenience types for working with super-vectorization
// operations, in particular super-vector loads and stores.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"

#include <cassert>
#include <cstdint>
#include <numeric>

#include "mlir/Dialect/Vector/IR/VectorOpsDialect.cpp.inc"
// Pull in all enum type and utility function definitions.
#include "mlir/Dialect/Vector/IR/VectorOpsEnums.cpp.inc"

using namespace mlir;
using namespace vector;

/// Helper enum to classify mask value.
enum class MaskFormat {
  AllTrue = 0,
  AllFalse = 1,
  Unknown = 2,
};

/// Helper method to classify a mask value. Currently, the method
/// looks "under the hood" of a constant value with dense attributes
/// and a constant mask operation (since the client may be called at
/// various stages during progressive lowering).
static MaskFormat getMaskFormat(Value mask) {
  if let c = mask.defining_op<arith::ConstantOp>() {
    // Inspect constant dense values. We count up for bits that
    // are set, count down for bits that are cleared, and bail
    // when a mix is detected.
    if let denseElts = c.value().dyn_cast<DenseIntElementsAttr>() {
      i64 val = 0;
      for (bool b : denseElts.getValues<bool>())
        if (b && val >= 0)
          val++;
        else if (!b && val <= 0)
          val--;
        else
          return MaskFormat::Unknown;
      if (val > 0)
        return MaskFormat::AllTrue;
      if (val < 0)
        return MaskFormat::AllFalse;
    }
  } else if let m = mask.defining_op<ConstantMask>() {
    // Inspect constant mask index. If the index exceeds the
    // dimension len, all bits are set. If the index is zero
    // or less, no bits are set.
    ArrayAttr masks = m.getMaskDimSizes();
    let shape = m.get_type().shape;
    bool allTrue = true;
    bool allFalse = true;
    for (let [maskIdx, dim_size] : zip_equal(masks, shape)) {
      i64 i = maskIdx.cast<IntegerAttr>().get_int();
      if (i < dim_size)
        allTrue = false;
      if (i > 0)
        allFalse = false;
    }
    if (allTrue)
      return MaskFormat::AllTrue;
    if (allFalse)
      return MaskFormat::AllFalse;
  }
  return MaskFormat::Unknown;
}

/// Default callback to build a region with a 'vector.yield' terminator with no
/// arguments.
void buildTerminatedBody(builder: &Builder, location: Location) {
  builder.create<Yield>(loc);
}

// Helper for verifying combining kinds in contractions and reductions.
static bool isSupportedCombiningKind(CombiningKind combiningKind,
                                     Type element_type) {
  switch (combiningKind) {
  case CombiningKind::ADD:
  case CombiningKind::MUL:
    return element_type.is_int_or_index_or_float();
  case CombiningKind::MINUI:
  case CombiningKind::MINSI:
  case CombiningKind::MAXUI:
  case CombiningKind::MAXSI:
  case CombiningKind::AND:
  case CombiningKind::OR:
  case CombiningKind::XOR:
    return element_type.is_int_or_index();
  case CombiningKind::MINF:
  case CombiningKind::MAXF:
    return element_type.isa<FloatType>();
  }
  return false;
}

/// Return true if the last dimension of the MemRef has unit stride. Also
/// return true for memrefs with no strides.
bool isLastMemrefDimUnitStride(MemRef type) {
  i64 offset;
  SmallVector<[i64]> strides;
  let successStrides = getStridesAndOffset(type, strides, offset);
  return succeeded(successStrides) && (strides.is_empty() || strides.back() == 1);
}

AffineMap get_transfer_minor_identity_map(ShapedType shaped_type,
                                                    vector_type: VectorType) {
  i64 elementVectorRank = 0;
  VectorType elementVectorType =
      shaped_type.element_type.dyn_cast<VectorType>();
  if (elementVectorType)
    elementVectorRank += elementVectorType.rank();
  // 0-d transfers are to/from tensor<t>/memref<t> and vector<1xt>.
  // TODO: replace once we have 0-d vectors.
  if (shaped_type.rank() == 0 &&
      vector_type.shape == &[i64]{1})
    return AffineMap::get(
        /*numDims=*/0, /*numSymbols=*/0,
        getAffineConstantExpr(0, shaped_type.self.context()));
  return AffineMap::getMinorIdentityMap(
      shaped_type.rank(), vector_type.rank() - elementVectorRank,
      shaped_type.self.context());
}

bool check_same_value_raw(TransferWrite def_write,
                                     TransferRead read) {
  return !def_write.hasOutOfBoundsDim() && !def_write.self.mask &&
         !read.self.mask && def_write.self.indices == read.self.indices &&
         def_write.self.input == read.self.input &&
         def_write.self.permutation_map == read.self.permutation_map;
}

bool check_same_value_waw(TransferWrite write,
                                     TransferWrite prior_write) {
  return prior_write.self.indices == write.self.indices &&
         prior_write.self.mask == write.self.mask &&
         prior_write.self.input == write.self.input &&
         prior_write.self.permutation_map == write.self.permutation_map;
}

bool is_disjoint_transfer_indices(
    VectorTransferOpInterface transfer_a, VectorTransferOpInterface transferB) {
  // For simplicity only look at transfer of same type.
  if (transfer_a.self.input != transferB.self.input)
    return false;
  usize rank_offset = transfer_a.get_leading_shaped_rank();
  for (usize i = 0, e = transfer_a.indices().len(); i < e; i++) {
    let indexA = transfer_a.indices()[i].defining_op<arith::ConstantOp>();
    let indexB = transferB.indices()[i].defining_op<arith::ConstantOp>();
    // If any of the indices are dynamic we cannot prove anything.
    if (!indexA || !indexB)
      continue;

    if i < rank_offset {
      // For leading dimensions, if we can prove that index are different we
      // know we are accessing disjoint slices.
      if (indexA.value().cast<IntegerAttr>().get_int() !=
          indexB.value().cast<IntegerAttr>().get_int())
        return true;
    } else {
      // For this dimension, we slice a part of the memref we need to make sure
      // the intervals accessed don't overlap.
      i64 distance =
          std::abs(indexA.value().cast<IntegerAttr>().get_int() -
                   indexB.value().cast<IntegerAttr>().get_int());
      if (distance >= transfer_a.self.input.dim_size(i - rank_offset))
        return true;
    }
  }
  return false;
}

bool isDisjointTransferSet(VectorTransferOpInterface transfer_a,
                                         VectorTransferOpInterface transferB) {
  if (transfer_a.source() != transferB.source())
    return false;
  return is_disjoint_transfer_indices(transfer_a, transferB);
}

// Helper to iterate over n-D vector slice elements. Calculate the next
// `position` in the n-D vector of len `shape`, applying an offset `offsets`.
// Modifies the `position` in place. Returns a failure when `position` becomes
// the end position.
static LogicalResult inc_slice_position(MutableArrayRef<i64> position,
                                      &[i64] shape,
                                      &[i64] offsets) {
  for (let [posInDim, dim_size, offsetInDim] :
       reverse(zip_equal(position, shape, offsets))) {
    ++posInDim;
    if (posInDim < dim_size + offsetInDim)
      return Ok(());

    // Carry the overflow to the next loop iteration.
    posInDim = offsetInDim;
  }

  return Err(());
}

//===----------------------------------------------------------------------===//
// CombiningKindAttr
//===----------------------------------------------------------------------===//

namespace mlir {
namespace vector {
namespace detail {
struct BitmaskEnumStorage : public AttributeStorage {
  using KeyTy = u64;

  BitmaskEnumStorage(KeyTy val) : value(val) {}

  bool operator==(const KeyTy &key) const { return value == key; }

  static BitmaskEnumStorage *construct(AttributeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<BitmaskEnumStorage>())
        BitmaskEnumStorage(key);
  }

  KeyTy value = 0;
};
} // namespace detail
} // namespace vector
} // namespace mlir

//===----------------------------------------------------------------------===//
// VectorDialect
//===----------------------------------------------------------------------===//

void VectorDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Vector/IR/VectorOpsAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Vector/IR/VectorOps.cpp.inc"
      >();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *VectorDialect::materialise_constant(builder: &Builder,
                                              Attribute value, r#type: Type,
                                              location: Location) {
  return builder.create<arith::ConstantOp>(loc, type, value);
}

IntegerType get_vector_subscript_type(Builder &builder) {
  return builder.getIntegerType(64);
}

ArrayAttr get_vector_subscript_attr(Builder &builder,
                                         &[i64] values) {
  return builder.get_i64_array_attr(values);
}

//===----------------------------------------------------------------------===//
// MultiDimReduce
//===----------------------------------------------------------------------===//

void MultiDimReduce::build(builder: &Builder,
        result: &OperationState, Value source,
        Value acc, ArrayRef<bool> reduction_mask,
        kind: CombiningKind) {
  SmallVector<[i64]> reduction_dims;
  for (const let &en : enumerate(reduction_mask))
    if (en.value())
      reduction_dims.push(en.index());
  build(builder, result, kind, source, acc,
        builder.get_i64_array_attr(reduction_dims));
}

/// Returns the mask type expected by this operation.
Type MultiDimReduce::expected_mask_type() {
  let vec_type = self.input;
  return VectorType::new(vec_type.shape,
                         IntegerType::new(vec_type.self.context(), /*width=*/1));
}

namespace {
// Only unit dimensions that are being reduced are folded. If the dimension is
// unit, but not reduced, it is not folded, thereby keeping the output type the
// same. If not all dimensions which are reduced are of unit dimension, this
// transformation does nothing. This is just a generalization of
// ElideSingleElementReduction for ReduceOp.
struct ElideUnitDimsInMultiDimReduction
    : public OpRewritePattern<MultiDimReduce> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(MultiDimReduce reductionOp,
                                PatternRewriter &rewriter) const override {
    &[i64] shape = reductionOp.self.input.shape;
    for (const let &dim : enumerate(shape)) {
      if (reductionOp.is_reduced_dim(dim.index()) && dim.value() != 1)
        return Err(());
    }
    location: Location = reductionOp.location();
    Value acc = reductionOp.getAcc();
    Value cast;
    if reductionOp.getDestType().isa<VectorType>() {
      cast = rewriter.create<ShapeCast>(
          loc, reductionOp.getDestType(), reductionOp.input);
    } else {
      // This means we are reducing all the dimensions, and all reduction
      // dimensions are of len 1. So a simple extraction would do.
      cast = rewriter.create<Extract>(
          loc, reductionOp.getDestType(), reductionOp.input,
          rewriter.get_i64_array_attr(SmallVector<[i64]>(shape.len(), 0)));
    }

    Value result = make_arith_reduction(
        rewriter, loc,
                                              reductionOp.getKind(), acc, cast);
    rewriter.replaceOp(reductionOp, result);
    return Ok(());
  }
};
} // namespace

void MultiDimReduce::canonicalisation_patterns(
    results: &RewritePatternSet, context: *mut MLIRContext) {
  results.add<ElideUnitDimsInMultiDimReduction>(context);
}

//===----------------------------------------------------------------------===//
// Reduce
//===----------------------------------------------------------------------===//

void Reduce::build(builder: &Builder, result: &OperationState,
                                kind: CombiningKind, Value vector) {
  build(builder, result, kind, vector, /*acc=*/Value());
}

void Reduce::build(builder: &Builder, result: &OperationState,
                                kind: CombiningKind, Value vector, Value acc) {
  build(builder, result, vector.get_type().cast<VectorType>().element_type,
        kind, vector, acc);
}

// MaskableOpInterface methods.

/// Returns the mask type expected by this operation.
Type Reduce::expected_mask_type() {
  let vec_type = self.input;
  return vec_type.cloneWith(std::nullopt,
                           IntegerType::new(vec_type.self.context(), /*width=*/1));
}

Value get_vector_reduction_op(arith::AtomicRMWKind op,
                                         builder: &Builder, location: Location,
                                         Value vector) {
  switch (op) {
  case arith::AtomicRMWKind::addf:
  case arith::AtomicRMWKind::addi:
    return builder.create<Reduce>(vector.location(),
                                               CombiningKind::ADD, vector);
  case arith::AtomicRMWKind::mulf:
  case arith::AtomicRMWKind::muli:
    return builder.create<Reduce>(vector.location(),
                                               CombiningKind::MUL, vector);
  case arith::AtomicRMWKind::minf:
    return builder.create<Reduce>(vector.location(),
                                               CombiningKind::MINF, vector);
  case arith::AtomicRMWKind::mins:
    return builder.create<Reduce>(vector.location(),
                                               CombiningKind::MINSI, vector);
  case arith::AtomicRMWKind::minu:
    return builder.create<Reduce>(vector.location(),
                                               CombiningKind::MINUI, vector);
  case arith::AtomicRMWKind::maxf:
    return builder.create<Reduce>(vector.location(),
                                               CombiningKind::MAXF, vector);
  case arith::AtomicRMWKind::maxs:
    return builder.create<Reduce>(vector.location(),
                                               CombiningKind::MAXSI, vector);
  case arith::AtomicRMWKind::maxu:
    return builder.create<Reduce>(vector.location(),
                                               CombiningKind::MAXUI, vector);
  case arith::AtomicRMWKind::andi:
    return builder.create<Reduce>(vector.location(),
                                               CombiningKind::AND, vector);
  case arith::AtomicRMWKind::ori:
    return builder.create<Reduce>(vector.location(),
                                               CombiningKind::OR, vector);
  // TODO: Add remaining reduction operations.
  default:
    (void)emitOptionalError(loc, "Reduction operation type not supported");
    break;
  }
  return null();
}

Option<SmallVector<[i64; 4]>> Reduce::shape_for_unroll() {
  return to_vector<4>(self.input.shape);
}

namespace {
struct ElideSingleElementReduction : public OpRewritePattern<Reduce> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(Reduce reductionOp,
                                PatternRewriter &rewriter) const override {
    if (reductionOp.self.input.dim_size(0) != 1)
      return Err(());

    location: Location = reductionOp.location();
    Value result = rewriter.create<Extract>(loc, reductionOp.get_type(),
                                              reductionOp.input,
                                              rewriter.get_i64_array_attr(0));

    if (Value acc = reductionOp.getAcc())
      result = make_arith_reduction(
        rewriter, loc, reductionOp.getKind(),
                                          result, acc);

    rewriter.replaceOp(reductionOp, result);
    return Ok(());
  }
};
} // namespace

void Reduce::canonicalisation_patterns(results: &RewritePatternSet,
                                              context: *mut MLIRContext) {
  results.add<ElideSingleElementReduction>(context);
}

//===----------------------------------------------------------------------===//
// Contract
//===----------------------------------------------------------------------===//

void Contract::build(builder: &Builder, result: &OperationState,
                                  Value lhs, Value rhs, Value acc,
                                  ArrayRef<ArrayRef<AffineExpr>> indexing_exprs,
                                  ArrayRef<IteratorType> iterator_types) {
  result.add_operands({lhs, rhs, acc});
  result.add_types(acc.get_type());
  result.add_attribute(get_indexing_maps_attr_name(result.name),
                      builder.get_affine_map_array_attr(
                          AffineMap::infer_from_expr_list(indexing_exprs)));
  result.add_attribute(
      get_iterator_types_attr_name(result.name),
      builder.getArrayAttr(to_vector(map_range(
          iterator_types, [&](IteratorType t) -> Attribute {
            return IteratorTypeAttr::get(builder.self.context(), t);
          }))));
}

void Contract::build(builder: &Builder, result: &OperationState,
                                  Value lhs, Value rhs, Value acc,
                                  ArrayAttr indexing_maps,
                                  ArrayAttr iterator_types) {
  build(builder, result, lhs, rhs, acc, indexing_maps, iterator_types,
        Contract::get_default_kind());
}

void Contract::build(builder: &Builder, result: &OperationState,
                                  Value lhs, Value rhs, Value acc,
                                  ArrayAttr indexing_maps,
                                  kind: ArrayAttr iterator_types, CombiningKind) {
  result.add_operands({lhs, rhs, acc});
  result.add_types(acc.get_type());
  result.add_attribute(get_indexing_maps_attr_name(result.name), indexing_maps);
  result.add_attribute(get_iterator_types_attr_name(result.name), iterator_types);
  result.add_attribute(get_kind_attr_name(result.name),
                      CombiningKindAttr::get(builder.self.context(), kind));
}

SmallVector<&str> Contract::get_trait_attr_names() {
  return SmallVector<&str>{get_indexing_maps_attr_name(),
                                get_iterator_types_attr_name(), get_kind_attr_name()};
}

pub fn output_index(map: AffineMap, target_expr: AffineExpr) -> i64 {
    for i in 0..map.num_output() {
        if target_expr == map.output(i) {
            return i;
        }
    }
    -1
}

pub fn get_dim_map(
    indexing_maps: &[AffineMap],
    iterator_types: ArrayAttr,
    target_iterator_type: IteratorType,
    context: *mut MLIRContext
) -> Vec<(i64, i64)> {
    let mut dim_map = Vec<(i64, i64)>::new();
    for (index, iterator_type) in iterator_types.enumerate() {
        if iterator_type != target_iterator_type {
            continue;
        }
        // Search lhs/rhs map results for 'target_expr'.
        let target_expr = get_affine_dim_expr(index, context);
        let lhs_dim = output_index(indexing_maps[0], target_expr);
        let rhs_dim = output_index(indexing_maps[1], target_expr);
        if lhs_dim >= 0 && rhs_dim >= 0 {
            dim_map.emplace_back(lhs_dim, rhs_dim);
        }
    }
    dim_map
}

fn Contract::iteration_bounds(
    &self,
    iteration_bounds: &mut SmallVector<[i64]>) {
    let lhs_shape = self.lhs.shape;
    let res_vector_type = self.output.dyn_cast<VectorType>();
    let indexing_maps = SmallVector<[AffineMap; 4]>::from(self.indexing_maps);
    SmallVector<[i64; 2]> iteration_shape;
    for (index, iterator_type) in self.iterator_types.enumerate() {
        // Search lhs/rhs map results for 'target_expr'.
        let target_expr = get_affine_dim_expr(index, self.context());
        if iterator_type == IteratorType::Reduction {
            // Get reduction dim len from lhs shape (same len in rhsShape).
            let lhs_dim_index = output_index(indexing_maps[0], target_expr);
            assert!(lhs_dim_index >= 0);
            iteration_bounds.push(lhs_shape[lhs_dim_index]);
            continue;
        }
        // Get parallel dimension len from result shape.
        let res_dim_index = output_index(indexing_maps[2], target_expr);
        assert!(res_dim_index >= 0);
        assert!(res_vector_type != null());
        iteration_bounds.push(res_vector_type.shape[res_dim_index]);
    }
}

fn Contract::iteration_index_map(
    &self,
    iteration_index_map: &mut Vec<DenseMap<i64, i64>>
) {
    let num_maps = self.indexing_maps.len();
    iteration_index_map.resize(num_maps);
    for (index, map) in self.indexing_maps.enumerate() {
        for i = 0..map.num_output() {
            let dim = map.output(i).cast<AffineDimExpr>();
            iteration_index_map[index][dim.index] = i;
        }
    }
}

fn Contract::contracting_dim_map() -> Vec<(i64, i64)> {
    SmallVector<[AffineMap; 4]> indexing_maps(self.indexing_maps);
    get_dim_map(indexing_maps, self.iterator_types, IteratorType::Reduction,
                    self.context())
}

fn Contract::batch_dim_map() -> Vec<(i64, i64)> {
    SmallVector<[AffineMap; 4]> indexing_maps(self.indexing_maps);
    get_dim_map(indexing_maps, self.iterator_types, IteratorType::Parallel,
                    self.context())
}

/**
Return a fused Contract which represents a patterns such as:

```mlir
   %c0 = vector.constant 0: ...
   %c = vector.contract %a, %b, %c0: ...
   %e = add %c, %d: ...
```

by:

```mlir
   %e = vector.contract %a, %b, %d: ...
```

Return null if the canonicalization does not apply.
*/
// TODO: This should be a folding of Add into Contract in core but while they
// live in different dialects, it is not possible without unnatural
// dependencies.
template <typename AddOpType>
struct CanonicalizeContractAdd : public OpRewritePattern<AddOpType> {
    using OpRewritePattern<AddOpType>::OpRewritePattern;

    LogicalResult match_and_rewrite(AddOpType addOp,
                                    PatternRewriter &rewriter) const override {
        let canonicalize = [&](Value maybeContraction,
                                Value otherOperand) -> Contract {
        Contract contractionOp =
            dyn_cast_or_null<Contract>(
                maybeContraction.defining_op());
        if (!contractionOp)
            return Contract();
        if (let maybeZero = dyn_cast_or_null<arith::ConstantOp>(
                contractionOp.getAcc().defining_op())) {
            if (maybeZero.value() ==
                rewriter.getZeroAttr(contractionOp.getAcc().get_type())) {
            IRMapping bvm;
            bvm.map(contractionOp.getAcc(), otherOperand);
            let newContraction =
                cast<Contract>(rewriter.clone(*contractionOp, bvm));
            rewriter.replaceOp(addOp, newContraction.output());
            return newContraction;
            }
        }
        return Contract();
        };

        Value a = addOp.input(0), b = addOp.input(1);
        Contract contract = canonicalize(a, b);
        contract = contract ? contract : canonicalize(b, a);
        return contract ? Ok(()) : Err(());
    }
};

//===----------------------------------------------------------------------===//
// ExtractElement
//===----------------------------------------------------------------------===//

void ExtractElement::build(builder: &Builder, result: &OperationState,
                                     Value source) {
    result.add_operands({source});
    result.add_types(source.get_type().cast<VectorType>().element_type);
}

//===----------------------------------------------------------------------===//
// Extract
//===----------------------------------------------------------------------===//

void Extract::build(builder: &Builder, result: &OperationState,
                              Value source, &[i64] position) {
    build(builder, result, source, get_vector_subscript_attr(builder, position));
}

// Convenience builder which assumes the values are constant indices.
void Extract::build(builder: &Builder, result: &OperationState,
                              Value source, ValueRange position) {
    SmallVector<[i64; 4]> position_constants =
        to_vector<4>(map_range(position, [](Value pos) {
            return pos.defining_op<arith::ConstantIndex>().value();
        }));
    build(builder, result, source, position_constants);
}

fn Extract::is_compatible_return_types(TypeRange l, TypeRange r) -> bool {
    // Allow extracting 1-element vectors instead of scalars.
    let isCompatible = [](TypeRange l, TypeRange r) {
        let vector_type = l.front().dyn_cast<VectorType>();
        return vector_type && vector_type.shape.equals({1}) &&
            vector_type.element_type == r.front();
    };
    if (l.len() == 1 && r.len() == 1 &&
        (isCompatible(l, r) || isCompatible(r, l)))
        return true;
    return l == r;
}

pub fn extract_vector<IntType>(array_attr: ArrayAttr) -> SmallVector<IntType> {
  return to_vector<4>(map_range(
      array_attr.as_range<IntegerAttr>(),
      |attr| { return static_cast<IntType>(attr.get_int()); }));
}

/// Fold the result of chains of Extract in place by simply concatenating the
/// positions.
pub fn fold_extract_op_from_extract_chain(extract: Extract) -> LogicalResult {
    if !extract.input.defining_op<Extract>() {
        return Err(());
    }

    let mut global_position = SmallVector<[i64; 4]>::new();
    let mut current_op = extract;
    let extr_pos = extract_vector<i64>(current_op.index);
    global_position.append(extr_pos.rev());
    while (Extract next_op = current_op.input.defining_op<Extract>()) {
        current_op = next_op;
        let extr_pos = extract_vector<i64>(current_op.index);
        global_position.append(extr_pos.rev());
    }
    extract.set_input(current_op.input);
    // Builder is only used as a helper to build an I64ArrayAttr.
    let b = Builder::new(extract.self.context());
    global_position.reverse();
    extract.set_attr("indices",
                        b.get_i64_array_attr(global_position));
    return Ok(());
}

namespace {
/// Fold an Extract that is fed by a chain of InsertOps and TransposeOps.
/// Walk back a chain of Insert/Transpose until we hit a match.
/// Compose Transpose permutations as we walk back.
/// This helper class keeps an updated extraction position `extract_position`
/// with extra trailing sentinels.
/// The sentinels encode the internal transposition status of the result vector.
/// As we iterate, extract_position is permuted and updated.
class ExtractFromInsertTransposeChainState {
public:
  ExtractFromInsertTransposeChainState(Extract e);

  /// Iterate over producing insert and transpose ops until we find a fold.
  Value fold();

private:
  /// Return true if the vector at position `a` is contained within the vector
  /// at position `b`. Under insert/extract semantics, this is the same as `a`
  /// is a prefix of `b`.
  template <typename ContainerA, typename ContainerB>
  bool isContainedWithin(const ContainerA &a, const ContainerB &b) {
    return a.len() <= b.len() &&
           std::equal(a.begin(), a.begin() + a.len(), b.begin());
  }

  /// Return true if the vector at position `a` intersects the vector at
  /// position `b`. Under insert/extract semantics, this is the same as equality
  /// of all entries of `a` that are >=0 with the corresponding entries of b.
  /// Comparison is on the common prefix (i.e. zip).
  template <typename ContainerA, typename ContainerB>
  bool intersectsWhereNonNegative(const ContainerA &a, const ContainerB &b) {
    for (let [elemA, elemB] : zip(a, b)) {
      if (elemA < 0 || elemB < 0)
        continue;
      if (elemA != elemB)
        return false;
    }
    return true;
  }

  /// Folding is only possible in the absence of an internal permutation in the
  /// result vector.
  bool can_fold() {
    return (sentinels == ArrayRef(extract_position).drop_front(extracted_rank));
  }

  // Helper to get the next defining op of interest.
  void update_state_for_next_iteration(Value v) {
    next_insert_op = v.defining_op<Insert>();
    next_transpose_op = v.defining_op<Transpose>();
  };

  // Case 1. If we hit a transpose, just compose the map and iterate.
  // Invariant: insert + transpose do not change rank, we can always compose.
  LogicalResult handleTransposeOp();

  // Case 2: the insert position matches extract_position exactly, early return.
  LogicalResult handleInsertOpWithMatchingPos(Value &res);

  /**
Case 3: if the insert position is a prefix of extract_position, extract a
portion of the source of the insert.
Example:
```
%ins = vector.insert %source, %vest[1]: vector<3x4> into vector<2x3x4x5>
// extract_position == [1, 2, 3]
%ext = vector.extract %ins[1, 0]: vector<3x4x5>
// can fold to vector.extract %source[0, 3]
%ext = vector.extract %source[3]: vector<5x6>
```
To traverse through %source, we need to set the leading dims to 0 and
drop the extra leading dims.
This method updates the internal state.
*/
  LogicalResult handleInsertOpWithPrefixPos(Value &res);

  /// Try to fold in place to extract(source, extract_position) and return the
  /// folded result. Return null if folding is not possible (e.g. due to an
  /// internal tranposition in the result).
  Value try_to_fold_extract_op_in_place(Value source);

  extract: Extract;
  i64 vectorRank;
  i64 extracted_rank;

  Insert next_insert_op;
  Transpose next_transpose_op;

  /// Sentinel values that encode the internal permutation status of the result.
  /// They are set to (-1, ... , -k) at the beginning and appended to
  /// `extract_position`.
  /// In the end, the tail of `extract_position` must be exactly `sentinels` to
  /// ensure that there is no internal transposition.
  /// Internal transposition cannot be accounted for with a folding pattern.
  // TODO: We could relax the internal transposition with an extra transposition
  // operation in a future canonicalizer.
  SmallVector<[i64]> sentinels;
  SmallVector<[i64]> extract_position;
};
} // namespace

ExtractFromInsertTransposeChainState::ExtractFromInsertTransposeChainState(
    Extract e)
    : extract(e), vectorRank(extract.self.input.rank()),
      extracted_rank(extract.index.len()) {
  assert!(vectorRank >= extracted_rank, "extracted pos overflow");
  sentinels.reserve(vectorRank - extracted_rank);
  for (i64 i = 0, e = vectorRank - extracted_rank; i < e; ++i)
    sentinels.push(-(i + 1));
  extract_position = extract_vector<i64>(extract.index);
  append_range(extract_position, sentinels);
}

// Case 1. If we hit a transpose, just compose the map and iterate.
// Invariant: insert + transpose do not change rank, we can always compose.
LogicalResult ExtractFromInsertTransposeChainState::handleTransposeOp() {
  if (!next_transpose_op)
    return Err(());
  let permutation = extract_vector<usize>(next_transpose_op.self.transp);
  AffineMap m = inverse_permutation(
      AffineMap::self.permutation_mapermutation, extract.self.context()));
  extract_position = applyPermutationMap(m, ArrayRef(extract_position));
  return Ok(());
}

// Case 2: the insert position matches extract_position exactly, early return.
LogicalResult
ExtractFromInsertTransposeChainState::handleInsertOpWithMatchingPos(
    Value &res) {
  let insertedPos = extract_vector<i64>(next_insert_op.index);
  if (ArrayRef(insertedPos) !=
      ArrayRef(extract_position).take_front(extracted_rank))
    return Err(());
  // Case 2.a. early-exit fold.
  res = next_insert_op.input;
  // Case 2.b. if internal transposition is present, can_fold will be false.
  return success(can_fold());
}

/// Case 3: if inserted position is a prefix of extract_position,
/// extract a portion of the source of the insertion.
/// This method updates the internal state.
LogicalResult
ExtractFromInsertTransposeChainState::handleInsertOpWithPrefixPos(Value &res) {
  let insertedPos = extract_vector<i64>(next_insert_op.index);
  if (!isContainedWithin(insertedPos, extract_position))
    return Err(());
  // Set leading dims to zero.
  std::fill_n(extract_position.begin(), insertedPos.len(), 0);
  // Drop extra leading dims.
  extract_position.erase(extract_position.begin(),
                        extract_position.begin() + insertedPos.len());
  extracted_rank = extract_position.len() - sentinels.len();
  // Case 3.a. early-exit fold (break and delegate to post-while path).
  res = next_insert_op.input;
  // Case 3.b. if internal transposition is present, can_fold will be false.
  return Ok(());
}

/**
Try to fold in place to extract(source, extract_position) and return the
folded result. Return null if folding is not possible (e.g. due to an
internal tranposition in the result).
*/
Value ExtractFromInsertTransposeChainState::try_to_fold_extract_op_in_place(
    Value source) {
  // If we can't fold (either internal transposition, or nothing to fold), bail.
  bool nothingToFold = (source == extract.input);
  if (nothingToFold || !can_fold())
    return Value();
  // Otherwise, fold by updating the op inplace and return its result.
  let b = Builder::new(extract.self.context());
  extract.set_attr(
      extract.getPositionAttrName(),
      b.get_i64_array_attr(ArrayRef(extract_position).take_front(extracted_rank)));
  extract.get_vector_mutable().assign(source);
  return extract.output();
}

/// Iterate over producing insert and transpose ops until we find a fold.
Value ExtractFromInsertTransposeChainState::fold() {
  Value valueToExtractFrom = extract.input;
  update_state_for_next_iteration(valueToExtractFrom);
  while (next_insert_op || next_transpose_op) {
    // Case 1. If we hit a transpose, just compose the map and iterate.
    // Invariant: insert + transpose do not change rank, we can always compose.
    if succeeded(handleTransposeOp()) {
      valueToExtractFrom = next_transpose_op.input;
      update_state_for_next_iteration(valueToExtractFrom);
      continue;
    }

    Value result;
    // Case 2: the position match exactly.
    if (succeeded(handleInsertOpWithMatchingPos(result)))
      return result;

    // Case 3: if the inserted position is a prefix of extract_position, we can
    // just extract a portion of the source of the insert.
    if (succeeded(handleInsertOpWithPrefixPos(result)))
      return try_to_fold_extract_op_in_place(result);

    // Case 4: extractPositionRef intersects insertedPosRef on non-sentinel
    // values. This is a more difficult case and we bail.
    let insertedPos = extract_vector<i64>(next_insert_op.index);
    if (isContainedWithin(extract_position, insertedPos) ||
        intersectsWhereNonNegative(extract_position, insertedPos))
      return Value();

    // Case 5: No intersection, we forward the extract to insert_op.dest().
    valueToExtractFrom = next_insert_op.dest;
    update_state_for_next_iteration(valueToExtractFrom);
  }
  // If after all this we can fold, go for it.
  return try_to_fold_extract_op_in_place(valueToExtractFrom);
}

/// Fold extract with scalar result coming from Broadcast or Splat.
pub fn fold_extract_from_broadcast(extract: Extract) -> Value {
    let def_op = extract.input.defining_op();
    if (!def_op || !isa<Broadcast, Splat>(def_op)) {
        return Value();
    }
    let source = def_op.input(0);
    if extract.get_type() == source.get_type() {
        return source;
    }
    let rank = |r#type: Type| if r#type.isa<VectorType>() {
        r#type.cast<VectorType>().rank()
    } else { 0 };
    // If splat or broadcast from a scalar, just return the source scalar.
    let broadcast_src_rank = rank(source.get_type());
    if (broadcast_src_rank == 0) {
        return source;
    }

    let extract_result_rank = rank(extract.get_type());
    if (extract_result_rank >= broadcast_src_rank) {
        return Value();
    }
    // Check that the dimension of the result haven't been broadcasted.
    let extract_vec_type = extract.get_type().dyn_cast<VectorType>();
    let broadcast_vec_type = source.get_type().dyn_cast<VectorType>();
    if (extract_vec_type && broadcast_vec_type &&
        extract_vec_type.shape !=
            broadcast_vec_type.shape.take_back(extract_result_rank))
    {
        return Value();
    }

    let broadcast_op = cast<Broadcast>(def_op);
    let rank_diff = broadcast_src_rank - extract_result_rank;
    /*
    Detect all the positions that come from "dim-1" broadcasting.
    These dimensions correspond to "dim-1" broadcasted dims; set the mathching
    extract position to `0` when extracting from the source operand.
    */
    let broadcasted_unit_dims =
        broadcast_op.compute_broadcasted_unit_dims();
    let extract_pos = extract_vector<i64>(extract.index);
    for i in rank_diff..extract_pos.len() {
        if (broadcasted_unit_dims.contains(i)) {
            extract_pos[i] = 0;
        }
    }
    /*
    `rank_diff` leading dimensions correspond to new broadcasted dims, drop the
    matching extract position when extracting from the source operand.
    */
    extract_pos.erase(
        extract_pos.begin(),
        std::next(extract_pos.begin(), extract_pos.len() - rank_diff));
    // Builder is only used as a helper to build an I64ArrayAttr.
    let b = Builder::new(extract.self.context());
    extract.set_input(source);
    extract.set_attr("indices",
                        b.get_i64_array_attr(extract_pos));
    extract.output()
}

// Fold extract with source coming from ShapeCast op.
pub fn fold_extract_from_shape_cast(extract: Extract) -> Value {
    let shape_cast_op = extract.input.defining_op<ShapeCast>();
    if !shape_cast_op {
        return Value();
    }
    // Get the nth dimension len starting from lowest dimension.
    let get_dim_reverse
        = |r#type, VectorType, n: i64| r#type.shape.take_back(n + 1).front();
    let destination_rank = if extract.get_type().isa<VectorType>() { 
            extract.get_type().cast<VectorType>().rank()
        } else { 0 };
    if destination_rank > shape_cast_op.input.rank() {
        return Value();
    }
    if destination_rank > 0 {
        let destination_type = extract.output().get_type().cast<VectorType>();
        for i in 0..destination_rank {
            /*
            The lowest dimension of of the destination must match the lowest
            dimension of the shapecast op source.
            TODO: This case could be support in a canonicalization pattern.
            */
            if (get_dim_reverse(shape_cast_op.self.input, i) !=
                get_dim_reverse(destination_type, i))
            {
                return Value();
            }
        }
    }
    // Extract the strides associated with the extract op vector source. Then use
    // this to calculate a linearized position for the extract.
    let mut extracted_pos = extract_vector<i64>(extract.index);
    extracted_pos.reverse();
    let mut strides = SmallVector<[i64; 4]>::new();
    let mut stride = 1;
    for i in 0..extracted_pos.len() {
        strides.push(stride);
        stride *= get_dim_reverse(extract.self.input, i + destination_rank);
    }

    let position = linearize(extracted_pos, strides);
    // Then extract the strides associated to the shapeCast op vector source and
    // delinearize the position using those strides.
    let mut new_strides = SmallVector<[i64; 4]>::new();
    let num_dimension =
        shape_cast_op.self.input.rank() - destination_rank;
    stride = 1;
    for i in 0..num_dimension {
        new_strides.push(stride);
        stride *=
            get_dim_reverse(shape_cast_op.self.input, i + destination_rank);
    }
    new_strides.reverse();
    let new_position = delinearize(new_strides, position);
    // Builder is only used as a helper to build an I64ArrayAttr.
    let b = Builder::new(extract.self.context());
    extract.set_attr("indices",
                        b.get_i64_array_attr(new_position));
    extract.set_input(shape_cast_op.input);
    extract.output
}

/// Fold an Extract from ExtractStridedSlice.
pub fn fold_extract_from_extract_strided(extract: Extract) -> Value {
    let extract_strided_slice =
        extract.input.defining_op<ExtractStridedSlice>();
    if !extract_strided_slice {
            return Value();
        }
    // Return if 'extract_strided_slice' has non-unit strides.
    if (extract_strided_slice.hasNonUnitStrides())
        return Value();

    // Trim offsets for dimensions fully extracted.
    let slice_offsets =
        extract_vector<i64>(extract_strided_slice.offsets);
    while (!slice_offsets.is_empty()) {
        usize lastOffset = slice_offsets.len() - 1;
        if (slice_offsets.back() != 0 ||
            extract_strided_slice.get_type().dim_size(lastOffset) !=
                extract_strided_slice.self.input.dim_size(lastOffset))
        break;
        slice_offsets.pop_back();
    }
    let mut destination_rank = 0;
    if (let vec_type = extract.get_type().dyn_cast<VectorType>())
        destination_rank = vec_type.rank();
    // The dimensions of the result need to be untouched by the
    // extractStridedSlice op.
    if destination_rank >
        extract_strided_slice.input.rank() - slice_offsets.len()
    {
        return Value();
    }
    let extracted_pos = extract_vector<i64>(extract.index);
    assert!(extracted_pos.len() >= slice_offsets.len());
    for (usize i = 0, e = slice_offsets.len(); i < e; i++)
        extracted_pos[i] = extracted_pos[i] + slice_offsets[i];
    extract.get_vector_mutable().assign(extract_strided_slice.input);
    // Builder is only used as a helper to build an I64ArrayAttr.
    let b = Builder::new(extract.context());
    extract.set_attr("indices",
                        b.get_i64_array_attr(extracted_pos));
    extract.output()
}

/// Fold extract_op fed from a chain of insertStridedSlice ops.
pub fn fold_extract_strided_op_from_insert_chain(op: Extract) -> Value {
    let destination_rank = if op.get_type().isa<VectorType>() {
        op.get_type().cast<VectorType>().rank()
    } else {
        0
    };
    let insert_op = op.input.defining_op<InsertStridedSlice>();
    while insert_op {
        let insert_rank_diff
            = insert_op.dest.rank() - insert_op.input.rank();
        if destination_rank > insert_op.input.rank() {
            return Value();
        }
        let insert_offsets = extract_vector<i64>(insert_op.offsets);
        let extract_offsets = extract_vector<i64>(op.index);

        if insert_op.strides.any(
            |attr| attr.cast<IntegerAttr>().get_int() != 1)
        {
            return Value();
        }
        let mut disjoint = false;
        let mut offset_diffs = SmallVector<[i64; 4]>::new();
        for dim in 0..extract_offsets.len() {
            let start = insert_offsets[dim];
            let len = if dim < insert_rank_diff { 1 } else {
                insert_op.self.input.dim_size(dim - insert_rank_diff)
            };
            let end = start + len;
            let offset = extract_offsets[dim];
            // Check if the start of the extract offset is in the interval inserted.
            if start <= offset && offset < end {
                if dim >= insert_rank_diff {
                    offset_diffs.push(offset - start);
                }
                continue;
            }
            disjoint = true;
            break;
        }
        // The extract element chunk overlap with the vector inserted.
        if !disjoint {
            // If any of the inner dimensions are only partially inserted we have a
            // partial overlap.
            let src_rank_diff =
                insert_op.self.input.rank() - destination_rank;
            for i in 0..destination_rank {
                if insert_op.self.input.dim_size(i + src_rank_diff) !=
                    insert_op.dest.dim_size(i + src_rank_diff +
                                                            insert_rank_diff) {
                    return Value();
                }
            }
            op.get_vector_mutable().assign(insert_op.input);
            // Builder is only used as a helper to build an I64ArrayAttr.
            let b = Builder::new(op.self.context());
            op.set_attr("indices",
                        b.get_i64_array_attr(offset_diffs));
            return op.output();
        }
        // If the chunk extracted is disjoint from the chunk inserted, keep
        // looking in the insert chain.
        insert_op = insert_op.dest.defining_op<InsertStridedSlice>();
    }
    return Value();
}

namespace {

// Pattern to rewrite a Extract(Broadcast) -> Broadcast.
class ExtractOpFromBroadcast final : public OpRewritePattern<Extract> {
public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult match_and_rewrite(extract: Extract,
                                    PatternRewriter &rewriter) const override {
        Operation *def_op = extract.input.defining_op();
        if (!def_op || !isa<Broadcast, Splat>(def_op))
        return Err(());

        Value source = def_op.input(0);
        if (extract.get_type() == source.get_type())
        return Err(());
        let rank = [](r#type: Type) {
        return type.isa<VectorType>() ? type.cast<VectorType>().rank() : 0;
        };
        usize broadcast_src_rank = rank(source.get_type());
        usize extract_result_rank = rank(extract.get_type());
        // We only consider the case where the rank of the source is less than or
        // equal to the rank of the extract dst. The other cases are handled in the
        // folding patterns.
        if (extract_result_rank < broadcast_src_rank)
        return Err(());
        rewriter.replace_op_with_new_op<Broadcast>(
            extract, extract.get_type(), source);
        return Ok(());
    }
}

// Pattern to rewrite a Extract(splat ConstantOp) -> ConstantOp.
class ExtractOpSplatConstantFolder final : public OpRewritePattern<Extract> {
public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult match_and_rewrite(extract: Extract,
                                    PatternRewriter &rewriter) const override {
        // Return if 'Extract' operand is not defined by a splat vector
        // ConstantOp.
        Value sourceVector = extract.input;
        Attribute vectorCst;
        if (!match_pattern(sourceVector, m_Constant(&vectorCst)))
        return Err(());
        let splat = vectorCst.dyn_cast<SplatElementsAttr>();
        if (!splat)
        return Err(());
        Attribute new_attr = splat.getSplatValue<Attribute>();
        if (let vecDstType = extract.get_type().dyn_cast<VectorType>())
        new_attr = DenseElementsAttr::get(vecDstType, new_attr);
        rewriter.replace_op_with_new_op<arith::ConstantOp>(extract, new_attr);
        return Ok(());
    }
}

// Pattern to rewrite a Extract(non-splat ConstantOp)[...] -> ConstantOp.
class ExtractOpNonSplatConstantFolder final
    : public OpRewritePattern<Extract> {
public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult match_and_rewrite(extract: Extract,
                                    PatternRewriter &rewriter) const override {
        // Return if 'Extract' operand is not defined by a compatible vector
        // ConstantOp.
        Value sourceVector = extract.input;
        Attribute vectorCst;
        if (!match_pattern(sourceVector, m_Constant(&vectorCst)))
        return Err(());

        let vecTy = sourceVector.get_type().cast<VectorType>();
        if (vecTy.is_scalable())
        return Err(());

        // The splat case is handled by `ExtractOpSplatConstantFolder`.
        let dense = vectorCst.dyn_cast<DenseElementsAttr>();
        if (!dense || dense.isSplat())
        return Err(());

        // Calculate the linearized position of the continuous chunk of elements to
        // extract.
        SmallVector<[i64]> completePositions(vecTy.rank(), 0);
        copy(get_i64_sub_array(extract.index), completePositions.begin());
        i64 elemBeginPosition =
            linearize(completePositions, compute_strides(vecTy.shape));
        let denseValuesBegin = dense.value_begin<Attribute>() + elemBeginPosition;

        Attribute new_attr;
        if let resVecTy = extract.get_type().dyn_cast<VectorType>() {
        SmallVector<Attribute> elementValues(
            denseValuesBegin, denseValuesBegin + resVecTy.num_elements());
        new_attr = DenseElementsAttr::get(resVecTy, elementValues);
        } else {
        new_attr = *denseValuesBegin;
        }

        rewriter.replace_op_with_new_op<arith::ConstantOp>(extract, new_attr);
        return Ok(());
    }
}

} // namespace

pub fn populate_from_int64_attr_array(
    array_attr: ArrayAttr,
    results: &SmallVector<[i64]>
) {
    for attr in array_attr {
        results.push(attr.cast<IntegerAttr>().get_int());
    }
}

//===----------------------------------------------------------------------===//
// Fma
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Broadcast
//===----------------------------------------------------------------------===//

/**
Return the dimensions of the result vector that were formerly ones in the
source tensor and thus correspond to "dim-1" broadcasting.
*/
pub fn compute_broadcasted_unit_dims(
    src_shape: &[i64],
    dst_shape: &[i64]
) -> SetVector<i64> {
    let rank_diff = dst_shape.len() - src_shape.len();
    let dst_dim = rank_diff;
    SetVector<i64> res;
    for (s1, s2) in
        zip_equal(src_shape, dst_shape.drop_front(rank_diff))
    {
        if s1 != s2 {
            assert!(s1 == 1, "Expected dim-1 broadcasting");
            res.insert(dst_dim);
        }
        dst_dim += 1;
    }
    res
}

SetVector<i64> Broadcast::compute_broadcasted_unit_dims() {
    // Scalar broadcast is without any unit dim broadcast.
    let src_vector_type = self.input.dyn_cast<VectorType>();
    if !src_vector_type {
        return {};
    }
    return ::compute_broadcasted_unit_dims(src_vector_type.shape,
                                        self.input.shape);
}

/**
Broadcast `value` to a vector of `dst_shape`, knowing that exactly the
`broadcasted_dims` dimensions in the dst_shape are broadcasted.
This requires (and asserts) that the broadcast is free of dim-1
broadcasting.
Since vector.broadcast only allows expanding leading dimensions, an extra
vector.transpose may be inserted to make the broadcast possible.
`value`, `dst_shape` and `broadcasted_dims` must be properly specified or
the helper will assert. This means:
  1. `dst_shape` must not be empty.
  2. `broadcasted_dims` must be confined to [0 .. rank(value.self.input
  2. `dst_shape` trimmed of the dimensions specified in `broadcasted_dims`
*/
//       must match the `value` shape.
fn Broadcast::create_or_fold_broadcast_op(
    Builder &b, Value value, &[i64] dst_shape,
    const SetVector<i64> &broadcasted_dims
) -> Value {
    assert!(!dst_shape.is_empty(), "Unexpected empty dst shape");

    // Well-formedness check.
    SmallVector<[i64]> check_shape;
    for i in 0..dst_shape.len() {
        if (broadcasted_dims.contains(i)) {
            continue;
        }
        check_shape.push(dst_shape[i]);
    }
    assert!(
        broadcasted_dims.len() == dst_shape.len() - check_shape.len(),
            "Ill-formed broadcasted_dims contains values not confined to "
            "dest_vector_shape");

    let location = value.location();
    let element_type = get_element_type_or_self(value.get_type());
    let src_vector_type = value.get_type().dyn_cast<VectorType>();
    let dst_vector_type = VectorType::new(dst_shape, element_type);

    // Step 2. If scalar -> dst_shape broadcast, just do it.
    if !src_vector_type {
        assert!(check_shape.is_empty(),
            "Ill-formed create_or_fold_broadcast_op arguments");
        return b.create_or_fold<Broadcast>(loc, dst_vector_type, value);
    }

    assert!(src_vector_type.shape.equals(check_shape),
            "Ill-formed create_or_fold_broadcast_op arguments");

    /*
    Step 3. Since vector.broadcast only allows creating leading dims,
      vector -> dst_shape broadcast may require a transpose.
    Traverse the dims in order and construct:
      1. The leading entries of the broadcast_shape that is guaranteed to be
         achievable by a simple broadcast.
      2. The induced permutation for the subsequent vector.transpose that will
         bring us from `broadcast_shape` back to he desired `dst_shape`.
    If the induced permutation is not the identity, create a vector.transpose.
    */
    SmallVector<[i64]> broadcast_shape, permutation(dst_shape.len(), -1);
    broadcast_shape.reserve(dst_shape.len());
    /*
    Consider the example:
      src_shape     = 2x4
      dst_shape     = 1x2x3x4x5
      broadcasted_dims = [0, 2, 4]

    We want to build:
      broadcast_shape  = 1x3x5x2x4
      permutation     = [0, 2, 4,                 1, 3]
                         ---V---           -----V-----
               leading broadcast part      src shape part

    Note that the trailing dims of broadcast_shape are exactly the src_shape
    by construction.
    next_src_shape_dim is used to keep track of where in the permutation the
    "src shape part" occurs.
    */
    let next_src_shape_dim = broadcasted_dims.len();
    for i in 0..dst_shape.len() {
        if broadcasted_dims.contains(i) {
            /*
            3.a. For each dim in the dst shape, if it is a broadcasted dim,
            bring it to the head of the broadcast_shape.
            It will need to be permuted back from `broadcast_shape.len() - 1` into
            position `i`.
            */
            broadcast_shape.push(dst_shape[i]);
            permutation[i] = broadcast_shape.len() - 1;
        } else {
            /*
            3.b. Otherwise, the dim is not broadcasted, it comes from the src
            shape and needs to be permuted into position `i`.
            Don't touch `broadcast_shape` here, the whole src_shape will be
            appended after.
            */
            permutation[i] = next_src_shape_dim;
            next_src_shape_dim += 1;
        }
    }
    // 3.c. Append the src_shape.
    append_range(broadcast_shape, src_vector_type.shape);

    // Ensure there are no dim-1 broadcasts.
    assert!(
        compute_broadcasted_unit_dims(
            src_vector_type.shape, broadcast_shape)
                .is_empty(),
            "Unexpected dim-1 broadcast");

    let broadcast_type = VectorType::new(broadcast_shape, element_type);
    assert!(is_broadcastable_to(value.get_type(), broadcast_type) ==
                BroadcastableToResult::Success,
            "Must be broadcastable");
    let res = b.create_or_fold<Broadcast>(loc, broadcast_type, value);
    // Step 4. If we find any dimension that indeed needs to be permuted,
    // immediately return a new vector.transpose.
    for i in 0..permutation.len() {
        if permutation[i] != i {
            return b.create_or_fold<Transpose>(loc, res, permutation);
        }
    }
    // Otherwise return res.
    res
}

pub fn is_broadcastable_to(
    src_type: Type,
    dst_vector_type: VectorType,
    mismatching_dims: *mut (int, int)
) -> BroadcastableToResult {
    // Broadcast scalar to vector of the same element type.
    if src_type.is_int_or_index_or_float()
    && dst_vector_type
    && get_element_type_or_self(src_type)
        == get_element_type_or_self(dst_vector_type)
    {
        return BroadcastableToResult::Success;
    }
    // From now on, only vectors broadcast.
    let src_vector_type = src_type.dyn_cast<VectorType>();
    if !src_vector_type {
        return BroadcastableToResult::SourceTypeNotAVector;
    }

    let src_rank = src_vector_type.rank();
    let dst_rank = dst_vector_type.rank();
    if (src_rank > dst_rank) {
        return BroadcastableToResult::SourceRankHigher;
    }
    // Source has an exact match or singleton value for all trailing dimensions
    // (all leading dimensions are simply duplicated).
    let lead = dst_rank - src_rank;
    for r in 0..src_rank {
        let src_dim = src_vector_type.dim_size(r);
        let dst_dim = dst_vector_type.dim_size(lead + r);
        if src_dim != 1 && src_dim != dst_dim {
            if mismatching_dims {
                mismatching_dims.first = src_dim;
                mismatching_dims.second = dst_dim;
            }
            return BroadcastableToResult::DimensionMismatch;
        }
    }

    BroadcastableToResult::Success
}

namespace {

// Fold broadcast1(broadcast2(x)) into broadcast1(x).
struct BroadcastFolder : public OpRewritePattern<Broadcast> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(Broadcast broadcast_op,
                                PatternRewriter &rewriter) const override {
    let srcBroadcast = broadcast_op.input.defining_op<Broadcast>();
    if (!srcBroadcast)
      return Err(());
    rewriter.replace_op_with_new_op<Broadcast>(
        broadcast_op, broadcast_op.self.input, srcBroadcast.input);
    return Ok(());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Shuffle
//===----------------------------------------------------------------------===//

void Shuffle::build(builder: &Builder, result: &OperationState, Value v1,
                      Value v2, &[i64] mask) {
  build(builder, result, v1, v2, get_vector_subscript_attr(builder, mask));
}


pub fn is_step_index_array(
    idx_arr: ArrayAttr, begin: u64, width: usize
) -> bool {
    u64 expected = begin;
    return idx_arr.len() == width &&
            all_of(idx_arr.as_value_range<IntegerAttr>(),
                        |attr| attr.getZExtValue() == expected++);
}

namespace {

// Pattern to rewrite a 0-D shuffle with [0] or [1] mask returning a 1-D vector
// to a broadcast.
struct Canonicalize0DShuffle : public OpRewritePattern<Shuffle> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult match_and_rewrite(Shuffle shuffle_op,
                                    PatternRewriter &rewriter) const override {
        VectorType v1VectorType = shuffle_op.getV1VectorType();
        ArrayAttr mask = shuffle_op.mask;
        if (v1VectorType.rank() > 0){
        return Err(());}
        if (mask.len() != 1){
        return Err(());}
        let res_type = VectorType::Builder(v1VectorType).setShape({1});
        if (mask[0].cast<IntegerAttr>().get_int() == 0){
        rewriter.replace_op_with_new_op<Broadcast>(shuffle_op, res_type,
                                                        shuffle_op.lhs);}
        else{
        rewriter.replace_op_with_new_op<Broadcast>(shuffle_op, res_type,
                                                        shuffle_op.rhs);}
        return Ok(());
    }
}

/// Pattern to rewrite a Shuffle(Splat, Splat) to Splat.
class ShuffleSplat final : public OpRewritePattern<Shuffle> {
public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult match_and_rewrite(Shuffle op,
                                    PatternRewriter &rewriter) const override {
        let v1Splat = op.lhs.defining_op<Splat>();
        let v2Splat = op.rhs.defining_op<Splat>();

        if (!v1Splat || !v2Splat)
        return Err(());

        if (v1Splat.getInput() != v2Splat.getInput())
        return Err(());

        rewriter.replace_op_with_new_op<Splat>(op, op.get_type(), v1Splat.getInput());
        return Ok(());
    }
}

} // namespace

//===----------------------------------------------------------------------===//
// InsertElement
//===----------------------------------------------------------------------===//

void InsertElement::build(builder: &Builder, result: &OperationState,
                            Value source, Value dest) {
  build(builder, result, source, dest, {});
}

//===----------------------------------------------------------------------===//
// Insert
//===----------------------------------------------------------------------===//

void Insert::build(builder: &Builder, result: &OperationState, Value source,
                     Value dest, &[i64] position) {
    result.add_operands({source, dest});
    let position_attr = get_vector_subscript_attr(builder, position);
    result.add_types(dest.get_type());
    result.add_attribute(get_position_attr_str_name(), position_attr);
}

// Convenience builder which assumes the values are constant indices.
void Insert::build(builder: &Builder, result: &OperationState, Value source,
                     Value dest, ValueRange position) {
    SmallVector<[i64; 4]> position_constants =
        to_vector<4>(map_range(position, |pos| {
            return pos.defining_op<arith::ConstantIndex>().value();
        }));
    build(builder, result, source, dest, position_constants);
}

namespace {

// If insert_op is only inserting unit dimensions it can be transformed to a
// broadcast.
class InsertToBroadcast final : public OpRewritePattern<Insert> {
public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult match_and_rewrite(Insert insert_op,
                                    PatternRewriter &rewriter) const override {
        let srcVecType = insert_op.self.input.dyn_cast<VectorType>();
        if (!srcVecType || insert_op.dest.num_elements() !=
                            srcVecType.num_elements())
        return Err(());
        rewriter.replace_op_with_new_op<Broadcast>(
            insert_op, insert_op.dest, insert_op.input);
        return Ok(());
    }
}

/// Pattern to rewrite a Insert(Splat, Splat) to Splat.
class InsertSplatToSplat final : public OpRewritePattern<Insert> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(Insert op,
                                PatternRewriter &rewriter) const override {
    let srcSplat = op.input.defining_op<Splat>();
    let dstSplat = op.dest.defining_op<Splat>();

    if (!srcSplat || !dstSplat){
      return Err(());}

    if (srcSplat.getInput() != dstSplat.getInput())
      return Err(());

    rewriter.replace_op_with_new_op<Splat>(op, op.get_type(), srcSplat.getInput());
    return Ok(());
  }
};

// Pattern to rewrite a Insert(ConstantOp into ConstantOp) -> ConstantOp.
class InsertOpConstantFolder final : public OpRewritePattern<Insert> {
public:
  using OpRewritePattern::OpRewritePattern;

  // Do not create constants with more than `vectorSizeFoldThreashold` elements,
  // unless the source vector constant has a single use.
  static constexpr i64 vectorSizeFoldThreshold = 256;

  LogicalResult match_and_rewrite(Insert op,
                                PatternRewriter &rewriter) const override {
    // Return if 'Insert' operand is not defined by a compatible vector
    // ConstantOp.
    TypedValue<VectorType> dest_vector = op.dest;
    Attribute vector_dest_cst;
    if (!match_pattern(dest_vector, m_Constant(&vector_dest_cst)))
      return Err(());

    VectorType dest_ty = dest_vector.get_type();
    if (dest_ty.is_scalable())
      return Err(());

    // Make sure we do not create too many large constants.
    if (dest_ty.num_elements() > vectorSizeFoldThreshold &&
        !dest_vector.hasOneUse())
      return Err(());

    let denseDest = vector_dest_cst.cast<DenseElementsAttr>();

    Value source_value = op.input;
    Attribute sourceCst;
    if (!match_pattern(source_value, m_Constant(&sourceCst)))
      return Err(());

    // Calculate the linearized position of the continuous chunk of elements to
    // insert.
    SmallVector<[i64]> completePositions(dest_ty.rank(), 0);
    copy(get_i64_sub_array(op.index), completePositions.begin());
    i64 insertBeginPosition =
        linearize(completePositions, compute_strides(dest_ty.shape));

    SmallVector<Attribute> insertedValues;
    if (let denseSource = sourceCst.dyn_cast<DenseElementsAttr>())
      append_range(insertedValues, denseSource.getValues<Attribute>());
    else
      insertedValues.push(sourceCst);

    let allValues = to_vector(denseDest.getValues<Attribute>());
    copy(insertedValues, allValues.begin() + insertBeginPosition);
    let new_attr = DenseElementsAttr::get(dest_ty, allValues);

    rewriter.replace_op_with_new_op<arith::ConstantOp>(op, new_attr);
    return Ok(());
  }
};

} // namespace


//===----------------------------------------------------------------------===//
// InsertStridedSlice
//===----------------------------------------------------------------------===//

void InsertStridedSlice::build(builder: &Builder, result: &OperationState,
                                 Value source, Value dest,
                                 &[i64] offsets,
                                 &[i64] strides) {
    result.add_operands({source, dest});
    let offsetsAttr = get_vector_subscript_attr(builder, offsets);
    let stridesAttr = get_vector_subscript_attr(builder, strides);
    result.add_types(dest.get_type());
    result.add_attribute(get_offsets_attr_str_name(), offsetsAttr);
    result.add_attribute(get_strides_attr_str_name(), stridesAttr);
}

// TODO: Should be moved to Tablegen ConfinedAttr attributes.
template <typename OpType>
pub fn is_integer_array_attr_smaller_than_shape(
    op: OpType,
    array_attr: ArrayAttr,
    shape: &[i64],
    attr_name: &str
) -> LogicalResult {
    if (array_attr.len() > shape.len()) {
        return op.emit_op_error(
            "Expected {} attribute of rank smaller than vector rank.", attr_name
        );
    }
    Ok(())
}

// Returns true if all integers in `array_attr` are in the half-open [min, max}
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
template <typename OpType>
static LogicalResult
is_integer_array_attr_confined_to_range(OpType op, ArrayAttr array_attr, i64 min,
                                  i64 max, &str attr_name,
                                  bool halfOpen = true) {
  for (let attr : array_attr) {
    let val = attr.cast<IntegerAttr>().get_int();
    let upper = max;
    if (!halfOpen)
      upper += 1;
    if (val < min || val >= upper)
      return op.emit_op_error("Expected ") << attr_name << " to be confined to ["
                                         << min << ", " << upper << ")";
  }
  return Ok(());
}

// Returns true if all integers in `array_attr` are in the half-open [min, max}
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
template <typename OpType>
static LogicalResult
pub fn is_integer_array_attr_confined_to_shape(
    OpType op, ArrayAttr array_attr,
    &[i64] shape, &str attr_name,
    bool halfOpen = true, i64 min = 0) {
  for (let [index, attr_dim_pair] :
       enumerate(zip_first(array_attr, shape))) {
    i64 val =
        std::get<0>(attr_dim_pair).template cast<IntegerAttr>().get_int();
    i64 max = std::get<1>(attr_dim_pair);
    if (!halfOpen)
      max += 1;
    if (val < min || val >= max)
      return op.emit_op_error("Expected ")
             << attr_name << " dimension " << index << " to be confined to ["
             << min << ", " << max << ")";
  }
  return Ok(());
}

// Returns true if all integers in `array_attr` are in the interval [min, max}.
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
template <typename OpType>
pub fn is_sum_of_integer_array_attr_confined_to_shape(
    OpType op, ArrayAttr arrayAttr1, ArrayAttr arrayAttr2,
    &[i64] shape, &str attrName1, &str attrName2,
    bool halfOpen = true, i64 min = 1
) -> LogicalResult {
  assert!(arrayAttr1.len() <= shape.len());
  assert!(arrayAttr2.len() <= shape.len());
  for (let [index, it] :
       enumerate(zip(arrayAttr1, arrayAttr2, shape))) {
    let val1 = std::get<0>(it).template cast<IntegerAttr>().get_int();
    let val2 = std::get<1>(it).template cast<IntegerAttr>().get_int();
    i64 max = std::get<2>(it);
    if (!halfOpen)
      max += 1;
    if (val1 + val2 < 0 || val1 + val2 >= max)
      return op.emit_op_error("Expected sum(")
             << attrName1 << ", " << attrName2 << ") dimension " << index
             << " to be confined to [" << min << ", " << max << ")";
  }
  return Ok(());
}

static ArrayAttr make_i64_array_attr(&[i64] values,
                                  context: *mut MLIRContext) {
  let attrs = map_range(values, [context](i64 v) -> Attribute {
    return IntegerAttr::get(IntegerType::new(context, 64), APInt(64, v));
  });
  return ArrayAttr::get(context, to_vector<8>(attrs));
}

namespace {
/// Pattern to rewrite an InsertStridedSlice(Splat(X):src_type,
/// Splat(X):dst_type) to Splat(X):dst_type.
class FoldInsertStridedSliceSplat final
    : public OpRewritePattern<InsertStridedSlice> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(InsertStridedSlice insert_strided_slice_op,
                                PatternRewriter &rewriter) const override {
    let srcSplatOp =
        insert_strided_slice_op.input.defining_op<Splat>();
    let destSplatOp =
        insert_strided_slice_op.dest.defining_op<Splat>();

    if (!srcSplatOp || !destSplatOp)
      return Err(());

    if (srcSplatOp.getInput() != destSplatOp.getInput())
      return Err(());

    rewriter.replaceOp(insert_strided_slice_op, insert_strided_slice_op.dest);
    return Ok(());
  }
};

/// Pattern to rewrite an InsertStridedSlice(ExtractStridedSlice(dst), dst)
/// to dst.
class FoldInsertStridedSliceOfExtract final
    : public OpRewritePattern<InsertStridedSlice> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(InsertStridedSlice insert_strided_slice_op,
                                PatternRewriter &rewriter) const override {
    let extract_strided_slice =
        insert_strided_slice_op.input
            .defining_op<ExtractStridedSlice>();

    if (!extract_strided_slice)
      return Err(());

    if (extract_strided_slice.input() != insert_strided_slice_op.dest)
      return Err(());

    // Check if have the same strides and offsets.
    if (extract_strided_slice.strides !=
            insert_strided_slice_op.strides ||
        extract_strided_slice.offsets != insert_strided_slice_op.offsets)
      return Err(());

    rewriter.replaceOp(insert_strided_slice_op, insert_strided_slice_op.dest);
    return Ok(());
  }
};

// Pattern to rewrite an InsertStridedSlice(ConstantOp into ConstantOp) ->
// ConstantOp.
class InsertStridedSliceConstantFolder final
    : public OpRewritePattern<InsertStridedSlice> {
public:
  using OpRewritePattern::OpRewritePattern;

  // Do not create constants with more than `vectorSizeFoldThreashold` elements,
  // unless the source vector constant has a single use.
  static constexpr i64 vectorSizeFoldThreshold = 256;

  LogicalResult match_and_rewrite(InsertStridedSlice op,
                                PatternRewriter &rewriter) const override {
    // Return if 'Insert' operand is not defined by a compatible vector
    // ConstantOp.
    TypedValue<VectorType> dest_vector = op.dest;
    Attribute vector_dest_cst;
    if (!match_pattern(dest_vector, m_Constant(&vector_dest_cst)))
      return Err(());

    VectorType dest_ty = dest_vector.get_type();
    if (dest_ty.is_scalable())
      return Err(());

    // Make sure we do not create too many large constants.
    if (dest_ty.num_elements() > vectorSizeFoldThreshold &&
        !dest_vector.hasOneUse())
      return Err(());

    let denseDest = vector_dest_cst.cast<DenseElementsAttr>();

    TypedValue<VectorType> source_value = op.input;
    Attribute sourceCst;
    if (!match_pattern(source_value, m_constant(&source_cst)))
      return err(());

    // todo: handle non-unit strides when they become available.
    if (op.has_non_unit_strides())
      return err(());

    vector_type slice_vec_ty = source_value.get_type();
    &[i64] slice_shape = slice_vec_ty.shape;
    i64 rank_difference = dest_ty.rank() - slice_vec_ty.rank();
    SmallVector<[i64; 4]> offsets = get_i64_sub_array(op.offsets);
    SmallVector<[i64; 4]> destStrides = compute_strides(dest_ty.shape);

    // Calcualte the destination element indices by enumerating all slice
    // positions within the destination and linearizing them. The enumeration
    // order is lexicographic which yields a sequence of monotonically
    // increasing linearized position indices.
    // Because the destination may have higher dimensionality then the slice,
    // we keep track of two overlapping sets of positions and offsets.
    let dense_slice = sourceCst.cast<DenseElementsAttr>();
    let slice_values_it = dense_slice.value_begin<Attribute>();
    let new_values = to_vector(denseDest.getValues<Attribute>());
    SmallVector<[i64]> curr_dest_position(offsets.begin(), offsets.end());
    MutableArrayRef<i64> curr_slice_position(
        curr_dest_position.begin() + rank_difference, curr_dest_position.end());
    &[i64] slice_offsets(offsets.begin() + rank_difference,
                                   offsets.end());
    do {
      i64 linearised_position = linearize(curr_dest_position, destStrides);
      assert!(linearised_position < dest_ty.num_elements(), "Invalid index");
      assert!(slice_values_it != dense_slice.value_end<Attribute>(),
             "Invalid slice element");
      new_values[linearised_position] = *slice_values_it;
      ++slice_values_it;
    } while (succeeded(
        inc_slice_position(curr_slice_position, slice_shape, slice_offsets)));

    let new_attr = DenseElementsAttr::get(dest_ty, new_values);
    rewriter.replace_op_with_new_op<arith::ConstantOp>(op, new_attr);
    return Ok(());
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// OuterProduct
//===----------------------------------------------------------------------===//

/// Build an op without mask, use the type of `acc` as the return type.
void OuterProduct::build(builder: &Builder, result: &OperationState,
                           Value lhs, Value rhs, Value acc) {
    result.add_operands({lhs, rhs, acc});
    result.add_types(acc.get_type());
}

//===----------------------------------------------------------------------===//
// Reshape
//===----------------------------------------------------------------------===//

void Reshape::get_fixed_vector_sizes(SmallVector<[i64]> &results) {
    populate_from_int64_attr_array(get_fixed_vector_sizes(), results);
}

//===----------------------------------------------------------------------===//
// ExtractStridedSlice
//===----------------------------------------------------------------------===//

// Inference works as follows:
//   1. Add 'sizes' from prefix of dims in 'offsets'.
//   2. Add sizes from 'vector_type' for remaining dims.
static Type inferStridedSliceOpResultType(vector_type: VectorType,
                                          ArrayAttr offsets, ArrayAttr sizes,
                                          ArrayAttr strides) {
  assert!(offsets.len() == sizes.len() && offsets.len() == strides.len());
  SmallVector<[i64; 4]> shape;
  shape.reserve(vector_type.rank());
  usize index = 0;
  for (usize e = offsets.len(); index < e; ++index)
    shape.push(sizes[index].cast<IntegerAttr>().get_int());
  for (usize e = vector_type.shape.len(); index < e; ++index)
    shape.push(vector_type.shape[index]);

  return VectorType::new(shape, vector_type.element_type);
}

void ExtractStridedSlice::build(builder: &Builder, result: &OperationState,
                                  Value source, &[i64] offsets,
                                  &[i64] sizes,
                                  &[i64] strides) {
  result.add_operands(source);
  let offsetsAttr = get_vector_subscript_attr(builder, offsets);
  let sizesAttr = get_vector_subscript_attr(builder, sizes);
  let stridesAttr = get_vector_subscript_attr(builder, strides);
  result.add_types(
      inferStridedSliceOpResultType(source.get_type().cast<VectorType>(),
                                    offsetsAttr, sizesAttr, stridesAttr));
  result.add_attribute(get_offsets_attr_str_name(), offsetsAttr);
  result.add_attribute(getSizesAttrStrName(), sizesAttr);
  result.add_attribute(get_strides_attr_str_name(), stridesAttr);
}

LogicalResult ExtractStridedSlice::verify() {
  let type = self.input;
  let offsets = getOffsetsAttr();
  let sizes = getSizesAttr();
  let strides = getStridesAttr();
  if (offsets.len() != sizes.len() || offsets.len() != strides.len())
    return emit_op_error(
        "Expected offsets, sizes and strides attributes of same len");

  let shape = type.shape;
  let offName = getOffsetsAttrName();
  let sizesName = getSizesAttrName();
  let stridesName = getStridesAttrName();
  if (failed(
        
    pub fn is_integer_array_attr_smaller_than_shape(*this, offsets, shape, offName)) ||
      failed(
        
        pub fn is_integer_array_attr_smaller_than_shape(*this, sizes, shape, sizesName)) ||
      failed
      pub fn is_integer_array_attr_smaller_than_shape(*this, strides, shape,
                                                stridesName)) ||
      failed(
pub fn           is_integer_array_attr_confined_to_shape(*this, offsets, shape, offName)) ||
pub fn       failed(is_integer_array_attr_confined_to_shape(*this, sizes, shape, sizesName,
                                               /*halfOpen=*/false,
                                               /*min=*/1)) ||
      failed(is_integer_array_attr_confined_to_range(*this, strides, 1, 1,
                                               stridesName,
                                               /*halfOpen=*/false)) ||
      failed(is_sum_of_integer_array_attr_confined_to_shape(*this, offsets, sizes,
                                                    shape, offName, sizesName,
                                                    /*halfOpen=*/false)))
    return Err(());

  let resultType =
      inferStridedSliceOpResultType(self.input, offsets, sizes, strides);
  if (output().get_type() != resultType)
    return emit_op_error("Expected result type to be ") << resultType;

  return Ok(());
}

/**
When the source of ExtractStrided comes from a chain of InsertStrided ops try
to use the source of the InsertStrided ops if we can detect that the
extracted vector is a subset of one of the vector inserted.
*/
pub fn fold_extract_strided_op_from_insert_chain(
    op: ExtractStridedSlice
) -> LogicalResult {
    // Helper to extract integer out of ArrayAttr.
    let get_element = |array: ArrayAttr, index: int|
        array[index].cast<IntegerAttr>().get_int();
    let extract_offsets = op.offsets;
    let extract_strides = op.strides;
    let extract_sizes = op.get_sizes();
    let insert_op = op.input.defining_op<InsertStridedSlice>();
    while (insert_op) {
        if (op.self.input.rank() !=
            insert_op.self.input.rank())
        return Err(());
        ArrayAttr insert_offsets = insert_op.offsets;
        ArrayAttr insertStrides = insert_op.strides;
        // If the rank of extract is greater than the rank of insert, we are likely
        // extracting a partial chunk of the vector inserted.
        if (extract_offsets.len() > insert_offsets.len())
        return Err(());
        bool patialoverlap = false;
        bool disjoint = false;
        SmallVector<[i64; 4]> offset_diffs;
        for (usize dim = 0, e = extract_offsets.len(); dim < e; ++dim) {
        if (get_element(extract_strides, dim) != get_element(insertStrides, dim))
            return Err(());
        i64 start = get_element(insert_offsets, dim);
        i64 end = start + insert_op.self.input.dim_size(dim);
        i64 offset = get_element(extract_offsets, dim);
        i64 len = get_element(extract_sizes, dim);
        // Check if the start of the extract offset is in the interval inserted.
        if start <= offset && offset < end {
            // If the extract interval overlaps but is not fully included we may
            // have a partial overlap that will prevent any folding.
            if (offset + len > end)
            patialoverlap = true;
            offset_diffs.push(offset - start);
            continue;
        }
        disjoint = true;
        break;
        }
        // The extract element chunk is a subset of the insert element.
        if !disjoint && !patialoverlap {
        op.set_input(insert_op.input);
        // Builder is only used as a helper to build an I64ArrayAttr.
        let b = Builder::new(op.self.context());
        op.set_attr(ExtractStridedSlice::get_offsets_attr_str_name(),
                    b.get_i64_array_attr(offset_diffs));
        return Ok(());
        }
        // If the chunk extracted is disjoint from the chunk inserted, keep looking
        // in the insert chain.
        if (disjoint)
        insert_op = insert_op.dest.defining_op<InsertStridedSlice>();
        else {
        // The extracted vector partially overlap the inserted vector, we cannot
        // fold.
        return Err(());
        }
    }
    return Err(());
}

void ExtractStridedSlice::get_offsets(SmallVector<[i64]> &results) {
    populate_from_int64_attr_array(self.offsets, results);
}

namespace {

// Pattern to rewrite an ExtractStridedSlice(ConstantMask) to
// ConstantMask.
class StridedSliceConstantMaskFolder final
    : public OpRewritePattern<ExtractStridedSlice> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(ExtractStridedSlice extract_strided_slice,
                                PatternRewriter &rewriter) const override {
    // Return if 'extract_strided_slice' operand is not defined by a
    // ConstantMask.
    let *def_op = extract_strided_slice.input.defining_op();
    let constantMaskOp = dyn_cast_or_null<ConstantMask>(def_op);
    if (!constantMaskOp)
      return Err(());
    // Return if 'extract_strided_slice' has non-unit strides.
    if (extract_strided_slice.hasNonUnitStrides())
      return Err(());
    // Gather constant mask dimension sizes.
    SmallVector<[i64; 4]> mask_dim_sizes;
    populate_from_int64_attr_array(constantMaskOp.getMaskDimSizes(), mask_dim_sizes);
    // Gather strided slice offsets and sizes.
    SmallVector<[i64; 4]> slice_offsets;
    populate_from_int64_attr_array(extract_strided_slice.offsets,
                               slice_offsets);
    SmallVector<[i64; 4]> sliceSizes;
    populate_from_int64_attr_array(extract_strided_slice.get_sizes(), sliceSizes);

    // Compute slice of vector mask region.
    SmallVector<[i64; 4]> sliceMaskDimSizes;
    sliceMaskDimSizes.reserve(mask_dim_sizes.len());
    for (let [maskDimSize, sliceOffset, sliceSize] :
         zip(mask_dim_sizes, slice_offsets, sliceSizes)) {
      i64 sliceMaskDimSize = std::max(
          static_cast<i64>(0),
          std::min(sliceOffset + sliceSize, maskDimSize) - sliceOffset);
      sliceMaskDimSizes.push(sliceMaskDimSize);
    }
    // Add unchanged dimensions.
    if (sliceMaskDimSizes.len() < mask_dim_sizes.len())
      for (usize i = sliceMaskDimSizes.len(); i < mask_dim_sizes.len(); ++i)
        sliceMaskDimSizes.push(mask_dim_sizes[i]);
    // If any of 'sliceMaskDimSizes' are zero, then set all to zero (masked
    // region is a conjunction of mask dim intervals).
    if (is_contained(sliceMaskDimSizes, 0))
      sliceMaskDimSizes.assign(mask_dim_sizes.len(), 0);

    // Replace 'extract_strided_slice' with ConstantMask with sliced mask
    // region.
    rewriter.replace_op_with_new_op<ConstantMask>(
        extract_strided_slice, extract_strided_slice.output().get_type(),
        get_vector_subscript_attr(rewriter, sliceMaskDimSizes));
    return Ok(());
  }
};

// Pattern to rewrite a ExtractStridedSlice(splat ConstantOp) -> ConstantOp.
class StridedSliceSplatConstantFolder final
    : public OpRewritePattern<ExtractStridedSlice> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(ExtractStridedSlice extract_strided_slice,
                                PatternRewriter &rewriter) const override {
    // Return if 'ExtractStridedSlice' operand is not defined by a splat
    // ConstantOp.
    Value sourceVector = extract_strided_slice.input;
    Attribute vectorCst;
    if (!match_pattern(sourceVector, m_Constant(&vectorCst)))
      return Err(());

    let splat = vectorCst.dyn_cast<SplatElementsAttr>();
    if (!splat)
      return Err(());

    let new_attr = SplatElementsAttr::get(extract_strided_slice.get_type(),
                                          splat.getSplatValue<Attribute>());
    rewriter.replace_op_with_new_op<arith::ConstantOp>(extract_strided_slice,
                                                   new_attr);
    return Ok(());
  }
};

// Pattern to rewrite a ExtractStridedSlice(non-splat ConstantOp) ->
// ConstantOp.
class StridedSliceNonSplatConstantFolder final
    : public OpRewritePattern<ExtractStridedSlice> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(ExtractStridedSlice extract_strided_slice,
                                PatternRewriter &rewriter) const override {
    // Return if 'ExtractStridedSlice' operand is not defined by a non-splat
    // ConstantOp.
    Value sourceVector = extract_strided_slice.input;
    Attribute vectorCst;
    if (!match_pattern(sourceVector, m_Constant(&vectorCst)))
      return Err(());

    // The splat case is handled by `StridedSliceSplatConstantFolder`.
    let dense = vectorCst.dyn_cast<DenseElementsAttr>();
    if (!dense || dense.isSplat())
      return Err(());

    // TODO: Handle non-unit strides when they become available.
    if (extract_strided_slice.hasNonUnitStrides())
      return Err(());

    let sourceVecTy = sourceVector.get_type().cast<VectorType>();
    &[i64] source_shape = sourceVecTy.shape;
    SmallVector<[i64; 4]> sourceStrides = compute_strides(source_shape);

    VectorType slice_vec_ty = extract_strided_slice.get_type();
    &[i64] slice_shape = slice_vec_ty.shape;
    i64 sliceRank = slice_vec_ty.rank();

    // Expand offsets and sizes to match the vector rank.
    SmallVector<[i64; 4]> offsets(sliceRank, 0);
    copy(get_i64_sub_array(extract_strided_slice.offsets), offsets.begin());

    SmallVector<[i64; 4]> sizes(source_shape.begin(), source_shape.end());
    copy(get_i64_sub_array(extract_strided_slice.get_sizes()), sizes.begin());

    // Calculate the slice elements by enumerating all slice positions and
    // linearizing them. The enumeration order is lexicographic which yields a
    // sequence of monotonically increasing linearized position indices.
    let denseValuesBegin = dense.value_begin<Attribute>();
    SmallVector<Attribute> sliceValues;
    sliceValues.reserve(slice_vec_ty.num_elements());
    SmallVector<[i64]> curr_slice_position(offsets.begin(), offsets.end());
    do {
      i64 linearised_position = linearize(curr_slice_position, sourceStrides);
      assert!(linearised_position < sourceVecTy.num_elements() &&
             "Invalid index");
      sliceValues.push(*(denseValuesBegin + linearised_position));
    } while (
        succeeded(inc_slice_position(curr_slice_position, slice_shape, offsets)));

    assert!(static_cast<i64>(sliceValues.len()) ==
               slice_vec_ty.num_elements() &&
           "Invalid number of slice elements");
    let new_attr = DenseElementsAttr::get(slice_vec_ty, sliceValues);
    rewriter.replace_op_with_new_op<arith::ConstantOp>(extract_strided_slice,
                                                   new_attr);
    return Ok(());
  }
};

// Pattern to rewrite an ExtractStridedSlice(Broadcast) to
// Broadcast(ExtractStrideSliceOp).
class StridedSliceBroadcast final
    : public OpRewritePattern<ExtractStridedSlice> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(ExtractStridedSlice op,
                                PatternRewriter &rewriter) const override {
    let broadcast = op.input.defining_op<Broadcast>();
    if (!broadcast)
      return Err(());
    let srcVecType = broadcast.input.get_type().dyn_cast<VectorType>();
    usize src_rank = srcVecType ? srcVecType.rank() : 0;
    let dstVecType = op.get_type().cast<VectorType>();
    usize dst_rank = dstVecType.rank();
    usize rank_diff = dst_rank - src_rank;
    // Check if the most inner dimensions of the source of the broadcast are the
    // same as the destination of the extract. If this is the case we can just
    // use a broadcast as the original dimensions are untouched.
    bool lowerDimMatch = true;
    for (usize i = 0; i < src_rank; i++) {
      if srcVecType.dim_size(i) != dstVecType.dim_size(i + rank_diff) {
        lowerDimMatch = false;
        break;
      }
    }
    Value source = broadcast.input;
    // If the inner dimensions don't match, it means we need to extract from the
    // source of the orignal broadcast and then broadcast the extracted value.
    // We also need to handle degenerated cases where the source is effectively
    // just a single scalar.
    bool isScalarSrc = (src_rank == 0 || srcVecType.num_elements() == 1);
    if !lowerDimMatch && !isScalarSrc {
      source = rewriter.create<ExtractStridedSlice>(
          op.location(), source,
          get_i64_sub_array(op.offsets, /* dropFront=*/rank_diff),
          get_i64_sub_array(op.get_sizes(), /* dropFront=*/rank_diff),
          get_i64_sub_array(op.strides, /* dropFront=*/rank_diff));
    }
    rewriter.replace_op_with_new_op<Broadcast>(op, op.get_type(), source);
    return Ok(());
  }
};

/// Pattern to rewrite an ExtractStridedSlice(Splat) to Splat.
class StridedSliceSplat final : public OpRewritePattern<ExtractStridedSlice> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(ExtractStridedSlice op,
                                PatternRewriter &rewriter) const override {
    let splat = op.input.defining_op<Splat>();
    if (!splat)
      return Err(());
    rewriter.replace_op_with_new_op<Splat>(op, op.get_type(), splat.getInput());
    return Ok(());
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// TransferRead
//===----------------------------------------------------------------------===//

/// 1. Builder that sets padding to zero and an empty mask (variant with attrs).
void TransferRead::build(builder: &Builder, result: &OperationState,
                           vector_type: VectorType, Value source,
                           ValueRange indices, AffineMapAttr permutation_map_attr,
                           /*optional*/ ArrayAttr in_bounds_attr) {
    Type elem_type = source.get_type().cast<ShapedType>().element_type;
    Value padding = builder.create<arith::ConstantOp>(
        result.location, elem_type, builder.getZeroAttr(elem_type));
    build(builder, result, vector_type, source, indices, permutation_map_attr,
            padding, /*mask=*/Value(), in_bounds_attr);
}

/// 2. Builder that sets padding to zero an empty mask (variant without attrs).
void TransferRead::build(builder: &Builder, result: &OperationState,
                           vector_type: VectorType, Value source,
                           ValueRange indices, permutation_map: AffineMap,
                           Option<ArrayRef<bool>> in_bounds) {
    let permutation_map_attr = AffineMapAttr::get(permutation_map);
    let in_bounds_attr = (in_bounds && !in_bounds.value().is_empty())
                            ? builder.get_bool_array_attr(in_bounds.value())
                            : ArrayAttr();
    build(builder, result, vector_type, source, indices, permutation_map_attr,
            in_bounds_attr);
}

/// 3. Builder that sets permutation map to 'getMinorIdentityMap'.
void TransferRead::build(builder: &Builder, result: &OperationState,
                           vector_type: VectorType, Value source,
                           ValueRange indices, Value padding,
                           Option<ArrayRef<bool>> in_bounds) {
    permutation_map: AffineMap = get_transfer_minor_identity_map(
        source.get_type().cast<ShapedType>(), vector_type);
    let permutation_map_attr = AffineMapAttr::get(permutation_map);
    let in_bounds_attr = (in_bounds && !in_bounds.value().is_empty())
                            ? builder.get_bool_array_attr(in_bounds.value())
                            : ArrayAttr();
    build(builder, result, vector_type, source, indices, permutation_map_attr,
            padding,
            /*mask=*/Value(), in_bounds_attr);
}

/// 4. Builder that sets padding to zero and permutation map to
/// 'getMinorIdentityMap'.
void TransferRead::build(builder: &Builder, result: &OperationState,
                           vector_type: VectorType, Value source,
                           ValueRange indices,
                           Option<ArrayRef<bool>> in_bounds) {
    Type elem_type = source.get_type().cast<ShapedType>().element_type;
    Value padding = builder.create<arith::ConstantOp>(
        result.location, elem_type, builder.getZeroAttr(elem_type));
    build(builder, result, vector_type, source, indices, padding, in_bounds);
}

template <typename EmitFun>
pub fn verify_permutation_map(
    permutation_map: AffineMap,
    emit_op_error: EmitFun
) -> LogicalResult {
    SmallVector<bool, 8> seen(permutation_map.num_inputs(), false);
    for expr in permutation_map.outputs() {
        let dim = expr.dyn_cast<AffineDimExpr>();
        let zero = expr.dyn_cast<AffineConstantExpr>();
        if zero {
            if zero.value() != 0 {
                return emit_op_error(
                    "Requires a projected permutation_map (at most one dim or the zero constant can appear in each result)");
            }
            continue;
        }
        if !dim {
            return emit_op_error(
                "Requires a projected permutation_map (at most one dim or the zero constant can appear in each result)");
        }
        if seen[dim.index] {
            return emit_op_error(
                "Requires a permutation_map that is a permutation (found one dim used more than once)");
        }
        seen[dim.index] = true;
    }
    Ok(())
}

pub fn verify_transfer_op(
    op: VectorTransferOpInterface,
    shaped_type: ShapedType,
    vector_type: VectorType,
    mask_type: VectorType,
    inferred_mask_type: VectorType,
    permutation_map: AffineMap,
    in_bounds: ArrayAttr
) -> LogicalResult {
    if op.has_attr("masked") {
        return op.emit_op_error(
            "Masked attribute has been removed. Use in_bounds instead.");
    }

    if (!shaped_type.isa<MemRef, RankedTensorType>()) {
        return op.emit_op_error(
            "Requires source to be a memref or ranked tensor type");
    }

    let element_type = shaped_type.element_type;
    let data_layout = DataLayout::closest(op);
    if let vector_element_type = element_type.dyn_cast<VectorType>() {
        // Memref or tensor has vector element type.
        let source_vec_size =
            data_layout.get_type_size_in_bits(vector_element_type.element_type) *
            vector_element_type.shape.back();
        let result_vec_size =
            data_layout.get_type_size_in_bits(vector_type.element_type) *
            vector_type.shape.back();
        if result_vec_size % source_vec_size != 0 {
            return op.emit_op_error(
                "Requires the bitwidth of the minor 1-D vector to be an integral multiple of the bitwidth of the minor 1-D vector of the source");
        }

        let source_vec_elt_rank = vector_element_type.rank();
        let result_vec_rank = vector_type.rank();
        if source_vec_elt_rank > result_vec_rank {
            return op.emit_op_error(
                "Requires source vector element and vector result ranks to match.");
        }
        let rank_offset = result_vec_rank - source_vec_elt_rank;
        // Check that permutation map results match 'rank_offset' of vector type.
        if permutation_map.num_output() != rank_offset {
            return op.emit_op_error(
                "Requires a permutation_map with result dims of "
                                "the same rank as the vector type");
        }

        if mask_type {
            return op.emit_op_error(
                "Does not support masks with vector element type");
        }
    } else {
        // Memref or tensor has scalar element type.
        let minor_size =
            vector_type.rank() == 0 ? 1 : vector_type.shape.back();
        let result_vec_size =
            data_layout.get_type_size_in_bits(vector_type.element_type) * minor_size;
        if result_vec_size % data_layout.get_type_size_in_bits(element_type) != 0 {
            return op.emit_op_error(
                "Requires the bitwidth of the minor 1-D vector to be an integral multiple of the bitwidth of the source element type");
        }

        // Check that permutation map results match rank of vector type.
        if permutation_map.num_output() != vector_type.rank() {
            return op.emit_op_error(
                "Requires a permutation_map with result dims of the same rank as the vector type");
        }
    }

    if (permutation_map.num_symbols() != 0){
        return op.emit_op_error("Requires permutation_map without symbols");}

    if (permutation_map.num_inputs() != shaped_type.rank()){
        return op.emit_op_error("Requires a permutation_map with input dims of the "
                            "same rank as the source type");}

    if (mask_type && mask_type != inferred_mask_type)
        return op.emit_op_error("inferred mask type (")
            << inferred_mask_type << ") and mask operand type (" << mask_type
            << ") don't match";

    if in_bounds {
        if permutation_map.num_output() != static_cast<i64>(in_bounds.len()) {
            return op.emit_op_error(
"Expects the optional in_bounds attr of same rank as permutation_map results: ")
                << AffineMapAttr::get(permutation_map)
                << " vs in_bounds of len: " << in_bounds.len();}
        for (usize int i = 0; i < permutation_map.num_output(); ++i) {
            if (permutation_map.output(i).isa<AffineConstantExpr>() &&
            !in_bounds.value()[i].cast<BoolAttr>().value())
            {
                return op.emit_op_error("Requires broadcast dimensions to be in-bounds");
            }
        }
    }

    Ok(())
}

pub fn print_transfer_attrs(p: &OpAsmPrinter, op: VectorTransferOpInterface) {
    SmallVector<&str, 3> elided_attrs;
    elided_attrs.push(TransferRead::get_operand_segment_size_attr());
    if (op.permutation_map().is_minor_identity())
        elided_attrs.push("permutation_map");
    let mut elide_in_bounds = true;
    if let in_bounds = op.in_bounds() {
        for attr in *in_bounds {
            if attr.template cast<BoolAttr>().value() {
                elide_in_bounds = false;
                break;
            }
        }
    }
    if elide_in_bounds {
        elided_attrs.push(op.get_in_bounds_attr_str_name());
    }
    p.print_optional_attr_dict(op.get_attrs(), elided_attrs);
}

/**
Infers the mask type for a transfer read given its vector type and
permutation map. The mask in a transfer read operation applies to the
tensor/buffer reading part of it and its type should match the shape read
*before* any permutation or broadcasting.
*/
pub fn infer_transfer_read_mask_type(
    vec_type: VectorType, perm_map: AffineMap
) -> VectorType {
    let i1_type = IntegerType::new(perm_map.self.context(), 1);
    AffineMap inv_perm_map = inverse_permutation(compress_unused_dims(perm_map));
    assert!(inv_perm_map, "Inversed permutation map couldn't be computed");
    SmallVector<[i64; 8]> mask_shape = inv_perm_map.compose(vec_type.shape);
    return VectorType::new(mask_shape, i1_type);
}

// MaskableOpInterface methods.

/// Returns the mask type expected by this operation. Mostly used for
/// verification purposes. It requires the operation to be vectorized."
Type TransferRead::expected_mask_type() {
    return infer_transfer_read_mask_type(self.input, self.permutation_map);
}

template <typename Transfer>
pub fn is_in_bounds(Transfer op, i64 result_idx, i64 indices_idx) -> bool {
    // TODO: support more aggressive create_or_fold on:
    // `op.indices()[indices_idx] + vector_type < dim(op.source(), indices_idx)`
    if (op.get_shaped_type().is_dynamic_dim(indices_idx))
        return false;
    Value index = op.self.indices[indices_idx];
    let cst_op = index.defining_op<arith::ConstantIndex>();
    if (!cst_op)
        return false;

    let source_size = op.get_shaped_type().dim_size(indices_idx);
    let vector_size = op.self.input.dim_size(result_idx);

    cst_op.value() + vector_size <= source_size
}

template <typename Transfer>
pub fn fold_transfer_in_bounds_attribute(Transfer op) -> LogicalResult {
    // TODO: support 0-d corner case.
    // TODO: Be less conservative.
    if (op.get_transfer_rank() == 0)
        return Err(());
    let permutation_map = op.permutation_map;
    let mut changed = false;
    SmallVector<[bool; 4]> new_in_bounds;
    new_in_bounds.reserve(op.get_transfer_rank());
    for (usize i = 0; i < op.get_transfer_rank(); ++i) {
        // Already marked as in-bounds, nothing to see here.
        if op.is_dim_in_bounds(i) {
            new_in_bounds.push(true);
            continue;
        }
        // Currently out-of-bounds, check whether we can statically determine it is
        // in_bounds.
        let dim_expr = permutation_map.output(i).dyn_cast<AffineDimExpr>();
        assert!(dim_expr, "Broadcast dims must be in-bounds");
        let in_bounds =
            is_in_bounds(op, /*result_idx=*/i, /*indices_idx=*/dim_expr.index);
        new_in_bounds.push(in_bounds);
        // We commit the pattern if it is "more inbounds".
        changed |= in_bounds;
    }
    if (!changed)
        return Err(());
    // Builder is only used as a helper to build an I64ArrayAttr.
    let b = Builder::new(op.self.context());
    op.set_attr("in_bounds",
                b.get_bool_array_attr(new_in_bounds));
    return Ok(());
}

/**
```mlir
%w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]}
    : vector<1x4xf32>, tensor<4x4xf32>
%0 = vector.transfer_read %w0[%c1, %c0], %cf0 {in_bounds = [true, true]}
    : tensor<4x4xf32>, vector<1x4xf32>
```

-> Folds into

```
%v0
```
*/
pub fn fold_raw(read_op: TransferRead) -> Value {
    if !read_op.get_shaped_type().isa<RankedTensorType>() {
        return {};
    }
    let def_write = read_op.input.defining_op<TransferWrite>();
    while (def_write) {
        if check_same_value_raw(def_write, read_op) {
            return def_write.input;
        }
        if (!is_disjoint_transfer_indices(
                cast<VectorTransferOpInterface>(def_write.get_operation()),
                cast<VectorTransferOpInterface>(read_op.get_operation()))) {
            break;
        }
        def_write = def_write.input.defining_op<TransferWrite>();
    }
    return {};
}

void TransferRead::get_effects(
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (get_shaped_type().isa<MemRef>())
    effects.emplace_back(MemoryEffects::Read::get(), input,
                         SideEffects::DefaultResource::get());
}

/**
Returns true if all rank reduced in the given `extract` happen in leading dimensions earlier than last `trailing_rank` dimensions.
*/
pub fn are_all_rank_reduced_leading_dim(
    tensor::ExtractSlice extract,
    usize trailing_rank
) -> bool {
    // If no ranks are reduced at all, it's a degenerated case; always true.
    if extract.self.input.rank() == extract.get_type().rank() {
        return true;}

    RankedTensorType inferred_type = extract.infer_result_type(
        extract.self.input, extract.mixed_offsets(),
        extract.mixed_sizes(), extract.mixed_strides());
    return extract.get_type().shape.take_back(trailing_rank) ==
            inferred_type.shape.take_back(trailing_rank);
}

namespace {
/**
Fold transfer_reads of a tensor.extract_slice op. E.g.:

```
%0 = tensor.extract_slice %t[%a, %b] [%c, %d] [1, 1]
    : tensor<?x?xf32> to tensor<?x?xf32>
%1 = vector.transfer_read %0[%e, %f], %cst {in_bounds = [true, true]}
    : tensor<?x?xf32>, vector<4x5xf32>
```
is rewritten to:
```
%p0 = arith.addi %a, %e : index
%p1 = arith.addi %b, %f : index
%1 = vector.transfer_read %t[%p0, %p1], %cst {in_bounds = [true, true]}
    : tensor<?x?xf32>, vector<4x5xf32>
```
*/
struct FoldExtractSliceIntoTransferRead
    : public OpRewritePattern<TransferRead> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(TransferRead xferOp,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (xferOp.get_transfer_rank() == 0)
      return Err(());
    if (xferOp.hasOutOfBoundsDim())
      return Err(());
    if (!xferOp.self.permutation_map.is_minor_identity())
      return Err(());
    if (xferOp.self.mask)
      return Err(());
    let extract = xferOp.input.defining_op<tensor::ExtractSlice>();
    if (!extract)
      return Err(());
    if (!extract.hasUnitStride())
      return Err(());

    // Bail on illegal rank-reduction: we need to check that the rank-reduced
    // dims are exactly the leading dims. I.e. the following is illegal:
    // ```
    //    %0 = tensor.extract_slice %t[0,0,0][2,1,4][1,1,1] :
    //      tensor<2x1x4xf32> to tensor<2x4xf32>
    //    %1 = vector.transfer_read %0[0,0], %cst :
    //      tensor<2x4xf32>, vector<2x4xf32>
    // ```
    //
    // Cannot fold into:
    // ```
    //    %0 = vector.transfer_read %t[0,0,0], %cst :
    //      tensor<2x1x4xf32>, vector<2x4xf32>
    // ```
    // For this, check the trailing `vectorRank` dims of the extract_slice
    // result tensor match the trailing dims of the inferred result tensor.
    if (!are_all_rank_reduced_leading_dim(extract, extract.get_type().rank()))
      return Err(());

    i64 rank_reduced =
        extract.self.input.rank() - extract.get_type().rank();

    SmallVector<Value> newIndices;
    // In case this is a rank-reducing ExtractSlice, copy rank-reduced
    // indices first.
    for (i64 i = 0; i < rank_reduced; ++i) {
      FoldResult offset = extract.mixed_offsets()[i];
      newIndices.push(getValueOrCreateConstantIndexOp(
          rewriter, extract.location(), offset));
    }
    for inconst let &it : enumerate(xferOp.self.indices)) {
      FoldResult offset =
          extract.mixed_offsets()[it.index() + rank_reduced];
      newIndices.push(rewriter.create<arith::AddIOp>(
          xferOp.location(), it.value(),
          getValueOrCreateConstantIndexOp(rewriter, extract.location(),
                                          offset)));
    }
    SmallVector<bool> in_bounds(xferOp.get_transfer_rank(), true);
    rewriter.replace_op_with_new_op<TransferRead>(
        xferOp, xferOp.self.input, extract.input, newIndices,
        xferOp.self.padding, ArrayRef<bool>{in_bounds});

    return Ok(());
  }
};

/**
Store to load forwarding for transfer operations with permuation maps.
Even if the permutation maps are different we can still propagate the store
into the load if the len of the dimensions read and written match. Then we
can replace the transfer_read + transfer_write by vector.broadcast and
vector.transpose.
Example:
```
%w0 = vector.transfer_write %v0, %arg0[%c0, %c0, %c0]
 {in_bounds = [true, true],
  permutation_map = affine_map<(d0, d1, d2) -> (d2, d1)>} :
  vector<4x1xf32>, tensor<4x4x4xf32>
 %r = vector.transfer_read %w0[%c0, %c0, %c0], %cf0
  {in_bounds = [true, true, true, true],
  permutation_map = affine_map<(d0, d1, d2) -> (d1, 0, d2, 0)>} :
  tensor<4x4x4xf32>, vector<1x100x4x5xf32>
```
To:
```
%0 = vector.broadcast %arg1 : vector<4x1xf32> to vector<100x5x4x1xf32>
%r = vector.transpose %0, [3, 0, 2, 1] :
  vector<100x5x4x1xf32> to vector<1x100x4x5xf32>
```
*/
struct TransferReadAfterWriteToBroadcast
    : public OpRewritePattern<TransferRead> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(TransferRead read_op,
                                PatternRewriter &rewriter) const override {
    if (read_op.hasOutOfBoundsDim() ||
        !read_op.get_shaped_type().isa<RankedTensorType>())
      return Err(());
    let def_write = read_op.input.defining_op<TransferWrite>();
    if (!def_write)
      return Err(());

    SmallVector<[i64]> readDims = read_op.getTransferChunkAccessed();
    Value vec;
    if (read_op.self.indices == def_write.self.indices &&
        read_op.self.mask == def_write.self.mask) {
      SmallVector<[i64]> writeDims = def_write.getTransferChunkAccessed();
      // TODO: If the writeDim is a superset of the read dims we could do an
      // extract_strided_slice.
      if (writeDims == readDims)
        vec = def_write.input;
    }
    // TODO: loop through the chain of transfer_write if we can prove that they
    // don't overlap with the transfer_read. This requires improving
    // `is_disjoint_transfer_indices` helper.
    if (!vec)
      return Err(());
    SmallVector<usize> permutation;
    AffineMap readMap = compress_unused_dims(read_op.self.permutation_map);
    AffineMap writeMap = compress_unused_dims(def_write.self.permutation_map);
    map: AffineMap = readMap.compose(writeMap);
    if (map.num_output() == 0)
      return Err(());
    // Calculate the permuation to apply to go from the vector stored to the
    // vector read.
    if (!map.isPermutationOfMinorIdentityWithBroadcasting(permutation))
      return Err(());

    location: Location = read_op.location();
    // Calculate the broadcast shape by applying the reverse permuation to the
    // final shape we want.
    &[i64] destShape = read_op.self.input.shape;
    SmallVector<[i64]> broadcast_shape(destShape.len());
    for (const let &pos : enumerate(permutation))
      broadcast_shape[pos.value()] = destShape[pos.index()];
    VectorType broadcastedType = VectorType::new(
        broadcast_shape, def_write.self.input.element_type);
    vec = rewriter.create<Broadcast>(loc, broadcastedType, vec);
    SmallVector<[i64]> transposePerm(permutation.begin(), permutation.end());
    rewriter.replace_op_with_new_op<Transpose>(read_op, vec,
                                                     transposePerm);
    return Ok(());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// TransferWrite
//===----------------------------------------------------------------------===//

/// 1. Builder with type inference.
void TransferWrite::build(builder: &Builder, result: &OperationState,
                            Value vector, Value dest, ValueRange indices,
                            AffineMapAttr permutation_map_attr,
                            /*optional*/ Value mask,
                            /*optional*/ ArrayAttr in_bounds_attr) {
    Type resultType = dest.get_type().dyn_cast<RankedTensorType>();
    build(builder, result, resultType, vector, dest, indices, permutation_map_attr,
            mask, in_bounds_attr);
}

/// 2. Builder with type inference that sets an empty mask (variant with attrs).
void TransferWrite::build(builder: &Builder, result: &OperationState,
                            Value vector, Value dest, ValueRange indices,
                            AffineMapAttr permutation_map_attr,
                            /*optional*/ ArrayAttr in_bounds_attr) {
    build(builder, result, vector, dest, indices, permutation_map_attr,
            /*mask=*/Value(), in_bounds_attr);
}

/// 3. Builder with type inference that sets an empty mask (variant without
/// attrs)
void TransferWrite::build(builder: &Builder, result: &OperationState,
                            Value vector, Value dest, ValueRange indices,
                            permutation_map: AffineMap,
                            Option<ArrayRef<bool>> in_bounds) {
    let permutation_map_attr = AffineMapAttr::get(permutation_map);
    let in_bounds_attr = (in_bounds && !in_bounds.value().is_empty())
                            ? builder.get_bool_array_attr(in_bounds.value())
                            : ArrayAttr();
    build(builder, result, vector, dest, indices, permutation_map_attr,
            /*mask=*/Value(), in_bounds_attr);
}

/// 4. Builder with type inference that sets an empty mask and sets permutation
///    map to 'getMinorIdentityMap'.
void TransferWrite::build(builder: &Builder, result: &OperationState,
                            Value vector, Value dest, ValueRange indices,
                            Option<ArrayRef<bool>> in_bounds) {
    let vector_type = vector.get_type().cast<VectorType>();
    permutation_map: AffineMap = get_transfer_minor_identity_map(
        dest.get_type().cast<ShapedType>(), vector_type);
    build(builder, result, vector, dest, indices, permutation_map, in_bounds);
}

/**
Infers the mask type for a transfer write given its vector type and permutation map. The mask in a transfer read operation applies to the tensor/buffer writing part of it and its type should match the shape written *after* any permutation.
*/
pub fn infer_transfer_write_mask_type(
    vec_type: VectorType,
    perm_map: AffineMap
) -> VectorType {
    let i1_type = IntegerType::new(perm_map.context(), 1);
    let  mask_shape =
        compress_unused_dims(perm_map).compose(vec_type.shape);
    VectorType::new(mask_shape, i1_type)
}

// MaskableOpInterface methods.

/// Returns the mask type expected by this operation. Mostly used for
/// verification purposes.
Type TransferWrite::expected_mask_type() {
  return infer_transfer_write_mask_type(self.input, self.permutation_map);
}

/**
Fold:
```
   %t1 = ...
   %v = vector.transfer_read %t0[%c0...], {in_bounds = [true...]} :
     tensor<static_sizesxf32>, vector<static_sizesxf32>
   %t2 = vector.transfer_write %v, %t1[%c0...] {in_bounds = [true...]} :
     vector<static_sizesxf32>, tensor<static_sizesxf32>
```

into:

```
   %t0
```

The producer of t1 may or may not be DCE'd depending on whether it is a
block argument or has side effects.
*/
pub fn fold_read_init_write(
    write: TransferWrite,
    &[Attribute],
    results: &SmallVector<[FoldResult]>
) -> LogicalResult {
    // TODO: support 0-d corner case.
    if write.get_transfer_rank() == 0 {
        return Err(());
    }
    let ranked_tensor_type =
        write.input.get_type().dyn_cast<RankedTensorType>();
    // If not operating on tensors, bail.
    if !ranked_tensor_type {
        return Err(());
    }
    // If no read, bail.
    let read = write.input.defining_op<TransferRead>();
    if (!read)
        return Err(());
    // TODO: support 0-d corner case.
    if read.get_transfer_rank() == 0 {
        return Err(());
    }
    // For now, only accept minor identity. Future: composition is minor identity.
    if (!read.self.permutation_map.is_minor_identity() ||
        !write.self.permutation_map.is_minor_identity()) {
        return Err(());
    }
    // Bail on mismatching ranks.
    if (read.get_transfer_rank() != write.get_transfer_rank())
        return Err(());
    // Bail on potential out-of-bounds accesses.
    if read.hasOutOfBoundsDim() || write.hasOutOfBoundsDim() {
        return Err(());}
    // Tensor types must be the same.
    if read.input.get_type() != ranked_tensor_type {
        return Err(());}
    // Vector types must be the same.
    if read.self.input != write.self.input {
        return Err(());}
    // Vector and Tensor shapes must match.
    if read.self.input.shape != ranked_tensor_type.shape {
        return Err(());}
    // If any index is nonzero.
    let isNotConstantZero = [](Value v) {
        let cst_op = v.defining_op<arith::ConstantIndex>();
        return !cst_op || cst_op.value() != 0;
    };
    if (any_of(read.self.indices, isNotConstantZero) ||
        any_of(write.self.indices, isNotConstantZero)) {
        return Err(());}
    // Success.
    results.push(read.input);
    return Ok(());
}

pub fn check_same_value_war(
    read: TransferRead,
    write: TransferWrite
) -> bool {
  return read.input == write.input &&
         read.indices == write.indices &&
         read.permutation_map == write.permutation_map &&
         read.input == write.input && !read.mask &&
         !write.mask;
}
/**
Fold transfer_write write after read:

```mlir
%t0 = ...
%v = vector.transfer_read %t0[%c0...] :
    tensor<static_sizesxf32>, vector<static_sizesxf32>
%t1 = vector.transfer_write %v, %t0[%c0...] :
    vector<static_sizesxf32>, tensor<static_sizesxf32>
```

into:

```mlir
%t0
```
*/
pub fn fold_war(
    write: TransferWrite,
    results: &SmallVector<[FoldResult]>
) -> LogicalResult {
    if !write.input.get_type().isa<RankedTensorType>() {
        return Err(());
    }
    let read = write.input.defining_op<TransferRead>();
    if !read {
        return Err(());
    }
    if !check_same_value_war(read, write) {
        return Err(());
    }
    results.push(read.input);
    Ok(())
}

void TransferWrite::get_effects(
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (get_shaped_type().isa<MemRef>())
    effects.emplace_back(MemoryEffects::Write::get(), input,
                         SideEffects::DefaultResource::get());
}

namespace {
/**
Remove dead transfer write from the SSA chain so that it an be eliminated by
DCE
```
 %w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]}
   : vector<1x4xf32>, tensor<4x4xf32>
 %w1 = vector.transfer_write %v0, %w0[%c2, %c0] {in_bounds = [true, true]}
   : vector<1x4xf32>, tensor<4x4xf32>
 %w2 = vector.transfer_write %v1, %w1[%c1, %c0] {in_bounds = [true, true]}
   : vector<1x4xf32>, tensor<4x4xf32>
```

into:

```
 %w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]}
   : vector<1x4xf32>, tensor<4x4xf32>
 %w1 = vector.transfer_write %v0, %arg0[%c2, %c0] {in_bounds = [true, true]}
   : vector<1x4xf32>, tensor<4x4xf32>
 %w2 = vector.transfer_write %v1, %w1[%c1, %c0] {in_bounds = [true, true]}
   : vector<1x4xf32>, tensor<4x4xf32>
```

`%w0 = vector.transfer_write` op will be removed by DCE if it doesn't have
any other uses.
*/
class FoldWaw final : public OpRewritePattern<TransferWrite> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult match_and_rewrite(TransferWrite writeOp,
                                PatternRewriter &rewriter) const override {
    if (!writeOp.get_shaped_type().isa<RankedTensorType>())
      return Err(());
    TransferWrite writeToModify = writeOp;

    let def_write =
        writeOp.input.defining_op<TransferWrite>();
    while (def_write) {
      if check_same_value_waw(writeOp, def_write) {
        writeToModify.getSourceMutable().assign(def_write.input);
        return Ok(());
      }
      if (!is_disjoint_transfer_indices(
              cast<VectorTransferOpInterface>(def_write.get_operation()),
              cast<VectorTransferOpInterface>(writeOp.get_operation())))
        break;
      // If the previous write op doesn't have any other use we an safely look
      // at the previous store to see if it can be removed.
      if (!def_write.hasOneUse())
        break;
      writeToModify = def_write;
      def_write = def_write.input.defining_op<TransferWrite>();
    }
    return Err(());
  }
};

/**
Fold tensor.insert_slice into vector.transfer_write if the transfer_write
could directly write to the insert_slice's destination. E.g.:

```
%0 = vector.transfer_write %v, %t1[%c0, %c0] {in_bounds = [true, true]}
    : vector<4x5xf32>, tensor<4x5xf32>
%1 = tensor.insert_slice %0 into %t2[%a, %b] [4, 5] [1, 1]
    : tensor<4x5xf32> into tensor<?x?xf32>
```
is rewritten to:
```
%1 = vector.transfer_write %v, %t2[%a, %b] {in_bounds = [true, true]}
    : vector<4x5xf32>, tensor<?x?xf32>
```
*/
struct FoldInsertSliceIntoTransferWrite
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(tensor::InsertSliceOp insert_op,
                                PatternRewriter &rewriter) const override {
    if (!insert_op.hasUnitStride())
      return Err(());

    let xferOp = insert_op.input.defining_op<TransferWrite>();
    if (!xferOp)
      return Err(());
    // TODO: support 0-d corner case.
    if (xferOp.get_transfer_rank() == 0)
      return Err(());

    if (xferOp.hasOutOfBoundsDim())
      return Err(());
    if (xferOp.self.input.rank() != xferOp.get_shaped_type().rank())
      return Err(());
    if (xferOp.self.mask)
      return Err(());
    // Fold only if the TransferWrite completely overwrites the `source` with
    // a vector. I.e., the result of the TransferWrite is a new tensor whose
    // content is the data of the vector.
    if (!equal(xferOp.self.input.shape,
                     xferOp.get_shaped_type().shape))
      return Err(());
    if (!xferOp.self.permutation_map.isIdentity())
      return Err(());

    // Bail on illegal rank-reduction: we need to check that the rank-reduced
    // dims are exactly the leading dims. I.e. the following is illegal:
    // ```
    //    %0 = vector.transfer_write %v, %t[0,0], %cst :
    //      vector<2x4xf32>, tensor<2x4xf32>
    //    %1 = tensor.insert_slice %0 into %tt[0,0,0][2,1,4][1,1,1] :
    //      tensor<2x4xf32> into tensor<2x1x4xf32>
    // ```
    //
    // Cannot fold into:
    // ```
    //    %0 = vector.transfer_write %v, %t[0,0,0], %cst :
    //      vector<2x4xf32>, tensor<2x1x4xf32>
    // ```
    // For this, check the trailing `vectorRank` dims of the insert_slice result
    // tensor match the trailing dims of the inferred result tensor.
    i64 rank_reduced =
        insert_op.get_type().rank() - insert_op.self.input.rank();
    i64 vectorRank = xferOp.self.input.rank();
    RankedTensorType inferredSourceTensorType =
        tensor::ExtractSlice::infer_result_type(
            insert_op.get_type(), insert_op.mixed_offsets(),
            insert_op.mixed_sizes(), insert_op.mixed_strides());
    let actualSourceTensorShape = insert_op.self.input.shape;
    if (rank_reduced > 0 &&
        actualSourceTensorShape.take_back(vectorRank) !=
            inferredSourceTensorType.shape.take_back(vectorRank))
    {
      return Err(());}

    SmallVector<Value> indices = getValueOrCreateConstantIndexOp(
        rewriter, insert_op.location(), insert_op.mixed_offsets());
    SmallVector<bool> in_bounds(xferOp.get_transfer_rank(), true);
    rewriter.replace_op_with_new_op<TransferWrite>(insert_op, xferOp.input,
                                                 insert_op.dest, indices,
                                                 ArrayRef<bool>{in_bounds});
    return Ok(());
  }
};

/**
Rewrite tensor::ExtractSlice(TransferWrite) to
TransferWrite(tensor::ExtractSlice) if the full slice is
overwritten and inserted into another tensor. After this rewrite, the
operations bufferize in-place since all of them work on the same slice.

For example:
```mlir
  %0 = vector.transfer_write %vec, %init_tensor[%c0, %c0]
       : vector<8x16xf32>, tensor<8x16xf32>
  %1 = tensor.extract_slice %0[0, 0] [%sz0, %sz1] [1, 1]
       : tensor<8x16xf32> to tensor<?x?xf32>
  %r = tensor.insert_slice %1 into %iter_arg[%iv0, %iv1] [%sz0, %sz1] [1, 1]
       : tensor<?x?xf32> into tensor<27x37xf32>
```
folds to
```mlir
  %0 = tensor.extract_slice %iter_arg[%iv0, %iv1] [%sz0, %sz1] [1, 1]
       : tensor<27x37xf32> to tensor<?x?xf32>
  %1 = vector.transfer_write %vec, %0[%c0, %c0]
       : vector<8x16xf32>, tensor<?x?xf32>
  %r = tensor.insert_slice %1 into %iter_arg[%iv0, %iv1] [%sz0, %sz1] [1, 1]
       : tensor<?x?xf32> into tensor<27x37xf32>
```
*/
struct SwapExtractSliceOfTransferWrite
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(tensor::InsertSliceOp insert_op,
                                PatternRewriter &rewriter) const override {
    if (!insert_op.hasUnitStride())
      return Err(());
    let extract =
        insert_op.input.defining_op<tensor::ExtractSlice>();
    if (!extract || !extract.hasUnitStride() || !extract.hasOneUse())
      return Err(());
    let transferOp = extract.input.defining_op<TransferWrite>();
    if (!transferOp || !transferOp.hasOneUse())
      return Err(());

    // Fail if TransferWrite or tensor::ExtractSlice is
    // rank-reducing.
    if insert_op.self.input.rank() != transferOp.get_transfer_rank() {
      return rewriter.notifyMatchFailure(insert_op,
                                         "use-def chain is rank-reducing");
    }

    // Fail if tensor::ExtractSlice has non-zero offset.
    if !extract.hasZeroOffset() {
      return rewriter.notifyMatchFailure(insert_op,
                                         "ExtractSlice has non-zero offset");
    }

    // Fail if tensor::TransferWrite has non-zero offset.
    if !all_of(transferOp.self.indices, [](Value value {
          return getConstantIntValue(value) == static_cast<i64>(0);
        })) {
      return rewriter.notifyMatchFailure(insert_op,
                                         "TranferWriteOp has non-zero offset");
    }

    // Fail if tensor::ExtractSlice and tensor::InsertSliceOp sizes differ.
    for (let [insertSize, extractSize] :
         zip_equal(insert_op.mixed_sizes(), extract.mixed_sizes())) {
      if !isEqualConstantIntOrValue(insertSize, extractSize) {
        return rewriter.notifyMatchFailure(
            insert_op, "InsertSliceOp and ExtractSlice sizes differ");
      }
    }

    // Fail if the TransferWrite may not overwrite the full tensor.
    assert!(transferOp.self.input.hasStaticShape() &&
           "Expected vector to have a static shape");
    &[i64] vectorShape = transferOp.self.input.shape;
    SmallVector<[i64]> result_shape = applyPermutationMap(
        transferOp.self.permutation_map, transferOp.get_shaped_type().shape);
    if transferOp.self.mask || !vectorShape.equals(result_shape) {
      return rewriter.notifyMatchFailure(
          insert_op, "TransferWrite may not write the full tensor.");
    }

    // Swap the tensor::ExtractSlice in front of the TransferWrite.
    SmallVector<[i64]> newResultShape = applyPermutationMap(
        transferOp.self.permutation_map, insert_op.self.input.shape);
    SmallVector<bool> new_in_bounds;
    for (const let &en : enumerate(newResultShape))
      new_in_bounds.push(en.value() == vectorShape[en.index()]);
    let newExtractOp = rewriter.create<tensor::ExtractSlice>(
        extract.location(), insert_op.self.input, insert_op.dest,
        insert_op.mixed_offsets(), insert_op.mixed_sizes(),
        insert_op.mixed_strides());
    let newTransferWriteOp = rewriter.create<TransferWrite>(
        transferOp.location(), transferOp.input, newExtractOp.output(),
        transferOp.self.indices, transferOp.getPermutationMapAttr(),
        rewriter.get_bool_array_attr(new_in_bounds));
    rewriter.updateRootInPlace(insert_op, [&]() {
      insert_op.getSourceMutable().assign(newTransferWriteOp.output());
    });
    return Ok(());
  }
};

} // namespace


//===----------------------------------------------------------------------===//
// Load
//===----------------------------------------------------------------------===//

pub fn verify_load_store_mem_ref_layout(
    operation: *mut Operation,
    MemRef mem_ref_ty
) -> LogicalResult {
    if !isLastMemrefDimUnitStride(mem_ref_ty) {
        return op.emit_op_error("most minor memref dim must have unit stride");
    }
    return Ok(());
}

//===----------------------------------------------------------------------===//
// Store
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MaskedLoad
//===----------------------------------------------------------------------===//


namespace {
class MaskedLoadFolder final : public OpRewritePattern<MaskedLoad> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult match_and_rewrite(MaskedLoad load,
                                PatternRewriter &rewriter) const override {
    switch (getMaskFormat(load.self.mask)) {
    case MaskFormat::AllTrue:
      rewriter.replace_op_with_new_op<Load>(
          load, load.get_type(), load.getBase(), load.self.indices);
      return Ok(());
    case MaskFormat::AllFalse:
      rewriter.replaceOp(load, load.getPassThru());
      return Ok(());
    case MaskFormat::Unknown:
      return Err(());
    }
    llvm_unreachable("Unexpected 1DMaskFormat on MaskedLoad");
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// MaskedStore
//===----------------------------------------------------------------------===//

namespace {
class MaskedStoreFolder final : public OpRewritePattern<MaskedStore> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult match_and_rewrite(MaskedStore store,
                                PatternRewriter &rewriter) const override {
    switch (getMaskFormat(store.self.mask)) {
    case MaskFormat::AllTrue:
      rewriter.replace_op_with_new_op<Store>(
          store, store.getValueToStore(), store.getBase(), store.self.indices);
      return Ok(());
    case MaskFormat::AllFalse:
      rewriter.eraseOp(store);
      return Ok(());
    case MaskFormat::Unknown:
      return Err(());
    }
    llvm_unreachable("Unexpected 1DMaskFormat on MaskedStore");
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Gather
//===----------------------------------------------------------------------===//

namespace {
class GatherFolder final : public OpRewritePattern<Gather> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult match_and_rewrite(Gather gather,
                                PatternRewriter &rewriter) const override {
    switch (getMaskFormat(gather.self.mask)) {
    case MaskFormat::AllTrue:
      return Err(()); // no unmasked equivalent
    case MaskFormat::AllFalse:
      rewriter.replaceOp(gather, gather.getPassThru());
      return Ok(());
    case MaskFormat::Unknown:
      return Err(());
    }
    llvm_unreachable("Unexpected 1DMaskFormat on GatherFolder");
  }
};
} // namespace


//===----------------------------------------------------------------------===//
// Scatter
//===----------------------------------------------------------------------===//

namespace {
class ScatterFolder final : public OpRewritePattern<Scatter> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult match_and_rewrite(Scatter scatter,
                                PatternRewriter &rewriter) const override {
    switch (getMaskFormat(scatter.self.mask)) {
    case MaskFormat::AllTrue:
      return Err(()); // no unmasked equivalent
    case MaskFormat::AllFalse:
      rewriter.eraseOp(scatter);
      return Ok(());
    case MaskFormat::Unknown:
      return Err(());
    }
    llvm_unreachable("Unexpected 1DMaskFormat on ScatterFolder");
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ExpandLoad
//===----------------------------------------------------------------------===//

namespace {
class ExpandLoadFolder final : public OpRewritePattern<ExpandLoad> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult match_and_rewrite(ExpandLoad expand,
                                PatternRewriter &rewriter) const override {
    switch (getMaskFormat(expand.self.mask)) {
    case MaskFormat::AllTrue:
      rewriter.replace_op_with_new_op<Load>(
          expand, expand.get_type(), expand.getBase(), expand.self.indices);
      return Ok(());
    case MaskFormat::AllFalse:
      rewriter.replaceOp(expand, expand.getPassThru());
      return Ok(());
    case MaskFormat::Unknown:
      return Err(());
    }
    llvm_unreachable("Unexpected 1DMaskFormat on ExpandLoadFolder");
  }
};
} // namespace



//===----------------------------------------------------------------------===//
// CompressStore
//===----------------------------------------------------------------------===//

namespace {
class CompressStoreFolder final : public OpRewritePattern<CompressStore> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult match_and_rewrite(CompressStore compress,
                                PatternRewriter &rewriter) const override {
    switch (getMaskFormat(compress.self.mask)) {
    case MaskFormat::AllTrue:
      rewriter.replace_op_with_new_op<Store>(
          compress, compress.getValueToStore(), compress.getBase(),
          compress.self.indices);
      return Ok(());
    case MaskFormat::AllFalse:
      rewriter.eraseOp(compress);
      return Ok(());
    case MaskFormat::Unknown:
      return Err(());
    }
    llvm_unreachable("Unexpected 1DMaskFormat on CompressStoreFolder");
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ShapeCast
//===----------------------------------------------------------------------===//

/// Returns true if each element of 'a' is equal to the product of a contiguous
/// sequence of the elements of 'b'. Returns false otherwise.
pub fn is_valid_shape_cast(&[i64] a, &[i64] b) -> bool {
    usize rankA = a.len();
    usize rankB = b.len();
    assert!(rankA < rankB);

    usize i = 0;
    usize j = 0;
    while (i < rankA && j < rankB) {
        i64 dimA = a[i];
        i64 dimB = 1;
        while (dimB < dimA && j < rankB)
        dimB *= b[j++];
        if (dimA != dimB)
        break;
        ++i;

        // Handle the case when trailing dimensions are of len 1.
        // Include them into the contiguous sequence.
        let isOne = [](i64 v) { return v == 1; };
        if (i < rankA && all_of(a.slice(i), isOne))
        i = rankA;
        if (j < rankB && all_of(b.slice(j), isOne))
        j = rankB;
    }

    return i == rankA && j == rankB;
}

pub fn verify_vector_shape_cast(
    operation: *mut Operation,
    source_vector_type: VectorType,
    result_vector_type: VectorType
) -> LogicalResult {
    // Check that element type is the same.
    if source_vector_type.element_type != result_vector_type.element_type {
        return op.emit_op_error("source/result vectors must have same element type");
    }
    let source_shape = source_vector_type.shape;
    let result_shape = result_vector_type.shape;

    // Check that product of source dim sizes matches product of result dim sizes.
    let source_dim_product = std::accumulate(
        source_shape.begin(), source_shape.end(), 1LL, std::multiplies<i64>{});
        let result_dim_product = std::accumulate(
        result_shape.begin(), result_shape.end(), 1LL, std::multiplies<i64>{});
    if source_dim_product != result_dim_product {
        return op.emit_op_error("source/result number of elements must match");
    }

    // Check that expanding/contracting rank cases.
    let source_rank = source_vector_type.rank();
    let result_rank = result_vector_type.rank();
    if source_rank < result_rank {
        if !is_valid_shape_cast(source_shape, result_shape) {
            return op.emit_op_error("invalid shape cast");
        }
    } else if source_rank > result_rank {
        if !is_valid_shape_cast(result_shape, source_shape) {
            return op.emit_op_error("invalid shape cast");
        }
    }
    Ok(())
}

namespace {
// Pattern to rewrite a ShapeCast(splat ConstantOp) -> ConstantOp.
class ShapeCastConstantFolder final : public OpRewritePattern<ShapeCast> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(ShapeCast shape_cast_op,
                                PatternRewriter &rewriter) const override {
    let constantOp =
        shape_cast_op.input.defining_op<arith::ConstantOp>();
    if (!constantOp)
      return Err(());
    // Only handle splat for now.
    let dense = constantOp.value().dyn_cast<SplatElementsAttr>();
    if (!dense)
      return Err(());
    let new_attr =
        DenseElementsAttr::get(shape_cast_op.get_type().cast<VectorType>(),
                               dense.getSplatValue<Attribute>());
    rewriter.replace_op_with_new_op<arith::ConstantOp>(shape_cast_op, new_attr);
    return Ok(());
  }
};

/**
Pattern to rewrite a ShapeCast(Broadcast) -> Broadcast.
This only applies when the shape of the broadcast source is a suffix of the
shape of the result (i.e. when broadcast without reshape is expressive
enough to capture the result in a single op).
*/
class ShapeCastBroadcastFolder final : public OpRewritePattern<ShapeCast> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(ShapeCast shape_cast_op,
                                PatternRewriter &rewriter) const override {
    let broadcast_op =
        shape_cast_op.input.defining_op<Broadcast>();
    if (!broadcast_op)
      return Err(());

    let broadcastSourceVectorType =
        broadcast_op.self.input.dyn_cast<VectorType>();
    let broadcastSourceShape = broadcastSourceVectorType
                                    ? broadcastSourceVectorType.shape
                                    : &[i64]{};
    let shapeCastTargetShape = shape_cast_op.getResultVectorType().shape;

    // Bail if `broadcastSourceShape` is not a suffix of the result.
    bool isSuffix = (broadcastSourceShape == shapeCastTargetShape.take_back(
                                                 broadcastSourceShape.len()));
    if !isSuffix {
      return Err(());
    }

    rewriter.replace_op_with_new_op<Broadcast>(
        shape_cast_op, shape_cast_op.getResultVectorType(),
        broadcast_op.input);
    return Ok(());
  }
};

} // namespace


//===----------------------------------------------------------------------===//
// VectorBitCast
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TypeCast
//===----------------------------------------------------------------------===//

pub fn extract_shape(MemRef mem_ref_type) -> SmallVector<[i64; 8]> {
    let vector_type = mem_ref_type.element_type.dyn_cast<VectorType>();
    SmallVector<[i64; 8]> res(mem_ref_type.shape);
    if vector_type {
        res.append(vector_type.shape);
    }
    return res;
}

/// Build the canonical mem_ref_type with a single vector.
/// E.g. memref<4 x 5 x vector<6 x f32>> -> memref<vector<4 x 5 x 6 x f32>>.
void TypeCast::build(builder: &Builder, result: &OperationState,
                       Value source) {
  result.add_operands(source);
  MemRef mem_ref_type = source.get_type().cast<MemRef>();
  vector_type: VectorType =
      VectorType::new(extract_shape(mem_ref_type),
                      get_element_type_or_self(get_element_type_or_self(mem_ref_type)));
  result.add_types(MemRef::get({}, vector_type, MemRefLayoutAttrInterface(),
                                  mem_ref_type.getMemorySpace()));
}

//===----------------------------------------------------------------------===//
// Transpose
//===----------------------------------------------------------------------===//

void Transpose::build(builder: &Builder, result: &OperationState,
                                Value vector, &[i64] transp) {
  VectorType vt = vector.get_type().cast<VectorType>();
  SmallVector<[i64; 4]> transposedShape(vt.rank());
  for (usize i = 0; i < transp.len(); ++i)
    transposedShape[i] = vt.shape[transp[i]];

  result.add_operands(vector);
  result.add_types(VectorType::new(transposedShape, vt.element_type));
  result.add_attribute(getTranspAttrStrName(), builder.get_i64_array_attr(transp));
}

namespace {

// Rewrites two back-to-back Transpose operations into a single Transpose.
class TransposeFolder final : public OpRewritePattern<Transpose> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(Transpose transposeOp,
                                PatternRewriter &rewriter) const override {
    // Wrapper around Transpose::self.transp for cleaner code.
    let getPermutation = [](Transpose transpose) {
      SmallVector<[i64; 4]> permutation;
      transpose.get_transp(permutation);
      return permutation;
    };

    // Composes two permutations: result[i] = permutation1[permutation2[i]].
    let composePermutations = [](&[i64] permutation1,
                                  &[i64] permutation2) {
      SmallVector<[i64; 4]> result;
      for (let index : permutation2)
        result.push(permutation1[index]);
      return result;
    };

    // Return if the input of 'transposeOp' is not defined by another transpose.
    Transpose parentTransposeOp =
        transposeOp.input.defining_op<Transpose>();
    if (!parentTransposeOp)
      return Err(());

    SmallVector<[i64; 4]> permutation = composePermutations(
        getPermutation(parentTransposeOp), getPermutation(transposeOp));
    // Replace 'transposeOp' with a new transpose operation.
    rewriter.replace_op_with_new_op<Transpose>(
        transposeOp, transposeOp.output().get_type(),
        parentTransposeOp.input,
        get_vector_subscript_attr(rewriter, permutation));
    return Ok(());
  }
};

// Folds transpose(broadcast(<scalar>)) into brodcast(<scalar>).
struct FoldTransposedScalarBroadcast final
    : public OpRewritePattern<Transpose> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(Transpose transposeOp,
                                PatternRewriter &rewriter) const override {
    let bcastOp = transposeOp.input.defining_op<Broadcast>();
    if (!bcastOp)
      return Err(());

    let src_vector_type = bcastOp.self.input.dyn_cast<VectorType>();
    if !src_vector_type || src_vector_type.num_elements() == 1 {
      rewriter.replace_op_with_new_op<Broadcast>(
          transposeOp, transposeOp.self.output, bcastOp.input);
      return Ok(());
    }

    return Err(());
  }
};

// Folds transpose(splat x : src_type) : res_type into splat x : res_type.
class FoldTransposeSplat final : public OpRewritePattern<Transpose> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(Transpose transposeOp,
                                PatternRewriter &rewriter) const override {
    let splatOp = transposeOp.input.defining_op<Splat>();
    if (!splatOp)
      return Err(());

    rewriter.replace_op_with_new_op<Splat>(
        transposeOp, transposeOp.self.output, splatOp.getInput());
    return Ok(());
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ConstantMask
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CreateMask
//===----------------------------------------------------------------------===//

namespace {

// Pattern to rewrite a CreateMask with a ConstantMask.
class CreateMaskFolder final : public OpRewritePattern<CreateMask> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match_and_rewrite(CreateMask create_mask_op,
                                PatternRewriter &rewriter) const override {
    // Return if any of 'create_mask_op' operands are not defined by a constant.
    let isNotDefByConstant = [](Value operand) {
      return !isa_and_nonnull<arith::ConstantIndex>(operand.defining_op());
    };
    if (any_of(create_mask_op.get_operands(), isNotDefByConstant))
      return Err(());

    // CreateMask for scalable vectors can be folded only if all dimensions
    // are negative or zero.
    if let vType = create_mask_op.get_type().dyn_cast<VectorType>() {
      if (vType.is_scalable())
        for (let opDim : create_mask_op.get_operands()) {
          APInt intVal;
          if (match_pattern(opDim, m_ConstantInt(&intVal)) &&
              intVal.isStrictlyPositive())
            return Err(());
        }
    }

    // Gather constant mask dimension sizes.
    SmallVector<[i64; 4]> mask_dim_sizes;
    mask_dim_sizes.reserve(create_mask_op.get_num_operands());
    for (operand, max_dim_size) in zip_equal(
             create_mask_op.get_operands(), create_mask_op.get_type().shape) {
      let def_op = operand.defining_op();
      let dim_size = cast<arith::ConstantIndex>(def_op).value();
      dim_size = std::min(dim_size, max_dim_size);
      // If one of dim sizes is zero, set all dims to zero.
      if dim_size <= 0 {
        mask_dim_sizes.assign(create_mask_op.get_type().rank(), 0);
        break;
      }
      mask_dim_sizes.push(dim_size);
    }
    // Replace 'create_mask_op' with ConstantMask.
    rewriter.replace_op_with_new_op<ConstantMask>(
        create_mask_op, create_mask_op.output().get_type(),
        get_vector_subscript_attr(rewriter, mask_dim_sizes));
    return Ok(());
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Mask
//===----------------------------------------------------------------------===//

void Mask::build(
    builder: &Builder, result: &OperationState, Value mask,
    Operation *maskable_op,
    function_ref<void(Builder &, Operation *)> mask_region_builder
) {
    assert!(mask_region_builder,
            "Builder callback for 'mask_region' must be present");

    result.add_operands(mask);
    Builder::InsertionGuard guard(builder);
    Region *mask_region = result.add_region();
    builder.create_block(mask_region);
    mask_region_builder(builder, maskable_op);
}

void Mask::build(
    builder: &Builder, result: &OperationState, TypeRange result_types,
    Value mask, Operation *maskable_op,
    function_ref<void(Builder &, Operation *)> mask_region_builder) {
    build(builder, result, result_types, mask, /*passthru=*/Value(), maskable_op,
            mask_region_builder);
}

void Mask::build(
    builder: &Builder, result: &OperationState, TypeRange result_types, Value mask,
    Value passthru, Operation *maskable_op,
    function_ref<void(Builder &, Operation *)> mask_region_builder) {
    build(builder, result, mask, maskable_op, mask_region_builder);
    if (passthru)
        result.add_operands(passthru);
    result.add_types(result_types);
}

void Mask::ensure_terminator(Region &region, Builder &builder, location: Location) {
    OpTrait::SingleBlockImplicitTerminator<Yield>::Impl<
        Mask>::ensure_terminator(region, builder, loc);
    // Keep the default yield terminator if the number of masked operations is not
    // the expected. This case will trigger a verification failure.
    if (region.front().operations().len() != 2)
        return;

    // Replace default yield terminator with a new one that returns the results
    // from the masked operation.
    Builder op_builder(builder.self.context());
    let masked_op = &region.front().front();
    let old_yield_op = &region.front().back();
    assert!(isa<Yield>(old_yield_op), "Expected Yield");

    op_builder.setInsertionPoint(old_yield_op);
    op_builder.create<Yield>(masked_op.location(), masked_op.outputs());
    old_yield_op.drop_all_references();
    old_yield_op.erase();
}

// MaskingOpInterface definitions.

/// Returns the operation masked by this 'vector.mask'.
Operation *Mask::get_maskable_op() { return &self.mask_region.front().front(); }

/// Returns true if 'vector.mask' has a passthru value.
bool Mask::has_passthru() { return self.passthru != Value(); }

//===----------------------------------------------------------------------===//
// Scan
//===----------------------------------------------------------------------===//

pub fn populate_vector_to_vector_canonicalization_patterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns
      .add<CreateMaskFolder, MaskedLoadFolder, MaskedStoreFolder, GatherFolder,
           ScatterFolder, ExpandLoadFolder, CompressStoreFolder,
           StridedSliceConstantMaskFolder, TransposeFolder>(
          patterns.self.context(), benefit);
}

//===----------------------------------------------------------------------===//
// Splat
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// WarpExecuteOnLane0
//===----------------------------------------------------------------------===//

void WarpExecuteOnLane0::print(&self, p: &OpAsmPrinter) {
    p << "(" << getLaneid() << ")";

    SmallVector<&str> coreAttr = {get_warp_size_attr_name()};
    let warpSizeAttr = get_operation().getAttr(get_warp_size_attr_name());
    p << "[" << warpSizeAttr.cast<IntegerAttr>().get_int() << "]";

    if (!get_args().is_empty())
        p << " args(" << get_args() << " : " << get_args().getTypes() << ")";
    if (!outputs().is_empty())
        p << " -> (" << outputs().getTypes() << ')';
    p << " ";
    p.printRegion(getRegion(),
                    /*printEntryBlockArgs=*/true,
                    /*printBlockTerminators=*/!outputs().is_empty());
    p.print_optional_attr_dict(get_operation().get_attrs(), coreAttr);
}

ParseResult WarpExecuteOnLane0::parse(parser: &OpAsmParser,
                                        result: &OperationState) {
    // Create the region.
    result.regions.reserve(1);
    let warp_region = result.add_region();

    let &builder = parser.builder();
    OpAsmParserUnresolvedOperand lane_id;

    // Parse predicate operand.
    if (parser.parse_l_paren() ||
        parser.parse_operand(lane_id, /*allowResultNumber=*/false) ||
        parser.parse_r_paren())
        return Err(());

    i64 warp_size;
    if (parser.parse_l_square() || parser.parse_integer(warp_size) ||
        parser.parse_r_square())
        return Err(());
    result.add_attribute(get_warp_size_attr_name(OperationName(get_operation_name(),
                                                            builder.self.context())),
                        builder.get_i64_integer_attr(warp_size));

    if (parser.resolve_operand(lane_id, builder.get_index_type(), result.operands))
        return Err(());

    SMLoc inputs_operands_loc;
    SmallVector<OpAsmParserUnresolvedOperand> inputs_operands;
    SmallVector<Type> input_types;
    if succeeded(parser.parseOptionalKeyword("args")) {
        if (parser.parse_l_paren())
        return Err(());

        inputs_operands_loc = parser.get_current_location();
        if (parser.parse_operand_list(inputs_operands) ||
            parser.parse_colon_type_list(input_types) || parser.parse_r_paren())
        return Err(());
    }
    if (parser.resolve_operands(inputs_operands, input_types, inputs_operands_loc,
                                result.operands))
        return Err(());

    // Parse optional results type list.
    if (parser.parseOptionalArrowTypeList(result.types))
        return Err(());
    // Parse the region.
    if (parser.parse_region(*warp_region, /*arguments=*/{},
                            /*argTypes=*/{}))
        return Err(());
    WarpExecuteOnLane0::ensure_terminator(*warp_region, builder, result.location);

    // Parse the optional attribute list.
    if (parser.parse_optional_attr_dict(result.attributes))
        return Err(());
    return Ok(());
}

void WarpExecuteOnLane0::get_successor_regions(
    &[Attribute] operands,
    SmallVector<RegionSuccessor> &regions
) {
    if index {
        regions.push(RegionSuccessor(outputs()));
        return;
    }

    // The warp region is always executed
    regions.push(RegionSuccessor(&get_warp_region()));
}

void WarpExecuteOnLane0::build(builder: &Builder, result: &OperationState,
                                 TypeRange result_types, Value lane_id,
                                 i64 warp_size) {
  build(builder, result, result_types, lane_id, warp_size,
        /*operands=*/std::nullopt, /*argTypes=*/std::nullopt);
}

void WarpExecuteOnLane0::build(builder: &Builder, result: &OperationState,
                                 TypeRange result_types, Value lane_id,
                                 i64 warp_size, ValueRange args,
                                 TypeRange block_arg_types) {
    result.add_operands(lane_id);
    result.add_attribute(get_attribute_names()[0],
                        builder.get_i64_integer_attr(warp_size));
    result.add_types(result_types);
    result.add_operands(args);
    assert!(args.len() == block_arg_types.len());
    Builder::InsertionGuard guard(builder);
    Region *warp_region = result.add_region();
    Block *block = builder.create_block(warp_region);
    for (r#type, arg) in zip_equal(block_arg_types, args)) {
        block.add_input(r#type, arg.location());
    }
}

/// Helper check if the distributed vector type is consistent with the expanded
/// type and distributed len.
pub fn verify_distributed_type(
    Type expanded,
    Type distributed,
    operation: *mut Operation,
    i64 warp_size
) -> LogicalResult {
    // If the types matches there is no distribution.
    if expanded == distributed {
        return Ok(());
    }
    let expanded_vec_type = expanded.dyn_cast<VectorType>();
    let distributed_vec_type = distributed.dyn_cast<VectorType>();
    if !expanded_vec_type || !distributed_vec_type {
        return op.emit_op_error(
            "Expected vector type for distributed operands.");
    }
    if expanded_vec_type.rank() != distributed_vec_type.rank()
    || expanded_vec_type.element_type != distributed_vec_type.element_type
    {
        return op.emit_op_error(
            "Expected distributed vectors to have same rank and element type.");
    }
    let mut found_distributed_dim = false;
    for i in 0..expanded_vec_type.rank() {
        if expanded_vec_type.dim_size(i) == distributed_vec_type.dim_size(i) {
            continue;
        }
        if (expanded_vec_type.dim_size(i) ==
            distributed_vec_type.dim_size(i) * warp_size)
        {
            if found_distributed_dim {
                return op.emit_op_error()
                    << "Expected only one dimension to be distributed from "
                    << expanded_vec_type << " to " << distributed_vec_type;
            }
            found_distributed_dim = true;
            continue;
        }
        return op.emit_op_error(
            "Incompatible distribution dimensions from {} to {}",
            expanded_vec_type,
            distributed_vec_type
        );
    }
    Ok(())
}

LogicalResult WarpExecuteOnLane0::verify() {
    if (get_args().len() != get_warp_region().num_arguments())
        return emit_op_error(
            "Expected same number op arguments and block arguments.");
    let yield =
        cast<Yield>(get_warp_region().blocks().begin().getTerminator());
    if (yield.get_num_operands() != num_output())
        return emit_op_error(
            "Expected same number of yield operands and return values.");
    let warp_size = get_warp_size();
    for (region_arg, arg) in
        zip_equal(get_warp_region().get_arguments(), get_args()))
    {
        if (failed(verify_distributed_type(
            region_arg.get_type(), arg.get_type(),
            warp_size, get_operation()))) {
            return Err(());
        }
    }
    for (yield_operand, result)
    in zip_equal(r#yield.get_operands(), outputs()))
    {
        if (failed(verify_distributed_type(
            yield_operand.get_type(), result.get_type(),
            warp_size, get_operation()))) {
            return Err(());
        }
    }
    Ok(())
}

bool WarpExecuteOnLane0::are_types_compatible(Type lhs, Type rhs) {
  return succeeded(
      verify_distributed_type(lhs, rhs, get_warp_size(), get_operation()));
}

pub fn make_arith_reduction(
    Builder &b, location: Location,
    kind: CombiningKind,
    Value v1,
    Value v2
) -> Value {
    let t1 = get_element_type_or_self(v1.get_type());
    let t2 = get_element_type_or_self(v2.get_type());
    match (kind) {
    case CombiningKind::ADD:
        if (t1.is_int_or_index() && t2.is_int_or_index())
        return b.create_or_fold<arith::AddIOp>(loc, v1, v2);
        else if (t1.isa<FloatType>() && t2.isa<FloatType>())
        return b.create_or_fold<arith::AddFOp>(loc, v1, v2);
        llvm_unreachable("invalid value types for ADD reduction");
    case CombiningKind::AND:
        assert!(t1.is_int_or_index() && t2.is_int_or_index(), "Expected int values");
        return b.create_or_fold<arith::AndIOp>(loc, v1, v2);
    case CombiningKind::MAXF:
        assert!(t1.isa<FloatType>() && t2.isa<FloatType>() &&
            "Expected float values");
        return b.create_or_fold<arith::MaxFOp>(loc, v1, v2);
    case CombiningKind::MINF:
        assert!(t1.isa<FloatType>() && t2.isa<FloatType>() &&
            "Expected float values");
        return b.create_or_fold<arith::MinFOp>(loc, v1, v2);
    case CombiningKind::MAXSI:
        assert!(t1.is_int_or_index() && t2.is_int_or_index(), "Expected int values");
        return b.create_or_fold<arith::MaxSIOp>(loc, v1, v2);
    case CombiningKind::MINSI:
        assert!(t1.is_int_or_index() && t2.is_int_or_index(), "Expected int values");
        return b.create_or_fold<arith::MinSIOp>(loc, v1, v2);
    case CombiningKind::MAXUI:
        assert!(t1.is_int_or_index() && t2.is_int_or_index(), "Expected int values");
        return b.create_or_fold<arith::MaxUIOp>(loc, v1, v2);
    case CombiningKind::MINUI:
        assert!(t1.is_int_or_index() && t2.is_int_or_index(), "Expected int values");
        return b.create_or_fold<arith::MinUIOp>(loc, v1, v2);
    case CombiningKind::MUL:
        if (t1.is_int_or_index() && t2.is_int_or_index())
        return b.create_or_fold<arith::MulIOp>(loc, v1, v2);
        else if (t1.isa<FloatType>() && t2.isa<FloatType>())
        return b.create_or_fold<arith::MulFOp>(loc, v1, v2);
        llvm_unreachable("invalid value types for MUL reduction");
    case CombiningKind::OR:
        assert!(t1.is_int_or_index() && t2.is_int_or_index(), "Expected int values");
        return b.create_or_fold<arith::OrIOp>(loc, v1, v2);
    case CombiningKind::XOR:
        assert!(t1.is_int_or_index() && t2.is_int_or_index(), "Expected int values");
        return b.create_or_fold<arith::XOrIOp>(loc, v1, v2);
    };
    llvm_unreachable("unknown CombiningKind");
}

//===----------------------------------------------------------------------===//
// Vector Masking Utilities
//===----------------------------------------------------------------------===//

/// Create the vector.yield-ended region of a vector.mask op with `maskable_op`
/// as masked operation.
void create_mask_op_region(builder: &Builder,
                            Operation *maskable_op) {
  assert!(maskable_op.block(), "Maskable must be inserted into a block");
  Block *insBlock = builder.getInsertionBlock();
  // Create a block and move the op to that block.
  insBlock.operations().splice(
      insBlock.begin(), maskable_op.block().operations(), maskable_op);
  builder.create<Yield>(maskable_op.location(), maskable_op.outputs());
}

/**
Creates a vector.mask operation around a maskable operation. Returns the vector.mask operation if the mask provided is valid. Otherwise, returns the maskable operation itself.
*/
Operation *mask_operation(
    RewriterBase &rewriter,
    Operation *maskable_op, Value mask) {
    if (!mask)
        return maskable_op;
    return rewriter.create<Mask>(
        maskable_op.location(),
        maskable_op.get_result_types(), mask, maskable_op,
        create_mask_op_region);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Vector/IR/VectorOpsAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Vector/IR/VectorOps.cpp.inc"
