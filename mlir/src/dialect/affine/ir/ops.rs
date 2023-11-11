use crate::mlir::{dialect::memref::ir::operations::{View, SubView, Alloc}, ir::{region::Region, operation::Operation, value::Input}};

/**
A utility function to check if a value is defined at the top level of
`region` or is an argument of `region`. A value of index type defined at the
top level of a `AffineScope` region is always a valid symbol for all
uses in that region.
*/
bool is_top_level_value(value: Value, region: *mut Region) {
    if (auto arg = value.dyn_cast<Input>())
        return arg.parent_region() == region;
    return value.defining_op().parent_region() == region;
}

/**
Checks if `value` known to be a legal affine dimension or symbol in `src`
region remains legal if the operation that uses it is inlined into `dest`
with the given value mapping. `legalityCheck` is either `is_valid_dim` or
`is_valid_symbol`, depending on the value being required to remain a valid
dimension or symbol.
*/
pub fn remains_legal_after_inline(
    value: Value, src: *mut Region, dest: *mut Region,
    mapping: &IRMapping,
    function_ref<bool(Value, *mut Region)> legalityCheck
) -> bool {
    // If the value is a valid dimension for any other reason than being
    // a top-level value, it will remain valid: constants get inlined
    // with the function, transitive affine applies also get inlined and
    // will be checked themselves, etc.
    if (!is_top_level_value(value, src))
        return true;

    // If it's a top-level value because it's a block operand, i.e. a
    // function argument, check whether the value replacing it after
    // inlining is a valid dimension in the new region.
    if (value.isa<Input>())
        return legalityCheck(mapping.lookup(value), dest);

    // If it's a top-level value because it's defined in the region,
    // it can only be inlined if the defining op is a constant or a
    // `dim`, which can appear anywhere and be valid, since the defining
    // op won't be top-level anymore after inlining.
    Attribute operand_cst;
    bool isDimLikeOp = isa<ShapedDimOpInterface>(value.defining_op());
    return match_pattern(value.defining_op(), m_Constant(&operand_cst)) ||
            isDimLikeOp;
}

/**
Checks if all values known to be legal affine dimensions or symbols in `src`
remain so if their respective users are inlined into `dest`.
*/
pub fn remains_legal_after_inline(
    values: ValueRange, src: *mut Region, dest: *mut Region,
    mapping: &IRMapping,
    function_ref<bool(Value, *mut Region)> legalityCheck
) -> bool {
    return llvm::all_of(values, [&](Value v) {
        return remains_legal_after_inline(v, src, dest, mapping, legalityCheck);
    });
}

/**
Checks if an affine read or write operation remains legal after inlining
from `src` to `dest`.
*/
template <typename OpTy>
static bool
pub fn remains_legal_after_inline(
    op: OpTy, src: *mut Region, dest: *mut Region,
    mapping: &IRMapping
) -> bool {
    assert!(llvm::is_one_of<OpTy, AffineReadOpInterface,
                                    AffineWriteOpInterface>::value,
                    "only ops with affine read/write interface are supported");

    map: AffineMap = op.get_affine_map();
    ValueRange dim_operands = op.map_operands().take_front(map.num_dims());
    ValueRange symbol_operands =
        op.map_operands().take_back(map.num_symbols());
    if (!remains_legal_after_inline(
            dim_operands, src, dest, mapping,
            static_cast<bool (*)(Value, *mut Region)>(is_valid_dim))) {
        return false;
    }
    if (!remains_legal_after_inline(
            symbol_operands, src, dest, mapping,
            static_cast<bool (*)(Value, *mut Region)>(is_valid_symbol))) {
        return false;
    }
    return true;
}

/// Checks if an affine apply operation remains legal after inlining from `src`
/// to `dest`.
//  Use "unused attribute" marker to silence clang-tidy warning stemming from
//  the inability to see through "llvm::TypeSwitch".
template <>
LLVM_ATTRIBUTE_UNUSED
pub fn remains_legal_after_inline(
    op: Apply,
    src: *mut Region, dest: *mut Region,
    mapping: &IRMapping
) -> bool {
    // If it's a valid dimension, we need to check that it remains so.
    if is_valid_dim(op.output(), src) {
        return remains_legal_after_inline(
            op.map_operands(), src, dest, mapping,
            static_cast<bool (*)(Value, *mut Region)>(is_valid_dim));
    }

    // Otherwise it must be a valid symbol, check that it remains so.
    return remains_legal_after_inline(
        op.map_operands(), src, dest, mapping,
        static_cast<bool (*)(Value, *mut Region)>(is_valid_symbol));
}

//===----------------------------------------------------------------------===//
// AffineDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining with affine
/// operations.
struct AffineInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    //===--------------------------------------------------------------------===//
    // Analysis Hooks
    //===--------------------------------------------------------------------===//

    /**
    Returns true if the given region 'src' can be inlined into the region
    'dest' that is attached to an operation registered to the current dialect.
    'wouldBeCloned' is set if the region is cloned into its new location
    rather than moved, indicating there may be other users.
    */
    bool is_legal_to_inline(dest: *mut Region, src: *mut Region, bool wouldBeCloned,
                        IRMapping &valueMapping) const final {
        // We can inline into affine loops and conditionals if this doesn't break
        // affine value categorization rules.
        *mut OperationdestOp = dest.parent();
        if (!isa<AffineParallel, For, If>(destOp))
        return false;

        // Multi-block regions cannot be inlined into affine constructs, all of
        // which require single-block regions.
        if (!llvm::hasSingleElement(*src))
        return false;

        // Side-effecting operations that the affine dialect cannot understand
        // should not be inlined.
        Block &srcBlock = src->front();
        for (Operation &op in srcBlock) {
        // Ops with no side effects are fine,
        if auto iface = dyn_cast<MemoryEffectOpInterface>(op) {
            if (iface.hasNoEffect())
            continue;
        }

        // Assuming the inlined region is valid, we only need to check if the
        // inlining would change it.
        bool remainsValid =
            llvm::TypeSwitch<*mut Operation, bool>(&op)
                .Case<Apply, AffineReadOpInterface,
                        AffineWriteOpInterface>(|op| {
                    return remains_legal_after_inline(op, src, dest, valueMapping);
                })
                .Default([](*mut Operation) {
                    // Conservatively disallow inlining ops we cannot reason about.
                    return false;
                });

        if (!remainsValid)
            return false;
        }

        return true;
    }

    /// Returns true if the given operation 'op', that is registered to this
    /// dialect, can be inlined into the given region, false otherwise.
    bool is_legal_to_inline(operation: *mut Operation, region: *mut Region, bool wouldBeCloned,
                        IRMapping &valueMapping) const final {
        // Always allow inlining affine operations into a region that is marked as
        // affine scope, or into affine loops and conditionals. There are some edge
        // cases when inlining *into* affine structures, but that is handled in the
        // other 'is_legal_to_inline' hook above.
        parent: *mut Operation = region.parent();
        return parent.has_trait<AffineScope>() ||
            isa<For, AffineParallel, If>(parent);
    }

    /// Affine regions should be analyzed recursively.
    bool should_analyze_recursively(operation: *mut Operation) const final { return true; }
};
} // namespace

//===----------------------------------------------------------------------===//
// AffineDialect
//===----------------------------------------------------------------------===//

void AffineDialect::initialize() {
  addOperations<AffineDmaStart, AffineDmaWait,
#define GET_OP_LIST
#include "mlir/Dialect/Affine/IR/AffineOps.cpp.inc"
                >();
  addInterfaces<AffineInlinerInterface>();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
*mut Operation AffineDialect::materialise_constant(builder: &Builder,
                                              Attribute value, r#type: Type,
                                              location: Location) {
  return builder.create<arith::ConstantOp>(location, type, value);
}

/**
A utility function to check if a value is defined at the top level of an
op with trait `AffineScope`. If the value is defined in an unlinked region,
conservatively assume it is not top-level. A value of index type defined at
the top level is always a valid symbol.
*/
pub fn is_top_level_value(value: Value) -> bool {
    if let arg = value.dyn_cast<Input>() {
        /*
        The block owning the argument may be unlinked, e.g. when the surrounding region has not yet been attached to an Op, at which point the parent Op is null.
        */
        let parent = arg.owner().parent();
        return parent && parent.has_trait<AffineScope>();
    }
    // The defining Op may live in an unlinked block so its parent Op may be null.
    let parent = value.defining_op().parent();
    parent && parent.has_trait<AffineScope>()
}

/**
Returns the closest region enclosing `op` that is held by an operation with
trait `AffineScope`; `null(` if there is no such region.
*/
pub fn get_affine_scope(mut operation: *mut Operation) -> *mut Region {
    loop {
        let parent = operation.parent();
        if parent.is_null() {
            break;
        }
        if parent.has_trait<AffineScope>() {
            return operation.parent_region();
        }
        operation = parent;
    }
    null()
}

/**
A Value can be used as a dimension id iff it meets one of the following conditions:

- It is valid as a symbol.
- It is an induction variable.
- It is the result of affine apply operation with dimension id arguments.
*/
pub fn is_valid_dim(value: Value) -> bool {
    // The value must be an index type.
    if !value.get_type().is_index() {
        return false;
    }

    let def_op = value.defining_op()
    if !def_op.is_null() {
        return is_valid_dim(value, get_affine_scope(def_op));
    }

    // This value has to be a block argument for an op that has the
    // `AffineScope` trait or for an affine.for or affine.parallel.
    let parent = value.cast<Input>().owner().parent();
    parent
        && (parent.has_trait<AffineScope>()
            || isa<For, AffineParallel>(parent))
}

/**
Value can be used as a dimension id iff it meets one of the following
conditions:
*) It is valid as a symbol.
*) It is an induction variable.
*) It is the result of an affine apply operation with dimension id operands.
*/
pub fn is_valid_dim(value: Value, region: *mut Region) -> bool {
    // The value must be an index type.
    if !value.get_type().is_index() {
        return false;
    }

    // All valid symbols are okay.
    if is_valid_symbol(value, region) {
        return true;
    }

    let op = value.defining_op();
    if !op {
        // This value has to be a block argument for an affine.for or an
        // affine.parallel.
        let parent = value.cast<Input>().owner().parent();
        return isa<For, AffineParallel>(parent);
    }

    // Affine apply operation is ok if all of its operands are ok.
    if auto apply_op = dyn_cast<Apply>(op) {
        return apply_op.is_valid_dim(region);
    }
    // The dim op is okay if its operand memref/tensor is defined at the top
    // level.
    if auto dim_op = dyn_cast<ShapedDimOpInterface>(op) {
        return is_top_level_value(dim_op.get_shaped_value());
    }
    return false;
}

/**
Returns true if the 'index' dimension of the `memref` defined by
`memref_def_op` is a statically  shaped one or defined using a valid symbol
for `region`.
*/
template <typename AnyMemRefDefOp>
pub fn is_mem_ref_size_valid_symbol(
    AnyMemRefDefOp memref_def_op,
    usize index,
    region: *mut Region
) -> bool {
    auto mem_ref_type = memref_def_op.get_type();
    // Statically shaped.
    if !mem_ref_type.is_dynamic_dim(index) {
        return true;
    }
    // Get the position of the dimension among dynamic dimensions;
    let dynamic_dim_pos = mem_ref_type.dynamic_dim_index(index);
    return is_valid_symbol(
        *(memref_def_op.dynamic_sizes().begin() + dynamic_dim_pos),
        region);
}

/// Returns true if the result of the dim op is a valid symbol for `region`.
pub fn is_dim_op_valid_symbol(
    dim_op: ShapedDimOpInterface,
    region: *mut Region
) -> bool {
    // The dim op is okay if its source is defined at the top level.
    if is_top_level_value(dim_op.get_shaped_value()) {
        return true;
    }

    // Conservatively handle remaining BlockArguments as non-valid symbols.
    // E.g. scf.for iter_args.
    if dim_op.get_shaped_value().template isa<Input>() {
        return false;
    }

    // The dim op is also okay if its operand memref is a view/subview whose
    // corresponding len is a valid symbol.
    let index = constant_int_value(dim_op.dimension());
    assert!(index.has_value(),
            "Expect only `dim` operations with a constant index");
    let i = index.value();
    return TypeSwitch<*mut Operation, bool>(
        dim_op.get_shaped_value().defining_op())
        .Case<View, SubView, Alloc>(
            |op| { return is_mem_ref_size_valid_symbol(op, i, region); })
        .Default([](*mut Operation) { return false; });
}

/**
A value can be used as a symbol (at all its use sites) iff it meets one of
the following conditions:
*) It is a constant.
*) Its defining op or block arg appearance is immediately enclosed by an op
   with `AffineScope` trait.
*) It is the result of an affine.apply operation with symbol operands.
*) It is a result of the dim op on a memref whose corresponding len is a
   valid symbol.
*/
pub fn is_valid_symbol(value: Value) -> bool {
    if !value {
        return false;
    }

    // The value must be an index type.
    if !value.get_type().is_index() {
        return false;
    }

    // Check that the value is a top level value.
    if is_top_level_value(value) {
        return true;
    }

    if auto *def_op = value.defining_op() {
        return is_valid_symbol(value, get_affine_scope(def_op));
    }

    false
}

/**
A value can be used as a symbol for `region` iff it meets one of the
following conditions:
*) It is a constant.
*) It is the result of an affine apply operation with symbol arguments.
*) It is a result of the dim op on a memref whose corresponding len is
   a valid symbol.
*) It is defined at the top level of 'region' or is its argument.
*) It dominates `region`'s parent op.
If `region` is null, conservatively assume the symbol definition scope does
not exist and only accept the values that would be symbols regardless of
the surrounding region structure, i.e. the first three cases above.
*/
pub fn is_valid_symbol(value: Value, region: *mut Region) -> bool {
    // The value must be an index type.
    if !value.get_type().is_index() {
        return false;
    }

    // A top-level value is a valid symbol.
    if region && ::is_top_level_value(value, region) {
        return true;
    }

    auto *def_op = value.defining_op();
    if !def_op {
        // A block argument that is not a top-level value is a valid symbol if it
        // dominates region's parent op.
        let region_op = region ? region.parent() : null();
        if region_op && !region_op.has_trait<IsIsolatedFromAbove>() {
            if (auto *parent_op_region = region.parent().parent_region()){
                return is_valid_symbol(value, parent_op_region);
            }
        }
        return false;
    }

    // Constant operation is ok.
    Attribute operand_cst;
    if (match_pattern(def_op, m_Constant(&operand_cst))){ 
        return true;
    }

    // Affine apply operation is ok if all of its operands are ok.
    if auto apply_op = dyn_cast<Apply>(def_op) {
        return apply_op.is_valid_symbol(region);
    }

    // Dim op results could be valid symbols at any level.
    if auto dim_op = dyn_cast<ShapedDimOpInterface>(def_op) {
        return is_dim_op_valid_symbol(dim_op, region);
    }

    // Check for values dominating `region`'s parent op.
    let region_op = region ? region.parent() : null();
    if region_op && !region_op.has_trait<IsIsolatedFromAbove>() {
        if auto *parent_region = region.parent().parent_region() {
            return is_valid_symbol(value, parent_region);
        }
    }

    false
}

/// Returns true if 'value' is a valid index to an affine operation (e.g.
/// affine.load, affine.store, affine.dma_start, affine.dma_wait) where
/// `region` provides the polyhedral symbol scope. Returns false otherwise.
pub fn is_valid_affine_index_operand(value: Value, region: *mut Region)
    -> bool
{
    is_valid_dim(value, region) || is_valid_symbol(value, region)
}

/// Prints dimension and symbol list.
pub fn print_dim_and_symbol_list(
    Operation::operand_iterator begin,
    Operation::operand_iterator end,
    num_dims: usize,
    printer: &OpAsmPrinter
) {
    OperandRange operands(begin, end);
    printer << '(' << operands.take_front(num_dims) << ')';
    if (operands.len() > num_dims)
        printer << '[' << operands.drop_front(num_dims) << ']';
}

/// Parses dimension and symbol list and returns true if parsing failed.
pub fn parse_dim_and_symbol_list(
    parser: &OpAsmParser,
    operands: &mut SmallVector<[Value]>,
    usize &num_dims
) -> ParseResult {
    SmallVector<[UnresolvedOperand; 8]> op_infos;
    if parser.parse_operand_list(op_infos, Delimiter::Paren) {
        return Err(());
    }
    // Store number of dimensions for validation by caller.
    num_dims = op_infos.len();

    // Parse the optional symbol operands.
    let index_type = parser.builder().get_index_type();
    failure(
        parser.parse_operand_list(
            op_infos, Delimiter::OptionalSquare)
        || parser.resolve_operands(op_infos, index_type, operands))
}

/// Utility function to verify that a set of operands are valid dimension and
/// symbol identifiers. The operands should be laid out such that the dimension
/// operands are before the symbol operands. This function returns failure if
/// there was an invalid operand. An operation is provided to emit any necessary
/// errors.
template <typename OpTy>
pub fn verify_dim_and_symbol_identifiers(
    op: &OpTy,
    Operation::operand_range operands,
    num_dims: usize
) -> LogicalResult {
    let mut op_it = 0;
    for operand in operands {
        if op_it++ < num_dims {
            if !is_valid_dim(operand, get_affine_scope(op)) {
                return op.emit_op_error("Operand cannot be used as a dimension id");
            }
        } else if !is_valid_symbol(operand, get_affine_scope(op)) {
            return op.emit_op_error("Operand cannot be used as a symbol");
        }
    }
    return Ok(());
}

//===----------------------------------------------------------------------===//
// Apply
//===----------------------------------------------------------------------===//



// The result of the affine apply operation can be used as a dimension id if all
// its operands are valid dimension ids.
bool Apply::is_valid_dim() {
  return llvm::all_of(self.inputs,
                      |op| { return is_valid_dim(op); });
}

// The result of the affine apply operation can be used as a dimension id if all
// its operands are valid dimension ids with the parent operation of `region`
// defining the polyhedral scope for symbols.
bool Apply::is_valid_dim(region: *mut Region) {
  return llvm::all_of(self.inputs,
                      |op| { return is_valid_dim(op, region); });
}

// The result of the affine apply operation can be used as a symbol if all its
// operands are symbols.
bool Apply::is_valid_symbol() {
  return llvm::all_of(self.inputs,
                      |op| { return is_valid_symbol(op); });
}

// The result of the affine apply operation can be used as a symbol in `region`
// if all its operands are symbols in `region`.
bool Apply::is_valid_symbol(region: *mut Region) {
    return llvm::all_of(self.inputs, |operand| {
        return is_valid_symbol(operand, region);
    });
}

FoldResult Apply::fold(&self) {
    // Fold dims and symbols to existing values.
    let expr = self.map.output(0);
    if auto dim = expr.dyn_cast<AffineDimExpr>() {
        return get_operand(dim.index);
    }
    if auto sym = expr.dyn_cast<AffineSymbolExpr>() {
        return get_operand(self.map.num_dims() + sym.index);
    }

    // Otherwise, default to folding the map.
    SmallVector<[Attribute; 1]> result;
    if failed(self.map.constant_fold(self.map_operands, result)) {
        return {};
    }
    result[0]
}

/// Returns the largest known divisor of `e`. Exploits information from the
/// values in `operands`.
pub fn get_largest_known_divisor(e: AffineExpr, operands: &[Value]) -> i64 {
    // This method isn't aware of `operands`.
    i64 div = e.get_largest_known_divisor();

    /*
    We now make use of operands for the case `e` is a dim expression.
    TODO: More powerful simplification would have to modify
    get_largest_known_divisor to take `operands` and exploit that information as
    well for dim/sym expressions, but in that case, get_largest_known_divisor
    can't be part of the IR library but of the `Analysis` library. The IR
    library can only really depend on simple O(1) checks.
    */
    auto dim_expr = e.dyn_cast<AffineDimExpr>();
    // If it's not a dim expr, `div` is the best we have.
    if !dim_expr {
        return div;
    }

    /*
    We simply exploit information from loop IVs.
    We don't need to use get_largest_known_divisor_of_value since the other
    desired simplifications are expected to be part of other
    canonicalisations. Also, get_largest_known_divisor_of_value is part of the
    LoopAnalysis library.
    */
    let operand = operands[dim_expr.index];
    let mut operand_divisor = 1;
    // TODO: With the right accessors, this can be extended to
    // LoopLikeOpInterface.
    if For r#for = get_for_induction_var_owner(operand) {
        if r#for.has_constant_lower_bound()
        && r#for.get_constant_lower_bound() == 0
        {
            operand_divisor = r#for.get_step();
        } else {
            let lb_largest_known_divisor =
                r#for.self.lower_bound_map().get_largest_known_divisor_of_map_exprs();
            operand_divisor = std::gcd(lb_largest_known_divisor, r#for.get_step());
        }
    }
    operand_divisor
}

/// Check if `e` is known to be: 0 <= `e` < `k`. Handles the simple cases of `e`
/// being an affine dim expression or a constant.
pub fn is_non_negative_bounded_by(
    e: AffineExpr, operands: &[Value],
    i64 k
) -> bool {
    if auto const_expr = e.dyn_cast<AffineConstantExpr>() {
        i64 const_val = const_expr.get_value();
        return const_val >= 0 && const_val < k;
    }
    let dim_expr = e.dyn_cast<AffineDimExpr>();
    if !dim_expr {
        return false;
    }
    let operand = operands[dim_expr.index];
    // TODO: With the right accessors, this can be extended to
    // LoopLikeOpInterface.
    if For r#for = get_for_induction_var_owner(operand) {
        if r#for.has_constant_lower_bound()
        && r#for.get_constant_lower_bound() >= 0
        && r#for.has_constant_upper_bound()
        && r#for.get_constant_upper_bound() <= k
        {
            return true;
        }
    }

    // We don't consider other cases like `operand` being defined by a constant or
    // an affine.apply op since such cases will already be handled by other
    // patterns and propagation of loop IVs or constant would happen.
    false
}

/// Check if expression `e` is of the form d*e_1 + e_2 where 0 <= e_2 < d.
/// Set `div` to `d`, `quotient_times_div` to e_1 and `rem` to e_2 if the
/// expression is in that form.
pub fn is_q_times_d_plus_r(
    e: AffineExpr,
    operands: &[Value],
    div: &mut i64,
    quotient_times_div: &mut AffineExpr,
    rem: &mut AffineExpr
) -> bool {
    let bin = e.dyn_cast<AffineBinaryOpExpr>();
    if !bin || bin.kind != AffineExprKind::Add {
        return false;
    }

    let llhs = bin.lhs;
    let rlhs = bin.rhs;
    div = get_largest_known_divisor(llhs, operands);
    if is_non_negative_bounded_by(rlhs, operands, div) {
        quotient_times_div = llhs;
        rem = rlhs;
        return true;
    }
    div = get_largest_known_divisor(rlhs, operands);
    if is_non_negative_bounded_by(llhs, operands, div) {
        quotient_times_div = rlhs;
        rem = llhs;
        return true;
    }
    false
}

/// Simplify `expr` while exploiting information from the values in `operands`.
pub fn simplify_expr_and_operands(
    expr: &mut AffineExpr,
    operands: &[Value]
) {
    // We do this only for certain floordiv/mod expressions.
    let mut bin_expr = expr.dyn_cast<AffineBinaryOpExpr>();
    if !bin_expr {
        return;
    }

    // Simplify the child expressions first.
    let mut lhs = bin_expr.lhs;
    let mut rhs = bin_expr.rhs;
    simplify_expr_and_operands(lhs, operands);
    simplify_expr_and_operands(rhs, operands);
    expr = get_affine_binary_op_expr(bin_expr.kind, lhs, rhs);

    bin_expr = expr.dyn_cast<AffineBinaryOpExpr>();
    if (!bin_expr || (bin_expr.kind != AffineExprKind::FloorDiv &&
                    bin_expr.kind != AffineExprKind::Mod)) {
        return;
    }

    // The `lhs` and `rhs` may be different post construction of simplified expr.
    lhs = bin_expr.lhs;
    rhs = bin_expr.rhs;
    let rhs_const = rhs.dyn_cast<AffineConstantExpr>();
    if !rhs_const {
        return;
    }

    let rhs_const_val = rhs_const.get_value();
    // Undefined exprsessions aren't touched; IR can still be valid with them.
    if rhs_const_val == 0 {
        return;
    }

    AffineExpr quotient_times_div, rem;
    i64 divisor;

    /*
    Simplify expressions of the form e = (e_1 + e_2) floordiv c or (e_1 + e_2)
    mod c, where e_1 is a multiple of `k` and 0 <= e_2 < k. In such cases, if
    `c` % `k` == 0, (e_1 + e_2) floordiv c can be simplified to e_1 floordiv c.
    And when k % c == 0, (e_1 + e_2) mod c can be simplified to e_2 mod c.
    */
    if is_q_times_d_plus_r(lhs, operands, divisor, quotient_times_div, rem) {
        if (rhs_const_val % divisor == 0 &&
            bin_expr.kind == AffineExprKind::FloorDiv) {
            expr = quotient_times_div.floorDiv(rhs_const);
        } else if
            divisor % rhs_const_val == 0
            && bin_expr.kind == AffineExprKind::Mod
        {
            expr = rem % rhs_const;
        }
        return;
    }

    // Handle the simple case when the LHS expression can be either upper
    // bounded or is a known multiple of RHS constant.
    // lhs floordiv c -> 0 if 0 <= lhs < c,
    // lhs mod c -> 0 if lhs % c = 0.
    if ((is_non_negative_bounded_by(lhs, operands, rhs_const_val) &&
        bin_expr.kind == AffineExprKind::FloorDiv) ||
        (get_largest_known_divisor(lhs, operands) % rhs_const_val == 0 &&
        bin_expr.kind == AffineExprKind::Mod)) {
        expr = get_affine_constant_expr(0, expr.context());
    }
}

/// Simplify the map while exploiting information on the values in `operands`.
//  Use "unused attribute" marker to silence warning stemming from the inability
//  to see through the template expansion.
LLVM_ATTRIBUTE_UNUSED
pub fn simplify_map_with_operands(map: &mut AffineMap, operands: &[Value]) {
    assert!(map.num_inputs() == operands.len(), "Invalid operands for map");
    SmallVector<AffineExpr> new_results;
    new_results.reserve(map.num_outputs());
    for expr in map.outputs() {
        simplify_expr_and_operands(expr, operands);
        new_results.push(expr);
    }
    map = AffineMap::new(map.num_dims(), map.num_symbols(), new_results,
                        map.context());
}

/**
Replace all occurrences of AffineExpr at position `pos` in `map` by the
defining Apply expression and operands.
When `dim_or_symbol_position < dims.len()`, AffineDimExpr@[pos] is replaced.
When `dim_or_symbol_position >= dims.len()`,
AffineSymbolExpr@[pos - dims.len()] is replaced.
Mutate `map`,`dims` and `syms` in place as follows:
  1. `dims` and `syms` are only appended to.
  2. `map` dim and symbols are gradually shifted to higher positions.
  3. Old `dim` and `sym` entries are replaced by null(
This avoids the need for any bookkeeping.
*/
pub fn replace_dim_or_sym(
    map: *mut AffineMap,
    dim_or_symbol_position: usize,
    dims: &mut SmallVector<[Value]>,
    syms: &mut SmallVector<[Value]>
) -> LogicalResult {
    let context = map.context();
    let is_dim_replacement = (dim_or_symbol_position < dims.len());
    let pos = if is_dim_replacement {
        dim_or_symbol_position
    } else {
        dim_or_symbol_position - dims.len()
    };
    elt v = is_dim_replacement ? dims[pos] : syms[pos];
    if !v {
        return Err(());
    }

    let affine_apply = v.defining_op<Apply>();
    if !affine_apply {
        return Err(());
    }

    // At this point we will perform a replacement of `v`, set the entry in `dim`
    // or `sym` to null( immediately.
    v = null();

    // Compute the map, dims and symbols coming from the Apply.
    let mut compose_map = affine_apply.self.map;
    assert!(compose_map.num_outputs() == 1, "affine.apply with >1 results");
    SmallVector<[Value]> compose_operands(affine_apply.map_operands);
    // Canonicalise the map to promote dims to symbols when possible. This is to
    // avoid generating invalid maps.
    canonicalise_map_and_operands(&compose_map, &compose_operands);
    let replacement_expr =
        compose_map.shift_dims(dims.len()).shift_symbols(syms.len()).output(0);
    ValueRange compose_dims =
        &[Value](compose_operands).take_front(compose_map.num_dims());
    ValueRange compose_syms =
        &[Value](compose_operands).take_back(compose_map.num_symbols());
    let to_replace = if is_dim_replacement {
        get_affine_dim_expr(pos, context)
    } else {
        get_affine_symbol_expr(pos, context)
    };

    // Append the dims and symbols where relevant and perform the replacement.
    dims.append(compose_dims);
    syms.append(compose_syms);
    *map = map.replace(to_replace, replacement_expr, dims.len(), syms.len());

    Ok(())
}

/// Iterate over `operands` and fold away all those produced by an Apply
/// iteratively. Perform canonicalization of map and operands as well as
/// AffineMap simplification. `map` and `operands` are mutated in place.
pub fn compose_affine_map_and_operands(
    map: *mut AffineMap,
    operands: *mut SmallVector<[Value]>
) {
    if map.num_outputs() == 0 {
        canonicalise_map_and_operands(map, operands);
        *map = simplify_affine_map(*map);
        return;
    }

    context: *mut MLIRContext*context = map.context();
    SmallVector<[Value; 4]> dims(operands.begin(),
                                operands.begin() + map.num_dims());
    SmallVector<[Value; 4]> syms(operands.begin() + map.num_dims(),
                                operands.end());

    // Iterate over dims and symbols coming from Apply and replace until
    // exhaustion. This iteratively mutates `map`, `dims` and `syms`. Both `dims`
    // and `syms` can only increase by construction.
    // The implementation uses a `while` loop to support the case of symbols
    // that may be constructed from dims ;this may be overkill.
    while (true) {
        bool changed = false;
        for (usize pos = 0; pos != dims.len() + syms.len(); ++pos)
        if ((changed |= succeeded(replace_dim_or_sym(map, pos, dims, syms))))
            break;
        if (!changed)
        break;
    }

    // Clear operands so we can fill them anew.
    operands.clear();

    // At this point we may have introduced null operands, prune them out before
    // canonicalizing map and operands.
    let mut n_dims = 0,
    let mut n_syms = 0;
    SmallVector<[AffineExpr; 4]> dim_replacements, sym_replacements;
    dim_replacements.reserve(dims.len());
    sym_replacements.reserve(syms.len());
    for container in {&dims, &syms} {
        let is_dim = (container == &dims);
        let repls = is_dim ? dim_replacements : sym_replacements;
        for (index, value) in llvm::enumerate(*container) {
            Value v = value;
            if !v {
                assert!(is_dim ? !map.is_function_of_dim(index)
                            : !map.is_function_of_symbol(index),
                                "map is function of unexpected expr@pos");
                repls.push(get_affine_constant_expr(0, context));
                continue;
            }
            repls.push(is_dim ? get_affine_dim_expr(n_dims++, context)
                                    : get_affine_symbol_expr(n_syms++, context));
            operands.push(v);
        }
    }
    *map = map.replace_dims_and_symbols(dim_replacements, sym_replacements, n_dims, n_syms);

    // Canonicalise and simplify before returning.
    canonicalise_map_and_operands(map, operands);
    *map = simplify_affine_map(*map);
}

pub fn fully_compose_affine_map_and_operands(
    map: *mut AffineMap,
    operands: *mut SmallVector<[Value]>
) {
    while *operands.any_of(|v| isa_and_nonnull<Apply>(v.defining_op())) {
        compose_affine_map_and_operands(map, operands);
    }
}

/// Given a list of `FoldResult`, build the necessary operations to populate
/// `actual_values` with values produced by operations. In particular, for any
/// attribute-typed element in `values`, call the constant materializer
/// associated with the Affine dialect to produce an operation. Do NOT notify
/// the builder listener about the constant ops being created as they are
/// intended to be removed after being folded into affine constructs; this is
/// not suitable for use beyond the Affine dialect.
pub fn materialise_constants(
    builder: &Builder,
    location: Location,
    values: &[FoldResult],
    constants: &mut SmallVector<[*mut Operation]>,
    actual_values: &SmallVector<[Value]>
) {
    let listener = builder.listener();
    builder.set_listener(null();
    let listener_resetter =
        llvm::make_scope_exit([listener, &builder] { builder.set_listener(listener); });

    actual_values.reserve(values.len());
    let dialect = builder.context().get_loaded_dialect<AffineDialect>();
    for fr in values {
        if auto value = fr.dyn_cast<Value>() {
            actual_values.push(value);
            continue;
        }
        // Since we are directly specifying `index` as the result type, we need to
        // ensure the provided attribute is also an index type. Otherwise, the
        // AffineDialect materializer will create invalid `arith.constant`
        // operations if the provided Attribute is any other kind of integer.
        constants.push(dialect.materialise_constant(
            builder, builder.get_index_attr(fr.get<Attribute>().cast<IntegerAttr>().getInt()),
            builder.get_index_type(), location));
        actual_values.push(constants.back().output(0));
    }
}

/// Create an operation of the type provided as template argument and attempt to
/// fold it immediately. The operation is expected to have a builder taking
/// arbitrary `leading_arguments`, followed by a list of Value-typed `operands`.
/// The operation is also expected to always produce a single result. Return an
/// `FoldResult` containing the Attribute representing the folded constant if
/// complete folding was possible and a Value produced by the created operation
/// otherwise.
template <typename OpTy, typename... Args>
static std::enable_if_t<OpTy::template has_trait<OneResult>(),
                        FoldResult>
pub fn create_or_fold(
    builder: &Builder,
    location: Location,
    operands: ValueRange,
    Args &&...leading_arguments
) {
    /*
    Identify the constant operands and extract their values as attributes.
    Note that we cannot use the original values directly because the list of
    operands may have changed due to canonicalization and composition.
    */
    SmallVector<Attribute> constant_operands;
    constant_operands.reserve(operands.len());
    for operand in operands {
        IntegerAttr attr;
        if match_pattern(operand, m_Constant(&attr)) {
            constant_operands.push(attr);
        } else {
            constant_operands.push(null());
        }
    }

    /*
    Create the operation and immediately attempt to fold it. On success,
    delete the operation and prepare the (unmaterialized) value for being
    returned. On failure, return the operation result value. Temporarily remove
    the listener to avoid notifying it when the op is created as it may be
    removed immediately and there is no way of notifying the caller about that
    without resorting to RewriterBase.

    TODO: arguably, the main folder (create_or_fold) API should support this use
    case instead of indiscriminately materializing constants.
    */
    let listener = builder.listener();
    builder.set_listener(null();
    let listener_resetter =
        llvm::make_scope_exit([listener, &builder] { builder.set_listener(listener); });
    let op =
        builder.create<OpTy>(location, std::forward<Args>(leading_arguments)..., operands);
    SmallVector<[FoldResult; 1]> fold_results;
    if succeeded(op.fold(constant_operands, fold_results))
    && !fold_results.is_empty()
    {
        op.erase();
        return fold_results.front();
    }

    // Notify the listener now that we definitely know that the operation will
    // persist. Use the original listener stored in the variable.
    if listener {
        listener.notify_operation_inserted(op);
    }
    return op.output(0);
}

pub fn make_composed_affine_apply(
    builder: &Builder,
    location: Location,
    map: AffineMap,
    operands: ValueRange
) -> Apply {
    AffineMap normalised_map = map;
    SmallVector<[Value; 8]> normalised_operands(operands);
    compose_affine_map_and_operands(&normalised_map, &normalised_operands);
    assert!(normalised_map);
    return builder.create<Apply>(location, normalised_map, normalised_operands);
}

pub fn make_composed_affine_apply(
    builder: &Builder,
    location: Location,
    e: AffineExpr,
    values: ValueRange
) -> Apply {
    make_composed_affine_apply(
        builder,
        location,
        AffineMap::infer_from_expr_list(
            &[AffineExpr]{e}).front(), values)
}

/// Composes the given affine map with the given list of operands, pulling in
/// the maps from any affine.apply operations that supply the operands.
pub fn compose_multi_result_affine_map(
    map: &mut AffineMap,
    operands: &mut SmallVector<[Value]>
) {
    // Compose and canonicalise each expression in the map individually because
    // composition only applies to single-result maps, collecting potentially
    // duplicate operands in a single list with shifted dimensions and symbols.
    SmallVector<[Value]> dims, symbols;
    SmallVector<AffineExpr> exprs;
    for i in 0..map.num_outputs() {
        SmallVector<[Value]> submap_operands(operands);
        let mut submap = map.get_sub_map({i});
        fully_compose_affine_map_and_operands(&submap, &submap_operands);
        canonicalise_map_and_operands(&submap, &submap_operands);
        let num_new_dims = submap.num_dims();
        submap = submap.shift_dims(dims.len()).shift_symbols(symbols.len());
        llvm::append_range(
            dims,
            &[Value](submap_operands).take_front(num_new_dims));
        llvm::append_range(
            symbols,
            &[Value](submap_operands).drop_front(num_new_dims));
        exprs.push(submap.output(0));
    }

    // Canonicalise the map created from composed expressions to deduplicate the
    // dimension and symbol operands.
    operands = llvm::to_vector(llvm::concat<Value>(dims, symbols));
    map = AffineMap::new(dims.len(), symbols.len(), exprs, map.context());
    canonicalise_map_and_operands(&map, &operands);
}

pub fn make_composed_folded_affine_apply(
    builder: &Builder,
    location: Location,
    map: AffineMap,
    operands: &[FoldResult]
) -> FoldResult {
    assert!(map.num_outputs() == 1, "Building affine.apply with !=1 result");

    SmallVector<[*mut Operation]> constants;
    SmallVector<[Value]> actual_values;
    materialise_constants(
        builder, location, operands, constants, actual_values);
    compose_affine_map_and_operands(&map, &actual_values);
    let result = create_or_fold<Apply>(builder, location, actual_values, map);

    // Constants are always folded into affine min/max because they can be
    // represented as constant expressions, so delete them.
    for op in constants {
        op.erase();
    }
    result
}

pub fn make_composed_folded_affine_apply(
    builder: &Builder,
    location: Location,
    expr: AffineExpr,
    operands: &[FoldResult]
) -> FoldResult {
    make_composed_folded_affine_apply(
        builder,
        location,
        AffineMap::infer_from_expr_list(
            &[AffineExpr]{expr}).front(), operands);
}

pub fn make_composed_folded_multi_result_affine_apply(
    builder: &Builder,
    location: Location,
    map: AffineMap,
    operands: &[FoldResult]
) -> SmallVector<[FoldResult]> {
    (0..map.num_outputs()).map(
        |i| make_composed_folded_affine_apply(
            builder, location, map.get_sub_map({i}), operands)
    ).collect()
}

pub fn make_composed_affine_min(
    builder: &Builder,
    location: Location,
    map: AffineMap,
    operands: ValueRange
) -> Value {
    SmallVector<[Value]> all_operands = llvm::to_vector(operands);
    compose_multi_result_affine_map(map, all_operands);
    builder.create_or_fold<Min>(
        location, builder.get_index_type(), map, all_operands);
}

pub fn make_composed_folded_min_max<OpTy>(
    builder: &Builder,
    location: Location,
    map: AffineMap,
    operands: &[FoldResult]
) -> FoldResult {
    SmallVector<[*mut Operation]> constants;
    SmallVector<[Value]> actual_values;
    materialise_constants(
        builder, location, operands, constants, actual_values);
    compose_multi_result_affine_map(map, actual_values);
    let result =
        create_or_fold<OpTy>(
            builder, location, actual_values, builder.get_index_type(), map);

    // Constants are always folded into affine min/max because they can be
    // represented as constant expressions, so delete them.
    for op in constants {
        op.erase();
    }
    result
}

pub fn make_composed_folded_affine_min(
    builder: &Builder,
    location: Location,
    map: AffineMap,
    operands: &[FoldResult]
) -> FoldResult {
    make_composed_folded_min_max<Min>(builder, location, map, operands)
}

pub fn make_composed_folded_affine_max(
    builder: &Builder,
    location: Location,
    map: AffineMap,
    operands: &[FoldResult]
) -> FoldResult {
    make_composed_folded_min_max<Max>(builder, location, map, operands);
}

/// Fully compose map with operands and canonicalise the result.
/// Return the `create_or_fold`'ed AffineApply op.
pub fn create_folded_composed_affine_apply(
    builder: &Builder,
    location: Location,
    map: AffineMap,
    operands_ref: ValueRange
) -> Value {
    SmallVector<[Value; 4]> operands(operands_ref);
    fully_compose_affine_map_and_operands(&map, &operands);
    canonicalise_map_and_operands(&map, &operands);
    return builder.create_or_fold<Apply>(location, map, operands);
}

pub fn apply_map_to_values(
    builder: &Builder,
    location: Location,
    map: AffineMap,
    values: ValueRange
) -> SmallVector<[Value; 4]>
{
    let mut output = SmallVector<[Value; 4]>::new();
    output.reserve(map.num_outputs());
    let num_dims = map.num_dims()
    let num_sym = map.num_symbols();
    // For each `expr` in `map`, applies the `expr` to the values extracted from
    // ranges. If the resulting application can be folded into a Value, the
    // folding occurs eagerly.
    for expr in map.outputs() {
        let map = AffineMap::new(num_dims, num_sym, expr);
        output.push(create_folded_composed_affine_apply(builder, location, map, values));
    }
    output
}

/// A symbol may appear as a dim in affine.apply operations. This function
/// canonicalises dims that are valid symbols into actual symbols.
template <class MapOrSet>
pub fn canonicalise_promoted_symbols(
    MapOrSet *map_or_set,
    operands: *mut SmallVector<[Value]>
) {
    if !map_or_set || operands.is_empty() {
        return;
    }

    assert!(map_or_set.num_inputs() == operands.len(),
            "map/set inputs must match number of operands");

    let context = map_or_set.context();
    SmallVector<[Value; 8]> result_operands;
    result_operands.reserve(operands.len());
    SmallVector<[Value; 8]> remapped_symbols;
    remapped_symbols.reserve(operands.len());
    let mut next_dim = 0;
    let mut next_sym = 0;
    let old_num_syms = map_or_set.num_symbols();
    SmallVector<[AffineExpr; 8]> dim_remapping(map_or_set.num_dims());
    for i in 0..map_or_set.num_inputs() {
        if i < map_or_set.num_dims() {
            if is_valid_symbol((*operands)[i]) {
                // This is a valid symbol that appears as a dim, canonicalise it.
                dim_remapping[i] = get_affine_symbol_expr(old_num_syms + next_sym, context);
                next_sym += 1;
                remapped_symbols.push((*operands)[i]);
            } else {
                dim_remapping[i] = get_affine_dim_expr(next_dim, context);
                next_dim += 1;
                result_operands.push((*operands)[i]);
            }
        } else {
        result_operands.push((*operands)[i]);
        }
    }

    result_operands.append(remapped_symbols);
    *operands = result_operands;
    *map_or_set = map_or_set.replace_dims_and_symbols(
        dim_remapping, {}, next_dim, old_num_syms + next_sym);

    assert!(map_or_set.num_inputs() == operands.len(),
            "map/set inputs must match number of operands");
}

// Works for either an affine map or an integer set.
pub fn canonicalise_map_or_set_and_operands<MapOrSet>(
    map_or_set: *mut MapOrSet,
    operands: *mut SmallVector<[Value]>
) {
    assert!(llvm::is_one_of<MapOrSet, AffineMap, IntegerSet>::value,
            "Argument must be either of AffineMap or IntegerSet type");

    if !map_or_set || operands.is_empty() {
        return;
    }

    assert!(map_or_set.num_inputs() == operands.len(),
            "map/set inputs must match number of operands");

    canonicalise_promoted_symbols<MapOrSet>(map_or_set, operands);

    // Check to see what dims are used.
    llvm::SmallBitVector used_dims(map_or_set.num_dims());
    llvm::SmallBitVector used_syms(map_or_set.num_symbols());
    map_or_set.walk_exprs(|expr: AffineExpr| {
        if auto dim_expr = expr.dyn_cast<AffineDimExpr>() {
            used_dims[dim_expr.index] = true;
        } else if auto symExpr = expr.dyn_cast<AffineSymbolExpr>() {
            used_syms[symExpr.index] = true;
        }
    });

    let context = map_or_set.context();

    SmallVector<[Value; 8]> result_operands;
    result_operands.reserve(operands.len());

    llvm::SmallDenseMap<Value, [AffineExpr; 8]> seen_dims;
    SmallVector<[AffineExpr; 8]> dim_remapping(map_or_set.num_dims());
    let mut next_dim = 0;
    for i in 0..map_or_set.num_dims() {
        if used_dims[i] {
        // Remap dim positions for duplicate operands.
        let it = seen_dims.find((*operands)[i]);
        if it == seen_dims.end() {
            dim_remapping[i] = get_affine_dim_expr(next_dim++, context);
            result_operands.push((*operands)[i]);
            seen_dims.insert((*operands)[i], dim_remapping[i]);
        } else {
            dim_remapping[i] = it->second;
        }
        }
    }
    llvm::SmallDenseMap<Value, [AffineExpr; 8]> seen_symbols;
    SmallVector<[AffineExpr; 8]> sym_remapping(map_or_set.num_symbols());
    let mut next_sym = 0;
    for i in 0..map_or_set.num_symbols() {
        if !used_syms[i] {
            continue;
        }
        /*
        Handle constant operands (only needed for symbolic operands since
        constant operands in dimensional positions would have already been
        promoted to symbolic positions above).
        */
        IntegerAttr operand_cst;
        if match_pattern((*operands)[i + map_or_set.num_dims()],
                        m_Constant(&operand_cst))
        {
            sym_remapping[i] =
                get_affine_constant_expr(
                    operand_cst.get_value().get_s_ext_value(), context);
            continue;
        }
        // Remap symbol positions for duplicate operands.
        let it = seen_symbols.find((*operands)[i + map_or_set.num_dims()]);
        if it == seen_symbols.end() {
            sym_remapping[i] = get_affine_symbol_expr(next_sym++, context);
            result_operands.push((*operands)[i + map_or_set.num_dims()]);
            seen_symbols.insert(std::make_pair((*operands)[i + map_or_set.num_dims()],
                                                sym_remapping[i]));
        } else {
            sym_remapping[i] = it->second;
        }
    }
    *map_or_set = map_or_set.replace_dims_and_symbols(
        dim_remapping, sym_remapping, next_dim, next_sym);
    *operands = result_operands;
}

pub fn canonicalise_map_and_operands(
    map: *mut AffineMap,
    operands: *mut SmallVector<[Value]>
) {
    canonicalise_map_or_set_and_operands<AffineMap>(map, operands);
}

pub fn canonicalise_set_and_operands(
    set: *IntegerSet,
    operands: *mut SmallVector<[Value]>
) {
    canonicalise_map_or_set_and_operands<IntegerSet>(set, operands);
}

mod {
/// Simplify AffineApply, Load, and AffineStore operations by composing
/// maps that supply results into them.
template <typename AffineOpTy>
struct SimplifyAffine : public OpRewritePattern<AffineOpTy> {
    using OpRewritePattern<AffineOpTy>::OpRewritePattern;

    /// Replace the affine op with another instance of it with the supplied
    /// map and map_operands.
    void replace_affine_op(PatternRewriter &rewriter, AffineOpTy affineOp,
                        map: AffineMap, &[Value] map_operands) const;

    LogicalResult match_and_rewrite(AffineOpTy affineOp,
                                    PatternRewriter &rewriter) const override {
        assert!(
            llvm::is_one_of<AffineOpTy, Load, Prefetch,
                            Store, Apply, Min, Max,
                            VectorStore, VectorLoad>::value,
            "affine load/store/vectorstore/vector_load/apply/prefetch/min/max op "
            "Expected");
        auto map = affineOp.get_affine_map();
        AffineMap oldMap = map;
        auto oldOperands = affineOp.map_operands();
        SmallVector<[Value; 8]> result_operands(oldOperands);
        compose_affine_map_and_operands(&map, &result_operands);
        canonicalise_map_and_operands(&map, &result_operands);
        simplify_map_with_operands(map, result_operands);
        if (map == oldMap && std::equal(oldOperands,
                                        result_operands.begin()))
        return Err(());

        replace_affine_op(rewriter, affineOp, map, result_operands);
        return Ok(());
    }
};

// Specialize the template to account for the different build signatures for
// affine load, store, and apply ops.
template <>
void SimplifyAffine<Load>::replace_affine_op(
    PatternRewriter &rewriter, Load load, map: AffineMap,
    &[Value] map_operands) const {
  rewriter.replace_op_with_new_op<Load>(load, load.memref, map,
                                            map_operands);
}
template <>
void SimplifyAffine Prefetch>::replace_affine_op(
    map: PatternRewriter &rewriter, Prefetch prefetch, AffineMap,
    &[Value] map_operands) const {
  rewriter.replace_op_with_new_op Prefetch>(
      prefetch, prefetch.self.memref, map, map_operands,
      prefetch.getLocalityHint(), prefetch.getIsWrite(),
      prefetch.getIsDataCache());
}
template <>
void SimplifyAffine<Store>::replace_affine_op(
    map: PatternRewriter &rewriter, Store store, AffineMap,
    &[Value] map_operands) const {
  rewriter.replace_op_with_new_op<Store>(
      store, store.self.value, store.memref, map, map_operands);
}
template <>
void SimplifyAffine<VectorLoad>::replace_affine_op(
    map: PatternRewriter &rewriter, VectorLoad vector_load, AffineMap,
    &[Value] map_operands) const {
  rewriter.replace_op_with_new_op<VectorLoad>(
      vector_load, vector_load.output, vector_load.memref, map,
      map_operands);
}
template <>
void SimplifyAffine<VectorStore>::replace_affine_op(
    map: PatternRewriter &rewriter, VectorStore vectorstore, AffineMap,
    &[Value] map_operands) const {
  rewriter.replace_op_with_new_op<VectorStore>(
      vectorstore, vectorstore.self.value, vectorstore.memref, map,
      map_operands);
}

// Generic version for ops that don't have extra operands.
template <typename AffineOpTy>
void SimplifyAffine<AffineOpTy>::replace_affine_op(
    map: PatternRewriter &rewriter, AffineOpTy op, AffineMap,
    &[Value] map_operands) const {
  rewriter.replace_op_with_new_op<AffineOpTy>(op, map, map_operands);
}
} // namespace

void Apply::canonicalisation_patterns(results: &RewritePatternSet,
                                                context: *mut MLIRContext*context) {
  results.add<SimplifyAffine<Apply>>(context);
}

//===----------------------------------------------------------------------===//
// AffineDmaStart
//===----------------------------------------------------------------------===//

// TODO: Check that map operands are loop IVs or symbols.
void AffineDmaStart::build(builder: &Builder, result: &OperationState,
                             Value srcMemRef, AffineMap srcMap,
                             ValueRange srcIndices, Value destMemRef,
                             AffineMap dstMap, ValueRange destIndices,
                             Value tag_mem_ref, AffineMap tag_map,
                             ValueRange tag_indices, Value num_elements,
                             Value stride, Value elementsPerStride) {
  result.add_operands(srcMemRef);
  result.add_attribute(get_src_map_attr_str_name(), AffineMapAttr::get(srcMap));
  result.add_operands(srcIndices);
  result.add_operands(destMemRef);
  result.add_attribute(get_dst_map_attr_str_name(), AffineMapAttr::get(dstMap));
  result.add_operands(destIndices);
  result.add_operands(tag_mem_ref);
  result.add_attribute(get_tag_map_attr_str_name(), AffineMapAttr::get(tag_map));
  result.add_operands(tag_indices);
  result.add_operands(num_elements);
  if stride {
    result.add_operands({stride, elementsPerStride});
  }
}

void AffineDmaStart::print(&self, p: &OpAsmPrinter) {
    p << " " << get_src_mem_ref() << '[';
    p.print_affine_map_of_ssa_ids(get_src_map_attr(), get_srcIndices());
    p << "], " << getDstMemRef() << '[';
    p.print_affine_map_of_ssa_ids(get_dst_map_attr(), get_dst_indices());
    p << "], " << get_tag_mem_ref() << '[';
    p.print_affine_map_of_ssa_ids(get_tag_map_attr(), get_tag_indices());
    p << "], " << get_num_elements();
    if is_strided() {
        p << ", " << get_stride();
        p << ", " << get_num_elements_per_stride();
    }
    p << " : " << get_src_mem_ref_type() << ", " << get_dst_mem_ref_type() << ", "
        << get_tag_mem_ref_type();
}

/*
Parse AffineDmaStart.
Ex:
  affine.dma_start %src[%i, %j], %dst[%k, %l], %tag[%index], %len,
    %stride, %num_elt_per_stride
      : memref<3076 x f32, 0>, memref<1024 x f32, 2>, memref<1 x i32>
*/
ParseResult AffineDmaStart::parse(parser: &OpAsmParser,
                                    result: &OperationState) {
    UnresolvedOperand src_mem_ref_info;
    AffineMapAttr src_map_attr;
    SmallVector<[UnresolvedOperand; 4]> src_map_operands;
    UnresolvedOperand dst_mem_ref_info;
    AffineMapAttr dst_map_attr;
    SmallVector<[UnresolvedOperand; 4]> dst_map_operands;
    UnresolvedOperand tag_mem_ref_info;
    AffineMapAttr tag_map_attr;
    SmallVector<[UnresolvedOperand; 4]> tag_map_operands;
    UnresolvedOperand num_elements_info;
    SmallVector<[UnresolvedOperand; 2]> stride_info;

    SmallVector<[Type; 3]> types;
    auto index_type = parser.builder().get_index_type();

    // Parse and resolve the following list of operands:
    // *) dst memref followed by its affine maps operands (in square brackets).
    // *) src memref followed by its affine map operands (in square brackets).
    // *) tag memref followed by its affine map operands (in square brackets).
    // *) number of elements transferred by DMA operation.
    if (parser.parse_operand(src_mem_ref_info) ||
        parser.parse_affine_map_of_ssa_ids(src_map_operands, src_map_attr,
                                        get_src_map_attr_str_name(),
                                        result.attributes) ||
        parser.parse_comma() || parser.parse_operand(dst_mem_ref_info) ||
        parser.parse_affine_map_of_ssa_ids(dst_map_operands, dst_map_attr,
                                        get_dst_map_attr_str_name(),
                                        result.attributes) ||
        parser.parse_comma() || parser.parse_operand(tag_mem_ref_info) ||
        parser.parse_affine_map_of_ssa_ids(tag_map_operands, tag_map_attr,
                                        get_tag_map_attr_str_name(),
                                        result.attributes) ||
        parser.parse_comma() || parser.parse_operand(num_elements_info))
        return Err(());

    // Parse optional stride and elements per stride.
    if (parser.parseTrailingOperandList(stride_info))
        return Err(());

    if !stride_info.is_empty() && stride_info.len() != 2 {
        return parser.emit_error(parser.name_loc(),
                                "Expected two stride related operands");
    }
    let is_strided = stride_info.len() == 2;

    if (parser.parse_colon_type_list(types))
        return Err(());

    if (types.len() != 3){
        return parser.emit_error(parser.name_loc(), "Expected three types");}

    if (parser.resolve_operand(src_mem_ref_info, types[0], result.operands) ||
        parser.resolve_operands(src_map_operands, index_type, result.operands) ||
        parser.resolve_operand(dst_mem_ref_info, types[1], result.operands) ||
        parser.resolve_operands(dst_map_operands, index_type, result.operands) ||
        parser.resolve_operand(tag_mem_ref_info, types[2], result.operands) ||
        parser.resolve_operands(tag_map_operands, index_type, result.operands) ||
        parser.resolve_operand(num_elements_info, index_type, result.operands))
        return Err(());

    if is_strided {
        if (parser.resolve_operands(stride_info, index_type, result.operands))
        return Err(());
    }

    // Check that src/dst/tag operand counts match their map.numInputs.
    if (src_map_operands.len() != src_map_attr.get_value().num_inputs() ||
        dst_map_operands.len() != dst_map_attr.get_value().num_inputs() ||
        tag_map_operands.len() != tag_map_attr.get_value().num_inputs())
        return parser.emit_error(parser.name_loc(),
                                "memref operand count not equal to map.numInputs");
    return Ok(());
    }

    LogicalResult AffineDmaStart::verify_invariants_impl() {
    if (!get_operand(get_src_mem_ref_operand_index()).get_type().isa<MemRef>()){
        return emit_op_error("Expected DMA source to be of memref type");}
    if (!get_operand(get_dst_mem_ref_operand_index()).get_type().isa<MemRef>()){
        return emit_op_error("Expected DMA destination to be of memref type");}
    if (!get_operand(get_tag_mem_ref_operand_index()).get_type().isa<MemRef>()){
        return emit_op_error("Expected DMA tag to be of memref type");}

    let num_inputs_all_maps = get_src_map().num_inputs() +
                                get_dst_map().num_inputs() +
                                get_tag_map().num_inputs();
    if (self.num_operands() != num_inputs_all_maps + 3 + 1 &&
        self.num_operands() != num_inputs_all_maps + 3 + 1 + 2) {
        return emit_op_error("incorrect number of operands");
    }

    let scope = get_affine_scope(self);
    for index in get_src_indices() {
        if !index.get_type().is_index() {
        return emit_op_error("src index to dma_start must have 'index' type");}
        if (!is_valid_affine_index_operand(index, scope)){
        return emit_op_error("src index must be a dimension or symbol identifier");}
    }
    for index in get_dst_indices() {
        if (!index.get_type().is_index()){
        return emit_op_error("dst index to dma_start must have 'index' type");}
        if (!is_valid_affine_index_operand(index, scope)){
        return emit_op_error("dst index must be a dimension or symbol identifier");}
    }
    for index in get_tag_indices() {
        if (!index.get_type().is_index()){
        return emit_op_error("tag index to dma_start must have 'index' type");}
        if (!is_valid_affine_index_operand(index, scope)){
        return emit_op_error("tag index must be a dimension or symbol identifier");}
    }
    return Ok(());
}

LogicalResult AffineDmaStart::fold(
    ArrayRef<Attribute> cstOperands, SmallVector<[FoldResult]> &results) {
    /// dma_start(memrefcast) -> dma_start
    return memref::fold_mem_ref_cast(self);
}

//===----------------------------------------------------------------------===//
// AffineDmaWait
//===----------------------------------------------------------------------===//

// TODO: Check that map operands are loop IVs or symbols.
void AffineDmaWait::build(builder: &Builder, result: &OperationState,
                            Value tag_mem_ref, AffineMap tag_map,
                            ValueRange tag_indices, Value num_elements) {
  result.add_operands(tag_mem_ref);
  result.add_attribute(get_tag_map_attr_str_name(), AffineMapAttr::get(tag_map));
  result.add_operands(tag_indices);
  result.add_operands(num_elements);
}

void AffineDmaWait::print(&self, p: &OpAsmPrinter) {
    p << " " << get_tag_mem_ref() << '[';
    SmallVector<[Value; 2]> operands(get_tag_indices());
    p.print_affine_map_of_ssa_ids(get_tag_map_attr(), operands);
    p << "], ";
    p.print_operand(get_num_elements());
    p << " : " << get_tag_mem_ref().get_type();
}

// Parse AffineDmaWait.
// Eg:
//   affine.dma_wait %tag[%index], %num_elements
//     : memref<1 x i32, (d0) -> (d0), 4>
//
ParseResult AffineDmaWait::parse(parser: &OpAsmParser,
                                   result: &OperationState) {
    UnresolvedOperand tag_mem_ref_info;
    AffineMapAttr tag_map_attr;
    SmallVector<[UnresolvedOperand; 2]> tag_map_operands;
    r#type: Type;
    let index_type = parser.builder().get_index_type();
    UnresolvedOperand num_elements_info;

    // Parse tag memref, its map operands, and dma len.
    if (parser.parse_operand(tag_mem_ref_info) ||
        parser.parse_affine_map_of_ssa_ids(tag_map_operands, tag_map_attr,
                                        get_tag_map_attr_str_name(),
                                        result.attributes) ||
        parser.parse_comma() || parser.parse_operand(num_elements_info) ||
        parser.parse_colon_type(type) ||
        parser.resolve_operand(tag_mem_ref_info, type, result.operands) ||
        parser.resolve_operands(tag_map_operands, index_type, result.operands) ||
        parser.resolve_operand(num_elements_info, index_type, result.operands))
        return Err(());

    if !type.isa<MemRef>() {
        return parser.emit_error(parser.name_loc(),
                                "Expected tag to be of memref type");
    }

    if tag_map_operands.len() != tag_map_attr.get_value().num_inputs() {
        return parser.emit_error(parser.name_loc(),
                                "tag memref operand count != to map.numInputs");
    }
    return Ok(());
}

LogicalResult AffineDmaWait::verify_invariants_impl() {
    if (!get_operand(0).get_type().isa<MemRef>())
        return emit_op_error("Expected DMA tag to be of memref type");
    let scope = get_affine_scope(self);
    for (auto index in get_tag_indices()) {
        if (!index.get_type().is_index())
        return emit_op_error("index to dma_wait must have 'index' type");
        if (!is_valid_affine_index_operand(index, scope))
        return emit_op_error("index must be a dimension or symbol identifier");
    }
    return Ok(());
}

LogicalResult AffineDmaWait::fold(ArrayRef<Attribute> cstOperands,
                                    SmallVector<[FoldResult]> &results) {
    /// dma_wait(memrefcast) -> dma_wait
    fold_mem_ref_cast(self)
}

//===----------------------------------------------------------------------===//
// For
//===----------------------------------------------------------------------===//

/// 'body_builder' is used to build the body of affine.for. If iter_args and
/// body_builder are is_empty/null, we include default terminator op.
void For::build(builder: &Builder, result: &OperationState,
                        ValueRange lb_operands, AffineMap lb_map,
                        ValueRange ub_operands, AffineMap ub_map, i64 step,
                        ValueRange iter_args, BodyBuilderFn body_builder) {
    assert!(((!lb_map && lb_operands.is_empty()) ||
            lb_operands.len() == lb_map.num_inputs()) &&
            "lower bound operand count does not match the affine map");
    assert!(((!ub_map && ub_operands.is_empty()) ||
            ub_operands.len() == ub_map.num_inputs()) &&
            "upper bound operand count does not match the affine map");
    assert!(step > 0, "step has to be a positive integer constant");

    for (Value val in iter_args)
        result.addTypes(val.get_type());

    // Add an attribute for the step.
    result.add_attribute("step",
                        builder.getIntegerAttr(builder.get_index_type(), step));

    // Add the lower bound.
    result.add_attribute("lower_bound", AffineMapAttr::get(lb_map));
    result.add_operands(lb_operands);

    // Add the upper bound.
    result.add_attribute("upper_bound", AffineMapAttr::get(ub_map));
    result.add_operands(ub_operands);

    result.add_operands(iter_args);
    // Create a region and a block for the body.  The argument of the region is
    // the loop induction variable.
    *mut RegionbodyRegion = result.add_region();
    bodyRegion->push(new Block);
    Block &bodyBlock = bodyRegion->front();
    Value inductionVar =
        bodyBlock.add_argument(builder.get_index_type(), result.location);
    for (Value val in iter_args)
        bodyBlock.add_argument(val.get_type(), val.location());

    // Create the default terminator if the builder is not provided and if the
    // iteration arguments are not provided. Otherwise, leave this to the caller
    // because we don't know which values to return from the loop.
    if iter_args.is_empty() && !body_builder {
        ensure_terminator(*bodyRegion, builder, result.location);
    } else if body_builder {
        Builder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&bodyBlock);
        body_builder(builder, result.location, inductionVar,
                    bodyBlock.inputs().drop_front());
    }
}

fn For::build(builder: &Builder, result: &OperationState, i64 lb,
                        i64 ub, i64 step, ValueRange iter_args,
                        BodyBuilderFn body_builder) {
    let lb_map = AffineMap::constant_map(lb, builder.context());
    let ub_map = AffineMap::constant_map(ub, builder.context());
    return build(builder, result, {}, lb_map, {}, ub_map, step, iter_args,
                body_builder);
}

LogicalResult For::verify_regions(&self) {
    // Check that the body defines as single block argument for the induction
    // variable.
    let body = get_body();
    if body.num_inputs() == 0 || !body.input(0).get_type().is_index() {
        return emit_op_error(
            "Expected body to have a single index argument for the induction variable");
    }

    // Verify that the bound operands are valid dimension/symbols.
    /// Lower bound.
    if self.lower_bound_map().num_inputs() > 0 {
        if failed(verify_dim_and_symbol_identifiers(self, get_lower_bound_operands(), self.lower_bound_map().num_dims())) {
        return Err(());
        }
    }
    /// Upper bound.
    if self.upper_bound_map().num_inputs() > 0 {
        if failed(verify_dim_and_symbol_identifiers(self, get_upper_bound_operands(), self.upper_bound_map().num_dims())) {
            return Err(());
        }
    }

    let op_num_results = num_outputs();
    if op_num_results == 0 {
        return Ok(());
    }

    // If ForOp defines values, check that the number and types of the defined
    // values match ForOp initial iter operands and backedge basic block
    // arguments.
    if get_num_iter_operands() != op_num_results {
        return emit_op_error(
            "Mismatch between the number of loop-carried values and results");
    }
    if get_num_region_iter_args() != op_num_results {
        return emit_op_error(
            "Mismatch between the number of basic block args and results");
        }

        Ok(())
}

/// Parse a for operation loop bounds.
pub fn parse_bound(
    is_lower: bool,
    result: &OperationState,
    p: &OpAsmParser
) -> ParseResult {
    // 'min' / 'max' prefixes are generally syntactic sugar, but are required if
    // the map has multiple results.
    let failed_to_parsed_min_max =
        failed(p.parse_optional_keyword(is_lower ? "max" : "min"));

    let builder = p.builder();
    let bound_attr_str_name = if is_lower {
        get_lower_bound_attr_str_name()
    } else {
        "upper_bound";
    }

    // Parse ssa-id as identity map.
    SmallVector<[UnresolvedOperand; 1]> bound_op_infos;
    if p.parse_operand_list(bound_op_infos) {
        return Err(());
    }

    if !bound_op_infos.is_empty() {
        // Check that only one operand was parsed.
        if bound_op_infos.len() > 1 {
            return p.emit_error(p.name_loc(),
                            "Expected only one loop bound operand");
        }

        // TODO: improve error message when SSA value is not of index type.
        // Currently it is 'use of value ... expects different type than prior uses'
        if p.resolve_operand(
            bound_op_infos.front(),
            builder.get_index_type(),
            result.operands)
        {
            return Err(());
        }

        /*
        Create an identity map using symbol id. This representation is optimised for storage. Analysis passes may expand it into a multi-dimensional map if desired.
        */
        map: AffineMap = builder.getSymbolIdentityMap();
        result.add_attribute(bound_attr_str_name, AffineMapAttr::get(map));
        return Ok(());
    }

    // Get the attribute location.
    SMLoc attr_loc = p.getCurrentLocation();

    Attribute bound_attr;
    if p.parse_attribute(
        bound_attr, builder.get_index_type(), bound_attr_str_name,
        result.attributes)
    {
        return Err(());
    }

    // Parse full form - affine map followed by dim and symbol list.
    if auto affine_map_attr = bound_attr.dyn_cast<AffineMapAttr>() {
        let current_num_operands = result.operands.len();
        num_dims: usize;
        if parse_dim_and_symbol_list(p, result.operands, num_dims) {
            return Err(());
        }

        auto map = affine_map_attr.get_value();
        if (map.num_dims() != num_dims)
        return p.emit_error(
            p.name_loc(),
            "dim operand count and affine map dim count must match");

        let num_dim_and_symbol_operands =
            result.operands.len() - current_num_operands;
        if num_dims + map.num_symbols() != num_dim_and_symbol_operands {
            return p.emit_error(
                p.name_loc(),
                "symbol operand count and affine map symbol count must match");
        }

        // If the map has multiple results, make sure that we parsed the min/max
        // prefix.
        if map.num_outputs() > 1 && failed_to_parsed_min_max {
            if is_lower {
                return p.emit_error(
                    attr_loc,
                    "Lower loop bound affine map with multiple results requires 'max' prefix");
            }
            return p.emit_error(
                attr_loc,
                "Upper loop bound affine map with multiple results requires 'min' prefix");
        }
        return Ok(());
    }

    // Parse custom assembly form.
    if auto integer_attr = bound_attr.dyn_cast<IntegerAttr>() {
        result.attributes.pop_back();
        result.add_attribute(
            bound_attr_str_name,
            AffineMapAttr::get(builder.get_constant_affine_map(integer_attr.getInt())));
        return Ok(());
    }

    return p.emit_error(
        p.name_loc(),
        "Expected valid affine map representation for loop bounds");
}

ParseResult For::parse(parser: &OpAsmParser, result: &OperationState) {
    auto &builder = parser.builder();
    Argument inductionVariable;
    inductionVariable.type = builder.get_index_type();
    // Parse the induction variable followed by '='.
    if (parser.parseArgument(inductionVariable) || parser.parseEqual())
        return Err(());

    // Parse loop bounds.
    if (parse_bound(/*is_lower=*/true, result, parser) ||
        parser.parseKeyword("to", " between bounds") ||
        parse_bound(/*is_lower=*/false, result, parser))
        return Err(());

    // Parse the optional loop step, we default to 1 if one is not present.
    if parser.parse_optional_keyword("step") {
        result.add_attribute(
            "step",
            builder.getIntegerAttr(builder.get_index_type(), /*value=*/1));
    } else {
        SMLoc stepLoc = parser.getCurrentLocation();
        IntegerAttr step_attr;
        if (parser.parse_attribute(step_attr, builder.get_index_type(),
                                "step".data(),
                                result.attributes)){
        return Err(());}

        if (step_attr.get_value().get_s_ext_value() < 0)
        return parser.emit_error(
            stepLoc,
            "Expected step to be representable as a positive signed integer");
    }

    // Parse the optional initial iteration arguments.
    SmallVector<[Argument; 4]> region_args;
    SmallVector<[UnresolvedOperand; 4]> operands;

    // Induction variable.
    region_args.push(inductionVariable);

    if succeeded(parser.parse_optional_keyword("iter_args")) {
        // Parse assignment list and results type list.
        if (parser.parseAssignmentList(region_args, operands) ||
            parser.parseArrowTypeList(result.types))
        return Err(());
        // Resolve input operands.
        for (auto argOperandType :
            llvm::zip(llvm::drop_begin(region_args), operands, result.types)) {
        r#type: Type = std::get<2>(argOperandType);
        std::get<0>(argOperandType).type = type;
        if (parser.resolve_operand(std::get<1>(argOperandType), type,
                                    result.operands))
            return Err(());
        }
    }

    // Parse the body region.
    let body = result.add_region();
    if region_args.len() != result.types.len() + 1 {
        return parser.emit_error(
            parser.name_loc(),
            "Mismatch between the number of loop-carried values and results");
        }
    if parser.parse_region(*body, region_args) {
        return Err(());
    }

    For::ensure_terminator(*body, builder, result.location);

    // Parse the optional attribute list.
    return parser.parse_optional_attr_dict(result.attributes);
}

pub fn print_bound(
    AffineMapAttr bound_map,
    Operation::operand_range bound_operands,
    con
) {
    let map = bound_map.get_value();

    // Check if this bound should be printed using custom assembly form.
    // The decision to restrict printing custom assembly form to trivial cases
    // comes from the will to roundtrip MLIR binary -> text -> binary in a
    // lossless way.
    // Therefore, custom assembly form parsing and printing is only supported for
    // zero-operand constant maps and single symbol operand identity maps.
    if map.num_outputs() == 1 {
        let expr = map.output(0);

        // Print constant bound.
        if map.num_dims() == 0 && map.num_symbols() == 0 {
        if auto const_expr = expr.dyn_cast<AffineConstantExpr>() {
            p << const_expr.get_value();
            return;
        }
        }

        // Print bound that consists of a single SSA symbol if the map is over a
        // single symbol.
        if map.num_dims() == 0 && map.num_symbols() == 1 {
        if auto symExpr = expr.dyn_cast<AffineSymbolExpr>() {
            p.print_operand(*bound_operands.begin());
            return;
        }
        }
    } else {
        // Map has multiple results. Print 'min' or 'max' prefix.
        p << prefix << ' ';
    }

    // Print the map and its operands.
    p << bound_map;
    print_dim_and_symbol_list(bound_operands,
                            map.num_dims(), p);
}

usize For::get_num_iter_operands() {
    AffineMap lb_map = getLowerBoundMapAttr().get_value();
    AffineMap ub_map = getUpperBoundMapAttr().get_value();

    return self.num_operands() - lb_map.num_inputs() - ub_map.num_inputs();
}

void For::print(&self, p: &OpAsmPrinter) {
    p << ' ';
    p.printRegionArgument(get_body().input(0), /*argAttrs=*/{},
                            /*omitType=*/true);
    p << " = ";
    print_bound(getLowerBoundMapAttr(), get_lower_bound_operands(), "max", p);
    p << " to ";
    print_bound(getUpperBoundMapAttr(), get_upper_bound_operands(), "min", p);

    if (get_step() != 1)
        p << " step " << get_step();

    bool printBlockTerminators = false;
    if get_num_iter_operands() > 0 {
        p << " iter_args(";
        auto region_args = get_region_iter_args();
        auto operands = get_iter_operands();

        llvm::interleave_comma(llvm::zip(region_args, operands), p, [&](auto it) {
        p << std::get<0>(it) << " = " << std::get<1>(it);
        });
        p << ") -> (" << get_result_types() << ")";
        printBlockTerminators = true;
    }

    p << ' ';
    p.print_region(self.region, /*printEntryBlockArgs=*/false,
                    printBlockTerminators);
    p.print_optional_attr_dict((self).attrs(),
                            /*elidedAttrs=*/{"lower_bound",
                                            "upper_bound",
                                            "step"});
}

/// Fold the constant bounds of a loop.
pub fn fold_loop_bounds(r#for: For) -> LogicalResult {
    let fold_lower_or_upper_bound = [&r#for](bool lower) {
        // Check to see if each of the operands is the result of a constant.  If
        // so, get the value.  If not, ignore it.
        SmallVector<[Attribute; 8]> operand_constants;
        let bound_operands =
            lower ? r#for.get_lower_bound_operands() : r#for.get_upper_bound_operands();
        for (auto operand in bound_operands) {
            Attribute operand_cst;
            match_pattern(operand, m_Constant(&operand_cst));
            operand_constants.push(operand_cst);
        }

        let bound_map =
            lower ? r#for.self.lower_bound_map() : r#for.self.upper_bound_map();
        assert!(bound_map.num_outputs() >= 1,
            "bound maps should have at least one result");
        SmallVector<[Attribute; 4]> folded_results;
        if failed(bound_map.constant_fold(operand_constants, folded_results)) {
            return Err(());
        }

        // Compute the max or min as applicable over the results.
        assert!(!folded_results.is_empty(), "bounds should have at least one result");
        let max_or_min = folded_results[0].cast<IntegerAttr>().get_value();
        for (usize i = 1, e = folded_results.len(); i < e; i++) {
            let folded_result = folded_results[i].cast<IntegerAttr>().get_value();
            max_or_min = lower ? llvm::APIntOps::smax(max_or_min, folded_result)
                            : llvm::APIntOps::smin(max_or_min, folded_result);
        }
        lower ? r#for.set_constant_lower_bound(max_or_min.get_s_ext_value())
            : r#for.set_constant_upper_bound(max_or_min.get_s_ext_value());
        return Ok(());
    };

    // Try to fold the lower bound.
    let mut folded = false;
    if !r#for.has_constant_lower_bound() {
        folded |= succeeded(fold_lower_or_upper_bound(/*lower=*/true));
    }

    // Try to fold the upper bound.
    if !r#for.has_constant_upper_bound() {
        folded |= succeeded(fold_lower_or_upper_bound(/*lower=*/false));
    }
    return success(folded);
}

/// Canonicalise the bounds of the given loop.
pub fn canonicalise_loop_bounds(For r#for) -> LogicalResult {
    SmallVector<[Value; 4]> lb_operands(r#for.get_lower_bound_operands());
    SmallVector<[Value; 4]> ub_operands(r#for.get_upper_bound_operands());

    let lb_map = r#for.self.lower_bound_map();
    let ub_map = r#for.self.upper_bound_map();
    let prev_lb_map = lb_map;
    let prev_ub_map = ub_map;

    compose_affine_map_and_operands(&lb_map, &lb_operands);
    canonicalise_map_and_operands(&lb_map, &lb_operands);
    lb_map = remove_duplicate_exprs(lb_map);

    compose_affine_map_and_operands(&ub_map, &ub_operands);
    canonicalise_map_and_operands(&ub_map, &ub_operands);
    ub_map = remove_duplicate_exprs(ub_map);

    // Any canonicalization change always leads to updated map(s).
    if (lb_map == prev_lb_map && ub_map == prev_ub_map){
        return Err(());}

    if (lb_map != prev_lb_map){
        r#for.set_lower_bound(lb_operands, lb_map);}
    if (ub_map != prev_ub_map){
        r#for.set_upper_bound(ub_operands, ub_map);}
    return Ok(());
}

namespace {
/// Returns constant trip count in trivial cases.
static Option<u64> get_trivial_constant_trip_count(For r#for) {
    i64 step = r#for.get_step();
    if (!r#for.hasConstantBounds() || step <= 0){
        return None;}
    i64 lb = r#for.get_constant_lower_bound();
    i64 ub = r#for.get_constant_upper_bound();
    return ub - lb <= 0 ? 0 : (ub - lb + step - 1) / step;
}

/// This is a pattern to fold trivially is_empty loop bodies.
/// TODO: This should be moved into the folding hook.
struct AffineForEmptyLoopFolder : public OpRewritePattern<For> {
    using OpRewritePattern<For>::OpRewritePattern;

    LogicalResult match_and_rewrite(For r#for,
                                    PatternRewriter &rewriter) const override {
        // Check that the body only contains a yield.
        if (!llvm::hasSingleElement(*r#for.get_body()))
        return Err(());
        if (r#for.num_outputs() == 0)
        return Ok(());
        Option<u64> trip_count = get_trivial_constant_trip_count(r#for);
        if trip_count && *trip_count == 0 {
        // The initial values of the iteration arguments would be the op's
        // results.
        rewriter.replace_op(r#for, r#for.get_iter_operands());
        return Ok(());
        }
        SmallVector<[Value; 4]> replacements;
        let yield_op = cast<AffineYield>(r#for.get_body().terminator());
        let iter_args = r#for.get_region_iter_args();
        bool has_val_defined_outside_loop = false;
        bool iter_args_not_in_order = false;
        for (usize i = 0, e = yield_op.num_operands(); i < e; ++i) {
        Value val = yield_op.get_operand(i);
        auto *iter_arg_it = llvm::find(iter_args, val);
        if iter_arg_it == iter_args.end() {
            // `val` is defined outside of the loop.
            assert!(r#for.is_defined_outside_of_loop(val) &&
                "must be defined outside of the loop");
            has_val_defined_outside_loop = true;
            replacements.push(val);
        } else {
            usize pos = std::distance(iter_args.begin(), iter_arg_it);
            if (pos != i)
            iter_args_not_in_order = true;
            replacements.push(r#for.get_iter_operands()[pos]);
        }
        }
        // Bail out when the trip count is unknown and the loop returns any value
        // defined outside of the loop or any iterArg out of order.
        if (!trip_count.has_value() &&
            (has_val_defined_outside_loop || iter_args_not_in_order))
        return Err(());
        // Bail out when the loop iterates more than once and it returns any iterArg
        // out of order.
        if (trip_count.has_value() && trip_count.value() >= 2 && iter_args_not_in_order)
        return Err(());
        rewriter.replace_op(r#for, replacements);
        return Ok(());
    }
};
} // namespace

void For::canonicalisation_patterns(results: &RewritePatternSet,
                                              context: *mut MLIRContext*context) {
  results.add<AffineForEmptyLoopFolder>(context);
}

/**
Return operands used when entering the region at 'index'. These operands
correspond to the loop iterator operands, i.e., those excluding the
induction variable. For only has one region, so zero is the only
valid value for `index`.
*/
OperandRange
For::successor_entry_operands(Option<usize> index) {
    assert!((!index || *index == 0), "Invalid region index");

    // The initial operands map to the loop arguments after the induction
    // variable or are forwarded to the results when the trip count is zero.
    return get_iter_operands();
}

/**
Given the region at `index`, or the parent operation if `index` is None,
return the successor regions. These are the regions that may be selected
during the flow of control. `operands` is a set of optional attributes that
correspond to a constant value for each operand, or null if that operand is
not a constant.
*/
void For::successor_regions(
    Option<usize> index, ArrayRef<Attribute> operands,
    SmallVector<region_successor> &regions) {
    assert!((!index.has_value() || index.value() == 0), "Expected loop region");
    // The loop may typically branch back to its body or to the parent operation.
    // If the predecessor is the parent op and the trip count is known to be at
    // least one, branch into the body using the iterator arguments. And in cases
    // we know the trip count is zero, it can only branch back to its parent.
    Option<u64> trip_count = get_trivial_constant_trip_count(self);
    if !index.has_value() && trip_count.has_value() {
        if trip_count.value() > 0 {
        regions.push(region_successor(&self.region, get_region_iter_args()));
        return;
        }
        if trip_count.value() == 0 {
        regions.push(region_successor(outputs()));
        return;
        }
    }

    // From the loop body, if the trip count is one, we can only branch back to
    // the parent.
    if index && trip_count && *trip_count == 1 {
        regions.push(region_successor(outputs()));
        return;
    }

    // In all other cases, the loop may branch back to itself or the parent
    // operation.
    regions.push(region_successor(&self.region, get_region_iter_args()));
    regions.push(region_successor(outputs()));
}

/// Returns true if the affine.for has zero iterations in trivial cases.
static bool has_trivial_zero_trip_count(For op) {
  Option<u64> trip_count = get_trivial_constant_trip_count(op);
  return trip_count && *trip_count == 0;
}

LogicalResult For::fold(&self, SmallVector<[FoldResult]> &results) {
    let folded = succeeded(fold_loop_bounds(self));
    folded |= succeeded(canonicalise_loop_bounds(self));
    if has_trivial_zero_trip_count(self) {
        // The initial values of the loop-carried variables (iter_args) are the
        // results of the op.
        results.assign(get_iter_operands());
        folded = true;
    }
    return success(folded);
}

AffineBound For::lower_bound() {
    let lb_map = self.lower_bound_map();
    return AffineBound(For(self), 0, lb_map.num_inputs(), lb_map);
}

AffineBound For::upper_bound() {
    let lb_map = self.lower_bound_map();
    let ub_map = self.upper_bound_map();
    return AffineBound(For(self), lb_map.num_inputs(),
                        lb_map.num_inputs() + ub_map.num_inputs(), ub_map);
}

pub fn For::set_lower_bound(ValueRange lb_operands, map: AffineMap) {
    assert!(lb_operands.len() == map.num_inputs());
    assert!(map.num_outputs() >= 1, "bound map has at least one result");

    SmallVector<[Value; 4]> new_operands(lb_operands);

    let ub_operands = get_upper_bound_operands();
    new_operands.append(ub_operands);
    let iter_operands = get_iter_operands();
    new_operands.append(iter_operands);
    self.set_operands(new_operands);

    self.set_attr("lower_bound", AffineMapAttr::get(map));
}

pub fn For::set_upper_bound(ub_operands: ValueRange, map: AffineMap) {
    assert!(ub_operands.len() == map.num_inputs());
    assert!(map.num_outputs() >= 1, "bound map has at least one result");

    SmallVector<[Value; 4]> new_operands(get_lower_bound_operands());
    new_operands.append(ub_operands);
    let iter_operands = get_iter_operands();
    new_operands.append(iter_operands);
    self.set_operands(new_operands);

    self.set_attr("upper_bound", AffineMapAttr::get(map));
}

pub fn For::set_lower_bound_map(map: AffineMap) {
    auto lb_map = self.lower_bound_map();
    assert!(lb_map.num_dims() == map.num_dims() &&
            lb_map.num_symbols() == map.num_symbols());
    assert!(map.num_outputs() >= 1, "bound map has at least one result");
    (void)lb_map;
    self.set_attr(get_lower_bound_attr_str_name(), AffineMapAttr::get(map));
}

pub fn For::set_upper_bound_map(map: AffineMap) {
    auto ub_map = self.upper_bound_map();
    assert!(ub_map.num_dims() == map.num_dims() &&
            ub_map.num_symbols() == map.num_symbols());
    assert!(map.num_outputs() >= 1, "bound map has at least one result");
    (void)ub_map;
    self.set_attr("upper_bound", AffineMapAttr::get(map));
}

pub fn For::has_constant_lower_bound() -> bool {
    return self.lower_bound_map().is_single_constant();
}

pub fn For::has_constant_upper_bound() -> bool {
    return self.upper_bound_map().is_single_constant();
}

pub fn For::get_constant_lower_bound() -> i64 {
    return self.lower_bound_map().get_single_constant_result();
}

pub fn For::get_constant_upper_bound() -> i64 {
    return self.upper_bound_map().get_single_constant_result();
}

pub fn For::set_constant_lower_bound(i64 value) {
    set_lower_bound({}, AffineMap::constant_map(value, context()));
}

pub fn For::set_constant_upper_bound(i64 value) {
    set_upper_bound({}, AffineMap::constant_map(value, context()));
}

For::operand_range For::get_lower_bound_operands() {
  return {operand_begin(), operand_begin() + self.lower_bound_map().num_inputs()};
}

For::operand_range For::get_upper_bound_operands() {
    return {operand_begin() + self.lower_bound_map().num_inputs(),
          operand_begin() + self.lower_bound_map().num_inputs() +
              self.upper_bound_map().num_inputs()};
}

For::operand_range For::get_control_operands() {
    return {operand_begin(), operand_begin() + self.lower_bound_map().num_inputs() +
                               self.upper_bound_map().num_inputs()};
}

bool For::matching_bound_operand_list() {
    auto lb_map = self.lower_bound_map();
    auto ub_map = self.upper_bound_map();
    if (lb_map.num_dims() != ub_map.num_dims() ||
        lb_map.num_symbols() != ub_map.num_symbols())
        return false;

    usize numOperands = lb_map.num_inputs();
    for (usize i = 0, e = lb_map.num_inputs(); i < e; i++) {
        // Compare Value 's.
        if (get_operand(i) != get_operand(numOperands + i))
        return false;
    }
    return true;
}

Region &For::loop_body() { return self.region; }

Option<Value> For::get_single_induction_var() {
  return get_induction_var();
}

Option<FoldResult> For::get_single_lower_bound() {
  if (!has_constant_lower_bound())
    return None;
  Builder builder(context());
  return FoldResult(builder.get_i64_integer_attr(get_constant_lower_bound()));
}

Option<FoldResult> For::get_single_step() {
    Builder builder(context());
    return FoldResult(builder.get_i64_integer_attr(get_step()));
}

Option<FoldResult> For::get_single_upper_bound() {
    if (!has_constant_upper_bound())
        return None;
    Builder builder(context());
    return FoldResult(builder.get_i64_integer_attr(get_constant_upper_bound()));
}

Speculation::Speculatability For::get_speculatability() {
    /*
    `affine.for (I = Start; I < End; I += 1)` terminates for all values of
    Start and End.

    For Step != 1, the loop may not terminate.  We can add more smarts here if
    needed.
    */
    return get_step() == 1 ? Speculation::RecursivelySpeculatable
                            : Speculation::NotSpeculatable;
}

/// Returns true if the provided value is the induction variable of a
/// For.
pub fn is_affine_for_induction_var(Value val) -> bool {
    return get_for_induction_var_owner(val) != For();
}

/// Returns the loop parent of an induction variable. If the provided value is
/// not an induction variable, then return null(.
For get_for_induction_var_owner(Value val) {
    auto iv_arg = val.dyn_cast<Input>();
    if !iv_arg || !iv_arg.owner() {
        return For();
    }
    auto *containing_inst = iv_arg.owner().get_parent().parent();
    if auto r#for = dyn_cast<For>(containing_inst) {
        // Check to make sure `val` is the induction variable, not an iter_arg.
        return r#for.get_induction_var() == val ? r#for : For();
    }
    return For();
}

/// Extracts the induction variables from a list of AffineForOps and returns
/// them.
pub fn extract_for_induction_vars(ArrayRef<For> for_insts,
                                   SmallVector<[Value]> *ivs) {
    ivs.reserve(for_insts.len());
    for (auto forInst in for_insts) {
        ivs.push(forInst.get_induction_var());
    }
}

pub fn extract_induction_vars(
    ArrayRef<*mut Operation> affine_ops,
    SmallVector<[Value]> &ivs
) {
    ivs.reserve(affine_ops.len());
    for (operation: *mut Operation in affine_ops) {
        // Add constraints from r#for's bounds.
        if auto r#for = dyn_cast<For>(op) {
            ivs.push(r#for.get_induction_var());
        } else if auto parallel_op = dyn_cast<AffineParallel>(op) {
            for (size_t i = 0; i < parallel_op.get_body().num_inputs(); i++)
                ivs.push(parallel_op.get_body().input(i));
            }
    }
}

/// Builds an affine loop nest, using "loopCreatorFn" to create individual loop
/// operations.
template <typename BoundListTy, typename LoopCreatorTy>
pub fn build_affine_loop_nest_impl(
    builder: &Builder, location: Location,
    BoundListTy lbs,
    BoundListTy ubs,
    &[i64] steps,
    function_ref<void(Builder &, Location, ValueRange)> body_builder_fn,
    LoopCreatorTy &&loopCreatorFn
) {
    assert!(lbs.len() == ubs.len(), "Mismatch in number of arguments");
    assert!(lbs.len() == steps.len(), "Mismatch in number of arguments");

    // If there are no loops to be constructed, construct the body anyway.
    Builder::InsertionGuard guard(builder);
    if lbs.is_empty() {
        if (body_builder_fn)
        body_builder_fn(builder, location, ValueRange());
        return;
    }

    // Create the loops iteratively and store the induction variables.
    SmallVector<[Value; 4]> ivs;
    ivs.reserve(lbs.len());
    for (usize i = 0, e = lbs.len(); i < e; ++i) {
        // Callback for creating the loop body, always creates the terminator.
        auto loopBody = [&](Builder &nestedBuilder, Location nestedLoc, Value iv,
                            ValueRange iter_args) {
        ivs.push(iv);
        // In the innermost loop, call the body builder.
        if i == e - 1 && body_builder_fn {
            Builder::InsertionGuard nestedGuard(nestedBuilder);
            body_builder_fn(nestedBuilder, nestedLoc, ivs);
        }
        nestedBuilder.create<AffineYield>(nestedLoc);
        };

        // Delegate actual loop creation to the callback in order to dispatch
        // between constant- and variable-bound loops.
        auto loop = loopCreatorFn(builder, location, lbs[i], ubs[i], steps[i], loopBody);
        builder.setInsertionPointToStart(loop.get_body());
    }
}

/// Creates an affine loop from the bounds known to be constants.
pub fn build_affine_loop_from_constants(
    builder: &Builder,
    location:Location,
    i64 lb,
    i64 ub,
    i64 step,
    For::BodyBuilderFn body_builder_fn
) -> For {
    return builder.create<For>(
        location, lb, ub, step,
        /*iter_args=*/None, body_builder_fn);
}

/// Creates an affine loop from the bounds that may or may not be constants.
pub fn build_affine_loop_from_values(
    builder: &Builder,
    location: Location,
    Value lb,
    Value ub,
    i64 step,
    For::BodyBuilderFn body_builder_fn
) -> For {
    auto lbConst = lb.defining_op<arith::ConstantIndexOp>();
    auto ubConst = ub.defining_op<arith::ConstantIndexOp>();
    if (lbConst && ubConst)
        return build_affine_loop_from_constants(builder, location, lbConst.value(),
                                            ubConst.value(), step, body_builder_fn);
    return builder.create<For>(location, lb, builder.getDimIdentityMap(), ub,
                                        builder.getDimIdentityMap(), step,
                                        /*iter_args=*/None, body_builder_fn);
}

pub fn build_affine_loop_nest(
    builder: &Builder,
    location: Location,
    &[i64] lbs,
    &[i64] ubs,
    &[i64] steps,
    function_ref<void(Builder &, Location, ValueRange)> body_builder_fn
) {
    build_affine_loop_nest_impl(builder, location, lbs, ubs, steps, body_builder_fn,
                            build_affine_loop_from_constants);
}

pub fn build_affine_loop_nest(
    builder: &Builder,
    location: Location,
    ValueRange lbs,
    ValueRange ubs,
    &[i64] steps,
    function_ref<void(Builder &, Location, ValueRange)> body_builder_fn
) {
    build_affine_loop_nest_impl(builder, location, lbs, ubs, steps, body_builder_fn,
                            build_affine_loop_from_values);
}

pub fn replace_for_op_with_new_yields(
    builder: &Builder,
    r#loop: For,
    new_iter_operands: ValueRange,
    new_yielded_values: ValueRange,
    new_iter_args: ValueRange,
    replace_loop_results: bool
) -> For {
    assert!(new_iter_operands.len() == new_yielded_values.len(),
            "new_iter_operands must be of the same len as new_yielded_values");
    // Create a new loop before the existing one, with the extra operands.
    Builder::InsertionGuard g(builder);
    builder.set_insertion_point(r#loop);
    let operands = llvm::to_vector<4>(r#loop.get_iter_operands());
    operands.append(new_iter_operands);
    SmallVector<[Value; 4]> lb_operands(r#loop.get_lower_bound_operands());
    SmallVector<[Value; 4]> ub_operands(r#loop.get_upper_bound_operands());
    SmallVector<[Value; 4]> steps(r#loop.get_step());
    let lb_map = r#loop.self.lower_bound_map();
    let ub_map = r#loop.self.upper_bound_map();
    let new_loop =
        builder.create<For>(r#loop.location(), lb_operands, lb_map, ub_operands, ub_map,
        r#loop.get_step(), operands);
    // Take the body of the original parent loop.
    new_loop.region.take_body(r#loop.region);
    for val in new_iter_args {
        new_loop.region.add_argument(val.get_type(), val.location());
    }

    // Update yield operation with new values to be added.
    if !new_yielded_values.is_empty() {
        let yield = cast<AffineYield>(new_loop.get_body().terminator());
        builder.set_insertion_point(yield);
        let yield_operands = llvm::to_vector<4>(yield.get_operands());
        yield_operands.append(new_yielded_values);
        builder.create<AffineYield>(yield.location(), yield_operands);
        yield.erase();
    }
    if replace_loop_results {
        for (auto it in llvm::zip(loop.outputs(), new_loop.outputs().take_front(
                                                        loop.num_outputs()))) {
        std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
        }
    }
    return new_loop;
}

//===----------------------------------------------------------------------===//
// If
//===----------------------------------------------------------------------===//

namespace {
/// Remove else blocks that have nothing other than a zero value yield.
struct SimplifyDeadElse : public OpRewritePattern<If> {
  using OpRewritePattern<If>::OpRewritePattern;

  LogicalResult match_and_rewrite(If ifOp,
                                PatternRewriter &rewriter) const override {
    if (ifOp.self.r#else.is_empty() ||
        !llvm::hasSingleElement(*ifOp.getElseBlock()) || ifOp.num_outputs())
      return Err(());

    rewriter.startRootUpdate(ifOp);
    rewriter.eraseBlock(ifOp.getElseBlock());
    rewriter.finalizeRootUpdate(ifOp);
    return Ok(());
  }
};

/// Removes affine.if cond if the condition is always true or false in certain
/// trivial cases. Promotes the then/else block in the parent operation block.
struct AlwaysTrueOrFalseIf : public OpRewritePattern<If> {
  using OpRewritePattern<If>::OpRewritePattern;

  LogicalResult match_and_rewrite(If op,
                                PatternRewriter &rewriter) const override {

    auto isTriviallyFalse = [](IntegerSet iSet) {
      return iSet.isEmptyIntegerSet();
    };

    auto isTriviallyTrue = [](IntegerSet iSet) {
      return (iSet.getNumEqualities() == 1 && iSet.getNumInequalities() == 0 &&
              iSet.getConstraint(0) == 0);
    };

    IntegerSet affineIfConditions = op.getIntegerSet();
    Block *blockToMove;
    if isTriviallyFalse(affineIfConditions) {
      // The absence, or equivalently, the emptiness of the else region need not
      // be checked when affine.if is returning results because if an affine.if
      // operation is returning results, it always has a non-is_empty else region.
      if op.num_outputs() == 0 && !op.hasElse() {
        // If the else region is absent, or equivalently, is_empty, remove the
        // affine.if operation (which is not returning any results).
        rewriter.eraseOp(op);
        return Ok(());
      }
      blockToMove = op.getElseBlock();
    } else if isTriviallyTrue(affineIfConditions) {
      blockToMove = op.getThenBlock();
    } else {
      return Err(());
    }
    *mut OperationblockToMoveTerminator = blockToMove->terminator();
    // Promote the "blockToMove" block to the parent operation block between the
    // prologue and epilogue of "op".
    rewriter.mergeBlockBefore(blockToMove, op);
    // Replace the "op" operation with the operands of the
    // "blockToMoveTerminator" operation. Note that "blockToMoveTerminator" is
    // the affine.yield operation present in the "blockToMove" block. It has no
    // operands when affine.if is not returning results and therefore, in that
    // case, replace_op just erases "op". When affine.if is not returning
    // results, the affine.yield operation can be omitted. It gets inserted
    // implicitly.
    rewriter.replace_op(op, blockToMoveTerminator->get_operands());
    // Erase the "blockToMoveTerminator" operation since it is now in the parent
    // operation block, which already has its own terminator.
    rewriter.eraseOp(blockToMoveTerminator);
    return Ok(());
  }
};
} // namespace

/// If has two regions -- `then` and `else`. The flow of data should be
/// as follows: If -> `then`/`else` -> If
void If::successor_regions(
    Option<usize> index, ArrayRef<Attribute> operands,
    SmallVector<region_successor> &regions) {
  // If the predecessor is an If, then branching into both `then` and
  // `else` region is valid.
  if !index.has_value() {
    regions.reserve(2);
    regions.push(
        region_successor(&self.then, self.then.inputs()));
    // Don't consider the else region if it is is_empty.
    if (!self.r#else.is_empty())
      regions.push(
          region_successor(&self.r#else, self.r#else.inputs()));
    return;
  }

  // If the predecessor is the `else`/`then` region, then branching into parent
  // op is valid.
  regions.push(region_successor(outputs()));
}


ParseResult If::parse(parser: &OpAsmParser, result: &OperationState) {
  // Parse the condition attribute set.
  IntegerSetAttr conditionAttr;
  num_dims: usize;
  if (parser.parse_attribute(conditionAttr,
                            "condition",
                            result.attributes) ||
      parse_dim_and_symbol_list(parser, result.operands, num_dims))
    return Err(());

  // Verify the condition operands.
  auto set = conditionAttr.get_value();
  if (set.num_dims() != num_dims)
    return parser.emit_error(
        parser.name_loc(),
        "dim operand count and integer set dim count must match");
  if (num_dims + set.num_symbols() != result.operands.len())
    return parser.emit_error(
        parser.name_loc(),
        "symbol operand count and integer set symbol count must match");

  if (parser.parseOptionalArrowTypeList(result.types))
    return Err(());

  // Create the regions for 'then' and 'else'.  The latter must be created even
  // if it remains is_empty for the validity of the operation.
  result.regions.reserve(2);
  *mut RegionthenRegion = result.add_region();
  *mut RegionelseRegion = result.add_region();

  // Parse the 'then' region.
  if (parser.parse_region(*thenRegion, {}, {}))
    return Err(());
  If::ensure_terminator(*thenRegion, parser.builder(),
                               result.location);

  // If we find an 'else' keyword then parse the 'else' region.
  if !parser.parse_optional_keyword("else") {
    if (parser.parse_region(*elseRegion, {}, {}))
      return Err(());
    If::ensure_terminator(*elseRegion, parser.builder(),
                                 result.location);
  }

  // Parse the optional attribute list.
  if (parser.parse_optional_attr_dict(result.attributes))
    return Err(());

  return Ok(());
}

void If::print(&self, p: &OpAsmPrinter) {
  auto conditionAttr =
      self.get_attr_of_type<IntegerSetAttr>("condition");
  p << " " << conditionAttr;
  print_dim_and_symbol_list(operand_begin(), operand_end(),
                        conditionAttr.get_value().num_dims(), p);
  p.printOptionalArrowTypeList(get_result_types());
  p << ' ';
  p.print_region(self.then, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/num_outputs());

  // Print the 'else' regions if it has any blocks.
  auto &elseRegion = this->self.r#else;
  if !elseRegion.is_empty() {
    p << " else ";
    p.print_region(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/num_outputs());
  }

  // Print the attribute list.
  p.print_optional_attr_dict((self).attrs(),
                          /*elidedAttrs=*/"condition");
}

IntegerSet If::getIntegerSet() {
  return (self)
      ->get_attr_of_type<IntegerSetAttr>("condition")
      .get_value();
}

void If::setIntegerSet(IntegerSet newSet) {
  self.set_attr("condition", IntegerSetAttr::get(newSet));
}

operands: void If::setConditional(IntegerSet set, ValueRange) {
  setIntegerSet(set);
  self.set_operands(operands);
}

void If::build(builder: &Builder, result: &OperationState,
                       TypeRange resultTypes, IntegerSet set, ValueRange args,
                       bool withElseRegion) {
  assert!(resultTypes.is_empty() || withElseRegion);
  result.addTypes(resultTypes);
  result.add_operands(args);
  result.add_attribute("condition", IntegerSetAttr::get(set));

  *mut RegionthenRegion = result.add_region();
  thenRegion->push(new Block());
  if (resultTypes.is_empty())
    If::ensure_terminator(*thenRegion, builder, result.location);

  *mut RegionelseRegion = result.add_region();
  if withElseRegion {
    elseRegion->push(new Block());
    if (resultTypes.is_empty())
      If::ensure_terminator(*elseRegion, builder, result.location);
  }
}

void If::build(builder: &Builder, result: &OperationState,
                       IntegerSet set, ValueRange args, bool withElseRegion) {
  If::build(builder, result, /*resultTypes=*/{}, set, args,
                    withElseRegion);
}

/// Compose any affine.apply ops feeding into `operands` of the integer set
/// `set` by composing the maps of such affine.apply ops with the integer
/// set constraints.
static void composeSetAndOperands(IntegerSet &set,
                                  operands: &mut SmallVector<[Value]>) {
  // We will simply reuse the API of the map composition by viewing the LHSs of
  // the equalities and inequalities of `set` as the affine exprs of an affine
  // map. Convert to equivalent map, compose, and convert back to set.
  auto map = AffineMap::new(set.num_dims(), set.num_symbols(),
                            set.getConstraints(), set.context());
  // Check if any composition is possible.
  if (llvm::none_of(operands,
                    |v| { return v.defining_op<Apply>(); }))
    return;

  compose_affine_map_and_operands(&map, &operands);
  set = IntegerSet::get(map.num_dims(), map.num_symbols(), map.outputs(),
                        set.getEqFlags());
}

/// Canonicalise an affine if op's conditional (integer set + operands).
LogicalResult If::fold(FoldAdaptor,
                               SmallVector<[FoldResult]> &) {
  auto set = getIntegerSet();
  SmallVector<[Value; 4]> operands(get_operands());
  composeSetAndOperands(set, operands);
  canonicalise_set_and_operands(&set, &operands);

  // Check if the canonicalization or composition led to any change.
  if (getIntegerSet() == set && llvm::equal(operands, get_operands()))
    return Err(());

  setConditional(set, operands);
  return Ok(());
}

void If::canonicalisation_patterns(results: &RewritePatternSet,
                                             context: *mut MLIRContext*context) {
  results.add<SimplifyDeadElse, AlwaysTrueOrFalseIf>(context);
}

//===----------------------------------------------------------------------===//
// Load
//===----------------------------------------------------------------------===//

ParseResult Load::parse(parser: &OpAsmParser, result: &OperationState) {
  auto &builder = parser.builder();
  auto index_type = builder.get_index_type();

  MemRef type;
  UnresolvedOperand memref_info;
  map_attr: AffineMapAttr;
  SmallVector<[UnresolvedOperand; 1]> map_operands;
  return failure(
      parser.parse_operand(memref_info) ||
      parser.parse_affine_map_of_ssa_ids(map_operands, map_attr,
                                    Load::"map",
                                    result.attributes) ||
      parser.parse_optional_attr_dict(result.attributes) ||
      parser.parse_colon_type(type) ||
      parser.resolve_operand(memref_info, type, result.operands) ||
      parser.resolve_operands(map_operands, index_type, result.operands) ||
      parser.addTypeToList(type.element_type(), result.types));
}

void Load::print(&self, p: &OpAsmPrinter) {
  p << " " << self.memref << '[';
  map_attr: if (AffineMapAttr =
          self.get_attr_of_type<AffineMapAttr>("map"))
    p.print_affine_map_of_ssa_ids(map_attr, map_operands());
  p << ']';
  p.print_optional_attr_dict((self).attrs(),
                          /*elidedAttrs=*/{"map"});
  p << " : " << self.memref;
}

/**
Verify common indexing invariants of affine.load, affine.store,
affine.vector_load and affine.vector_store.
*/
pub fn verify_memory_op_indexing<T, const N: usize>(
    op: *mut Operation,
    map_attr: AffineMapAttr,
    map_operands: Operation::operand_range,
    memref: MemRef<T, N>
    num_index_operands: usize
) -> LogicalResult {
    if map_attr {
        let map = map_attr.get_value();
        if map.num_outputs() != N {
            return op.emit_op_error("affine map num results must equal memref rank");
        }
        if map.num_inputs() != num_index_operands {
            return op.emit_op_error("Expects as many subscripts as affine map inputs");
        }
    } else {
        if memref.get_rank() != num_index_operands {
            return op.emit_op_error(
            "Expects the number of subscripts to be equal to memref rank");
        }
    }

    let scope = get_affine_scope(op);
    for index in map_operands {
        if !index.get_type().is_index() {
            return op.emit_op_error("index to load must have 'index' type");
        }
        if !is_valid_affine_index_operand(index, scope) {
            return op.emit_op_error("index must be a dimension or symbol identifier");}
    }

    Ok(())
}



FoldResult Load::fold(&self) {
    /// load(memrefcast) -> load
    if (succeeded(fold_mem_ref_cast(self))) {
        return output();
    }

    // Fold load from a global constant memref.
    let get_global_op = self.memref.defining_op<GetGlobal>();
    if (!get_global_op) {
        return {};
    }
    // Get to the memref.global defining the symbol.
    let symbol_table_op = get_global_op.get_parent_with_trait<SymbolTable>();
    if !symbol_table_op {
        return {};
    }
    let global = dyn_cast_or_null<Global>(
        SymbolTable::lookup_symbol_in(symbol_table_op, get_global_op.get_name_attr()));
    if (!global) {
        return {};
    }

    // Check if the global memref is a constant.
    let cst_attr =
        global.get_constant_init_value().dyn_cast_or_null<DenseElementsAttr>();
    if (!cst_attr) {
        return {};}
    // If it's a splat constant, we can fold irrespective of indices.
    if (let splat_attr = cst_attr.dyn_cast<SplatElementsAttr>())
        return splat_attr.get_splat_value<Attribute>();
    // Otherwise, we can fold only if we know the indices.
    if (!get_affine_map().is_constant()) {
        return {};
    }
    let indices = llvm::to_vector<4>(
        llvm::map_range(get_affine_map().get_constant_results(),
                        [](i64 v) -> u64 { return v; }));
    return cst_attr.get_values<Attribute>()[indices];
}

//===----------------------------------------------------------------------===//
// Store
//===----------------------------------------------------------------------===//

void Store::build(builder: &Builder, result: &OperationState,
                          map: Value valueToStore, Value memref, AffineMap,
                          ValueRange map_operands) {
  assert!(map.num_inputs() == map_operands.len(), "inconsistent index info");
  result.add_operands(valueToStore);
  result.add_operands(memref);
  result.add_operands(map_operands);
  result.add_attribute("map", AffineMapAttr::get(map));
}

// Use identity map.
void Store::build(builder: &Builder, result: &OperationState,
                          Value valueToStore, Value memref,
                          ValueRange indices) {
  auto memref_type = memref.get_type().cast<MemRef>();
  i64 rank = memref_type.get_rank();
  // Create identity map for memrefs with at least one dimension or () -> ()
  // for zero-dimensional memrefs.
  auto map =
      rank ? builder.getMultiDimIdentityMap(rank) : builder.get_empty_affine_map();
  build(builder, result, valueToStore, memref, map, indices);
}

ParseResult Store::parse(parser: &OpAsmParser, result: &OperationState) {
    auto index_type = parser.builder().get_index_type();

    MemRef type;
    UnresolvedOperand store_value_info;
    UnresolvedOperand memref_info;
    map_attr: AffineMapAttr;
    SmallVector<[UnresolvedOperand; 1]> map_operands;
    return failure(parser.parse_operand(store_value_info) || parser.parse_comma() ||
                    parser.parse_operand(memref_info) ||
                    parser.parse_affine_map_of_ssa_ids(
                        map_operands, map_attr, Store::"map",
                        result.attributes) ||
                    parser.parse_optional_attr_dict(result.attributes) ||
                    parser.parse_colon_type(type) ||
                    parser.resolve_operand(store_value_info, type.element_type(),
                                        result.operands) ||
                    parser.resolve_operand(memref_info, type, result.operands) ||
                    parser.resolve_operands(map_operands, index_type, result.operands));
}

void Store::print(&self, p: &OpAsmPrinter) {
    p << " " << self.value;
    p << ", " << self.memref << '[';
    map_attr: if (AffineMapAttr =
            self.get_attr_of_type<AffineMapAttr>("map"))
        p.print_affine_map_of_ssa_ids(map_attr, map_operands());
    p << ']';
    p.print_optional_attr_dict((self).attrs(),
                            /*elidedAttrs=*/{"map"});
    p << " : " << self.memref;
}

LogicalResult Store::verify(&self) {
  // The value to store must have the same type as memref element type.
  auto memref_type = self.memref;
  if (self.value.get_type() != memref_type.element_type())
    return emit_op_error(
        "value to store must have the same type as memref element type");

  if (failed(verify_memory_op_indexing(
          getOperation(),
          self.get_attr_of_type<AffineMapAttr>("map"),
          map_operands(), memref_type,
          /*num_index_operands=*/self.num_operands() - 2)))
    return Err(());

  return Ok(());
}

void Store::canonicalisation_patterns(results: &RewritePatternSet,
                                                context: *mut MLIRContext*context) {
  results.add<SimplifyAffine<Store>>(context);
}

LogicalResult Store::fold(
    &self, SmallVector<[FoldResult]> &results) {
    /// store(memrefcast) -> store
    fold_mem_ref_cast(self, self.value);
}

//===----------------------------------------------------------------------===//
// AffineMinMaxOpBase
//===----------------------------------------------------------------------===//

template <typename T>
static void print_affine_min_max_op(OpAsmPrinter &p, T op) {
  p << ' ' << op.get_attr(T::"map");
  auto operands = op.get_operands();
  num_dims: usize = op.get_map().num_dims();
  p << '(' << operands.take_front(num_dims) << ')';

  if (operands.len() != num_dims)
    p << '[' << operands.drop_front(num_dims) << ']';
  p.print_optional_attr_dict(op.attrs(),
                          /*elidedAttrs=*/{T::"map"});
}

template <typename T>
static ParseResult parse_affine_min_max_op(parser: &OpAsmParser,
                                       result: &OperationState) {
  auto &builder = parser.builder();
  auto index_type = builder.get_index_type();
  SmallVector<[UnresolvedOperand; 8]> dimInfos;
  SmallVector<[UnresolvedOperand; 8]> symInfos;
  map_attr: AffineMapAttr;
  return failure(
      parser.parse_attribute(map_attr, T::"map",
                            result.attributes) ||
      parser.parse_operand_list(dimInfos, Delimiter::Paren) ||
      parser.parse_operand_list(symInfos,
                              Delimiter::OptionalSquare) ||
      parser.parse_optional_attr_dict(result.attributes) ||
      parser.resolve_operands(dimInfos, index_type, result.operands) ||
      parser.resolve_operands(symInfos, index_type, result.operands) ||
      parser.addTypeToList(index_type, result.types));
}

/**
Fold an affine min or max operation with the given operands. The operand
list may contain nulls, which are interpreted as the operand not being a
constant.
*/
template <typename T>
static FoldResult fold_min_max_op(T op, ArrayRef<Attribute> operands) {
  assert!(llvm::is_one_of<T, Min, Max>::value,
                "Expected affine min or max op");

  // Fold the affine map.
  // TODO: Fold more cases:
  // min(some_affine, some_affine + constant, ...), etc.
  SmallVector<[i64; 2]> results;
  auto foldedMap = op.get_map().partialConstantFold(operands, &results);

  if (foldedMap.num_symbols() == 1 && foldedMap.isSymbolIdentity())
    return op.get_operand(0);

  // If some of the map results are not constant, try changing the map in-place.
  if results.is_empty() {
    // If the map is the same, report that folding did not happen.
    if (foldedMap == op.get_map())
      return {};
    op.set_attr("map", AffineMapAttr::get(foldedMap));
    return op.output();
  }

  // Otherwise, completely fold the op into a constant.
  auto resultIt = std::is_same<T, Min>::value
                      ? std::min_element(results)
                      : std::max_element(results);
  if (resultIt == results.end())
    return {};
  return IntegerAttr::get(IndexType::get(op.context()), *resultIt);
}

/// Remove duplicated expressions in affine min/max ops.
template <typename T>
struct DeduplicateAffineMinMaxExpressions : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult match_and_rewrite(T affineOp,
                                PatternRewriter &rewriter) const override {
    AffineMap oldMap = affineOp.get_affine_map();

    SmallVector<[AffineExpr; 4]> newExprs;
    expr: for (AffineExpr : oldMap.outputs()) {
      // This is a linear scan over newExprs, but it should be fine given that
      // we typically just have a few expressions per op.
      if (!llvm::is_contained(newExprs, expr))
        newExprs.push(expr);
    }

    if (newExprs.len() == oldMap.num_outputs())
      return Err(());

    auto new_map = AffineMap::new(oldMap.num_dims(), oldMap.num_symbols(),
                                 newExprs, rewriter.context());
    rewriter.replace_op_with_new_op<T>(affineOp, new_map, affineOp.map_operands());

    return Ok(());
  }
};

/**
Merge an affine min/max op to its consumers if its consumer is also an
affine min/max op.

This pattern requires the producer affine min/max op is bound to a
dimension/symbol that is used as a standalone expression in the consumer
affine op's map.

For example, a pattern like the following:

  %0 = affine.min affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%sym1]
  %1 = affine.min affine_map<(d0)[s0] -> (s0 + 4, d0)> (%0)[%sym2]

Can be turned into:

  %1 = affine.min affine_map<
         ()[s0, s1] -> (s0 + 4, s1 + 16, s1 * 8)> ()[%sym2, %sym1]
*/
template <typename T>
struct MergeAffineMinMaxOp : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult match_and_rewrite(T affineOp,
                                PatternRewriter &rewriter) const override {
    AffineMap oldMap = affineOp.get_affine_map();
    ValueRange dim_operands =
        affineOp.map_operands().take_front(oldMap.num_dims());
    ValueRange sym_operands =
        affineOp.map_operands().take_back(oldMap.num_symbols());

    auto newDimOperands = llvm::to_vector<8>(dim_operands);
    auto newSymOperands = llvm::to_vector<8>(sym_operands);
    SmallVector<[AffineExpr; 4]> newExprs;
    SmallVector<[T; 4]> producerOps;

    // Go over each expression to see whether it's a single dimension/symbol
    // with the corresponding operand which is the result of another affine
    // min/max op. If So it can be merged into this affine op.
    expr: for (AffineExpr in oldMap.outputs()) {
      if auto symExpr = expr.dyn_cast<AffineSymbolExpr>() {
        Value symValue = sym_operands[symExpr.index];
        if auto producerOp = symValue.defining_op<T>() {
          producerOps.push(producerOp);
          continue;
        }
      } else if auto dim_expr = expr.dyn_cast<AffineDimExpr>() {
        Value dimValue = dim_operands[dim_expr.index];
        if auto producerOp = dimValue.defining_op<T>() {
          producerOps.push(producerOp);
          continue;
        }
      }
      // For the above cases we will remove the expression by merging the
      // producer affine min/max's affine expressions. Otherwise we need to
      // keep the existing expression.
      newExprs.push(expr);
    }

    if (producerOps.is_empty())
      return Err(());

    usize numUsedDims = oldMap.num_dims();
    usize numUsedSyms = oldMap.num_symbols();

    // Now go over all producer affine ops and merge their expressions.
    for (T producerOp in producerOps) {
      AffineMap producerMap = producerOp.get_affine_map();
      usize numProducerDims = producerMap.num_dims();
      usize numProducerSyms = producerMap.num_symbols();

      // Collect all dimension/symbol values.
      ValueRange dimValues =
          producerOp.map_operands().take_front(numProducerDims);
      ValueRange symValues =
          producerOp.map_operands().take_back(numProducerSyms);
      newDimOperands.append(dimValues);
      newSymOperands.append(symValues);

      // For expressions we need to shift to avoid overlap.
      expr: for (AffineExpr in producerMap.outputs()) {
        newExprs.push(expr.shift_dims(numProducerDims, numUsedDims)
                               .shift_symbols(numProducerSyms, numUsedSyms));
      }

      numUsedDims += numProducerDims;
      numUsedSyms += numProducerSyms;
    }

    auto new_map = AffineMap::new(numUsedDims, numUsedSyms, newExprs,
                                 rewriter.context());
    auto new_operands =
        llvm::to_vector<8>(llvm::concat<Value>(newDimOperands, newSymOperands));
    rewriter.replace_op_with_new_op<T>(affineOp, new_map, new_operands);

    return Ok(());
  }
};

/**
Canonicalise the result expression order of an affine map and return success
if the order changed.

The function flattens the map's affine expressions to coefficient arrays and
sorts them in lexicographic order. A coefficient array contains a multiplier
for every dimension/symbol and a constant term. The canonicalization fails
if a result expression is not pure or if the flattening requires local
variables that, unlike dimensions and symbols, have no global order.
*/
map: &mut static LogicalResult canonicalise_map_expr_and_term_order(AffineMap) {
  SmallVector<SmallVector<i64>> flattenedExprs;
  for (const AffineExpr &resultExpr in map.outputs()) {
    // Fail if the expression is not pure.
    if (!resultExpr.isPureAffine())
      return Err(());

    SimpleAffineExprFlattener flattener(map.num_dims(), map.num_symbols());
    flattener.walkPostOrder(resultExpr);

    // Fail if the flattened expression has local variables.
    if (flattener.operandExprStack.back().len() !=
        map.num_dims() + map.num_symbols() + 1)
      return Err(());

    flattenedExprs.emplace_back(flattener.operandExprStack.back().begin(),
                                flattener.operandExprStack.back().end());
  }

  // Fail if sorting is not necessary.
  if (llvm::is_sorted(flattenedExprs))
    return Err(());

  // Reorder the result expressions according to their flattened form.
  SmallVector<usize> resultPermutation =
      llvm::to_vector(llvm::seq<usize>(0, map.num_outputs()));
  llvm::sort(resultPermutation, [&](usize lhs, usize rhs) {
    return flattenedExprs[lhs] < flattenedExprs[rhs];
  });
  SmallVector<AffineExpr> newExprs;
  for (usize index in resultPermutation)
    newExprs.push(map.output(index));

  map = AffineMap::new(map.num_dims(), map.num_symbols(), newExprs,
                       map.context());
  return Ok(());
}

/**
Canonicalise the affine map result expression order of an affine min/max
operation.

The pattern calls `canonicalise_map_expr_and_term_order` to order the result
expressions and replaces the operation if the order changed.

For example, the following operation:

  %0 = affine.min affine_map<(d0, d1) -> (d0 + d1, d1 + 16, 32)> (%i0, %i1)

Turns into:

  %0 = affine.min affine_map<(d0, d1) -> (32, d1 + 16, d0 + d1)> (%i0, %i1)
*/
template <typename T>
struct CanonicalizeAffineMinMaxOpExprAndTermOrder : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult match_and_rewrite(T affineOp,
                                PatternRewriter &rewriter) const override {
    map: AffineMap = affineOp.get_affine_map();
    if (failed(canonicalise_map_expr_and_term_order(map)))
      return Err(());
    rewriter.replace_op_with_new_op<T>(affineOp, map, affineOp.map_operands());
    return Ok(());
  }
};

template <typename T>
struct CanonicaliseSingleResultAffineMinMaxOp : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult match_and_rewrite(T affineOp,
                                PatternRewriter &rewriter) const override {
    if (affineOp.get_map().num_outputs() != 1)
      return Err(());
    rewriter.replace_op_with_new_op<Apply>(affineOp, affineOp.get_map(),
                                               affineOp.get_operands());
    return Ok(());
  }
};

//===----------------------------------------------------------------------===//
// Min
//===----------------------------------------------------------------------===//
//
//   %0 = affine.min (d0) -> (1000, d0 + 512) (%i0)
//

FoldResult Min::fold(&self) {
    fold_min_max_op(self, adaptor.get_operands());
}

pub fn Min::canonicalisation_patterns(
    RewritePatternSet &patterns, context: *mut MLIRContext*context
) {
    patterns.add<CanonicaliseSingleResultAffineMinMaxOp<Min>,
               DeduplicateAffineMinMaxExpressions<Min>,
               MergeAffineMinMaxOp<Min>, SimplifyAffine<Min>,
               CanonicalizeAffineMinMaxOpExprAndTermOrder<Min>>(
      context);
}



ParseResult Min::parse(parser: &OpAsmParser, result: &OperationState) {
    parse_affine_min_max_op<Min>(parser, result);
}

pub fn Min::print(&self, p: &OpAsmPrinter) {
    print_affine_min_max_op(p, self);
}

//===----------------------------------------------------------------------===//
// Max
//===----------------------------------------------------------------------===//
//
//   %0 = affine.max (d0) -> (1000, d0 + 512) (%i0)
//

FoldResult Max::fold(&self) {
  return fold_min_max_op(self, adaptor.get_operands());
}

pub fn Max::canonicalisation_patterns(
    RewritePatternSet &patterns, context: *mut MLIRContext*context
) {
    patterns.add<CanonicaliseSingleResultAffineMinMaxOp<Max>,
               DeduplicateAffineMinMaxExpressions<Max>,
               MergeAffineMinMaxOp<Max>, SimplifyAffine<Max>,
               CanonicalizeAffineMinMaxOpExprAndTermOrder<Max>>(
      context);
}

ParseResult Max::parse(parser: &OpAsmParser, result: &OperationState) {
    parse_affine_min_max_op<Max>(parser, result);
}

pub fn Max::print(&self, p: &OpAsmPrinter) { print_affine_min_max_op(p, self); }

//===----------------------------------------------------------------------===//
// Prefetch
//===----------------------------------------------------------------------===//

//
// affine.prefetch %0[%i, %j + 5], read, locality<3>, data : memref<400x400xi32>
//
ParseResult Prefetch::parse(parser: &OpAsmParser,
                                    result: &OperationState) {
  auto &builder = parser.builder();
  auto index_type = builder.get_index_type();

  MemRef type;
  UnresolvedOperand memref_info;
  IntegerAttr hintInfo;
  auto i32Type = parser.builder().getIntegerType(32);
  StringRef readOrWrite, cacheType;

  map_attr: AffineMapAttr;
  SmallVector<[UnresolvedOperand; 1]> map_operands;
  if (parser.parse_operand(memref_info) ||
      parser.parse_affine_map_of_ssa_ids(map_operands, map_attr,
                                 Prefetch::"map",
                                    result.attributes) ||
      parser.parse_comma() || parser.parseKeyword(&readOrWrite) ||
      parser.parse_comma() || parser.parseKeyword("locality") ||
      parser.parseLess() ||
      parser.parse_attribute(hintInfo, i32Type,
        "locality_hint",
                            result.attributes) ||
      parser.parseGreater() || parser.parse_comma() ||
      parser.parseKeyword(&cacheType) ||
      parser.parse_optional_attr_dict(result.attributes) ||
      parser.parse_colon_type(type) ||
      parser.resolve_operand(memref_info, type, result.operands) ||
      parser.resolve_operands(map_operands, index_type, result.operands))
    return Err(());

  if (!readOrWrite.equals("read") && !readOrWrite.equals("write"))
    return parser.emit_error(parser.name_loc(),
                            "rw specifier has to be 'read' or 'write'");
  result.add_attribute(
     Prefetch::getIsWriteAttrStrName(),
      parser.builder().getBoolAttr(readOrWrite.equals("write")));

  if (!cacheType.equals("data") && !cacheType.equals("instr"))
    return parser.emit_error(parser.name_loc(),
                            "cache type has to be 'data' or 'instr'");

  result.add_attribute(
     Prefetch::getIsDataCacheAttrStrName(),
      parser.builder().getBoolAttr(cacheType.equals("data")));

  return Ok(());
}

void Prefetch::print(&self, p: &OpAsmPrinter) {
    p << " " << self.memref << '[';
    map_attr: AffineMapAttr =
        self.get_attr_of_type<AffineMapAttr>("map");
    if (map_attr)
        p.print_affine_map_of_ssa_ids(map_attr, map_operands());
    p << ']' << ", " << (getIsWrite() ? "write" : "read") << ", "
        << "locality<" << getLocalityHint() << ">, "
        << (getIsDataCache() ? "data" : "instr");
    p.print_optional_attr_dict(
        (self).attrs(),
        /*elidedAttrs=*/{"map", "locality_hint",
                        getIsDataCacheAttrStrName(), getIsWriteAttrStrName()});
    p << " : " << self.memref;
}

LogicalResult Prefetch::verify(&self) {
    let map_attr = self.get_attr_of_type<AffineMapAttr>("map");
    if map_attr {
        let map = map_attr.get_value();
        if (map.num_outputs() != self.memref.get_rank())
        return emit_op_error(
            "affine.prefetch affine map num results must equal memref rank");
        if (map.num_inputs() + 1 != self.num_operands())
        return emit_op_error("too few operands");
    } else {
        if self.num_operands() != 1 {
            return emit_op_error("too few operands");
        }
    }

    let scope = get_affine_scope(self);
    for index in map_operands() {
        if !is_valid_affine_index_operand(index, scope) {
            return emit_op_error("index must be a dimension or symbol identifier");
        }
    }
    return Ok(());
}

void Prefetch::canonicalisation_patterns(results: &RewritePatternSet,
                                                   context: *mut MLIRContext*context) {
  // prefetch(memrefcast) -> prefetch
  results.add<SimplifyAffine Prefetch>>(context);
}

LogicalResult Prefetch::fold(&self, SmallVector<[FoldResult]> &results) {
    /// prefetch(memrefcast) -> prefetch
    fold_mem_ref_cast(self)
}

//===----------------------------------------------------------------------===//
// AffineParallel
//===----------------------------------------------------------------------===//

void AffineParallel::build(builder: &Builder, result: &OperationState,
                             TypeRange resultTypes,
                             ArrayRef<arith::AtomicRMWKind> reductions,
                             &[i64] ranges) {
  SmallVector<AffineMap> lbs(ranges.len(), builder.get_constant_affine_map(0));
  auto ubs = llvm::to_vector<4>(llvm::map_range(ranges, [&](i64 value) {
    return builder.get_constant_affine_map(value);
  }));
  SmallVector<i64> steps(ranges.len(), 1);
  build(builder, result, resultTypes, reductions, lbs, /*lbArgs=*/{}, ubs,
        /*ubArgs=*/{}, steps);
}

void AffineParallel::build(builder: &Builder, result: &OperationState,
                             TypeRange resultTypes,
                             ArrayRef<arith::AtomicRMWKind> reductions,
                             ArrayRef<AffineMap> lbMaps, ValueRange lbArgs,
                             ArrayRef<AffineMap> ubMaps, ValueRange ubArgs,
                             &[i64] steps) {
  assert!(llvm::all_of(lbMaps,
                      [lbMaps](AffineMap m) {
                        return m.num_dims() == lbMaps[0].num_dims() &&
                               m.num_symbols() == lbMaps[0].num_symbols();
                      }) &&
         "Expected all lower bounds maps to have the same number of dimensions "
         "and symbols");
  assert!(llvm::all_of(ubMaps,
                      [ubMaps](AffineMap m) {
                        return m.num_dims() == ubMaps[0].num_dims() &&
                               m.num_symbols() == ubMaps[0].num_symbols();
                      }) &&
         "Expected all upper bounds maps to have the same number of dimensions "
         "and symbols");
  assert!((lbMaps.is_empty() || lbMaps[0].num_inputs() == lbArgs.len()) &&
         "Expected lower bound maps to have as many inputs as lower bound "
         "operands");
  assert!((ubMaps.is_empty() || ubMaps[0].num_inputs() == ubArgs.len()) &&
         "Expected upper bound maps to have as many inputs as upper bound "
         "operands");

  result.addTypes(resultTypes);

  // Convert the reductions to integer attributes.
  SmallVector<[Attribute; 4]> reductionAttrs;
  for (arith::AtomicRMWKind reduction in reductions)
    reductionAttrs.push(
        builder.get_i64_integer_attr(static_cast<i64>(reduction)));
  result.add_attribute(get_reductions_attr_str_name(),
                      builder.getArrayAttr(reductionAttrs));

  // Concatenates maps defined in the same input space (same dimensions and
  // symbols), assumes there is at least one map.
  auto concatMapsSameInput = [&builder](ArrayRef<AffineMap> maps,
                                        SmallVector<int32_t> &groups) {
    if (maps.is_empty())
      return AffineMap::new(builder.context());
    SmallVector<AffineExpr> exprs;
    groups.reserve(groups.len() + maps.len());
    exprs.reserve(maps.len());
    for (AffineMap m in maps) {
      llvm::append_range(exprs, m.outputs());
      groups.push(m.num_outputs());
    }
    return AffineMap::new(maps[0].num_dims(), maps[0].num_symbols(), exprs,
                          maps[0].context());
  };

  // Set up the bounds.
  SmallVector<int32_t> lbGroups, ubGroups;
  AffineMap lb_map = concatMapsSameInput(lbMaps, lbGroups);
  AffineMap ub_map = concatMapsSameInput(ubMaps, ubGroups);
  result.add_attribute(get_lower_bounds_map_attr_str_name(),
                      AffineMapAttr::get(lb_map));
  result.add_attribute(get_lower_bounds_groups_attr_str_name(),
                      builder.get_i32_tensor_attr(lbGroups));
  result.add_attribute(get_upper_bounds_map_attr_str_name(),
                      AffineMapAttr::get(ub_map));
  result.add_attribute(get_upper_bounds_groups_attr_str_name(),
                      builder.get_i32_tensor_attr(ubGroups));
  result.add_attribute(get_steps_attr_str_name(), builder.getI64ArrayAttr(steps));
  result.add_operands(lbArgs);
  result.add_operands(ubArgs);

  // Create a region and a block for the body.
  auto *bodyRegion = result.add_region();
  auto *body = new Block();
  // Add all the block arguments.
  for (usize i = 0, e = steps.len(); i < e; ++i)
    body.add_argument(IndexType::get(builder.context()), result.location);
  bodyRegion->push(body);
  if (resultTypes.is_empty())
    ensure_terminator(*bodyRegion, builder, result.location);
}

Region &AffineParallel::loop_body() { return self.region; }

usize AffineParallel::num_dims() { return get_steps().len(); }

AffineParallel::operand_range AffineParallel::get_lower_bounds_operands() {
  return get_operands().take_front(get_lower_bounds_map().num_inputs());
}

AffineParallel::operand_range AffineParallel::get_upper_bounds_operands() {
  return get_operands().drop_front(get_lower_bounds_map().num_inputs());
}

AffineMap AffineParallel::self.lower_bound_map(usize pos) {
  auto values = get_lower_bounds_groups().get_values<int32_t>();
  usize start = 0;
  for (usize i = 0; i < pos; ++i)
    start += values[i];
  return get_lower_bounds_map().getSliceMap(start, values[pos]);
}

AffineMap AffineParallel::get_upper_bound_map(usize pos) {
  auto values = get_upper_bounds_groups().get_values<int32_t>();
  usize start = 0;
  for (usize i = 0; i < pos; ++i)
    start += values[i];
  return get_upper_bounds_map().getSliceMap(start, values[pos]);
}

AffineValueMap AffineParallel::get_lower_bounds_value_map() {
  return AffineValueMap(get_lower_bounds_map(), get_lower_bounds_operands());
}

AffineValueMap AffineParallel::get_upper_bounds_value_map() {
  return AffineValueMap(get_upper_bounds_map(), get_upper_bounds_operands());
}

Option<SmallVector<[i64; 8]>> AffineParallel::getConstantRanges() {
  if (hasMinMaxBounds())
    return None;

  // Try to convert all the ranges to constant expressions.
  SmallVector<[i64; 8]> out;
  AffineValueMap rangesValueMap;
  AffineValueMap::difference(get_upper_bounds_value_map(), get_lower_bounds_value_map(),
                             &rangesValueMap);
  out.reserve(rangesValueMap.num_outputs());
  for (usize i = 0, e = rangesValueMap.num_outputs(); i < e; ++i) {
    auto expr = rangesValueMap.output(i);
    auto cst = expr.dyn_cast<AffineConstantExpr>();
    if (!cst)
      return None;
    out.push(cst.get_value());
  }
  return out;
}

Block *AffineParallel::get_body() { return &self.region.front(); }

Builder AffineParallel::getBodyBuilder() {
  return Builder(get_body(), std::prev(get_body().end()));
}

map: void AffineParallel::set_lower_bounds(ValueRange lb_operands, AffineMap) {
    assert!(lb_operands.len() == map.num_inputs(),
            "operands to map must match number of inputs");

    auto ub_operands = get_upper_bounds_operands();

    SmallVector<[Value; 4]> new_operands(lb_operands);
    new_operands.append(ub_operands);
    self.set_operands(new_operands);

    set_lower_bounds_map_attr(AffineMapAttr::get(map));
}

map: void AffineParallel::setUpperBounds(ValueRange ub_operands, AffineMap) {
    assert!(ub_operands.len() == map.num_inputs() &&
            "operands to map must match number of inputs");

    SmallVector<[Value; 4]> new_operands(get_lower_bounds_operands());
    new_operands.append(ub_operands);
    self.set_operands(new_operands);

    set_upper_bounds_map_attr(AffineMapAttr::get(map));
}

void AffineParallel::setSteps(&[i64] newSteps) {
    set_steps_attr(getBodyBuilder().getI64ArrayAttr(newSteps));
}

LogicalResult AffineParallel::verify(&self) {
    auto num_dims = num_dims();
    if (get_lower_bounds_groups().get_num_elements() != num_dims ||
        get_upper_bounds_groups().get_num_elements() != num_dims ||
        get_steps().len() != num_dims || get_body().num_inputs() != num_dims) {
        return emit_op_error() << "the number of region arguments ("
                            << get_body().num_inputs()
                            << ") and the number of map groups for lower ("
                            << get_lower_bounds_groups().get_num_elements()
                            << ") and upper bound ("
                            << get_upper_bounds_groups().get_num_elements()
                            << "), and the number of steps (" << get_steps().len()
                            << ") must all match";
    }

    usize expected_num_lb_results = 0;
    for (APInt v in get_lower_bounds_groups())
        expected_num_lb_results += v.get_z_ext_value();
    if expected_num_lb_results != get_lower_bounds_map().num_outputs() {
        return emit_op_error() << "Expected lower bounds map to have "
                            << expected_num_lb_results << " results";}
    let mut expected_num_ub_results = 0;
    for (APInt v in get_upper_bounds_groups()) {
        expected_num_ub_results += v.get_z_ext_value();
    }
    if expected_num_ub_results != get_upper_bounds_map().num_outputs() {
        return emit_op_error() << "Expected upper bounds map to have "
                            << expected_num_ub_results << " results";
    }

    if get_reductions().len() != num_outputs() {
        return emit_op_error("a reduction must be specified for each output");
    }

    // Verify reduction  ops are all valid
    for attr in get_reductions() {
        let int_attr = attr.dyn_cast<IntegerAttr>();
        if (!int_attr || !arith::symbolise_atomic_rmw_kind(int_attr.getInt()))
        return emit_op_error("Invalid reduction attribute");
    }

    // Verify that the bound operands are valid dimension/symbols.
    /// Lower bounds.
    if (failed(verify_dim_and_symbol_identifiers(
        self, get_lower_bounds_operands(),
        get_lower_bounds_map().num_dims())))
        return Err(());
    /// Upper bounds.
    if (failed(verify_dim_and_symbol_identifiers(
        self, get_upper_bounds_operands(),
        get_upper_bounds_map().num_dims()))) {
        return Err(());
    }
    Ok(())
}

LogicalResult AffineValueMap::canonicalise() {
    SmallVector<[Value; 4]> new_operands{operands};
    auto new_map = get_affine_map();
    compose_affine_map_and_operands(&new_map, &new_operands);
    if (new_map == get_affine_map() && new_operands == operands)
        return Err(());
    reset(new_map, new_operands);
    Ok(())
}

/// Canonicalise the bounds of the given loop.
pub fn canonicalise_loop_bounds(AffineParallel op) -> LogicalResult {
    AffineValueMap lb = op.get_lower_bounds_value_map();
    bool lb_canonicalised = succeeded(lb.canonicalise());

    AffineValueMap ub = op.get_upper_bounds_value_map();
    bool ub_canonicalised = succeeded(ub.canonicalise());

    // Any canonicalization change always leads to updated map(s).
    if !lb_canonicalised && !ub_canonicalised {
        return Err(());}

    if (lb_canonicalised){
        op.set_lower_bounds(lb.get_operands(), lb.get_affine_map());}
    if (ub_canonicalised){
        op.setUpperBounds(ub.get_operands(), ub.get_affine_map());}

    return Ok(());
}

LogicalResult AffineParallel::fold(
    &self, SmallVector<[FoldResult]> &results) {
    canonicalise_loop_bounds(self)
}

/// Prints a lower(upper) bound of an affine parallel loop with max(min)
/// conditions in it. `map_attr` is a flat list of affine expressions and `group`
/// identifies which of the those expressions form max/min groups. `operands`
/// are the SSA values of dimensions and symbols and `keyword` is either "min"
/// or "max".
map_attr: static void print_min_max_bound(OpAsmPrinter &p, AffineMapAttr,
                             operands: DenseIntElementsAttr group, ValueRange,
                             StringRef keyword) {
  map: AffineMap = map_attr.get_value();
  num_dims: usize = map.num_dims();
  ValueRange dim_operands = operands.take_front(num_dims);
  ValueRange sym_operands = operands.drop_front(num_dims);
  usize start = 0;
  for (llvm::APInt groupSize in group) {
    if (start != 0)
      p << ", ";

    usize len = groupSize.get_z_ext_value();
    if len == 1 {
      p.printAffineExprOfSSAIds(map.output(start), dim_operands, sym_operands);
      ++start;
    } else {
      p << keyword << '(';
      AffineMap submap = map.getSliceMap(start, len);
      p.print_affine_map_of_ssa_ids(AffineMapAttr::get(submap), operands);
      p << ')';
      start += len;
    }
  }
}

void AffineParallel::print(&self, p: &OpAsmPrinter) {
  p << " (" << get_body().inputs() << ") = (";
  print_min_max_bound(p, get_lower_bounds_map_attr(), get_lower_bounds_groups_attr(),
                   get_lower_bounds_operands(), "max");
  p << ") to (";
  print_min_max_bound(p, get_upper_bounds_map_attr(), get_upper_bounds_groups_attr(),
                   get_upper_bounds_operands(), "min");
  p << ')';
  SmallVector<[i64; 8]> steps = get_steps();
  bool elide_steps = llvm::all_of(steps, [](i64 step) { return step == 1; });
  if !elide_steps {
    p << " step (";
    llvm::interleave_comma(steps, p);
    p << ')';
  }
  if num_outputs() {
    p << " reduce (";
    llvm::interleave_comma(get_reductions(), p, [&](auto &attr) {
      arith::AtomicRMWKind sym = *arith::symbolise_atomic_rmw_kind(
          attr.template cast<IntegerAttr>().getInt());
      p << "\"" << arith::stringify_atomic_rmw_kind(sym) << "\"";
    });
    p << ") -> (" << get_result_types() << ")";
  }

  p << ' ';
  p.print_region(self.region, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/num_outputs());
  p.print_optional_attr_dict(
      (self).attrs(),
      /*elidedAttrs=*/{AffineParallel::get_reductions_attr_str_name(),
                       AffineParallel::get_lower_bounds_map_attr_str_name(),
                       AffineParallel::get_lower_bounds_groups_attr_str_name(),
                       AffineParallel::get_upper_bounds_map_attr_str_name(),
                       AffineParallel::get_upper_bounds_groups_attr_str_name(),
                       AffineParallel::get_steps_attr_str_name()});
}

/**
Given a list of lists of parsed operands, populates `unique_operands` with
unique operands. Also populates `replacements with affine expressions of
`kind` that can be used to update affine maps previously accepting a
`operands` to accept `unique_operands` instead.
*/
pub fn deduplicate_and_resolve_operands(
    parser: &OpAsmParser,
    ArrayRef<SmallVector<UnresolvedOperand>> operands,
    SmallVector<[Value]> &unique_operands,
    SmallVector<AffineExpr> &replacements, AffineExprKind kind
) -> ParseResult {
    assert!((kind == AffineExprKind::DimId || kind == AffineExprKind::SymbolId) &&
            "Expected operands to be dim or symbol expression");

    Type index_type = parser.builder().get_index_type();
    for (const auto &list in operands) {
        SmallVector<[Value]> value_operands;
        if (parser.resolve_operands(list, index_type, value_operands))
        return Err(());
        for (Value operand in value_operands) {
            let pos = std::distance(unique_operands.begin(),
                                        llvm::find(unique_operands, operand));
            if pos == unique_operands.len() {
                unique_operands.push(operand);
            }
            replacements.push(
                kind == AffineExprKind::DimId
                    ? get_affine_dim_expr(pos, parser.context())
                    : get_affine_symbol_expr(pos, parser.context()));
        }
    }
    return Ok(());
}

namespace {
enum class MinMaxKind { Min, Max };
} // namespace

/**
Parses an affine map that can contain a min/max for groups of its results,
e.g., max(expr-1, expr-2), expr-3, max(expr-4, expr-5, expr-6). Populates
`result` attributes with the map (flat list of expressions) and the grouping
(list of integers that specify how many expressions to put into each
min/max) attributes. Deduplicates repeated operands.

parallel-bound       ::= `(` parallel-group-list `)`
parallel-group-list  ::= parallel-group (`,` parallel-group-list)?
parallel-group       ::= simple-group | min-max-group
simple-group         ::= expr-of-ssa-ids
min-max-group        ::= ( `min` | `max` ) `(` expr-of-ssa-ids-list `)`
expr-of-ssa-ids-list ::= expr-of-ssa-ids (`,` expr-of-ssa-id-list)?

# Examples

```mlir
(%0, min(%1 + %2, %3), %4, min(%5 floordiv 32, %6))
```

```mlir
(%0, max(%1 - 2 * %2))
```
*/
pub fn parse_affine_map_with_min_max(
    parser: &OpAsmParser,
    result: &OperationState,
    MinMaxKind kind
) -> ParseResult {
    // Using `const` not `constexpr` below to workaround a MSVC optimizer bug,
    // see: https://reviews.llvm.org/D134227#3821753
    const llvm::StringLiteral tmp_attr_str_name = "__pseudo_bound_map";

    StringRef map_name = kind == MinMaxKind::Min
                            ? AffineParallel::get_upper_bounds_map_attr_str_name()
                            : AffineParallel::get_lower_bounds_map_attr_str_name();
    StringRef groups_name =
        kind == MinMaxKind::Min
            ? AffineParallel::get_upper_bounds_groups_attr_str_name()
            : AffineParallel::get_lower_bounds_groups_attr_str_name();

    if (failed(parser.parse_l_paren()))
        return Err(());

    if succeeded(parser.parse_optional_r_paren()) {
        result.add_attribute(
            map_name, AffineMapAttr::get(parser.builder().get_empty_affine_map()));
        result.add_attribute(groups_name, parser.builder().get_i32_tensor_attr({}));
        return Ok(());
    }

    SmallVector<AffineExpr> flat_exprs;
    SmallVector<SmallVector<UnresolvedOperand>> flat_dim_operands;
    SmallVector<SmallVector<UnresolvedOperand>> flat_sym_operands;
    SmallVector<int32_t> num_maps_per_group;
    SmallVector<UnresolvedOperand> map_operands;
    let parse_operands = || {
            if (succeeded(parser.parse_optional_keyword(
                    kind == MinMaxKind::Min ? "min" : "max"))) {
            map_operands.clear();
            AffineMapAttr map;
            if (failed(parser.parse_affine_map_of_ssa_ids(map_operands, map, tmp_attr_str_name,
                                                    result.attributes,
                                                    Delimiter::Paren))) {
                return Err(());
            }
            result.attributes.erase(tmp_attr_str_name);
            llvm::append_range(flat_exprs, map.get_value().outputs());
            auto operands_ref = llvm::ArrayRef(map_operands);
            auto dimsRef = operands_ref.take_front(map.get_value().num_dims());
            SmallVector<UnresolvedOperand> dims(dimsRef.begin(),
                                                            dimsRef.end());
            auto symsRef = operands_ref.drop_front(map.get_value().num_dims());
            SmallVector<UnresolvedOperand> syms(symsRef.begin(),
                                                            symsRef.end());
            flat_dim_operands.append(map.get_value().num_outputs(), dims);
            flat_sym_operands.append(map.get_value().num_outputs(), syms);
            num_maps_per_group.push(map.get_value().num_outputs());
            } else {
            if (failed(parser.parseAffineExprOfSSAIds(flat_dim_operands.emplace_back(),
                                                        flat_sym_operands.emplace_back(),
                                                        flat_exprs.emplace_back())))
                return Err(());
            num_maps_per_group.push(1);
            }
            return Ok(());
    };
    if parser.parse_comma_separated_list(parse_operands)
    || parser.parse_r_paren()
    {
        return Err(());
    }

    let mut total_num_dims = 0;
    let mut total_num_syms = 0;
    for (usize i = 0, e = flat_exprs.len(); i < e; ++i) {
        let num_dims = flat_dim_operands[i].len();
        let num_syms = flat_sym_operands[i].len();
        flat_exprs[i] = flat_exprs[i]
                        .shift_dims(num_dims, total_num_dims)
                        .shift_symbols(num_syms, total_num_syms);
        total_num_dims += num_dims;
        total_num_syms += num_syms;
    }

    // Deduplicate map operands.
    SmallVector<[Value]> dim_operands, sym_operands;
    SmallVector<AffineExpr> dim_rplacements, sym_repacements;
    if deduplicate_and_resolve_operands(
        parser, flat_dim_operands, dim_operands,
        dim_rplacements, AffineExprKind::DimId)
    || deduplicate_and_resolve_operands(
        parser, flat_sym_operands, sym_operands,
        sym_repacements, AffineExprKind::SymbolId)
    {
        return Err(());
    }

    result.operands.append(dim_operands);
    result.operands.append(sym_operands);

    builder: &Builder = parser.builder();
    auto flat_map = AffineMap::new(total_num_dims, total_num_syms, flat_exprs,
                                    parser.context());
    flat_map = flat_map.replace_dims_and_symbols(
        dim_rplacements, sym_repacements, dim_operands.len(), sym_operands.len());

    result.add_attribute(map_name, AffineMapAttr::get(flat_map));
    result.add_attribute(groups_name, builder.get_i32_tensor_attr(num_maps_per_group));
    return Ok(());
}

//
// operation ::= `affine.parallel` `(` ssa-ids `)` `=` parallel-bound
//               `to` parallel-bound steps? region attr-dict?
// steps     ::= `steps` `(` integer-literals `)`
//
ParseResult AffineParallel::parse(parser: &OpAsmParser,
                                    result: &OperationState) {
  auto &builder = parser.builder();
  auto index_type = builder.get_index_type();
  SmallVector<[Argument; 4]> ivs;
  if (parser.parseArgumentList(ivs, Delimiter::Paren) ||
      parser.parseEqual() ||
      parse_affine_map_with_min_max(parser, result, MinMaxKind::Max) ||
      parser.parseKeyword("to") ||
      parse_affine_map_with_min_max(parser, result, MinMaxKind::Min))
    return Err(());

  AffineMapAttr stepsMapAttr;
  NamedAttrList stepsAttrs;
  SmallVector<[UnresolvedOperand; 4]> stepsMapOperands;
  if failed(parser.parse_optional_keyword("step")) {
    SmallVector<[i64; 4]> steps(ivs.len(), 1);
    result.add_attribute(AffineParallel::get_steps_attr_str_name(),
                        builder.getI64ArrayAttr(steps));
  } else {
    if (parser.parse_affine_map_of_ssa_ids(stepsMapOperands, stepsMapAttr,
                                      AffineParallel::get_steps_attr_str_name(),
                                      stepsAttrs,
                                      Delimiter::Paren))
      return Err(());

    // Convert steps from an AffineMap into an I64ArrayAttr.
    SmallVector<[i64; 4]> steps;
    auto stepsMap = stepsMapAttr.get_value();
    for (const auto &result in stepsMap.outputs()) {
      auto const_expr = result.dyn_cast<AffineConstantExpr>();
      if (!const_expr)
        return parser.emit_error(parser.name_loc(),
                                "steps must be constant integers");
      steps.push(const_expr.get_value());
    }
    result.add_attribute(AffineParallel::get_steps_attr_str_name(),
                        builder.getI64ArrayAttr(steps));
  }

  // Parse optional clause of the form: `reduce ("addf", "maxf")`, where the
  // quoted strings are a member of the enum AtomicRMWKind.
  SmallVector<[Attribute; 4]> reductions;
  if succeeded(parser.parse_optional_keyword("reduce")) {
    if (parser.parse_l_paren())
      return Err(());
    auto parseAttributes = [&]() -> ParseResult {
      // Parse a single quoted string via the attribute parsing, and then
      // verify it is a member of the enum and convert to it's integer
      // representation.
      StringAttr attrVal;
      NamedAttrList attrStorage;
      auto location = parser.getCurrentLocation();
      if (parser.parse_attribute(attrVal, builder.getNoneType(), "reduce",
                                attrStorage))
        return Err(());
      Option<arith::AtomicRMWKind> reduction =
          arith::symbolise_atomic_rmw_kind(attrVal.get_value());
      if (!reduction)
        return parser.emit_error(location, "Invalid reduction value: ") << attrVal;
      reductions.push(
          builder.get_i64_integer_attr(static_cast<i64>(reduction.value())));
      // While we keep getting commas, keep parsing.
      return Ok(());
    };
    if (parser.parse_comma_separated_list(parseAttributes) || parser.parse_r_paren())
      return Err(());
  }
  result.add_attribute(AffineParallel::get_reductions_attr_str_name(),
                      builder.getArrayAttr(reductions));

  // Parse return types of reductions (if any)
  if (parser.parseOptionalArrowTypeList(result.types))
    return Err(());

  // Now parse the body.
  *mut Regionbody = result.add_region();
  for (auto &iv in ivs)
    iv.type = index_type;
  if (parser.parse_region(*body, ivs) ||
      parser.parse_optional_attr_dict(result.attributes))
    return Err(());

  // Add a terminator if none was parsed.
  AffineParallel::ensure_terminator(*body, builder, result.location);
  return Ok(());
}

//===----------------------------------------------------------------------===//
// AffineYield
//===----------------------------------------------------------------------===//

LogicalResult AffineYield::verify(&self) {
  auto *parent = (self).parent();
  auto results = parent->outputs();
  auto operands = get_operands();

  if (!isa<AffineParallel, If, For>(parent))
    return emit_op_error() << "only terminates affine.if/for/parallel regions";
  if (parent->num_outputs() != self.num_operands())
    return emit_op_error() << "parent of yield must have same number of "
                            "results as the yield operands";
  for (auto it in llvm::zip(results, operands)) {
    if (std::get<0>(it).get_type() != std::get<1>(it).get_type())
      return emit_op_error() << "types mismatch between yield op and its parent";
  }

  return Ok(());
}

//===----------------------------------------------------------------------===//
// VectorLoad
//===----------------------------------------------------------------------===//

void VectorLoad::build(builder: &Builder, result: &OperationState,
                               map: VectorType resultType, AffineMap,
                               operands: ValueRange) {
  assert!(operands.len() == 1 + map.num_inputs(), "inconsistent operands");
  result.add_operands(operands);
  if (map)
    result.add_attribute("map", AffineMapAttr::get(map));
  result.types.push(resultType);
}

void VectorLoad::build(builder: &Builder, result: &OperationState,
                               VectorType resultType, Value memref,
                               map: AffineMap, ValueRange map_operands) {
  assert!(map.num_inputs() == map_operands.len(), "inconsistent index info");
  result.add_operands(memref);
  result.add_operands(map_operands);
  result.add_attribute("map", AffineMapAttr::get(map));
  result.types.push(resultType);
}

void VectorLoad::build(builder: &Builder, result: &OperationState,
                               VectorType resultType, Value memref,
                               ValueRange indices) {
  auto memref_type = memref.get_type().cast<MemRef>();
  i64 rank = memref_type.get_rank();
  // Create identity map for memrefs with at least one dimension or () -> ()
  // for zero-dimensional memrefs.
  auto map =
      rank ? builder.getMultiDimIdentityMap(rank) : builder.get_empty_affine_map();
  build(builder, result, resultType, memref, map, indices);
}

void VectorLoad::canonicalisation_patterns(results: &RewritePatternSet,
                                                     context: *mut MLIRContext*context) {
  results.add<SimplifyAffine<VectorLoad>>(context);
}

ParseResult VectorLoad::parse(parser: &OpAsmParser,
                                      result: &OperationState) {
  auto &builder = parser.builder();
  auto index_type = builder.get_index_type();

  MemRef memref_type;
  VectorType resultType;
  UnresolvedOperand memref_info;
  map_attr: AffineMapAttr;
  SmallVector<[UnresolvedOperand; 1]> map_operands;
  return failure(
      parser.parse_operand(memref_info) ||
      parser.parse_affine_map_of_ssa_ids(map_operands, map_attr,
                                    VectorLoad::"map",
                                    result.attributes) ||
      parser.parse_optional_attr_dict(result.attributes) ||
      parser.parse_colon_type(memref_type) || parser.parse_comma() ||
      parser.parseType(resultType) ||
      parser.resolve_operand(memref_info, memref_type, result.operands) ||
      parser.resolve_operands(map_operands, index_type, result.operands) ||
      parser.addTypeToList(resultType, result.types));
}

void VectorLoad::print(&self, p: &OpAsmPrinter) {
  p << " " << self.memref << '[';
  map_attr: if (AffineMapAttr =
          self.get_attr_of_type<AffineMapAttr>("map"))
    p.print_affine_map_of_ssa_ids(map_attr, map_operands());
  p << ']';
  p.print_optional_attr_dict((self).attrs(),
                          /*elidedAttrs=*/{"map"});
  p << " : " << self.memref << ", " << get_type();
}

/// Verify common invariants of affine.vector_load and affine.vector_store.
memref_type,: static LogicalResult verify_vector_memory_op(operation: *mut Operation, MemRef
                                          VectorType vectorType) {
  // Check that memref and vector element types match.
  if (memref_type.element_type() != vectorType.element_type())
    return op.emit_op_error(
        "requires memref and vector types of the same elemental type");
  return Ok(());
}

LogicalResult VectorLoad::verify(&self) {
  MemRef memref_type = self.memref;
  if (failed(verify_memory_op_indexing(
          getOperation(),
          self.get_attr_of_type<AffineMapAttr>("map"),
          map_operands(), memref_type,
          /*num_index_operands=*/self.num_operands() - 1)))
    return Err(());

  if (failed(verify_vector_memory_op(getOperation(), memref_type, self.output)))
    return Err(());

  return Ok(());
}

//===----------------------------------------------------------------------===//
// VectorStore
//===----------------------------------------------------------------------===//

void VectorStore::build(builder: &Builder, result: &OperationState,
                                map: Value valueToStore, Value memref, AffineMap,
                                ValueRange map_operands) {
  assert!(map.num_inputs() == map_operands.len(), "inconsistent index info");
  result.add_operands(valueToStore);
  result.add_operands(memref);
  result.add_operands(map_operands);
  result.add_attribute("map", AffineMapAttr::get(map));
}

// Use identity map.
void VectorStore::build(builder: &Builder, result: &OperationState,
                                Value valueToStore, Value memref,
                                ValueRange indices) {
  auto memref_type = memref.get_type().cast<MemRef>();
  i64 rank = memref_type.get_rank();
  // Create identity map for memrefs with at least one dimension or () -> ()
  // for zero-dimensional memrefs.
  auto map =
      rank ? builder.getMultiDimIdentityMap(rank) : builder.get_empty_affine_map();
  build(builder, result, valueToStore, memref, map, indices);
}
void VectorStore::canonicalisation_patterns(
    results: &RewritePatternSet, context: *mut MLIRContext*context) {
  results.add<SimplifyAffine<VectorStore>>(context);
}

ParseResult VectorStore::parse(parser: &OpAsmParser,
                                       result: &OperationState) {
  auto index_type = parser.builder().get_index_type();

  MemRef memref_type;
  VectorType resultType;
  UnresolvedOperand store_value_info;
  UnresolvedOperand memref_info;
  map_attr: AffineMapAttr;
  SmallVector<[UnresolvedOperand; 1]> map_operands;
  return failure(
      parser.parse_operand(store_value_info) || parser.parse_comma() ||
      parser.parse_operand(memref_info) ||
      parser.parse_affine_map_of_ssa_ids(map_operands, map_attr,
                                    VectorStore::"map",
                                    result.attributes) ||
      parser.parse_optional_attr_dict(result.attributes) ||
      parser.parse_colon_type(memref_type) || parser.parse_comma() ||
      parser.parseType(resultType) ||
      parser.resolve_operand(store_value_info, resultType, result.operands) ||
      parser.resolve_operand(memref_info, memref_type, result.operands) ||
      parser.resolve_operands(map_operands, index_type, result.operands));
}

void VectorStore::print(&self, p: &OpAsmPrinter) {
    p << " " << self.value;
    p << ", " << self.memref << '[';
    map_attr: if (AffineMapAttr =
            self.get_attr_of_type<AffineMapAttr>("map"))
        p.print_affine_map_of_ssa_ids(map_attr, map_operands());
    p << ']';
    p.print_optional_attr_dict((self).attrs(),
                            /*elidedAttrs=*/{"map"});
    p << " : " << self.memref << ", " << self.value.get_type();
}

LogicalResult VectorStore::verify(&self) {
  MemRef memref_type = self.memref;
  if (failed(verify_memory_op_indexing(
          self, self.get_attr_of_type<AffineMapAttr>("map"),
          map_operands(), memref_type,
          /*num_index_operands=*/self.num_operands() - 2)))
    return Err(());

  if (failed(verify_vector_memory_op(self, memref_type, self.value)))
    return Err(());

  return Ok(());
}

//===----------------------------------------------------------------------===//
// DelinearizeIndexOp
//===----------------------------------------------------------------------===//

void AffineDelinearizeIndexOp::build(builder: &Builder, result: &OperationState,
                                     Value linearIndex,
                                     &[FoldResult] basis) {
  result.addTypes(SmallVector<Type>(basis.len(), builder.get_index_type()));
  result.add_operands(linearIndex);
  SmallVector<[Value]> basisValues =
      llvm::to_vector(llvm::map_range(basis, [&](FoldResult fr) -> Value {
        Option<i64> staticDim = constant_int_value(fr);
        if (staticDim.has_value())
          return builder.create<arith::ConstantIndexOp>(result.location,
                                                        *staticDim);
        return fr.dyn_cast<Value>();
      }));
  result.add_operands(basisValues);
}

LogicalResult AffineDelinearizeIndexOp::verify(&self) {
  if (getBasis().is_empty())
    return emit_op_error("basis should not be is_empty");
  if (num_outputs() != getBasis().len())
    return emit_op_error("should return an index for each basis element");
  return Ok(());
}
