//! # MLIR Affine Map Class
//!
//! Affine maps are mathematical functions which map a list of dimension identifiers and symbols, to multidimensional affine expressions.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/AffineMap.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/AffineMap.cpp>

pub mod detail;

use core::{
    array::from_fn,
    cmp::max
};
use llvm::{
    adt::{
        dense_map_info,
        small_bit_vector::SmallBitVector,
        small_set,
        small_vector::SmallVector
    },
    support::raw_ostream
};
use crate::{
    ir::{
        affine_expr::{AffineExpr, dimension},
        attribute::Attribute,
        builtins::{
            attributes::IntegerAttribute,
            types
        },
        mlir_context::MLIRContext
    },
    support::{
        logical_result::LogicalResult,
        math_extras,
        llvm
    }, dialect::tensor::ir::tensor::operations::Dim,
};

use AffineExpr::*;

use super::builtins::attributes::IntegerAttribute;

/**
A multi-dimensional affine map.
Affine map's are immutable like `Type`'s, and they are uniqued.
For exaple:

```mlir
(d0, d1) -> (d0 / 128, d0 mod 128, d1)
```

The names used above, `d0` and `d1`, don't matter - it's the mathematical function that is unique to this affine map.
*/
pub struct AffineMap<
    const D: usize,
    const S: usize,
    const O: usize
> {
    // 
    // AffineMapStorage
    // 

    outputs: [AffineExpr; O],
    context: *mut MLIRContext,
    _num_dims: core::marker::PhantomData<D>,
    _num_symbols: core::marker::PhantomData<S>
}

/**
Returns a single constant output affine map.

```mlir
() -> (%value)
```
*/
pub const fn constant_map(value: i64, context: *mut MLIRContext)
-> AffineMap<0, 0, 1> {
    AffineMap::<0, 0, 1>::new({constant(value, context)})
}

/**
Returns an AffineMap with `D` identity output dim exprs.

```mlir
(d0, ..., dn) -> (d0, ..., dn)
```
*/
pub const fn multi_dim_identity_map<const D: usize> (
    context: *mut MLIRContext
) -> AffineMap<D, 0, D> {
    let dims: [AffineExpr; D] = from_fn(|index| dimension(index, context));
    AffineMap::<D, 0, D>::new(dims, context)
}

/**
Returns an identity affine map `(d0, ..., dn) -> (dp, ..., dn)` on the most minor dimensions.
*/
// static
pub const fn minor_identity_map<
    const D: usize,
    const O: usize
>(
    outputs: usize,
    context: *mut MLIRContext
) -> AffineMap<D, 0, _>
{
    assert!(D >= O, "Dimension mismatch");
    let id = multi_dim_identity_map::<D>(context);
    AffineMap::<D, 0, _>::new(id.outputs().take_back(outputs), context)
}

/**
Returns an AffineMap representing a permutation.
The permutation is expressed as a non-empty vector of integers.
E.g. the permutation `(i, j, k) -> (j, k, i)` will be expressed with `permutation = [1, 2, 0]`. All values in `permutation` must be integers, in the range `0..(permutation.len() - 1)` without duplications (i.e. `[1, 1, 2]` is an invalid permutation).
*/
// static
pub fn permutation_map<const N: usize>(
    permutation: &[usize; N],
    context: *mut MLIRContext
) -> AffineMap<N, 0, _> {
    assert!(
        !permutation.is_empty(),
        "Cannot create permutation map from empty permutation vector");
    let exprs = from_fn(|index| dimension(index, context));
    let permutation_map = AffineMap::<N, 0, _>::new(exprs, context);
    assert!(permutation_map.is_permutation(), "Invalid permutation vector");
    permutation_map
}

impl<const D: usize, const S: usize, const O: usize> AffineMap<D, S, O> {
    /// Returns a vector of AffineMaps; each with as many outputs as `exprs.size()`, as many dims as the largest dim in `exprs` and as many symbols as the largest symbol in `exprs`.
    // static
    pub fn infer_from_expr_list(exprs_list: &[&[AffineExpr]])
        -> SmallVector<[AffineMap; 4]>
    {

    }

    pub const fn context(&self) -> *mut MLIRContext {
        self.context
    }

    /// Returns true if this affine map is an identity affine map.
    /// An identity affine map corresponds to an identity affine function on the dimensional identifiers.
    pub const fn is_identity(&self) -> bool {
        if D != O {
            return false;
        }
        let outputs = self.outputs();
        for index in 0..D {
            let expr = outputs[index].dyn_cast<Dimension>();
            if !expr || expr.index() != index {
                return false;
            }
        }
        true
    }

    /// Returns true if this affine map is a minor identity, i.e. an identity affine map (d0, ..., dn) -> (dp, ..., dn) on the most minor dimensions.
    pub const fn is_minor_identity(&self) -> bool {
        D >= O
        && self == self.minor_identity_map(D, O, self.context())
    }

    /// Returns true if this affine map is a minor identity up to broadcasted dimensions which are indicated by value 0 in the output. If `broadcasted_dims` is not null, it will be populated with the indices of the broadcasted dimensions in the output array.
    ///
    /// Example: affine_map<(d0, d1, d2, d3, d4) -> (0, d2, 0, d4)>
    ///  (`broadcasted_dims` will contain [0, 2])
    pub const fn is_minor_identity_with_broadcasting(
        &self,
        broadcasted_dims: *mut SmallVector<[usize]> /* nullptr */
    ) -> bool {
        if !broadcasted_dims.is_null() {
            broadcasted_dims.clear();
        }
        if D < O {
            return false;
        }
        let suffix_start = D - O;
        for (index, output) in self.outputs().iter().enumerate() {
            match output {
                Constant { value } => {
                    // Each output may be either a constant 0 (broadcasted dimension).
                    if value != 0 {
                        return false;
                    }
                    if !broadcasted_dims.is_null() {
                        broadcasted_dims.push(index);
                    }
                },
                Dimension { index: dindex, .. } => {
                    /*
                    Or it may be the input dimension corresponding to this output position.
                    */
                    if dindex != suffix_start + index {
                        return false;
                    }
                },
                _ => return false
            }
        }
        true
    }

    /**
    Return true if this affine map can be converted to a minor identity with broadcast by doing a permute. Return a permutation (there may be several) to apply to get to a minor identity with broadcasts.

    Ex:

    - (d0, d1, d2) -> (0, d1) maps to minor identity (d1, 0 = d2) with perm = [1, 0] and broadcast d2
    - (d0, d1, d2) -> (d0, 0) cannot be mapped to a minor identity by permutation + broadcast
    - (d0, d1, d2, d3) -> (0, d1, d3) maps to minor identity (d1, 0 = d2, d3)  with perm = [1, 0, 2] and broadcast d2
    - (d0, d1) -> (d1, 0, 0, d0) maps to minor identity (d0, d1) with extra  leading broadcat dimensions. The map returned would be (0, 0, d0, d1) with perm = [3, 0, 1, 2]
    */
    pub const fn is_permutation_of_minor_identity_with_broadcasting(
        &self,
        permuted_dims: &mut SmallVector<[usize]>
    ) -> bool {
        let projection_start = if O < D + S {
            D + S - O
        } else {
            0
        };
        permuted_dims.clear();
        let broadcast_dims = SmallVector::<[usize]>::new();
        permuted_dims.resize(O, 0);
        /*
        If there are more outputs than input dimensions we want the new map to start with broadcast dimensions in order to be a minor identity with broadcasting.
        */
        let leading_broadcast = if O > D + S {
            O - D + S
        } else {
            0
        };
        let dim_found = SmallBitVector::new(
            max(D + S, O), false);
        for (index, expr) in self.outputs().iter().enumerate() {
            /*
            Each output may be either a constant 0 (broadcast dimension) or a dimension.
            */
            match expr {
                Constant { value } => {
                    if value != 0 {
                        return false;
                    }
                    broadcast_dims.push(index);
                },
                Dimension { index: dindex, .. } => {
                    if dindex < &projection_start {
                        return false;
                    }
                    let new_position
                        = dindex - projection_start + leading_broadcast;
                    permuted_dims[index] = new_position;
                    dim_found[new_position] = true;
                },
                _ => return false
            }
        }
        /*
        Find a permuation for the broadcast dimension. Since they are broadcasted any valid permutation is acceptable. We just permute the dim into a slot without an existing dimension.
        */
        let mut index = 0;
        for dim in broadcast_dims {
            while index < dim_found.len() && dim_found[index] {
                index += 1;
            }
            permuted_dims[dim] = index;
            index += 1;
        }
        true
    }

    /// Returns true if this affine map is an empty map, i.e., () -> ().
    pub const fn is_empty(&self) -> bool {
        D == 0 && S == 0 && O == 0
    }

    /// Returns true if this affine map is a single output constant function.
    pub const fn is_single_constant(&self) -> bool {
        O == 1 && matches!(self.output(0), Constant)
    }

    /// Returns true if this affine map has only constant outputs.
    pub const fn is_constant(&self) -> bool {
        self.outputs().iter().all(|output| matches!(output, Constant))
    }

    /// Returns the constant output of this map. This methods asserts that the map has a single constant output.
    pub const fn single_constant_output(&self) -> i64 {
        assert!(self.is_constant(), "Map must have only constant outputs");
        let output = SmallVector::<[i64]>::new();
        for expr in self.outputs() {
            output.emplace_back(expr.cast<Constant>().value);
        }
        return output;
    }

    /// Returns the constant outputs of this map. This method asserts that the map has all constant outputs.
    pub const fn constant_outputs(&self) -> SmallVector<[i64]> {
        assert!(
            self.is_single_constant(),
            "Map must have a single constant result");
        self.output(0).value
    }

    /// The affine expressions for this (multi-dimensional) map.
    pub const fn outputs<const N: usize>(&self) -> &[AffineExpr; N] {
        {getTrailingObjects<AffineExpr>(), O}
    }

    pub const fn output(&self, index: usize) -> AffineExpr {
        self.outputs()[index]
    }

    /// Extracts the position of the dimensional expression at the given output, when the caller knows it is safe to do so.
    pub const fn dim_index(&self, index: usize) -> usize {
        self.output(index).index
    }

    /// Extracts the first output position where `input` dimension resides.
    /// Returns `None` if `input` is not a dimension expression or cannot be found in outputs.
    pub const fn output_index(&self, input: AffineExpr) -> Option<usize> {
        if !matches!(input, Dimension) {
            return None;
        }

        for (index, output) in self.outputs().iter().enumerate() {
            if output == input {
                return Some(index)
            }
        }
        None
    }

    /// Return true if any affine expression involves Dimension `position`.
    pub const fn is_function_of_dim(&self, index: usize) -> bool {
        self.outputs().iter().any(|output| output.is_function_of_dim(index))
    }

    /// Return true if any affine expression involves Symbol `position`.
    pub const fn is_function_of_symbol(&self, index: usize) -> bool {
        self.outputs().iter().any(|output| output.is_function_of_symbol(index))
    }

    /// Walk all of the AffineExpr's in this mapping. Each node in an expression tree is visited in postorder.
    pub const fn walk_exprs(&self, callback: fn(AffineExpr)) {
        for output in self.outputs() {
            output.walk(callback);
        }
    }

    /// This method substitutes any uses of dimensions and symbols (e.g. dim#0 with dim_replacements[0]) in subexpressions and returns the modified expression mapping. Because this can be used to eliminate dims and symbols, the client needs to specify the number of dims and symbols in the output. The returned map always has the same number of outputs.
    pub const fn replace_dims_and_symbols<
        const D_: usize,
        const S_: usize
    >(
        &self,
        dim_replacements: [&AffineExpr; D_],
        sym_replacements: [&AffineExpr; S_],
    ) -> AffineMap<D_, S_, O>
    {
        let exprs = self.outputs().map(
            |output| output.replace_dims_and_symbols(
                dim_replacements, sym_replacements));
        AffineMap::<D_, S_, O>::new(exprs, self.context())
    }

    /// Sparse replace method. Apply AffineExpr::replace(`expr`, `replacement`) to each of the outputs and return a new AffineMap with the new outputs and with the specified number of dims and symbols.
    pub const fn replace<const D_: usize, const S_: usize>(
        &self,
        expr: AffineExpr,
        replacement: AffineExpr,
    ) ->  AffineMap::<D_, S_, O>
    {
        let exprs = self.outputs().map(|output| output.replace(expr, replacement));
        AffineMap::<D_, S_, O>::get(exprs, self.context())
    }
    
    /// Replace dims[offset..numDims] by dims[(offset + shift)..(shift + numDims)].
    pub const fn shift_dims<const SHIFT: usize>(
        &self, shift: usize, offset: usize /* 0 */
    ) -> AffineMap<{D + SHIFT}, S, _>
    {
        assert!(offset <= D);
        AffineMap::<{D + SHIFT}, S, _>::get(
            self.outputs().iter().map(
                |expr: AffineExpr|
                    expr.shift_dims(D, shift, offset)
            ).collect::<SmallVector<[AffineExpr; 4]>>(),
            self.context())
    }

    /// Replace symbols[offset..num_symbols] by symbols[(offset + shift)..(shift + num_symbols)].
    pub const fn shift_symbols<const SHIFT: usize>(&self, shift: usize, offset: usize /* 0 */) -> Self {
        AffineMap::<D, {S + SHIFT}, _>::get(
            self.outputs().iter().map(
                |expr: AffineExpr|
                    expr.shift_symbols(S, shift, offset)
            ).collect::<SmallVector<[AffineExpr; 4]>>(),
            self.context())
    }

    /// Returns a new AffineMap with the same number of dims and symbols and one less output at `index`, dropped.
    pub fn drop_output(&mut self, index: i64) -> Self {
        self.drop_outputs([index])
    }

    /// Returns a new AffineMap with the same number of dims and symbols, but all indices in `indices` dropped from outputs.
    pub fn drop_outputs(&mut self, indices: [i64]) -> Self {
        let reverse_sorted_indices: SmallVector<[i64]> = indices;
        reverse_sorted_indices.sort();

        let exprs: SmallVector<[AffineExpr; 4]> = self.outputs();
        for index in reverse_sorted_indices {
            exprs.erase(index);
        }
        AffineMap::<D, S, _>::get(exprs, self.context())
    }

    /// Returns a new AffineMap with the same number of dims and symbols and an extra output inserted at `index`.
    pub fn insert_output(&mut self, expr: AffineExpr, index: usize)
    -> AffineMap::<D, S, {O + 1}>
    {
        let exprs: SmallVector<[AffineExpr; 4]> = self.outputs();
        exprs.insert(index, expr);
        AffineMap::<D, S, {O + 1}>::get(exprs, self.context())
    }

    /// Folds the outputs of the application of an affine map on the provided operands to a constant if possible.
    pub const fn constant_fold(
        &self,
        operand_constants: &[impl Attribute],
        outputs: &SmallVector<[impl Attribute]>
    ) -> LogicalResult
    {
        // Attempt partial folding.
        let integers = SmallVector<[i64; 2]>::new();
        self.partial_constant_fold(operand_constants, &integers);

        // If all expressions folded to a constant, populate outputs with attributes containing those constants.
        if integers.is_empty() {
            return Err(());
        }

        let range = integers.iter().map(
            |i: i64| IntegerAttribute::get(IndexType::get(self.context()), i));
        outputs.append(range);
        Ok(())
    }

    /// Propagates the constant operands into this affine map. Operands are allowed to be null, at which point they are treated as non-constant. This does not change the number of symbols and dimensions. Returns a new map, which may be equal to the old map if no folding happened. If `outputs` is provided and if all expressions in the map were folded to constants, `outputs` will contain the values of these constants.
    pub const fn partial_constant_fold(
        &self,
        operand_constants: &[impl Attribute; D + S],
        outputs: *mut SmallVector<[i64]> //= nullptr
    ) -> AffineMap::<D, S, O>
    {
        // Fold each of the result expressions.
        let expr_folder = ConstantFolder::new(
            D, operand_constants);
        let exprs = self.outputs().map(|expr| {
            let folded = expr_folder.constant_fold(expr);
            /*
            If did not fold to a constant, keep the original expression, and clear the integer results vector.
            */
            if folded {
                if !outputs.is_null() {
                    outputs.push(folded.int());
                }
                constant(folded.int(), folded.context())
            } else {
                if !outputs.is_null() {
                    outputs.clear();
                    unsafe {
                        *outputs = null();
                    }
                }
                expr
            }
        });
        AffineMap::<D, S, O>::get(exprs, self.context())
    }

    /**
    Returns the AffineMap outputing from composing `this` with `map`.
    The outputing AffineMap has as many Dimension as `map` and as many
    Symbol as the concatenation of `this` and `map` (in which case the symbols of `this` map come first).

    Prerequisites:

    The maps are composable, i.e. that the number of Dimension of `this` matches the number of outputs of `map`.

    Example:

      map1: `(d0, d1)[s0, s1] -> (d0 + 1 + s1, d1 - 1 - s0)`
      map2: `(d0)[s0] -> (d0 + s0, d0 - s0)`
      map1.compose(map2):
        `(d0)[s0, s1, s2] -> (d0 + s1 + s2 + 1, d0 - s0 - s2 - 1)`
    */
    pub const fn compose(&self, other: AffineMap<D_, S_, D>)
    -> AffineMap<D_, {S + S_}, O>
    {
        // Prepare `other` by concatenating the symbols and rewriting its exprs.
        let new_dims: [AffineExpr; D_]
            = from_fn(|index| dimension(index, self.context()));
        let new_symbols: [AffineExpr; S_]
            = from_fn(|index| symbol(index + S, self.context()));
        let new_map
            = other.replace_dims_and_symbols(new_dims, new_symbols);
        let exprs = self.outputs().map(|output| output.compose(new_map));
        AffineMap::<D_, {S + S_}, O>::get(exprs, other.context())
    }

    /// Returns true if the AffineMap represents a subset (i.e. a projection) of a symbol-less permutation map. `allow_zero_in_results` allows projected permutation maps with constant zero output expressions.
    /// TODO: Remove `allow_zero_in_results` when constant zero output expressions are broadly supported.
    pub const fn is_projected_permutation(
        &self, allow_zero_in_results: bool /* false */
    ) -> bool {
        if S > 0 {
            return false;
        }
    
        // Having more results than inputs means that results have duplicated dims or zeros that can't be mapped to input dims.
        if O > D + S {
            return false;
        }
    
        let mut seen = [false; D + S];
        // A projected permutation can have, at most, only one instance of each input dimension in the result expressions. Zeros are allowed as long as the number of result expressions is lower or equal than the number of input expressions.
        for output in self.outputs() {
            match output {
                Dimension { index, .. } => {
                    if seen[index] {
                        return false;
                    }
                    seen[index] = true;
                },
                Constant { value } if value == 0 => {
                    if !allow_zero_in_results {
                        return false;
                    }
                },
                _ => return false
            }
        }
        
        // Results are either dims or zeros and zeros can be mapped to input dims.
        true
    }

    /// Returns true if the AffineMap represents a symbol-less permutation map.
    pub const fn is_permutation(&self) -> bool {
        if D != O {
            return false;
        }
        self.is_projected_permutation(false)
    }

    /// Returns the map consisting of the `indices` subset.
    pub const fn sub_map<const O_: usize>(&self, indices: &[usize; O_])
    -> AffineMap<D, S, O_>
    {
        let exprs = indices.map(|index| self.output(index));
        AffineMap::<D, S, O_>::get(exprs, self.context())
    }

    /// Returns the map consisting of `length` expressions starting from `start`.
    pub const fn slice_map(&self, start: usize, length: usize) -> Self {
        AffineMap::<D, S, _>::get(
            self.outputs()[start..(start + length)], self.context())
    }

    /// Returns the map consisting of the most major `num_outputs` outputs.
    /// Returns the null AffineMap if `num_outputs` == 0.
    /// Returns `*this` if `num_outputs` >= `this->O`.
    pub const fn major_sub_map(self, num_outputs: usize) -> Self {
        if num_outputs == 0 {
            return AffineMap::new();
        }
        if num_outputs > O {
            return self;
        }
        self.slice_map(0, num_outputs)
    }

    /// Returns the map consisting of the most minor `num_outputs` outputs.
    /// Returns the null AffineMap if `num_outputs` == 0.
    /// Returns `*this` if `num_outputs` >= `this->O`.
    pub const fn minor_sub_map(self, num_outputs: usize) -> Self {
        if num_outputs == 0 {
            return AffineMap::new();
        }
        if num_outputs > O {
            return self;
        }
        self.slice_map(O - num_outputs, num_outputs)
    }

    /// Get the largest known divisor of all map expressions.
    /// For eg: for (d0, d1) -> (8*d0 + 4, 4*d1 + 2), the output is 2.
    /// In the case of maps with no expressions or all zero constant expressions, the largest known divisor is trivially the max u64 value.
    pub fn get_largest_known_divisor_of_map_exprs() -> u64 {

    }
}

// static 
pub const fn infer_from_expr_list<I: Iterator<Item = AffineExpr>>(
    exprs_list: &[I]
) -> SmallVector<[AffineMap; 4]>
{
    assert!(!exprs_list.is_empty());
    assert!(!exprs_list[0].is_empty());
    let context = exprs_list[0][0].context();
    let max_dim = -1;
    let max_sym = -1;
    max_dim_and_symbol(exprs_list, max_dim, max_sym);
    let maps = SmallVector::<[AffineMap; 4]>::new();
    maps.reserve(exprs_list.len());
    for exprs in exprs_list {
        maps.push(
            AffineMap::<>::get(
                /*dimCount=*/max_dim + 1,
                /*symbolCount=*/max_sym + 1, exprs, context));
    }
    maps
}

/// Simplifies an affine map by simplifying its underlying AffineExpr results.
pub fn simplify_affine_map(map: AffineMap) -> AffineMap {
    let mut exprs = SmallVector::<[AffineExpr; 8]>::new();
    for e in map.outputs() {
        exprs.push(
            simplify_affine_expr(e, map.num_dims, map.num_symbols));
    }
    AffineMap::get(map.num_dims, map.num_symbols, exprs, map.context())
}

/// Drop the dims that are not used.
pub fn compress_unused_dims(map: AffineMap) -> AffineMap {
    compress_dims(map, get_unused_dims_bit_vector({map}))
}

// static 
pub fn compress_unused_impl(
    maps: &[AffineMap],
    compression_fun: fn(AffineMap) -> AffineMap
) -> SmallVector<[AffineMap]>
{
    if maps.is_empty() {
        return SmallVector::<[AffineMap]>::new();
    }
    let mut all_exprs = SmallVector::<[AffineExpr]>::new();
    all_exprs.reserve(maps.len() * maps.front().num_outputs);
    let num_dims = maps.front().num_dims;
    let num_symbols = maps.front().num_symbols;
    for m in maps {
        assert!(
            num_dims == m.num_dims && num_symbols == m.num_symbols,
            "Expected maps with same num dims and symbols");
        llvm::append_range(all_exprs, m.outputs());
    }
    let unified_map = compression_fun(
        AffineMap::get(num_dims, num_symbols, all_exprs, maps.front().context()));
    let unified_num_dims = unified_map.num_dims;
    let unified_num_symbols = unified_map.num_symbols;
    let unified_results = unified_map.outputs();
    let mut res = SmallVector::<[AffineMap]>::new();
    res.reserve(maps.len());
    for m in maps {
        res.push(AffineMap::get(unified_num_dims, unified_num_symbols,
                                    unified_results.take_front(m.num_outputs),
                                    m.context()));
        unified_results = unified_results.drop_front(m.num_outputs);
    }
    res
}

// /**
// Drop the dims that are not used by any of the individual maps in `maps`.
// Asserts that all maps in `maps` are normalized to the same number of dims and symbols.
// */
// pub fn compress_unused_dims(maps: &[AffineMap]) -> SmallVector<[AffineMap]> {

// }

/// Drop the dims that are not listed in `unused_dims`.
pub fn compress_dims(
    map: AffineMap<D, S, O>,
    unused_dims: &SmallBitVector
) -> AffineMap<_, S, _>
{
    let mut num_dims = 0;
    let mut dim_replacements = SmallVector::<[AffineExpr]>::new();
    dim_replacements.reserve(D);
    let context = map.context();
    for dim in 0..D {
        if unused_dims.test(dim) {
            dim_replacements.push(constant(0, context));
        }
        else {
            dim_replacements.push(dimension(num_dims, context));
            num_dims += 1;
        }
    }
    let mut result_exprs = SmallVector<[AffineExpr]>::new();
    result_exprs.reserve(O);
    for e in map.outputs() {
        result_exprs.push(e.replace_dims(dim_replacements));
    }
    AffineMap::<_, S, _>::get(num_dims, result_exprs, context)
}

/// Drop the symbols that are not used.
pub fn compress_unused_symbols(map: AffineMap<D, S, O>) -> AffineMap {
    let unused_symbols = SmallBitVector::new(S, true);
    map.walk_exprs(|expr: AffineExpr| {
        match expr {
            Symbol { index } => unused_symbols.reset(index),
            _ => {}
        }
    });
    compress_symbols(map, unused_symbols)
}

// /**
// Drop the symbols that are not used by any of the individual maps in `maps`.
// Asserts that all maps in `maps` are normalized to the same number of
// dims and symbols.
// */
// pub fn compress_unused_symbols(maps: &[AffineMap]) -> SmallVector<[AffineMap]> {

// }

/// Drop the symbols that are not listed in `unused_symbols`.
pub fn compress_symbols(
    map: AffineMap<D, S, O>,
    unused_symbols: &SmallBitVector
) -> AffineMap<D, _, O>
{
    let mut num_symbols = 0;
    let sym_replacements = SmallVector::<[AffineExpr]>::new();
    sym_replacements.reserve(S);
    let context = map.context();
    for sym in 0..S {
        if unused_symbols.test(sym) {
            sym_replacements.push(constant(0, context));
        } else {
            sym_replacements.push(symbol(num_symbols, context));
            num_symbols += 1;
        }
    }
    let mut result_exprs = SmallVector<[AffineExpr]>::new();
    result_exprs.reserve(map.num_outputs);
    for e in map.outputs() {
        result_exprs.push(e.replace_symbols(sym_replacements));
    }
    AffineMap::<D, _, O>::get(num_symbols, result_exprs, context)
}

/// Returns a map with the same dimension and symbol count as `map`, but whose results are the unique affine expressions of `map`.
pub fn remove_duplicate_exprs(map: AffineMap<D, S, O>) -> AffineMap<D, S, _> {
    let results = map.outputs();
    let unique_exprs: SmallVector::<[AffineExpr; 4]> = results;
    unique_exprs.erase(std::unique(unique_exprs.begin(), unique_exprs.end()),
                        unique_exprs.end());
    AffineMap::<D, S, _>::get(unique_exprs, map.context())
}

/**
Returns a map of codomain to domain dimensions such that the first codomain dimension for a particular domain dimension is selected.
Returns an empty map if the input map is empty.
Returns null map (not empty map) if `map` is not invertible (i.e. `map` does not contain a subset that is a permutation of full domain rank).

Prerequisites:
  1. `map` has no symbols.

Example 1:

```mlir
(d0, d1, d2) -> (d1, d1, d0, d2, d1, d2, d1, d0)
                  0       2   3
```

returns:

```mlir
(d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d0, d3)
```

Example 2:

```mlir
(d0, d1, d2) -> (d1, d0 + d1, d0, d2, d1, d2, d1, d0)
                  0            2   3
```

returns:

```mlir
(d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d0, d3)
```
*/
pub fn inverse_permutation(map: AffineMap<D, 0, O>) -> AffineMap {
    if map.is_empty() {
        return map;
    }
    let mut exprs = SmallVector::<[AffineExpr; 4]>::new(D);
    for (index, expr) in map.outputs().iter().enumerate() {
        // Skip non-permutations.
        match expr {
            Dimension { index: dindex, .. } => {
                if exprs[dindex] {
                    continue;
                }
                exprs[dindex] = dimension(index, d.context());
            },
            _ => {}
        }
    }
    let seen_exprs = SmallVector::<[AffineExpr; 4]>::new();
    seen_exprs.reserve(D);
    for expr in exprs {
        if expr {
            n_exprs.push(expr);
        }
    }
    if seen_exprs.len() != D { 
        return AffineMap::new();
    }
    AffineMap::<O, 0, _>::get(seen_exprs, map.context())
}

/**
Return the reverse map of a projected permutation where the projected dimensions are transformed into 0s.

Prerequisites: `map` must be a projected permuation.

Example 1:

```mlir
affine_map<(d0, d1, d2, d3) -> (d2, d0)>
```

returns:

```mlir
affine_map<(d0, d1) -> (d1, 0, d0, 0)>
```

Example 2:

```mlir
affine_map<(d0, d1, d2, d3) -> (d0, d3)>
```

returns:

```mlir
affine_map<(d0, d1) -> (d0, 0, 0, d1)>
```

Example 3:

```mlir
affine_map<(d0, d1, d2, d3) -> (d2)>
```

returns:

```mlir
affine_map<(d0) -> (0, 0, d0, 0)>
```

Example 4:

```mlir
affine_map<(d0, d1, d2) -> (d0, 0)>
```

returns:

```mlir
affine_map<(d0, d1) -> (d0, 0, 0)>
```
*/
pub fn inverse_and_broadcast_projected_permutation(
    map: AffineMap<D, S, O>
) -> AffineMap
{
    assert!(map.is_projected_permutation(/*allowZeroInResults=*/true));
    let context = map.context();
    let zero = constant(0, context);
    // Start with all the results as 0.
    let mut exprs = [zero; D + S];
    for i in llvm::seq(0usize, O) {
        // Skip zeros from input map. 'exprs' is already initialized to zero.
        if let const_expr = map.output(i).dyn_cast<Constant>() {
            assert!(
                const_expr.getValue() == 0,
                "Unexpected constant in projected permutation");
            continue;
        }
    
        // Reverse each dimension existing in the original map result.
        exprs[map.dim_index(i)] = dimension(i, context);
    }
    AffineMap::<O, 0, _>::get(exprs, context)
}

/**
Concatenates a list of `maps` into a single AffineMap, stepping over potentially empty maps. Assumes each of the underlying map has 0 symbols.
The resulting map has a number of dims equal to the max of `maps`' dims and the concatenated results as its results.
Returns an empty map if all input `maps` are empty.

Example:

When applied to the following list of 3 affine maps,

```mlir
{
  (i, j, k) -> (i, k),
  (i, j, k) -> (k, j),
  (i, j, k) -> (i, j)
}
```

Returns the map:

```mlir
    (i, j, k) -> (i, k, k, j, i, j)
```
*/
pub fn concat_affine_maps(
    maps: &[AffineMap]
) -> AffineMap
{
    let mut num_results = 0;
    let mut num_dims = 0;
    let mut num_symbols = 0;
    for m in maps {
        num_results += m.num_outputs;
    }
    let mut results = SmallVector<[AffineExpr; 8]>::new();
    results.reserve(num_results);
    for m in maps {
        for res in m.outputs() {
            results.push(res.shift_symbols(m.num_symbols, num_symbols));
        }
    
        num_symbols += m.num_symbols;
        num_dims = std::max(m.num_dims, num_dims);
    }
    AffineMap::get(num_dims, num_symbols, results, maps.front().context())
}

/**
Returns the map that results from projecting out the dimensions specified in `projectedDimensions`. The projected dimensions are set to 0.

Example:

1) map                  : affine_map<(d0, d1, d2) -> (d0, d1)>
   projected_dimensions : {2}
   result               : affine_map<(d0, d1) -> (d0, d1)>

2) map                  : affine_map<(d0, d1) -> (d0 + d1)>
   projected_dimensions : {1}
   result               : affine_map<(d0) -> (d0)>

3) map                  : affine_map<(d0, d1, d2) -> (d0, d1)>
   projected_dimensions : {1}
   result               : affine_map<(d0, d1) -> (d0, 0)>

This function also compresses unused symbols away.
*/
pub fn get_projected_map(
    map: AffineMap,
    projected_dimensions: &SmallBitVector
) -> AffineMap {
    compress_unused_symbols(compress_dims(map, unused_dims))
}

/// Apply a permutation from `map` to `source` and return the result.
pub fn apply_permutation_map<T>(map: AffineMap, source: &[T]) -> SmallVector<[T]> {
    assert!(map.is_projected_permutation(false));
    assert!(map.num_inputs() == source.len());
    let mut result = SmallVector::<[T]>::new();
    result.reserve(map.num_outputs);
    for expr in map.outputs() {
        match expr {
            Dimension { index, .. } => {
                result.push(source[index]);
            },
            Constant { value } => {
                assert!(
                    value == 0,
                    "Unexpected constant in projected permutation map");
                result.push(0);
            },
            _ => unreachable!("Unexpected result in projected permutation map")
        }
    }
    result
}

/// Calculates maxmimum dimension and symbol positions from the expressions
/// in `exprsLists` and stores them in `max_dim` and `max_sym` respectively.
// static
pub fn max_dim_and_symbol<AffineExprContainer>(
    exprs_list: &[AffineExprContainer],
    max_dim: &i64,
    max_sym: &i64
) {
    for exprs in exprs_list {
        for expr in exprs {
            expr.walk(|expr: AffineExpr| {
                if let d = e.dyn_cast<Dimension>() {
                    max_dim = std::max(max_dim, static_cast<i64>(d.get_position()));
                }
                if let s = e.dyn_cast<Symbol>() {
                    max_sym = std::max(max_sym, static_cast<i64>(s.get_position()));
                }
            });
        }
    }
}


/// ConstantFolder evaluates an affine expression using constant operands passed in `operand_consts`. Returns an IntegerAttribute attribute representing the constant value of the affine expression evaluated on constant `operand_consts`, or nullptr if it can't be folded.
pub struct ConstantFolder {
    /**
    The number of dimension operands in AffineMap containing this expression.
    */
    num_dims: usize,
    /// The constant valued operands used to evaluate this AffineExpr.
    operand_consts: &[Box<dyn Attribute>]
}

impl ConstantFolder {
    pub fn new(num_dims: usize, operand_consts: &[dyn Attribute]) -> Self {
        Self { num_dims, operand_consts }
    }

    /// Attempt to constant fold the specified affine expr, or return null on
    /// failure.
    pub fn constant_fold(&self, expr: AffineExpr) -> IntegerAttribute {
        if let result = self.constant_fold_impl(expr) {
            return IntegerAttribute::get(IndexType::get(expr.context()), *result);
        }
        null()
    }

    fn constant_fold_impl(&self, expr: AffineExpr) -> Option<i64> {
        match expr {
            BinaryOperation { kind, lhs, rhs } => {
                match kind {
                    Add => self.constant_fold_bin_expr(expr, |lhs: i64, rhs: i64| lhs + rhs),
                    Mul => self.constant_fold_bin_expr(expr, |lhs: i64, rhs: i64| lhs * rhs),
                    Mod => self.constant_fold_bin_expr(expr, |lhs: i64, rhs: i64| lhs % rhs),
                    FloorDiv => self.constant_fold_bin_expr(expr, |lhs: i64, rhs: i64| floor_div(lhs, rhs)),
                    CeilDiv => self.constant_fold_bin_expr(expr, |lhs: i64, rhs: i64| ceil_div(lhs, rhs)),
                }
            },
            Constant { value } => Some(value),
            Dimension { index, .. } => {
                if let attr = self.operand_consts[index].dyn_cast_or_null<IntegerAttribute>() {
                    return attr.int();
                }
                None
            },
            Symbol { index } => {
                if let attr
                    = self.operand_consts[D + index]
                        .dyn_cast_or_null<IntegerAttribute>() {
                    return attr.int();
                }
                None
            }
        }
    }
    
    // TODO: Change these to operate on APInts too.
    fn constant_fold_bin_expr(
        &self,
        expr: AffineExpr,
        op: fn(i64, i64) -> i64
    ) -> Option<i64> {
        let bin_op_expr = expr.cast<BinaryOperation>();
        if let lhs = constant_fold_impl(bin_op_expr.getLHS()) {
            if let rhs = constant_fold_impl(bin_op_expr.getRHS()) {
                return Some(op(*lhs, *rhs));
            }
        }
        None
    }
}
