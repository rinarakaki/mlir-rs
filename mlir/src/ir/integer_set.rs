/*!
MLIR Integer Set Class

Integer sets are sets of points from the integer lattice constrained by affine equality/inequality constraints. This class is meant to represent integer sets in the IR - for 'affine.if' operations and as attributes of other operations. It is typically expected to contain only a handful of affine constraints, and is immutable like an affine map. Integer sets are not unique'd unless the number of constraints they contain are below a certain threshold - although affine expressions that make up its equalities and inequalities are themselves unique.

This class is not meant for affine analysis and operations like set operations, emptiness checks, or other math operations for analysis and transformation. For the latter, use FlatAffineConstraints.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/IntegerSet.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/IntegerSet.cpp>
*/

pub mod detail;

use crate::mlir::ir::affine_expr::AffineExpr;

use super::affine_expr::constant;

/// An integer set representing a conjunction of one or more affine equalities and inequalities. An integer set in the IR is immutable like the affine map, but integer sets are not unique'd unless the number of constraints in them is below `kUniquingThreshold`. The affine expressions that make up the equalities and inequalities of an integer set are themselves unique and are allocated by the bump pointer allocator.
pub struct IntegerSet<const N: usize> {
    /// Array of affine constraints: a constraint is either an equality (affine_expr == 0) or an inequality (affine_expr >= 0).
    constraints: &'static [AffineExpr; N],

    // Bits to check whether a constraint is an equality or an inequality.
    kinds: [bool; N]
}

impl<const N: usize> IntegerSet<N> {
    pub fn new(
        num_dims: usize,
        num_symbols: usize,
        constraints: &[AffineExpr; N],
        kinds: &[bool; N]
    ) -> Self {

    }

    // Returns the canonical empty IntegerSet (i.e. a set with no integer points).
    pub fn new_empty(
        num_dims: usize,
        num_symbols: usize,
        context: *mut MLIRContext
    ) -> Self {
        let one = constant(1, context);
        // 1 == 0.
        Self::new(numDims, num_symbols, one, true)
    }
}
