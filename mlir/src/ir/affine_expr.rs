//! MLIR Affine Expr Class 
//!
//! An affine expression is an affine combination of dimension identifiers and symbols, including ceildiv/floordiv/mod by a constant integer.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/AffineExpr.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/AffineExpr.cpp>

pub mod detail;
pub mod visitor;

use core::{
    cmp::PartialEq,
    fmt::Display,
    hash::Hash,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};
use llvm::{
    adt::{
        dense_map::DenseMap,
        dense_map_info,
        hashing,
        small_vector::SmallVector
    },
    support::casting
};
use crate::{
    ir::{
        affine_map::AffineMap,
        mlir_context::MLIRContext
    },
    support::{
        llvm,
        storage_uniquer::{Storage, StorageAllocator}
    }
};

#[derive(PartialEq)]
pub enum Kind {
    Add,
    /// RHS of mul is always a constant or a symbolic expression.
    Mul,
    /// RHS of mod is always a constant or a symbolic expression with a positive value.
    Mod,
    /// RHS of floordiv is always a constant or a symbolic expression.
    FloorDiv,
    /// RHS of ceildiv is always a constant or a symbolic expression.
    CeilDiv,
}

#[derive(Hash, PartialEq)]
pub enum AffineExpr {
    /// Affine binary operation expression. An affine binary operation could be an add, mul, floordiv, ceildiv, or a modulo operation. (Subtraction is represented through a multiply by -1 and add.) These expressions are always constructed in a simplified form. For eg., the LHS and RHS operands can't both be constants. There are additional canonicalising rules depending on the op type: see checks in the constructor.
    BinaryOperation {
        kind: Kind,
        lhs: Box<AffineExpr>,
        rhs: Box<AffineExpr>
    },
    /// A dimensional identifier appearing in an affine expression.
    Dimension {
        kind: Kind,
        index: usize
    },
    /// A symbolic identifier appearing in an affine expression.
    Symbol {
        index: usize
    },
    /// An integer constant appearing in affine expression.
    Constant {
        /// The constant.
        value: i64
    }
  
    // /// This is a marker for the last affine binary op. The range of binary
    // /// op's is expected to be this element and earlier.
    // LAST_AFFINE_BINARY_OP = CeilDiv,
}

use AffineExpr::*;

/// Base type for affine expression.
/// AffineExpr's are immutable value types with intuitive operators to operate on chainable, lightweight compositions.
/// An AffineExpr is an interface to the underlying storage type pointer.
impl AffineExpr {
    pub fn context(&self) -> &MLIRContext {
        self.context
    }

    // /// Return the classification for this type.
    // fn kind(&self) -> AffineExprKind {
    //     self.kind
    // }
  
    // void print(raw_ostream &os) const;
    // void dump() const;
  
    /// Returns true if this expression is made out of only symbols and constants, i.e., it does not involve dimensional identifiers.
    fn is_symbolic_or_constant(&self) -> bool {
        match self {
            BinaryOperation { lhs, rhs, .. } =>
                   lhs.is_symbolic_or_constant()
                && rhs.is_symbolic_or_constant(),
            Dimension => false,
            Symbol => true,
            Constant => true
        }
    }
  
    /// Returns true if this is a pure affine expression, i.e., multiplication, floordiv, ceildiv, and mod is only allowed w.r.t constants.
    fn is_pure_affine(&self) -> bool {
        match self {
            BinaryOperation { kind, lhs, rhs } => {
                use Kind::*;
                match kind {
                    Add => lhs.is_pure_affine() && rhs.is_pure_affine(),
                    Mul => 
                        /*
                        TODO: Canonicalise the constants in binary operators to the RHS when possible, allowing this to merge into the next case.
                        */
                           lhs.is_pure_affine()
                        && rhs.is_pure_affine()
                        && (
                               matches!(lhs, Constant) 
                            || matches!(rhs, Constant) 
                        )
                    ,
                    FloorDiv | CeilDiv | Mod =>
                        lhs.is_pure_affine() && matches!(rhs, Constant) 
                }
            },
            Dimension => true,
            Symbol => true,
            Constant => true
        }
    }
  
    /// Returns the greatest known integral divisor of this affine expression. The result is always positive.
    fn largest_known_divisor(&self) -> i64 {
        match self {
            BinaryOperation { kind, lhs, rhs } => {
                use Kind::*;
                match kind {
                    CeilDiv | FloorDiv => {
                        // If the RHS is a constant and divides the known divisor on the LHS, the quotient is a known divisor of the expression.
                        // Leave alone undefined expressions.
                        if let Constant { value } = rhs && value != 0 {
                            let lhs_div = lhs.largest_known_divisor();
                            if lhs_div % value == 0 {
                                return lhs_div / value;
                            }
                        }
                        return 1;
                    }
                    Mul =>
                          lhs.largest_known_divisor()
                        * rhs.largest_known_divisor(),
                    Add | Mod => {
                        std::gcd(lhs.largest_known_divisor() as u64,
                                 rhs.largest_known_divisor() as u64)
                    }
                }
            },
            Dimension => 1,
            Symbol => 1,
            Constant { value } => value.abs()
        }
    }
  
    /// Return true if the affine expression is a multiple of 'factor'.
    fn is_multiple_of(&self, factor: i64) -> bool {
        match self {
            BinaryOperation { kind, lhs, rhs } => {
                use Kind::*;
                match kind {
                    Mul => {
                        /*
                        It's probably not worth optimising this further (to not traverse the whole sub-tree under - it that would require a version of is_multiple_of that on a 'false' return also returns the largest known divisor).
                        */
                        let l = lhs.largest_known_divisor();
                        let u = rhs.largest_known_divisor();
                           l % factor == 0
                        || u % factor == 0
                        || (l * u) % factor == 0
                    },
                    Add | FloorDiv | CeilDiv | Mod =>
                        std::gcd(lhs.largest_known_divisor() as u64,
                                 rhs.largest_known_divisor() as u64)
                        % factor == 0
                }
            },
            Dimension => factor * factor == 1,
            Symbol => factor * factor == 1,
            Constant { value } => value % factor == 0
        }
    }
  
    /// Return true if the affine expression involves AffineDimExpr `index`.
    pub fn is_function_of_dim(self, index: usize) -> bool {
        match self {
            BinaryOperation { lhs, rhs, .. } =>
                   lhs.is_function_of_dim(index)
                || rhs.is_function_of_dim(index),
            Dimension => self == dimension(index, self.context()),
            Symbol => false,
            Constant => false
        }
    }
  
    /// Return true if the affine expression involves AffineSymbolExpr `index`.
    pub fn is_function_of_symbol(self, index: usize) -> bool {
        match self {
            BinaryOperation { lhs, rhs, .. } => {
                   lhs.is_function_of_symbol(index)
                || rhs.is_function_of_symbol(index)
            },
            Dimension => false,
            Symbol => self == symbol(index, self.context()),
            Constant => false
        }
    }
  
    /// Walk all of the AffineExpr's in this expression in postorder.
    pub fn walk(&self, callback: fn(Self)) {
        AffineExprWalker::new(callback).walk_post_order(&self);
    }
  
    /// This method substitutes any uses of dimensions and symbols (e.g. dim#0 with dim_replacements[0]) and returns the modified expression tree.
    /// This is a dense replacement method: a replacement must be specified for every single dim and symbol.
    pub fn replace_dims_and_symbols(
        self,
        dim_replacements: [&Self],
        sym_replacements: [&Self]
    ) -> Self
    {
        match self {
            BinaryOperation { kind, lhs, rhs } => {
                let new_lhs = lhs.replace_dims_and_symbols(dim_replacements, sym_replacements);
                let new_rhs = rhs.replace_dims_and_symbols(dim_replacements, sym_replacements);
                if new_lhs == lhs && new_rhs == rhs {
                    return self;
                }
                binary_operation(kind, new_lhs, new_rhs)
            },
            Dimension { index, .. } => {
                if index >= dim_replacements.len() {
                    self
                } else {
                    dim_replacements[index]
                }
            },
            Symbol { index } => {
                if index >= dim_replacements.len() {
                    self
                } else {
                    dim_replacements[index]
                }
            },
            Constant => self
        }
    }   
  
    /// Dim-only version of replace_dims_and_symbols.
    pub fn replace_dims(&self, dim_replacements: &[Self]) -> Self {
        self.replace_dims_and_symbols(dim_replacements, [])
    }
  
    /// Symbol-only version of replace_dims_and_symbols.
    pub fn replace_symbols(&self, sym_replacements: &[Self]) -> Self {
        self.replace_dims_and_symbols([], sym_replacements)
    }
  
    /// Sparse replace method. Replace `expr` by `replacement` and return the modified expression tree.
    pub fn replace(&self, expr: Self, replacement: Self) -> Self {
        let mut map = DenseMap::<Self, Self>::new();
        map.insert(expr, replacement);
        self.replace_(map)
    }
  
    /// Sparse replace method. If `*this` appears in `map` replaces it by `map[*this]` and return the modified expression tree. Otherwise traverse `*this` and apply replace with `map` on its subexpressions.
    pub fn replace_(self, map: &DenseMap<Self, Self>) -> Self {
        match self {
            BinaryOperation { lhs, rhs, .. } => {
                if let Some(expr) = map.find(&self) {
                    return expr;
                }
                let new_lhs = lhs.replace_(map);
                let new_rhs = rhs.replace_(map);
                if new_lhs == lhs && new_rhs == rhs {
                    return self
                }
                binary_operation(self.kind(), new_lhs, new_rhs)
            },
            Dimension { .. } | Symbol { .. } | Constant { .. } => self
        }
    }
  
    /// Replace dims[offset..num_dims] by dims[(offset + shift)..(shift + num_dims)].
    pub fn shift_dims(
        &self,
        num_dims: usize,
        shift: usize,
        offset: usize /* 0 */
    ) -> Self
    {
        let mut dims = SmallVector::<[Self; 4]>::new();
        for index in 0..offset {
            dims.push(dimension(index, self.context()));
        }
        for index in offset..num_dims {
            dims.push(dimension(index + shift, self.context()));
        }
        self.replace_dims_and_symbols(dims.into(), &[])
    }
  
    /// Replace symbols[offset ... num_symbols) by symbols[offset + shift ... shift + num_symbols).
    pub fn shift_symbols(&self, num_symbols: usize, shift: usize, offset: usize /* 0 */
    ) -> Self
    {
        let mut symbols = SmallVector::<[Self; 4]>::new();
        for index in 0..offset { 
            symbols.push(symbol(index, self.context()));
        }
        for index in offset..num_symbols {
            symbols.push(symbol(index + shift, self.context()));
        }
        self.replace_dims_and_symbols(&[], symbols.into())
    }

    /// Compose with an AffineMap.
    /// Returns the composition of this AffineExpr with `map`.
    ///
    /// Prerequisites:
    /// `this` and `map` are composable, i.e. that the number of AffineDimExpr of `this` is smaller than the number of results of `map`. If a result of a map does not have a corresponding AffineDimExpr, that result simply does not appear in the produced AffineExpr.
    ///
    /// Example:
    ///   expr: `d0 + d2`
    ///   map:  `(d0, d1, d2)[s0, s1] -> (d0 + s1, d1 + s0, d0 + d1 + d2)`
    ///   returned expr: `d0 * 2 + d1 + d2 + s1`
    pub fn compose(&self, map: AffineMap) -> Self {
        // let dim_replacements: SmallVector::<[Self; 8]> = ;
        self.replace_dims_and_symbols(map.outputs(), &[])
    }
}

// : AffineExprVisitor
pub struct AffineExprWalker {
    callback: fn(AffineExpr)  // std::function
}

impl AffineExprWalker {
    pub fn new(callback: Fn(AffineExpr) /* std::function */) -> Self {
        Self { callback }
    }

    // pub fn visit_affine_binary_op_expr(&self, expr: BinaryOperation) {
    //     self.callback(expr);
    // }

    // pub fn visit_constant_expr(&self, expr: Constant) {
    //     self.callback(expr);
    // }

    // pub fn visit_dim_expr(&self, expr: Dimension) {
    //     self.callback(expr);
    // }

    // pub fn visit_symbol_expr(&self, expr: Symbol) {
    //     self.callback(expr);
    // }
}

// inline AffineExpr operator+(int64_t val, AffineExpr expr) { return expr + val; }
// inline AffineExpr operator*(int64_t val, AffineExpr expr) { return expr * val; }
// inline AffineExpr operator-(int64_t val, AffineExpr expr) {
//   return expr * (-1) + val;
// }

impl Add<i64> for AffineExpr {
    type Output = Self;

    fn add(self, rhs: i16) -> Self::Output {
        self + constant(rhs, self.context())
    }
}

impl Add for AffineExpr {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        if let Some(simplified) = simplify_add(self, other) {
            return simplified;
        }

        let uniquer = self.context().affine_uniquer();
        uniquer.get::<BinaryOperation>(
            /*initFn=*/{}, static_cast::<usize>(Kind::Add), self, other)
    }
}

/// Simplify add expression. Return nullptr if it can't be simplified.
pub const fn simplify_add(lhs: AffineExpr, rhs: AffineExpr) -> AffineExpr {
    match (lhs, rhs) {
        // Fold if both LHS, RHS are a constant.
        (Constant { value: lhs_value }, Constant { value: rhs_value })
            =>  constant(lhs_value + rhs_value, lhs.context()),
        // Canonicalise so that only the RHS is a constant. (4 + d0 becomes d0 + 4).
        // If only one of them is a symbolic expressions, make it the RHS.
        (Constant, rhs) => rhs + lhs,
        (lhs, rhs)
        if lhs.is_symbolic_or_constant() && !rhs.is_symbolic_or_constant()
        => rhs + lhs,
        // Addition with a zero is a noop, return the other input.
        (lhs, Constant { value }) if value == 0 => lhs,
        // Fold successive additions like (d0 + 2) + 3 into d0 + 5.
        (
            BinaryOperation {
                kind: Kind::Add,
                lhs,
                rhs: Constant { value: lrhs_value }
            },
            Constant { value: rhs_value }
        ) => lhs + (lrhs_value + rhs_value)
    }
    // At this point, if there was a constant, it would be on the right.
    
    // Detect "c1 * expr + c_2 * expr" as "(c1 + c2) * expr".
    // c1 is rRhsConst, c2 is rLhsConst; first_expr, second_expr are their
    // respective multiplicands.
    let rLhsConst: i64;
    let rRhsConst: u64;
    let first_expr: AffineExpr;
    let second_expr: AffineExpr;
    if let BinaryOperation {
        kind: Kind::Mul,
        lhs,
        rhs: Constant { value: lrhs_value }
    } = lhs {
      rLhsConst = lrhs_value;
      first_expr = lhs;
    } else {
      rLhsConst = 1;
      first_expr = lhs;
    }
  
    if let BinaryOperation {
        kind: Kind::Mul,
        lhs,
        rhs: Constant { value: rrhs_value }
    } = rhs {
      rRhsConst = rrhs_value;
      second_expr = lhs;
    } else {
      rRhsConst = 1;
      second_expr = rhs;
    }
  
    if rLhsConst && rRhsConst && first_expr == second_expr {
      return binary_operation(
          Kind::Mul, first_expr,
          constant(*rLhsConst + *rRhsConst, lhs.getContext()));
        }
  
    // When doing successive additions, bring constant to the right: turn (d0 + 2) + d1 into (d0 + d1) + 2.
    if let BinaryOperation {
        kind: Kind::Add,
        lhs,
        rhs: lrhs
    } = lhs
    && matches!(lrhs, Constant)
    {
        return lhs + rhs + lrhs;
    }
  
    // Detect and transform "expr - q * (expr floordiv q)" to "expr mod q", where
    // q may be a constant or symbolic expression. This leads to a much more
    // efficient form when 'c' is a power of two, and in general a more compact
    // and readable form.
  
    // Process '(expr floordiv c) * (-c)'.
    if !matches!(rhs, BinaryOperation) {
        return None;
    }
  
    if let BinaryOperation {
        // Check if rlhs is of the form (expr floordiv q) * q, where q is a
        // symbolic expression.
        lhs: BinaryOperation {
            kind: Kind::Mul,
            lhs: rllhs,
            rhs: rlrhs
        },
        rhs: Constant { value: rrhs_value },
        ..
    } = rhs
    && rrhs_value == -1
    {
        // Check rllhs = expr floordiv q.
        if let BinaryOperation {
            kind,
            lhs: rlllhs,
            rhs: rllrhs
        } = rllhs {
            if !matches!(kind, Kind::FloorDiv)
            // Check rlrhs = q.
            && rllrhs = rlrhs
            && lhs == rlllhs
            {
                return lhs % rlrhs;
            } else {
                return None;
            }
        } else {
            return None;
        }
    }

    // Process rlhs, which is 'expr floordiv c'.
    if let BinaryOperation {
        lhs: BinaryOperation {
            kind: Kind::FloorDiv,
            lhs: rllhs,
            rhs: rlrhs
        },
        rhs: rrhs,
        ..
    } = rhs
    && lhs == rllhs && rlrhs == -rrhs
    {
        return lhs % rlrhs;
    } else {
        return None;
    }
    
    None
}

impl Neg for AffineExpr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * constant(-1, self.context())
    }
}

impl Sub<i64> for AffineExpr {
    type Output = Self;

    fn sub(self, rhs: i16) -> Self::Output {
        self + (-rhs)
    }
}

impl Sub for AffineExpr {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl Mul<i64> for AffineExpr {
    type Output = Self;

    fn mul(self, rhs: i16) -> Self::Output {
        self * constant(rhs, self.context())
    }
}

impl Mul for AffineExpr {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        if let Some(simplified) = simplify_mul(self, other) {
            return simplified;
        }

        let uniquer = self.context().affine_uniquer();
        uniquer.get::<BinaryOperation>(
            /*initFn=*/{}, static_cast::<usize>(Kind::Mul), self, other)
    }
}

/// Simplify a multiply expression. Return nullptr if it can't be simplified.
pub const fn simplify_mul(lhs: AffineExpr, rhs: AffineExpr) -> Option<AffineExpr> {
    if let Constant { value: lhs_value } = lhs
    && let Constant { value: rhs_value } = rhs
    {
        return Some(constant(lhs_value * rhs_value, lhs.context()));
    };

    assert!(lhs.is_symbolic_or_constant() || rhs.is_symbolic_or_constant());
  
    // Canonicalise the mul expression so that the constant/symbolic term is the RHS. If both the lhs and rhs are symbolic, swap them if the lhs is a constant. (Note that a constant is trivially symbolic).
    if !rhs.is_symbolic_or_constant()
    || matches!(lhs, Constant) {
        // At least one of them has to be symbolic.
        return Some(rhs * lhs);
    }
  
    // At this point, if there was a constant, it would be on the right.
  
    // Multiplication with a one is a noop, return the other input.
    match rhs {
        Constant { value } if value == 1 => return Some(lhs),
        Constant { value } if value == 0 => return Some(value),
        _ => {}
    };

    match lhs {
        BinaryOperation {
            kind: Kind::Mul,
            lhs: llhs,
            rhs: Constant { value: lrhs_value }
        } => {
            // Fold successive multiplications: eg: (d0 * 2) * 3 into d0 * 6.
            if let Constant { value: rhs_value } = rhs {
                return llhs * (lrhs_value * rhs_value);
            }
            // When doing successive multiplication, bring constant to the right: turn (d0d * 2) * d1 into (d0 * d1) * 2.
            else {
                return Some((lhs * rhs) * Constant { value: lrhs_value })
            }
        }
    }
  
    None
}

impl Rem<i64> for AffineExpr {
    type Output = Self;

    fn rem(self, rhs: i16) -> Self::Output {
        self % constant(rhs, self.context())
    }
}

impl Rem for AffineExpr {
    type Output = Self;

    fn rem(self, other: Self) -> Self::Output {
        if let Some(simplified) = simplify_mod(self, other) {
            return simplified;
        }

        let uniquer = self.context().affine_uniquer();
        uniquer.get::<BinaryOperation>(
            /*initFn=*/{}, static_cast::<usize>(Kind::Mod), self, other)
    }
}

pub const fn simplify_mod(lhs: AffineExpr, rhs: AffineExpr)
    -> Option<AffineExpr>
{
    let rhs_value = match rhs {
        // mod w.r.t zero or negative numbers is undefined and preserved as is.
        Constant { value } if value >= 1 => value,
        _ => return None
    };
    if let Constant { value } = lhs {
        constant(r#mod(value, rhs_value), lhs.context())
    }

    // Fold modulo of an expression that is known to be a multiple of a constant to zero if that constant is a multiple of the modulo factor. Eg: (i * 128) mod 64 is folded to 0, and less trivially, (i*(j*4*(k*32))) mod 128 = 0.
    if lhs.largest_known_divisor() % rhs_value == 0 {
        return Some(constant(0, lhs.context()));
    }
  
    // Simplify (expr1 + expr2) mod divConst when either expr1 or expr2 is known to be a multiple of divConst.
    match lhs {
        BinaryOperation { kind: Kind::Add, lhs, rhs } => {
            let llhs_div = lhs.largest_known_divisor();
            let lrhs_div = rhs.largest_known_divisor();
            // rhsConst is known to be a positive constant.
            if llhs_div % rhs_value == 0 {
                return rhs % rhs_value;
            }
            if lrhs_div % rhs_value == 0 {
                return lhs % rhs_value;
            }
        },
        BinaryOperation {
            kind: Kind::Mod,
            lhs,
            rhs: Constant { value }
        } if value >= 1 => {
            if r#mod(value, rhs_value) == 0 {
                return lhs % rhs_value;
            }
        },
        _ => {}
    };

    None
}

// impl Display for AffineExpr {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {

//     }
// }

// impl BinaryOperation {
//     pub fn construct(
//         allocator: &StorageUniquer::StorageAllocator,
//         kind: Kind,
//         lhs: AffineExpr,
//         rhs: AffineExpr
//     ) -> Self {
//         let result = allocator.allocate<BinaryOperation>();
//         Self { kind, lhs, rhs }
//         result.context = result.lhs.context();
//         return result;
//     }
// }

// These free functions allow clients of the API not to use classes in detail.

pub fn dimension(index: usize, context: *mut MLIRContext) -> AffineExpr {

}

pub fn symbol(index: usize, context: *mut MLIRContext) -> AffineExpr {

}

pub fn constant(constant: i64, context: *mut MLIRContext) -> AffineExpr {
    let assign_ctx = (AffineConstantExprStorage *storage) {
        storage.context = context;
    };
    
    letuniquer = context.affine_uniquer();
    return uniquer.get::<Constant { constant }>(assign_ctx, constant);
}

pub fn binary_operation(
    kind: Kind, lhs: AffineExpr, rhs: AffineExpr
) -> AffineExpr
{
    use Kind::*;
    match kind {
        Add => lhs + rhs,
        Mul => lhs * rhs,
        FloorDiv => lhs.floor_div(rhs),
        CeilDiv => lhs.ceil_div(rhs),
        Mod => lhs % rhs,
    }
}

/// Constructs an affine expression from a flat ArrayRef. If there are local identifiers (neither dimensional nor symbolic) that appear in the sum of products expression, 'local_exprs' is expected to have the AffineExpr
/// for it, and is substituted into. The ArrayRef 'eq' is expected to be in the
/// format [dims, symbols, locals, constant term].
pub fn get_affine_expr_from_flat_form(
    flat_exprs: &[i64],
    num_dims: usize,
    num_symbols: usize,
    local_exprs: &[AffineExpr],
    context: *mut MLIRContext
) -> AffineExpr
{

}

/// Simplify an affine expression by flattening and some amount of simple analysis. This has complexity linear in the number of nodes in 'expr'.
/// Returns the simplified expression, which is the same as the input expression if it can't be simplified. When `expr` is semi-affine, a simplified semi-affine expression is constructed in the sorted order of dimension and symbol positions.
pub fn simplify_affine_expr(
    expr: AffineExpr,
    num_dims: usize,
    num_symbols: usize
) -> AffineExpr
{

}
