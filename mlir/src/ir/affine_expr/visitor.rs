/*!
# MLIR AffineExpr Visitor Class

This file defines the AffineExpr visitor class.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/AffineExprVisitor.h>
*/

use crate::{
    mlir::ir::affine_expr::AffineExpr,
    llvm::adt::small_vector::SmallVector
};

use AffineExpr::*;

/**
Base class for AffineExpr visitors/walkers.

AffineExpr visitors are used when you want to perform different actions for different kinds of AffineExprs without having to use lots of casts and a big switch instruction.

To define your own visitor, inherit from this class, specifying your new type for the 'SubClass' template parameter, and "override" visitXXX functions in your class. This class is defined in terms of statically resolved overloading, not virtual functions.

For example, here is a visitor that counts the number of for AffineDimExprs in an AffineExpr.

```cpp
/// Declare the class.  Note that we derive from AffineExprVisitor instantiated with our new subclasses_ type.

struct DimExprCounter : public AffineExprVisitor {
    usize numDimExprs;
    DimExprCounter() : numDimExprs(0) {}
    void visit_dim_expr(AffineDimExpr expr) { ++numDimExprs; }
};
```

And this class would be used like this:

```cpp
DimExprCounter dec;
dec.visit(affineExpr);
numDimExprs = dec.numDimExprs;
```

AffineExprVisitor provides visit methods for the following binary affine op expressions:
AffineBinaryAddOpExpr, AffineBinaryMulOpExpr,
AffineBinaryModOpExpr, AffineBinaryFloorDivOpExpr,
AffineBinaryCeilDivOpExpr. Note that default implementations of these methods will call the general AffineBinaryOpExpr method.

In addition, visit methods are provided for the following affine expressions: AffineConstantExpr, AffineDimExpr, and AffineSymbolExpr.

Note that if you don't implement visitXXX for some affine expression type, the visitXXX method for Instruction superclass will be invoked.

Note that this class is specifically designed as a template to avoid virtual function call overhead. Defining and using a AffineExprVisitor is just as efficient as having your own switch instruction over the instruction opcode.
*/
pub trait AffineExprVisitor<T> {
    /*
    Interface code - This is the public interface of the AffineExprVisitor that you use to visit affine expressions...
    */

    /// Function to walk an AffineExpr (in post order).
    fn walk_post_order(&self, expr: AffineExpr) -> T {
        match expr {
            BinaryOperation { kind, lhs, rhs } => {
                match kind {
                    Add => {
                        self.walk_operands_post_order(expr);
                        self.visit_add_expr(expr)
                    },
                    Mul => {
                        self.walk_operands_post_order(expr);
                        self.visit_mul_expr(expr)
                    },
                    Mod => {
                        self.walk_operands_post_order(expr);
                        self.visit_mod_expr(expr)
                    },
                    FloorDiv => {
                        self.walk_operands_post_order(expr);
                        self.visit_floor_div_expr(expr)
                    },
                    CeilDev => {
                        self.walk_operands_post_order(expr);
                        self.visit_ceil_div_expr(expr)
                    }
                }
            },
            Constant => self.visit_constant_expr(expr),
            Dimension => self.visit_dim_expr(expr),
            Symbol => self.visit_symbol_expr(expr),
        }
    }

    /// Function to visit an AffineExpr.
    fn visit(&self, expr: AffineExpr) -> T {
        match expr {
            BinaryOperation { kind, lhs, rhs } => {
                match kind {
                    Add => self.visit_add_expr(expr),
                    Mul => self.visit_mul_expr(expr),
                    Mod => self.visit_mod_expr(expr),
                    FloorDiv => self.visit_floor_div_expr(expr),
                    CeilDev => self.visit_ceil_div_expr(expr)
                }
            },
            Constant => self.visit_constant_expr(expr),
            Dimension => self.visit_dim_expr(expr),
            Symbol => self.visit_symbol_expr(expr),
        }
    }

    /*
    Visitation functions... these functions provide default fallbacks in case the user does not specify what to do for a particular instruction type.
    The default behaviour is to generalise the instruction type to its subtype and try visiting the subtype.  All of this should be inlined perfectly, because there are no virtual functions to get in the way.
    */

    // Default visit methods. Note that the default op-specific binary op visit
    // methods call the general visit_affine_binary_op_expr visit method.
    fn visit_affine_binary_op_expr(&self, expr: AffineExpr) -> T;

    fn visit_add_expr(&self, expr: AffineExpr) {
        self.visit_affine_binary_op_expr(expr);
    }
    fn visit_mul_expr(&self, expr: AffineExpr) {
        self.visit_affine_binary_op_expr(expr);
    }
    fn visit_mod_expr(&self, expr: AffineExpr) {
        self.visit_affine_binary_op_expr(expr);
    }
    fn visit_floor_div_expr(&self, expr: AffineExpr) {
        self.visit_affine_binary_op_expr(expr);
    }
    fn visit_ceil_div_expr(&self, expr: AffineExpr) {
        self.visit_affine_binary_op_expr(expr);
    }
    fn visit_constant_expr(&self, expr: AffineExpr);

    fn visit_dim_expr(&self, expr: AffineExpr);

    fn visit_symbol_expr(&self, expr: AffineExpr);
}

/**
This class is used to flatten a pure affine expression (AffineExpr, which is in a tree form) into a sum of products (w.r.t constants) when possible, and in that process simplifying the expression. For a modulo, floordiv, or a ceildiv expression, an additional identifier, called a local identifier, is introduced to rewrite the expression as a sum of product affine expression. Each local identifier is always and by construction a floordiv of a pure add/mul affine function of dimensional, symbolic, and other local identifiers, in a non-mutually recursive way. Hence, every local identifier can ultimately always be recovered as an affine function of dimensional and symbolic identifiers (involving floordiv's); note however that by AffineExpr construction, some floordiv combinations are converted to mod's. The result of the flattening is a flattened expression and a set of constraints involving just the local variables.

d2 + (d0 + d1) floordiv 4  is flattened to d2 + q where 'q' is the local variable introduced, with local_var_cst containing 4*q <= d0 + d1 <= 4*q + 3.

The simplification performed includes the accumulation of contributions for each dimensional and symbolic identifier together, the simplification of floordiv/ceildiv/mod expressions and other simplifications that in turn happen as a result. A simplification that this flattening naturally performs is of simplifying the numerator and denominator of floordiv/ceildiv, and folding a modulo expression to a zero, if possible. Three examples are below:

(d0 + 3 * d1) + d0) - 2 * d1) - d0    simplified to     d0 + d1
(d0 - d0 mod 4 + 4) mod 4             simplified to     0
(3 * d0 + 2 * d1 + d0) floordiv 2 + d1    simplified to     2 * d0 + 2 * d1

The way the flattening works for the second example is as follows: d0 % 4 is replaced by d0 - 4 * q with q being introduced: the expression then simplifies to: (d0 - (d0 - 4q) + 4) = 4q + 4, modulo of which w.r.t 4 simplifies to zero. Note that an affine expression may not always be expressible purely as a sum of products involving just the original dimensional and symbolic identifiers due to the presence of modulo/floordiv/ceildiv expressions that may not be eliminated after simplification; in such cases, the final expression can be reconstructed by replacing the local identifiers with their corresponding explicit form stored in 'local_exprs' (note that each of the explicit forms itself would have been simplified).

The expression walk method here performs a linear time post order walk that performs the above simplifications through visit methods, with partial results being stored in 'operand_expr_stack'. When a parent expr is visited, the flattened expressions corresponding to its two operands would already be on the stack - the parent expression looks at the two flattened expressions and combines the two. It pops off the operand expressions and pushes the combined result (although this is done in-place on its LHS operand expr).
When the walk is completed, the flattened form of the top-level expression would be left on the stack.

A flattener can be repeatedly used for multiple affine expressions that bind to the same operands, for example, for all result expressions of an AffineMap or AffineValueMap. In such cases, using it for multiple expressions is more efficient than creating a new flattener for each expression since common identical div and mod expressions appearing across different expressions are mapped to the same local identifier (same column position in 'local_var_cst').
*/
pub struct SimpleAffineExprFlattener {
    /**
    Flattend expression layout: [dims, symbols, locals, constant]
    Stack that holds the LHS and RHS operands while visiting a binary op expr.
    In future, consider adding a prepass to determine how big the SmallVector's will be, and linearise this to `Vec<i64>` to prevent SmallVector moves on re-allocation.
    */
    operand_expr_stack: Vec<SmallVector<[i64; 8]>>,

    num_dims: usize,
    num_symbols: usize,

    /// Number of newly introduced identifiers to flatten mod/floordiv/ceildiv's.
    num_locals: usize,

    /**
    AffineExpr's corresponding to the floordiv/ceildiv/mod expressions for which new identifiers were introduced; if the latter do not get canceled out, these expressions can be readily used to reconstruct the AffineExpr (tree) form. Note that these expressions themselves would have been simplified (recursively) by this pass. Eg. d0 + (d0 + 2*d1 + d0) ceildiv 4 will be simplified to d0 + q, where q = (d0 + d1) ceildiv 2. (d0 + d1) ceildiv 2 would be the local expression stored for q.
    */
    local_exprs: SmallVector<[AffineExpr; 4]>
}

impl SimpleAffineExprFlattener {
    /**
    Add a local identifier (needed to flatten a mod, floordiv, ceildiv expr).
    The local identifier added is always a floordiv of a pure add/mul affine function of other identifiers, coefficients of which are specified in dividend and with respect to a positive constant divisor. local_expr is the simplified tree expression (AffineExpr) corresponding to the quantifier.
    */
    // virtual
    fn add_local_floor_div_id(
        &self, dividend: &[i64], divisor: i64, local_expr: AffineExpr
    ) {
        assert!(divisor > 0, "Positive constant divisor expected.");
        for sub_expr in self.operand_expr_stack {
            sub_expr.insert(self.local_var_start_index() + self.num_locals, 0);
        }
        self.local_exprs.push(local_expr);
        self.num_locals += 1;
        // dividend and divisor are not used here; an override of this method uses it.
    }

    /**
    Add a local identifier (needed to flatten a mod, floordiv, ceildiv, mul expr) when the rhs is a symbolic expression. The local identifier added may be a floordiv, ceildiv, mul or mod of a pure affine/semi-affine function of other identifiers, coefficients of which are specified in the lhs of the mod, floordiv, ceildiv or mul expression and with respect to a symbolic rhs expression. `local_expr` is the simplified tree expression (AffineExpr) corresponding to the quantifier.
    */
    //virtual
    fn add_local_id_semi_affine(&mut self, local_expr: AffineExpr) {
        for sub_expr in self.operand_expr_stack {
            sub_expr.insert(self.local_var_start_index() + self.num_locals, 0);
        }
        self.local_exprs.push(local_expr);
        self.num_locals += 1;
    }

// private:
    /**
    Adds `expr`, which may be mod, ceildiv, floordiv or mod expression representing the affine expression corresponding to the quantifier introduced as the local variable corresponding to `expr`. If the quantifier is already present, we put the coefficient in the proper index of `result`, otherwise we add a new local variable and put the coefficient there.
    */
    fn add_local_variable_semi_affine(
        &self,
        expr: AffineExpr,
        result: &SmallVector<[i64]>,
    ) {
        let loc = self.find_local_id(expr);
        if loc.is_none() {
            self.add_local_id_semi_affine(expr);
        }
        result.fill(0);
        if let Some(index) = loc {
            result[self.local_var_start_index() + index] = 1;
        } else {
            result[self.local_var_start_index() + self.num_locals - 1] = 1;
        }
    }

    /**
    t = expr floordiv c  <=> t = q, c * q <= expr <= c * q + c - 1
    A floordiv is thus flattened by introducing a new local variable q, and replacing that expression with 'q' while adding the constraints
    c * q <= expr <= c * q + c - 1 to local_var_cst (done by FlatAffineConstraints::add_local_floor_div).

    A ceildiv is similarly flattened:
    t = expr ceildiv c  <=> t = (expr + c - 1) floordiv c
    */
    fn visit_div_expr(&self, expr: AffineBinaryOpExpr, is_ceil: bool) {
        assert!(self.operand_expr_stack.len() >= 2);

        let context = expr.context();
        let rhs = self.operand_expr_stack.last().unwrap();
        self.operand_expr_stack.pop();
        let lhs = self.operand_expr_stack.last().unwrap();

        // TODO
    }

    fn find_local_id(&self, local_expr: AffineExpr) -> Option<usize> {
        self.local_exprs.iter().position(|expr| expr == local_expr)
    }

    #[inline]
    const fn num_cols(&self) -> usize {
        self.num_dims + self.num_symbols + self.num_locals + 1
    }
    
    #[inline]
    const fn constant_index(&self) -> usize {
        self.num_cols() - 1
    }

    #[inline]
    const fn local_var_start_index(&self) -> usize {
        self.num_dims + self.num_symbols
    }

    // inline unsigned get_symbol_start_index(&self) const { return num_dims; }

    // inline unsigned get_dim_start_index(&self) const { return 0; }
}

impl AffineExprVisitor for SimpleAffineExprFlattener {
    // Visitor method overrides.
    fn visit_mul_expr(&self, expr: AffineExpr) {

    }

    fn visit_add_expr(&self, expr: AffineExpr) {
        assert!(self.operand_expr_stack.len() >= 2);
        let rhs = self.operand_expr_stack.last().unwrap();
        let lhs = self.operand_expr_stack[self.operand_expr_stack.len() - 2];
        assert!(lhs.len() == rhs.len());
        // Update the LHS in place.
        for i in 0..rhs.len() {
            lhs[i] += rhs[i];
        }
        // Pop off the RHS.
        self.operand_expr_stack.pop_back();
    }

    fn visit_dim_expr(&self, expr: AffineExpr) {

    }

    fn visit_symbol_expr(&self, expr: AffineExpr) {

    }

    fn visit_constant_expr(&self, expr: AffineExpr) {

    }

    fn visit_ceil_div_expr(&self, expr: AffineExpr) {
        self.visit_div_expr(expr, true)
    }

    fn visit_floor_div_expr(&self, expr: AffineExpr) {
        self.visit_div_expr(expr, false)
    }

    /**
    t = expr mod c   <=>  t = expr - c*q and c*q <= expr <= c*q + c - 1

    A mod expression "expr mod c" is thus flattened by introducing a new local
    variable q (= expr floordiv c), such that expr mod c is replaced with
    'expr - c * q' and c * q <= expr <= c * q + c - 1 are added to localVarCst.
    */
    fn visit_mod_expr(&self, expr: AffineExpr);
}
