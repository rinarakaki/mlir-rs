/*!
# MLIR Affine Expr Storage Details

This holds implementation details of AffineExpr. Ideally it would not be exposed and would be kept local to AffineExpr.cpp however, MLIRContext.cpp needs to know the sizes for placement-new style Allocation.

- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/AffineExprDetail.h>
*/

use crate::mlir::{
    ir,
    support::storage_uniquer::{Storage, Allocator}
};
