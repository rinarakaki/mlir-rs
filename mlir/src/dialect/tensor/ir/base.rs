/*!
# Base Definitions for Tensor Dialect

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Tensor/IR/TensorBase.td>
*/

use crate::mlir::ir::dialect::Dialect;

/**
The `tensor` dialect is intended to hold core tensor creation and manipulation ops, which are not strongly associated with any particular other dialect or domain abstraction. The primary smoke test of this is ops that make sense for any tensor element type.

We leave it to other dialects to hold the vast swath of possible computations one might want to do on a tensor.

The `tensor` type is (for better or for worse) used to represent all kinds of things, and supports an open-ended set of element types. Examples:

- representing large, dense aggregations of primitive types, suitable for high-performance numerical computing.
- representing shapes in the `shape` dialect, which consist of small 1D tensors of `index` data type.
- representing aggregations of strings or “variant” types.
- representing large, sparse aggregations of primitive types, suitable for high-performance numerical computing.

Thus, for the `tensor` dialect, we prefer for now to constrain the scope as much as possible. The expectation is that at some point in the future, the `tensor` dialect’s scope may be broadened through a careful discussion of the tradeoffs.

The `tensor` type is actually a builtin type (it lives in the builtin dialect), and does not live in this dialect.
*/
pub struct TensorDialect {
}
