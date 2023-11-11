/*!
# Base Dialect Definition File

This file contains the base set of constructs for defining `Dialect` classes.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/DialectBase.td>
*/

pub struct Dialect {
    name: &'static str
}
