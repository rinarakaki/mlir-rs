/*!
# Function-Like Op Utilities

This file provides utility functions for implementing function-like operations, in particular, parsing, printing and verification components common to function-like operations.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/FunctionImplementation.h>
*/

/// A named class for passing around the variadic flag.
pub struct VariadicFlag {
    /// Underlying storage.
    variadic: bool
}
