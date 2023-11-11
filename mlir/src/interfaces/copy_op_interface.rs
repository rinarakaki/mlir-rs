/*!
# Copy Operations Interface

This file implements the operation interface for copy-like operations.

Defines the interface for copy-like operations.

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/CopyOpInterface.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/CopyOpInterface.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Interfaces/CopyOpInterface.cpp>
*/

use crate::mlir::ir::value::Value;

/// A copy-like operation is one that copies from source value to target value.
pub trait CopyOpInterface {
    /// Returns the source value for this copy operation
    fn source(&self) -> Value;

    fn target(&self) -> Value;
}
