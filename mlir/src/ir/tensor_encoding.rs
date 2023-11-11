//! # MLIR Tensor Encoding Declarations
//!
//! Defines the interfaces associated with tensor encoding attributes.
//!
//! - include
//!   - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/TensorEncoding.h>
//!   - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/TensorEncoding.td>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/TensorEncoding.cpp>

use crate::mlir::ir::{
    affine_map,
    operation::definition
};

/// Verifies an encoding attribute for a tensor.
pub trait VerifiableTensorEncoding: AttributeInterface {
    /// Verifies the encoding is valid for a tensor type with the given shape and element type. Generates a diagnostic using the supplied callback on failure.
    fn verify_encoding(
        &self,
        shape: &[u64],
        element_type: Type,
        emit_error: fn() -> Diagnostic
    ) -> LogicalResult;
}
