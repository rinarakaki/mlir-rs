/*!
# Utilities for Handling Success/Failure

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Support/LogicalResult.h>
*/

// use crate::{
//     mlir::support::llvm,
// };

#[must_use]
pub type LogicalResult = Result<(), Box<dyn std::error::Error>>;

/**
This class provides support for representing a failure result, or a valid value of type `T`. This allows for integrating with LogicalResult, while also providing a value on the success path.
*/
#[must_use]
pub struct FailureOr<T> {
    _unimplemented: std::marker::PhantomData<T>
}
