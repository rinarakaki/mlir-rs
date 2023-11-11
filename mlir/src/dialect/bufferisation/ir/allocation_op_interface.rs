/*!
This file implements the operation interface for allocation ops.

Defines the interface with allocation-related methods. It is used by the
buffer deallocation pass.

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Bufferization/IR/AllocationOpInterface.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Bufferization/IR/AllocationOpInterface.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Bufferization/IR/AllocationOpInterface.cpp>
*/

use crate::mlir::ir::{
    builders::Builder,
    operation::Operation,
    value::Value
  };

/**
This interface provides general allocation-related methods that are designed for allocation operations. For example, it offers the ability to construct associated deallocation and clone operations that are compatible with the current allocation operation.
*/
pub trait AllocationOpInterface {
    /**
    Builds a deallocation operation using the provided builder and the current allocation value (which refers to the current Op implementing this interface). The allocation value is a result of the current operation implementing this interface. If there is no compatible deallocation operation, this method can return ::std::nullopt.
    */
    fn build_dealloc(builder: &Builder, alloc: Value) -> Option<Operation>;

    /**
    Builds a clone operation using the provided builder and the current allocation value (which refers to the current Op implementing this interface). The allocation value is a result of the current operation implementing this interface. If there is no compatible clone operation, this method can return ::std::nullopt.
    */
    fn build_clone(builder: &Builder, alloc: Value) -> Option<Value>;
}
