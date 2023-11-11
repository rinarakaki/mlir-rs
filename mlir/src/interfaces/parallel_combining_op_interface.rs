/*!
This file implements the operation interface for ops that parallel combining operations.

Defines the interface for ops that perform parallel combining operations.

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/ParallelCombiningOpInterface.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/ParallelCombiningOpInterface.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Interfaces/ParallelCombiningOpInterface.cpp>
*/

/**
A parallel combining op is an op with a region.

This is useful as a terminator to parallel operations that iterate over some set and return tensors while avoiding tight coupling between the iterating op, the combining op and the individual subtensor producing ops.
*/

pub struct ParallelCombiningOpInterface {
  
}
