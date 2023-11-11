/*!
# Run all IR through a functor

Run all IR passed in through a user supplied functor.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ExecutionEngine/Orc/IRTransformLayer.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/ExecutionEngine/Orc/IRTransformLayer.cpp>
*/

use crate::llvm::execution_engine::orc::layer::IRLayer;

/**
A layer that applies a transform to emitted modules.
The transform function is responsible for locking the ThreadSafeContext before operating on the module.
*/
pub struct IRTransformLayer {
  
}
