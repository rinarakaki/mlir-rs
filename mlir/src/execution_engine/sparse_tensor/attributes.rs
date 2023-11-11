/*!
This header defines various macros for using C++ attributes whenever they're supported by the compiler.  These macros are the same as the versions in the LLVMSupport library, but we define our own versions in order to avoid introducing that dependency just for the sake of these macros.  (If we ever do end up depending on LLVMSupport, then we should remove this header and use "llvm/Support/Compiler.h" instead.)

This file is part of the lightweight runtime support library for sparse tensor manipulations.  The functionality of the support library is meant to simplify benchmarking, testing, and debugging MLIR code operating on sparse tensors.  However, the provided functionality is **not** part of core MLIR itself.

- include <https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/ExecutionEngine/SparseTensor/Attributes.h>
*/
