/*!
This header is not part of the public API.  It is placed in the includes directory only because that's required by the implementations of template-classes.

This file defines an extremely lightweight API for fatal errors (not arising from assertions).  The API does not attempt to be sophisticated in any way, it's just the usual "I give up" style of error reporting.

This file is part of the lightweight runtime support library for sparse tensor manipulations.  The functionality of the support library is meant to simplify benchmarking, testing, and debugging MLIR code operating on sparse tensors.  However, the provided functionality is **not** part of core MLIR itself.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/ExecutionEngine/SparseTensor/ErrorHandling.h>
*/
