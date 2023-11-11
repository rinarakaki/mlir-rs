/*!
This file contains the definitions of the branch interfaces defined in `ControlFlowInterfaces.td`.

 - include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/ControlFlowInterfaces.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/ControlFlowInterfaces.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Interfaces/ControlFlowInterfaces.cpp>
*/

use crate::{
    mlir::ir::{
        builtins::types,
        operation::definition
    },
    llvm::adt::small_ptr_set
};

/**
This class models how operands are forwarded to block arguments in control flow. It consists of a number, denoting how many of the successors block arguments are produced by the operation, followed by a range of operands that are forwarded. The produced operands are passed to the first few block arguments of the successor, followed by the forwarded operands.
It is unsupported to pass them in a different order.

An example operation with both of these concepts would be a branch-on-error operation, that internally produces an error object on the error path:

```text
  invoke %function(%0)
    label ^success ^error(%1 : i32)

    ^error(%e: !error, %arg0 : i32):
      ...
```

This operation would return an instance of SuccessorOperands with a produced operand count of 1 (mapped to %e in the successor) and a forwarded operands range consisting of %1 in the example above (mapped to %arg0 in the successor).
*/
pub struct SuccessorOperands {
}

/**
This class represents a successor of a region. A region successor can either be another region, or the parent operation. If the successor is a region, this class represents the destination region, as well as a set of arguments from that region that will be populated when control flows into the region.
If the successor is the parent operation, this class represents an optional set of results that will be populated when control returns to the parent operation.

This interface assumes that the values from the current region that are used to populate the successor inputs are the operands of the return-like terminator operations in the blocks within this region.
*/
pub struct RegionSuccessor {
}

/**
This trait indicates that a terminator operation is "return-like". This means that it exits its current region and forwards its operands as "exit" values to the parent region. Operations with this trait are not permitted to contain successors or produce results.
*/
pub struct ReturnLike<ConcreteType> {
    _unimplemented: std::marker::PhantomData<ConcreteType>
}
