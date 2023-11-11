/*!
# Cast Interfaces for Operations

This file contains the definitions of the cast interfaces defined in
`CastInterfaces.td`.

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/CastInterfaces.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/CastInterfaces.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Interfaces/CastInterfaces.cpp>
*/

use crate::mlir::ir::operation::definition;

/**
A cast-like operation is one that converts from a set of input types to a set of output types. The arity of the inputs may be from 0-N, whereas the arity of the outputs may be anything from 1-N. Cast-like operations are trivially removable in cases where they produce an No-op, i.e when the input types and output types match 1-1.
*/
pub trait CastOpInterface: OpInterface {
    /**
    Returns true if the given set of input and result types are compatible to cast using this cast operation.
    */
    fn are_cast_compatible(
        inputs: TypeRange,
        outputs: TypeRange
    ) -> bool;

    /// Attempt to fold the given cast operation.
    fn foldTrait(
        operation: *mut Operation,
        operands: &[Attribute],
        results: &SmallVector<[FoldResult]>
    ) -> LogicalResult {
        impl::fold_cast_interface_op(operation, operands, results)
    }

//   let verify = [{
//     return impl::verifyCastInterfaceOp($_op, ConcreteOp::are_cast_compatible);
//   }];
}
