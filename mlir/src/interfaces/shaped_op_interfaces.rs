/*!
# Interfaces for Shaped Operaions

This file implements the operation interface for ops that parallel combining operations.

Defines the interface for ops that perform parallel combining operations.

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/ShapedOpInterfaces.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/ShapedOpInterfaces.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Interfaces/ShapedOpInterfaces.cpp>
*/

// Ops that return the dimension of a shaped value.
def ShapedDimOpInterface : OpInterface<"ShapedDimOpInterface"> {
    let description = [{
        An interface for ops that return the dimension of a shaped value (such as a
        tensor or a memref).  It provides access to the source shaped value and to
        the dimension.
    }];
    let cppNamespace = "::mlir";

    let methods = [
        InterfaceMethod<
        /*desc=*/[{
            Return the shaped value operand. This is the value that the dimension
            is taken from.
        }],
        /*retTy=*/Value,
        /*methodName=*/"getShapedValue",
        /*args=*/(ins)
        >,
        InterfaceMethod<
        /*desc=*/[{
            Return the dimension operand. This can be a constant or an SSA value.
        }],
        /*retTy=*/"::mlir::OpFoldResult",
        /*methodName=*/"getDimension",
        /*args=*/(ins)
        >
    ];

    let verify = [{
        return verifyShapedDimOpInterface($_op);
    }];
}
