/*!
# MaskingOpInterface Defs

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Vector/Interfaces/MaskingOpInterface.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Vector/Interfaces/MaskingOpInterface.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Vector/Interfaces/MaskingOpInterface.cpp>
*/

/**
The 'MaskingOpInterface' defines an vector operation that can apply masking to its own or other vector operations.
*/
pub trait MaskingOpInterface: OpInterface {
  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the mask value of this masking operation.",
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getMask",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"">,
    InterfaceMethod<
      /*desc=*/"Returns the operation masked by this masking operation.",
      // TODO: Return a MaskableOpInterface when interface infra can handle
      // dependences between interfaces.
      /*retTy=*/"Operation *",
      /*methodName=*/"getMaskableOp",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"">,
    InterfaceMethod<
      /*desc=*/"Returns true if the masking operation has a passthru value.",
      /*retTy=*/"bool",
      /*methodName=*/"hasPassthru",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"">,
    InterfaceMethod<
      /*desc=*/"Returns the passthru value of this masking operation.",
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getPassthru",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"">,
  ];
}
