/*!
- include <https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/TableGen/Interfaces.h>
- lib <https://github.com/llvm/llvm-project/tree/main/mlir/lib/TableGen/Interfaces.cpp>
*/


/**
Wrapper class with helper methods for accessing InterfaceMethod defined in TableGen.
*/
pub struct InterfaceMethod {
}

 // This struct represents a single method argument.
struct Argument<'a> {
    r#type: &'a str,
    r#name: &'a str
}

/**
Wrapper class with helper methods for accessing Interfaces defined in TableGen.
*/
pub struct Interface {

}
