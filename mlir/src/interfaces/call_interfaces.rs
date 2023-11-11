/*!
This file contains the definitions of the call interfaces defined in `CallInterfaces.td`.

This file contains a set of interfaces that can be used to define information related to call-like and callable operations. Each of which are defined along with the respective interface below.

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/CallInterfaces.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/CallInterfaces.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Interfaces/CallInterfaces.cpp>
*/

use crate::{
    mlir::ir::{
        operation::{
            Operation,
            base::{OpInterface, SymbolRefAttr}
        },
        region::Region,
        symbol_table,
        r#type::Type,
        value::Value
    },
    llvm::adt::pointer_union::PointerUnion
};

/**
Interface for call-like operations.

A call-like operation is one that transfers control from one sub-routine to another. These operations may be traditional direct calls `call @foo`, or indirect calls to other operations `call_indirect %foo`. An operation that uses this interface, must *not* also provide the `CallableOpInterface`.
*/
pub trait CallOpInterface : OpInterface {
    /**
    Returns the callee of this call-like operation. A `callee` is either a reference to a symbol, via SymbolRefAttr, or a reference to a defined SSA value. If the reference is an SSA value, the SSA value corresponds to a region of a lambda-like operation.
    */
    fn callable_for_callee(&self) -> dyn CallInterfaceCallable;
  
    /**
    Returns the operands within this call that are used as arguments to the callee.
    */
    fn arg_operands(&self) -> Operation::operand_range;
}

/**
Interface for callable operations.

A callable operation is one who represents a potential sub-routine, and may be a target for a call-like operation (those providing the CallOpInterface above). These operations may be traditional functional operation `func @foo(...)`, as well as function producing operations `%foo = dialect.create_function(...)`. These operations may only contain a single region, or subroutine.
*/
pub trait CallableOpInterface: OpInterface {
    /**
    Returns the region on the current operation that is callable. This may return null in the case of an external callable object, e.g. an external function.
    */
    fn callable_region(&self) -> Option<Region>;
  
    /**
    Returns the results types that the callable region produces when executed.
    */
    fn callable_results(&self) -> &[dyn Type];
}

/**
A callable is either a symbol, or an SSA value, that is referenced by a
call-like operation. This represents the destination of the call.
*/
pub trait CallInterfaceCallable: PointerUnion<SymbolRefAttr, Value> {
//     type PointerUnion<SymbolRefAttr, Value>::PointerUnion;
}
