/*!
# Side Effect Base

This file contains base class definitions for side effect interfaces, i.e. the customisable interfaces that provide information about which effects are  applied by an operation.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/SideEffectInterfaceBase.td>
*/

// A generic resource that can be attached to a general base side effect.
pub struct Resource {

}

/**
A base interface used to query information about the side effects applied to an operation. This template class takes the name of the derived interface class, as well as the name of the base effect class.
*/
pub trait EffectOpInterfaceBase {

}
