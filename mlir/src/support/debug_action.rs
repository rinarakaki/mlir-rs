/*!
# Debug Action Support

This file contains definitions for the debug action framework. This framework allows for external entities to control certain actions taken by the compiler by registering handler functions. A debug action handler provides the internal implementation for the various queries on a debug action, such as whether it should execute or not.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Support/DebugAction.h>
*/

/**
This class represents manages debug actions, and orchestrates the communication between action queries and action handlers. An action handler is either an action specific handler, i.e. a derived class of `MyActionType::Handler`, or a generic handler, i.e. a derived class of `DebugActionManager::GenericHandler`. For more details on action specific handlers, see the definition of `DebugAction::Handler` below. For more details on generic handlers, see `DebugActionManager::GenericHandler` below.
*/
pub struct DebugActionManager {

}

/**
A debug action is a specific action that is to be taken by the compiler, that can be toggled and controlled by an external user. There are no constraints on the granularity of an action, it could be as simple as "perform this fold" and as complex as "run this pass pipeline". Via template parameters `ParameterTs`, a user may provide the set of argument types that are provided when handling a query on this action. Derived classes are expected to provide the following:

- static llvm::StringRef getTag()
  - This method returns a unique string identifier, similar to a command
      line flag or DEBUG_TYPE.
- static llvm::StringRef getDescription()
  - This method returns a short description of what the action represents.

This class provides a handler class that can be derived from to handle instances of this action. The parameters to its query methods map 1-1 to the types on the action type.
*/
// template <typename Derived, typename... ParameterTs>
pub struct DebugAction<Derived> {

}
