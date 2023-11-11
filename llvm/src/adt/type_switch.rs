/*!
This file implements the TypeSwitch template, which mimics a switch()
statement whose cases are type names.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/TypeSwitch.h>
*/

use crate::llvm::{
    adt::stl_extras,
    support::casting
};

pub struct TypeSwitchBase<DerivedT, T> {
    _unimplemented: std::marker::PhantomData<DerivedT>,
    _unimplemented1: std::marker::PhantomData<T>
}

/**
This class implements a switch-like dispatch statement for a value of 'T'
using dyn_cast functionality. Each `Case<T>` takes a callable to be invoked
if the root value `isa<T>`, the callable is invoked with the result of
`dyn_cast<T>()` as a parameter.

Example:

```cpp
Operation *op = ...;
LogicalResult result = TypeSwitch<Operation *, LogicalResult>(op)
  .Case<ConstantOp>([](ConstantOp op) { ... })
  .Default([](Operation *op) { ... });
```
*/
pub struct TypeSwitch<T, ResultT = ()> {
    /// The pointer to the result of this switch statement, once known,
    /// null before that.
    result: Option<ResultT>,
    _unimplemented: std::marker::PhantomData<T>
}

// impl<T, ResultT> TypeSwitch<T, ResultT> {
//     type BaseT = TypeSwitchBase<TypeSwitch<T, ResultT>, T>;
// }
