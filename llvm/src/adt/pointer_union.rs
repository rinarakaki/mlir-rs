/*!
# Discriminated Union of Two Pointers

This file defines the PointerUnion class, which is a discriminated union of pointer types.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/PointerUnion.h>
*/

use crate::llvm::{
    adt::{
        dense_map_info,
        pointer_int_pair,
        stl_extras
    },
    support::{
        casting,
        pointer_like_type_traits
    }
};

/**
A discriminated union of two or more pointer types, with the discriminator in the low bit of the pointer.

This implementation is extremely efficient in space due to leveraging the low bits of the pointer, while exposing a natural and type-safe API.

Common use patterns would be something like this:

```cpp
let P = PointerUnion<*mut int, *mut float>;
P = (int*)0;
printf("%d %d", P.is<int*>(), P.is<float*>());  // prints "1 0"
X = P.get<int*>();     // ok.
Y = P.get<float*>();   // runtime assertion failure.
Z = P.get<double*>();  // compile time failure.
P = (float*)0;
Y = P.get<float*>();   // ok.
X = P.get<int*>();     // runtime assertion failure.
PointerUnion<int*, int*> Q; // compile time failure.
```
*/
pub struct PointerUnion<P0, P1> {
    _unimplemented: std::marker::PhantomData<P0>,
    _unimplemented1: std::marker::PhantomData<P1>
}
