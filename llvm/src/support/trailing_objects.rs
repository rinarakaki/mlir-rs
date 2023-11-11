/*!
# Variable-Length Classes

This header defines support for implementing classes that have some trailing object (or arrays of objects) appended to them. The main purpose is to make it obvious where this idiom is being used, and to make the usage more idiomatic and more difficult to get wrong.

The TrailingObject template abstracts away the reinterpret_cast, pointer arithmetic, and size calculations used for the allocation and access of appended arrays of objects, and takes care that they are all allocated at their required alignment. Additionally, it ensures that the base type is final -- deriving from a class that expects data appended immediately after it is typically not safe.

Users are expected to derive from this template, and provide numTrailingObjects implementations for each trailing type except the last, e.g. like this sample:

```cpp
class VarLengthObj : private TrailingObjects<VarLengthObj, int, double> {
  friend TrailingObjects;

  unsigned NumInts, NumDoubles;
  size_t numTrailingObjects(OverloadToken<int>) const { return NumInts; }
};
```

You can access the appended arrays via 'getTrailingObjects', and determine the size needed for allocation via `additionalSizeToAlloc` and `totalSizeToAlloc`.

All the methods implemented by this class are are intended for use by the implementation of the class, not as part of its interface (thus, private inheritance is suggested).

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/TrailingObjects.h>
*/

#[allow(non_camel_case_types)]
type int = i32;

use crate::llvm::support::{
    align_of,
    alignment,
    compiler,
    math_extras,
    type_traits
};

/// The base class for TrailingObjects* classes.
pub struct TrailingObjectsBase {

}

/**
OverloadToken's purpose is to allow specifying function overloads for different types, without actually taking the types as parameters. (Necessary because member function templates cannot be specialised, so overloads must be used instead of specialisation.)
*/
struct OverloadToken<T> {}

/**
TrailingObjectsImpl is somewhat complicated, because it is a recursively inheriting template, in order to handle the template varargs. Each level of inheritance picks off a single trailing type then recurses on the rest. The `Align`, `BaseTy`, and `TopTrailingObj` arguments are passed through unchanged through the recursion. `PrevTy` is, at each level, the type handled by the level right above it.
*/
pub struct TrailingObjects<
    const Align: int,
    BaseTy,
    TopTrailingObj,
    PrevTy,
> {

}

/// See the file comment for details on the usage of the
/// TrailingObjects type.
pub struct TrailingObjects<Base, TrailingTys> {

}
