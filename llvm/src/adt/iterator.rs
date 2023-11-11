/*!
# Utilities for Using and Defining Iterators

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/iterator.h>
*/

use std::{
    cmp::PartialEq,
    iter::Iterator,
    ops::{Deref, Index}
};

use crate::llvm::adt::iterator_range;

/**
CRTP base class which implements the entire standard iterator facade in terms of a minimal subset of the interface.

Use this when it is reasonable to implement most of the iterator functionality in terms of a core subset. If you need special behaviour or there are performance implications for this, you may want to override the relevant members instead.

Note, one abstraction that this does *not* provide is implementing subtraction in terms of addition by negating the difference. Negation isn't always information preserving, and I can see very reasonable iterator designs where this doesn't work well. It doesn't really force much added boilerplate anyways.

Another abstraction that this doesn't provide is implementing increment in terms of addition of one. These aren't equivalent for all iterator categories, and respecting that adds a lot of complexity for little gain.

Iterators are expected to have const rules analogous to pointers, with a single, const-qualified operator*() that returns ReferenceT. This matches the second and third pointers in the following example:

```cpp
int Value;
{ int *I = &Value; }             // ReferenceT 'int&'
{ int *const I = &Value; }       // ReferenceT 'int&'; const
{ const int *I = &Value; }       // ReferenceT 'const int&'
{ const int *const I = &Value; } // ReferenceT 'const int&'; const
```

If an iterator facade returns a handle to its own state, then T (and PointerT and ReferenceT) should usually be const-qualified. Otherwise, if clients are expected to modify the handle itself, the field can be declared mutable or use const_cast.

Classes wishing to use `iterator_facade_base` should implement the following methods:

Forward Iterators:

(All of the following methods)

- DerivedT &operator=(const DerivedT &R);
- bool operator==(const DerivedT &R) const;
- T &operator*() const;
- DerivedT &operator++();

Bidirectional Iterators:

(All methods of forward iterators, plus the following)

- DerivedT &operator--();

Random-Sccess Iterators:

(All methods of bidirectional iterators excluding the following)

- DerivedT &operator++();
- DerivedT &operator--();
(and plus the following)
- bool operator<(const DerivedT &RHS) const;
- DifferenceTypeT operator-(const DerivedT &R) const;
- DerivedT &operator+=(DifferenceTypeT N);
- DerivedT &operator-=(DifferenceTypeT N);

*/
pub trait IteratorFacade: Iterator {
    
}

/**
A proxy object for computing a reference via indirecting a copy of an iterator. This is used in APIs which need to produce a reference via indirection but for which the iterator object might be a temporary. The proxy preserves the iterator internally and exposes the indirected reference via a conversion operator.
*/
struct ReferenceProxy<I> {
    iterator: I
}

/**
A proxy object for computing a pointer via indirecting a copy of a reference. This is used in APIs which need to produce a pointer but for which the reference might be a temporary. The proxy preserves the reference internally and exposes the pointer via a arrow operator.
*/
struct PointerProxy<T> {
    item: T
}

/**
CRTP base class for adapting an iterator to a different type.

This class can be used through CRTP to adapt one iterator into another.
Typically this is done through providing in the derived class a custom `operator*` implementation. Other methods can be overridden as well.
*/
pub trait IteratorAdaptor: IteratorFacade {
}

/**
An iterator type that allows iterating over the pointees via some other iterator.

The typical usage of this is to expose a type that iterates over Ts, but which is implemented with some iterator over T*s:

```cpp
using iterator = pointee_iterator<SmallVector<T *>::iterator>;
```
*/
pub trait PointeeIterator: IteratorAdaptor {

}

pub trait PointerIterator: IteratorAdaptor {

}
