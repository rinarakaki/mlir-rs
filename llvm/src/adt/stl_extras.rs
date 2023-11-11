/*!
# Useful STL Related Functions

This file contains some templates that are useful if you are working with the STL at all.

No library is required when using these functions.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/STLExtras.h>
*/

use std::{
    io::Write,
    iter::Iterator,
    convert::Into
};

use crate::llvm::{
    adt::{
        array_ref,
        stl_forward_compat,
        stl_functional_extras,
        identity,
        iterator::IteratorAdaptor,
        iterator_range
    },
    config::abi_breaking,
    support::error_handling
};

// L379
/// Returns true if the given container only contains a single element.
pub fn has_single_element<T: Iterator>(iterator: &T) -> bool{
    iterator.count() == 1
}

// L443
/**
A base type of mapped iterator, that is useful for building derived iterators that do not need/want to store the map function (as in mapped_iterator). These iterators must simply provide a `mapElement` method that defines how to map a value of the iterator to the provided reference type.
*/
impl<I: IteratorAdaptor> MappedIterator for I {
    fn current(&self) -> I::Item {
        
    }

    // fn operator*() -> &I::Item {
    //     return static_cast<&DerivedT>(*this).map_element(*this.I);
    // }
}

// L1400
/**
This class provides an implementation of a range of indexed_accessor_iterators where the base is not indexable. Ranges with bases that are offsetable should derive from indexed_accessor_range_base instead. Derived range classes are expected to implement the following static method:

- ReferenceT dereference(const BaseT &base, ptrdiff_t index)
    - Dereference an iterator pointing to a parent base at the given index.
*/
pub trait IndexedAccessorRange {
}

/**
An STL-style algorithm similar to std::for_each that applies a second functor between every pair of elements.

This provides the control flow logic to, for example, print a comma-separated list:

```
interleave(
    names.iter(),
    |name: &str| { os << name; },
    || { os << ", "; }
);
```
*/
#[inline]
pub fn interleave<
    I: Iterator,
    UnaryFunctor: Fn(Iterator::Item) + Into<&str>,
    NullaryFunctor: Fn() + Into<&str>,
>(
    iterator: I,
    each_fn: UnaryFunctor,
    between_fn: NullaryFunctor
) {
    if iterator.count() = 0 {
        return;
    }
    
    each_fn(iterator.next());
    for item in iterator {
        between_fn();
        each_fn(item);
    }
}

/// Overload of interleave for the common case of string separator.
#[inline]
pub fn interleave__<
    I: Iterator,
    W: Write,
    UnaryFunctor: Fn(I::Item),
    T = Iterator::Item
>(
    iteraor: &I,
    writer: &W,
    each_fn: UnaryFunctor,
    separator: &str
) {
    interleave(iteraor, each_fn, || { writer.write(buf); });
}

#[inline]
pub fn interleave_comma_<
    I: Iterator,
    W: Write,
    UnaryFunctor: Fn(I::Item),
>(iterator: &I, writer: &W, each_fn: UnaryFunctor) {
    interleave__(iterator, writer, each_fn, ", ");
}

// L2100
#[inline]
pub fn interleave_comma<I: Iterator, W: Write>(iterator: &I, writer: &W)
where
    I::Item: Into<&[u8]>
{
    interleave_comma_(iterator, writer, |item: &I::Item| writer.write(item));
}
