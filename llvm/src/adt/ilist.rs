/*!
# Intrusive Linked List Template

This file defines classes to implement an intrusive doubly linked list class (i.e. each node of the list must contain a next and previous field for the list.

The ilist class itself should be a plug in replacement for list. This list replacement does not provide a constant time size() method, so be careful to use empty() when you really want to know if it's empty.

The ilist class is implemented as a circular list.  The list itself contains a sentinel node, whose Next points at begin() and whose Prev points at rbegin(). The sentinel node itself serves as end() and rend().

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/ilist.h>
*/

pub mod base;
pub mod iterator;
pub mod node;
pub mod simple_ilist;

/**
A wrapper around an intrusive list with callbacks and non-intrusive ownership.

This wraps a purely intrusive list (like simple_ilist) with a configurable traits class.  The traits can implement callbacks and customise the ownership semantics.

This is a subset of ilist functionality that can safely be used on nodes of polymorphic types, i.e. a heterogeneous list with a common base class that holds the next/prev pointers.  The only state of the list itself is an ilist_sentinel, which holds pointers to the first and last nodes in the list.
*/
pub struct IPList<IntrusiveList, Traits> {
    _unimplemented: std::marker::PhantomData<IntrusiveList>,
    _unimplemented1: std::marker::PhantomData<Traits>
}

/**
An intrusive list with ownership and callbacks specified/controlled by ilist_traits, only with API safe for polymorphic types.

The `Options` parameters are the same as those for `simple_ilist`. See there for a description of what's available.
*/
pub struct IList<T> {
    _unimplemented: std::marker::PhantomData<T>
}
