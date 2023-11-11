/*!
- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/simple_ilist.h>
*/

use crate::llvm::{
    adt::ilist::{
      base,
      iterator,
      node,
      node::options
    },
    support::compiler
};

/**
A simple intrusive list implementation.

This is a simple intrusive list for a `T` that inherits from `ilist_node<T>`.  The list never takes ownership of anything inserted in it.

Unlike `iplist<T>` and `ilist<T>`, `simple_ilist<T>` never deletes values, and has no callback traits.

The API for adding nodes include `push_front()`, `push_back()`, and `insert()`.  These all take values by reference (not by pointer), except for the range version of `insert()`.

There are three sets of API for discarding nodes from the list: `remove()`, which takes a reference to the node to remove, `erase()`, which takes an iterator or iterator range and returns the next one, and `clear()`, which empties out the container.  All three are constant time operations.  None of these deletes any nodes; in particular, if there is a single node in the list, then these have identical semantics:

- `L.remove(L.front());`
- `L.erase(L.begin());`
- `L.clear();`

As a convenience for callers, there are parallel APIs that take a `Disposer` (such as `std::default_delete<T>`): `removeAndDispose()`, `eraseAndDispose()`, and `clearAndDispose()`. These have different names because the extra semantic is otherwise non-obvious. They are equivalent
to calling `std::for_each()` on the range to be discarded.

The currently available \p Options customize the nodes in the list.  The same options must be specified in the `ilist_node` instantiation for compatibility (although the order is irrelevant).
- Use `ilist_tag` to designate which ilist_node for a given `T` this list should use.  This is useful if a type `T` is part of multiple, independent lists simultaneously.
- Use `ilist_sentinel_tracking` to always (or never) track whether a node is a sentinel.  Specifying `true` enables the `ilist_node::isSentinel()` API. Unlike `ilist_node::isKnownSentinel()`, which is only appropriate for assertions, `ilist_node::isSentinel()` is appropriate for real logic.

Here are examples of `Options` usage:
- `simple_ilist<T>` gives the defaults.
- `simple_ilist<T,ilist_sentinel_tracking<true>>` enables the `ilist_node::isSentinel()` API.
- `simple_ilist<T,ilist_tag<A>,ilist_sentinel_tracking<false>>` specifies a tag of A and that tracking should be off (even when LLVM_ENABLE_ABI_BREAKING_CHECKS are enabled).
- `simple_ilist<T,ilist_sentinel_tracking<false>,ilist_tag<A>>` is equivalent to the last.

See `is_valid_option` for steps on adding a new option.
*/
pub struct SimpleIList<T, Options> {
    _unimplemented: std::marker::PhantomData<T>,
    _unimplemented1: std::marker::PhantomData<Options>
}
