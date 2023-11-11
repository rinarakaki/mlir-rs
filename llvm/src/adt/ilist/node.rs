/*!
- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/ilist_node.h>
*/

pub mod base;
pub mod options;

/**
An intrusive list node.

A base class to enable membership in intrusive lists, including `simple_ilist`, `iplist`, and `ilist`. The first template parameter is the `value_type` for the list.

An ilist node can be configured with compile-time options to change behaviour and/or add API.

By default, an `ilist_node` knows whether it is the list sentinel (an instance of `ilist_sentinel`) if and only if LLVM_ENABLE_ABI_BREAKING_CHECKS.  The function `isKnownSentinel()` always returns `false` tracking is off.  Sentinel tracking steals a bit from the "prev" link, which adds a mask operation when decrementing an iterator, but enables bug-finding assertions in `ilist_iterator`.

To turn sentinel tracking on all the time, pass in the `ilist_sentinel_tracking<true>` template parameter. This also enables the `isSentinel()` function. The same option must be passed to the intrusive list.  (`ilist_sentinel_tracking<false>` turns sentinel tracking off all the time.)

A type can inherit from ilist_node multiple times by passing in different `ilist_tag` options.  This allows a single instance to be inserted into multiple lists simultaneously, where each list is given the same tag.

```cpp
struct A {};
struct B {};
struct N : ilist_node<N, ilist_tag<A>>, ilist_node<N, ilist_tag<B>> {};

void foo() {
  simple_ilist<N, ilist_tag<A>> ListA;
  simple_ilist<N, ilist_tag<B>> ListB;
  N N1;
  ListA.push_back(N1);
  ListB.push_back(N1);
}
```

See `is_valid_option` for steps on adding a new option.
*/
pub struct Node<T> {}

/**
An access class for ilist_node private API.

This gives access to the private parts of ilist nodes. Nodes for an ilist should friend this class if they inherit privately from ilist_node.

Using this class outside of the ilist implementation is unsupported.
*/
struct NodeAccess {
}
