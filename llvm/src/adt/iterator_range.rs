/*!
# A Tange Adaptor for Iterators

This provides a very simple, boring adaptor for a begin and end iterator into a range type. This should be used to build range views that work well with range based for loops and range based constructors.

Note that code here follows more standards-based coding conventions as it is mirroring proposed interfaces for standardisation.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/iterator_range.h>
*/


/**
A range adaptor for a pair of iterators.

This just wraps two iterators into a range-compatible interface. Nothing fancy at all.
*/
pub struct IteratorRange<Iterator> {

} 
