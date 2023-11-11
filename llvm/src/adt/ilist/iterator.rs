/*!
# Intrusive List Iterator

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/ilist_iterator.h>
*/

use crate::llvm::adt::ilist::node::Node;

pub struct IListIterator {
    node_pointer: *mut Node<OptionsT>   // = nullptr;
}
