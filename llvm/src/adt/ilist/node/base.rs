/*!
- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/ilist_node_base.h>
*/

use std::default::Default;

use crate::llvm::adt::pointer_int_pair;

// type Link<T> = Rc<RefCell<Node<T>>>;

/**
Base class for ilist nodes.

Optionally tracks whether this node is the sentinel.
*/
#[derive(Default)]
pub struct Node {
    prev: Option<Self>,
    next: Option<Self>
}

// public:
//     void setPrev(ilist_node_base *Prev) { this->Prev = Prev; }
//     void setNext(ilist_node_base *Next) { this->Next = Next; }
//     ilist_node_base *getPrev() const { return Prev; }
//     ilist_node_base *getNext() const { return Next; }
// }
