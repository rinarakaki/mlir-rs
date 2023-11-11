/*!
- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/ilist_base.h>
*/

use crate::llvm::adt::ilist_node_base::Node;

impl Node {
    fn insert_before(next: &Self, n: &Self) {
        let prev = *next.prev();
        n.set_next(&next);
        n.set_prev(&prev);
        prev.set_next(&n);
        next.set_prev(&n);
    }
    
    fn remove(n: &Self) {
        let prev = n.prev();
        let next = n.next();
        next.set_prev(prev);
        prev.set_next(next);
    
        // Not strictly necessary, but helps catch a class of bugs.
        n.set_prev(None);
        n.set_next(None);
    }
}
