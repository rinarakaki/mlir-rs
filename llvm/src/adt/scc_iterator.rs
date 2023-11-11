/*!
This builds on the llvm/ADT/GraphTraits.h file to find the strongly
connected components (SCCs) of a graph in O(N+E) time using Tarjan's DFS
algorithm.

The SCC iterator has the important property that if a node in SCC S1 has an
edge to a node in SCC S2, then it visits S1 *after* S2.

To visit S1 *before* S2, use the scc_iterator on the Inverse graph. (NOTE:
This requires some simple wrappers and is not supported yet.)

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/SCCIterator.h>
*/

use crate::llvm::adt::{
    dense_map,
    graph_traits,
    iterator
};
