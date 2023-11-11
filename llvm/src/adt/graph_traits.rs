/*!
This file defines the little `GraphTraits<X>` template class that should be
specialised by classes that want to be iteratable by generic graph
iterators.

This file also defines the marker class Inverse that is used to iterate over
graphs in a graph defined, inverse ordering...

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/GraphTraits.h>
*/

/**
GraphTraits - This class should be specialised by different graph types...
which is why the default version is empty.

This template evolved from supporting `BasicBlock` to also later supporting
more complex types (e.g. CFG and DomTree).

GraphTraits can be used to create a view over a graph interpreting it
differently without requiring a copy of the original graph. This could
be achieved by carrying more data in NodeRef. See LoopBodyTraits for one
example.
*/
pub struct GraphTraits<Graph> {
    _unimplemented: std::marker::PhantomData<Graph>,
}
