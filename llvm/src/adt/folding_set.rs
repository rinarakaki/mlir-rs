/*!
# Uniquing Hash Set

This file defines a hash set that can be used to remove duplication of nodes in a graph.  This code was originally created by Chris Lattner for use with SelectionDAGCSEMap, but was isolated to provide use across the llvm code set.

- include <https://github.com/llvm/llvm-project/tree/main/llvm/include/llvm/ADT/FoldingSet.h>
*/

use crate::llvm::adt::small_vector::SmallVector;

/**
Implements the folding set functionality. The main structure is an array of buckets.  Each bucket is indexed by the hash of the nodes it contains.  The bucket itself points to the nodes contained in the bucket via a singly linked list. The last node in the list points back to the bucket to facilitate node removal.
*/
pub struct FoldingSetBase;

/**
This class is used to gather all the unique data bits of a node. When all the bits are gathered this class is used to produce a hash value for the node.
*/
pub struct FoldingSetNodeID {
    bits: SmallVector<[usize; 32]>
}

/**
This template class is used to instantiate a specialised implementation of the folding set to the node class T.  T must be a subclass of FoldingSetNode and implement a Profile function.

Note that this set type is movable and move-assignable. However, its moved-from state is not a valid state for anything other than move-assigning and destroying. This is primarily to enable movable APIs that incorporate these objects.
*/
pub struct FoldingSet<T>;

/**
This template class is a further refinement of FoldingSet which provides a context argument when calling `Profile` on its nodes. Currently, that argument is fixed at initialisation time.

T must be a subclass of FoldingSetNode and implement a Profile
function with signature
  void Profile(FoldingSetNodeID &, Ctx);
*/
pub struct ContextualFoldingSet<T, Context>;

/**
This template class combines a FoldingSet and a vector to provide the interface of FoldingSet but with deterministic iteration order based on the insertion order. T must be a subclass of FoldingSetNode and implement a Profile function.
*/
pub struct FoldingSetVector<
    T,
    Vector = SmallVector<[T; 8]>
> {
    set: FoldingSet<T>,
    vector: Vector
}

/**
This is the common iterator support shared by all folding sets, which knows how to walk the folding set hash table.
*/
pub struct FoldingSetIterator;

/**
This is the common bucket iterator support shared by all folding sets, which knows how to walk a particular bucket of a folding set hash table.
*/
pub struct FoldingSetBucketIterator;
