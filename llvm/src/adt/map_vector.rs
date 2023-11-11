/*!
This file implements a map that provides insertion order iteration. The
interface is purposefully minimal. The key is assumed to be cheap to copy
and 2 copies are kept, one for indexing in a DenseMap, one for iteration in
a std::vector.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/MapVector.h>
*/

use crate::llvm::adt::{
    dense_map::DenseMap,
    small_vector
};


/// This class implements a map that also provides access to all stored values
/// in a deterministic order. The values are kept in a std::vector and the
/// mapping is done with DenseMap from Keys to indexes in that vector.
pub struct MapVector<
    K, V,
    Map = DenseMap<K, usize>,
    Vector = Vec<(K, V)>
> {
    _unimplemented: std::marker::PhantomData<K>,
    _unimplemented1: std::marker::PhantomData<V>,
    _unimplemented2: std::marker::PhantomData<Map>,
    _unimplemented3: std::marker::PhantomData<Vector>
}
