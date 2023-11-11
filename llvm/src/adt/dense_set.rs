/*!
# Dense Probed Hash Table

This file defines the DenseSet and SmallDenseSet classes.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/DenseSet.h>
*/

use crate::llvm::{
    adt::{
        dense_map::{DenseMap, SmallDenseMap},
        dense_map_info::DenseMapInfo
    },
    support::{
        math_extras,
        type_traits
    }
};

// Use the empty base class trick so we can create a DenseMap where the buckets
// contain only a single item.
pub struct DenseSetPair<K> {
    key: K
}

/// Implements a dense probed hash-table based set.
pub struct DenseSet<T, ValueInfo = DenseMapInfo<T>> {
    map: DenseMap<T, DenseSetEmpty, ValueInfo, DenseSetPair<T>>,
}

pub struct ConstIterator {
}

/**
Implements a dense probed hash-table based set with some number of buckets stored inline.
*/
pub type SmallDenseSet<
    T,
    const INLINE_BUCKETS: usize = 4,
    ValueInfo = DenseMapInfo<T>
> = DenseSet<
    T,
    SmallDenseMap<T, DenseSetEmpty, INLINE_BUCKETS, ValueInfo, DenseSetPair<T>>,
    ValueInfo>;
