/*!
# ThreadLocalCache class

This file contains a definition of the ThreadLocalCache class. This class  provides support for defining thread local objects with non-static duration.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Support/ThreadLocalCache.h>
*/

use crate::{
    mlir::support::llvm,
    llvm::{
        adt::{
            dense_map::SmallDenseMap,
            small_vector::SmallVector
        },
        support::{
            managed_static,
            mutex::SmartMutex,
            thread_local
        }
    }
};

/**
This class provides support for defining a thread local object with non static storage duration. This is very useful for situations in which a data cache has very large lock contention.
*/
pub struct ThreadLocalCache<T> {
    /**
    Owning pointers to all of the values that have been constructed for this object in the static cache.
    */
    instances: SmallVector<std::shared_ptr<T>, 1>,

    /**
    A mutex used when a new thread instance has been added to the cache for this object.
    */
    instance_mutex: SmartMutex<true>
}

/**
The type used for the static thread_local cache. This is a map between an instance of the non-static cache and a weak reference to an instance of ValueT. We use a weak reference here so that the object can be destroyed without needing to lock access to the cache itself.
*/
struct CacheType {

}
