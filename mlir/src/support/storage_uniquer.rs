/*!
# Common Storage Class Uniquer

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Support/StorageUniquer.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Support/StorageUniquer.cpp>
*/

#[allow(non_camel_case_types)]
type unsigned = u16;

use std::any::TypeId;

use crate::{
    mlir::support::{
        llvm,
        thread_local_cache::ThreadLocalCache,
    },
    llvm::{
        adt::{
            dense_map::DenseMap,
            dense_set::DenseSet,
        },
        support::{
            allocator::BumpPtrAllocator,
            rw_mutex::SmartRWMutex
        }
    }
};

/**
A utility class to get or create instances of 'storage classes'. These storage classes must derive from 'StorageUniquer::Storage'.

For non-parametric storage classes, i.e. singleton classes, nothing else is needed. Instances of these classes can be created by calling `get` without trailing arguments.

Otherwise, the parametric storage classes may be created with `get`, and must respect the following:

- Define a type alias, KeyTy, to a type that uniquely identifies the
     instance of the storage class.
     * The key type must be constructible from the values passed into the
       getComplex call.
     * If the KeyTy does not have an llvm::DenseMapInfo specialisation, the
       storage class must define a hashing method:
        'static unsigned hashKey(const KeyTy &)'

- Provide a method, 'bool operator==(const KeyTy &) const', to
     compare the storage instance against an instance of the key type.

- Provide a static construction method:
       `DerivedStorage *construct(StorageAllocator &, const KeyTy &key)`
     that builds a unique instance of the derived storage. The arguments to
     this function are an allocator to store any uniqued data and the key
     type for this storage.

- Provide a cleanup method:
       `void cleanup()`
     that is called when erasing a storage instance. This should cleanup any
     fields of the storage as necessary and not attempt to free the memory
     of the storage itself.

Storage classes may have an optional mutable component, which must not take part in the unique immutable key. In this case, storage classes may be mutated with `mutate` and must additionally respect the following:

- Provide a mutation method:
       `LogicalResult mutate(StorageAllocator &, <...>)`
     that is called when mutating a storage instance. The first argument is
     an allocator to store any mutable data, and the remaining arguments are
     forwarded from the call site. The storage can be mutated at any time
     after creation. Care must be taken to avoid excessive mutation since
     the allocated storage can keep containing previous states. The return
     value of the function is used to indicate whether the mutation was
     successful, e.g., to limit the number of mutations or enable deferred
     one-time assignment of the mutable component.

All storage classes must be registered with the uniquer via `registerParametricStorageType` or `registerSingletonStorageType` using an appropriate unique `TypeId` for the storage class.
*/
pub struct StorageUniquer {
    // ==================
    // StorageUniquerImpl
    // ==================
    
    /// Map of type ids to the storage uniquer to use for registered objects.
    parametric_uniquers: DenseMap<TypeId, std::unique_ptr<ParametricStorageUniquer>>,

    /**
    Map of type ids to a singleton instance when the storage class is a singleton.
    */
    singleton_instances: DenseMap<TypeId, Option<Storage>>,

    /// Allocator used for uniquing singleton instances.
    singleton_allocator: StorageAllocator,

    /// Flag specifying if multi-threading is enabled within the uniquer.
    threading_is_enabled: bool,  // = true;
}

/**
This class acts as the base storage that all storage classes must derived from.
*/
#[repr(align(8))]
pub struct Storage {
}

/// This is a utility allocator used to allocate memory for instances of derived types.
pub struct StorageAllocator {
    /// The raw allocator for type storage objects.
    allocator: BumpPtrAllocator
}

/**
This class represents a uniquer for storage instances of a specific type that has parametric storage. It contains all of the necessary data to unique storage instances in a thread safe way. This allows for the main uniquer to bucket each of the individual sub-types removing the need to lock the main uniquer itself.
*/
pub struct ParametricStorageUniquer {
    /**
    A thread local cache for storage objects. This helps to reduce the lock contention when an object already existing in the cache.
    */
    local_cache: ThreadLocalCache<StorageTypeSet>,

    // /**
    // A set of uniquer shards to allow for further bucketing accesses for instances of this storage type. Each shard is lazily initialised to reduce the overhead when only a small amount of shards are in use.
    // */
    // shards: std::unique_ptr<std::atomic<Shard *>[]>,

    /// The number of available shards.
    num_shards: usize,

    /// Function to used to destruct any allocated storage instances.
    destructor_fn: fn(Option<Storage>)  // function_ref,
}

/// A lookup key for derived instances of storage objects.
pub struct LookupKey {
    /// The known hash value of the key.
    hash_value: unsigned,

    /// An equality function for comparing with an existing storage instance.
    is_equal: fn(Option<Storage>) -> bool  // function_ref
}

/**
A utility wrapper object representing a hashed storage object. This class contains a storage object and an existing computed hash value.
*/
struct HashedStorage {
    hash_value: unsigned,
    storage: Option<Storage>
}

/// Storage info for derived TypeStorage objects.
struct StorageKeyInfo {
}

type StorageTypeSet = DenseSet<HashedStorage, StorageKeyInfo>;

/**
This class represents a single shard of the uniquer. The uniquer uses a set of shards to allow for multiple threads to create instances with less lock contention.
*/
struct Shard {
    /// The set containing the allocated storage instances.
    instances: StorageTypeSet,

    /// Allocator to use when constructing derived instances.
    allocator: StorageAllocator,

    // #[cfg(LLVM_ENABLE_THREADS)]
    /// A mutex to keep uniquing thread-safe.
    mutex: SmartRWMutex<true>
}
