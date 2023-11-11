/*!
This file defines the DenseMap class.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/DenseMap.h>
*/

use std::{
    cmp::PartialEq,
    iter::Iterator,
    mem::{align_of, size_of},
    ops::Drop
};
  
use crate::llvm::{
    adt::{
        dense_map_info::DenseMapInfo,
        epoch_tracker
    },
    support::{
        align_of::AlignedCharArrayUnion,
        compiler,
        math_extras,
        mem_alloc::{allocate_buffer, deallocate_buffer},
        reverse_iteration,
        // type_traits::
    }
};

// We extend a pair to allow users to override the bucket type with their own
// implementation without requiring two members.
pub struct DenseMapPair<K, V>(K, V);

impl<K, V> DenseMapPair<K, V>
{
    // use (K, V)::pair;

    fn first(&mut self) -> &mut K {
        &mut self.0
    }

    // const fn get_first(&self) -> const &K {
    //     self.0
    // }

    fn second(&mut self) -> &mut V {
        &mut self.1
    }

    // const fn get_second(&self) -> const &V {
    //     self.1
    // }
}

pub struct DenseMap<
    K, V,
    KeyInfo = DenseMapInfo<K>,
    Bucket = DenseMapPair<K, V>
> {
    _unimplemented: std::marker::PhantomData<KeyInfo>,
    buckets: Option<Bucket>,
    num_entries: usize,
    num_tombstones: usize,
    num_buckets: usize
}

impl<K, V, KeyInfo, Bucket> DenseMap<K, V, KeyInfo, Bucket>
{
    // --------------------------------------------------------------------
    // DenseMapBase
    // --------------------------------------------------------------------

    /// lookup - Return the entry for the specified key, or a default
    /// constructed value if no such entry exists.
    pub const fn lookup(&self, value: const_arg_type_t<K>) -> V {
        let bucket = Bucket::new();
        if self.lookup_bucket_for(value, bucket) {
            bucket.second()
        } else {
            V::new()
        }
    }

    /**
    LookupBucketFor - Lookup the appropriate bucket for Val, returning it in
    FoundBucket.  If the bucket contains the key and a value, this returns
    true, otherwise it returns a bucket with an empty marker or tombstone and
    returns false.
    */
    const fn lookup_bucket_for<Key>(
        &self, 
        value: &Key,
        found_bucket: &mut Bucket
    ) -> bool
    {
        let buckets_ptr = self.buckets();
        let num_buckets = self.num_buckets();

        if num_buckets == 0 {
            *found_bucket = None;
            return false;
        }
      
        // FoundTombstone - Keep track of whether we find a tombstone while probing.
        let found_tombstone = None;
        let empty_key = self.empty_key();
        let tombstone_key = self.tombstone_key();
    }

    // --------------------------------------------------------------------
    // DenseMap
    // --------------------------------------------------------------------

    const fn num_entries(&self) -> usize {
        self.num_entries
    }
    
    fn set_num_entries(&self, num: usize) {
        self.num_entries = num;
    }
    
    const fn num_tombstones(&self) -> usize {
        self.num_tombstones
    }
    
    fn set_num_tombstones(&self, num: usize) {
        self.num_tombstones = num;
    }
    
    const fn buckets(&self) -> Option<Bucket> {
        self.buckets;
    }
    
    const fn num_buckets(&self) -> usize {
        self.num_buckets;
    }
    
    fn allocate_buckets(&self, num: usize) -> bool {
        self.num_buckets = num;
        if self.num_buckets == 0 {
            self.buckets = None;
            return false;
        }
    
        let buckets = static_cast::<Option<Bucket>>(
            allocate_buffer(
                size_of::<Bucket>() * self.num_buckets, align_of::<Bucket>()));
        true
    }
}

impl<K, V, KeyInfo, Bucket> Drop for DenseMap<K, V, KeyInfo, Bucket> {
    fn drop(&mut self) {
        self.destroy_all();
        deallocate_buffer(
            self.buckets, size_of::<Bucket>() * self.num_buckets,
            align_of::<Bucket>()
        );
    }
}

pub struct SmallDenseMap {
    
}

pub struct DenseMapIterator {

}
