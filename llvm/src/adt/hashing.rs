/*!
# Utilities for Hashing

This file implements the newly proposed standard C++ interfaces for hashing arbitrary data and building hash functions for user-defined types. This interface was originally proposed in N3333[1] and is currently under review for inclusion in a future TR and/or standard.

The primary interfaces provide are comprised of one type and three functions:

- `hash_code` class is an opaque type representing the hash code for some data. It is the intended product of hashing, and can be used to implement hash tables, checksumming, and other common uses of hashes. It is not an integer type (although it can be converted to one) because it is risky to assume much about the internals of a `hash_code`. In particular, each execution of the program has a high probability of producing a different `hash_code` for a given input. Thus their values are not stable to save or persist, and should only be used during the execution for the construction of hashing datastructures.

- `hash_value` is a function designed to be overloaded for each user-defined type which wishes to be used within a hashing context. It should be overloaded within the user-defined type's namespace and found via ADL. Overloads for primitive types are provided by this library.

- `hash_combine` and `hash_combine_range` are functions designed to aid programmers in easily and intuitively combining a set of data into a single `hash_code` for their object. They should only logically be used within the implementation of a `hash_value` routine or similar context.

Note that `hash_combine_range` contains very special logic for hashing a contiguous array of integers or pointers. This logic is *extremely* fast, on a modern Intel "Gainestown" Xeon (Nehalem uarch) @2.2 GHz, these were benchmarked at over 6.5 GiB/s for large keys, and <20 cycles/hash for keys under 32-bytes.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/Hashing.h>
*/
  
use crate::llvm::support::{
    data_types,
    error_handling,
    swap_byte_order,
    type_traits
};

/**
An opaque object representing a hash code.

This object represents the result of hashing some entity. It is intended to be used to implement hashtables or other hashing-based data structures.
While it wraps and exposes a numeric value, this value should not be trusted to be stable or predictable across processes or executions.

In order to obtain the `hash_code` for an object `x`:

```cpp
using llvm::hash_value;
llvm::hash_code code = hash_value(x);
```
*/
pub struct HashCode {
    value: usize
}
