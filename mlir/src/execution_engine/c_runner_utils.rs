/*!
# Utils for Debugging MLIR Execution

This file declares basic classes and functions to manipulate structured MLIR types at runtime. Entities in this file must be compliant with C++11 and be retargetable, including on targets without a C++ runtime.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/ExecutionEngine/CRunnerUtils.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/ExecutionEngine/CRunnerUtils.cpp>
*/

use std::ops::Index;

pub const fn is_power_of2(n: isize) -> bool {
    (!(n & (n - 1)))
}

pub const fn next_power_of2(n: isize) -> usize {
    if n <= 1 {
        1
    } else if is_power_of2(n) {
        n
    } else {
        2 * next_power_of2((n + 1) / 2)
    }
}

/// StridedMemRef descriptor type with static rank.
pub struct StridedMemRefType<T, const N: isize> {
    base_ptr: *mut T,
    data: *mut T,
    offset: i64,
    sizes: [i64; N],
    strides: [i64; N]
}
