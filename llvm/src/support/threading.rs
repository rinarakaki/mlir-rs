/*!
This file declares helper functions for running LLVM in a multi-threaded
environment.

This file defines helper functions for running LLVM in a multi-threaded
environment.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/Threading.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/Threading.cpp>
*/

use crate::llvm::{
    adt::{
        bit_vector,
    },
    config::{
        config,
        llvm_config
    },
    support::compiler
};
