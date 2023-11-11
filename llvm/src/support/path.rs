/*!
This file declares the llvm::sys::path namespace. It is designed after
TR2/boost filesystem (v3), but modified to remove exception handling and the
path class.

This file implements the operating system Path API.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/Path.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/Path.cpp>
*/

use crate::llvm::{
    adt::{
        array_ref,
        scope_exit,
        iterator,
        twine
    },
    config::{
        config,
        llvm_config
    },
    support::{
        data_types,
        endian,
        errc,
        error_handling,
        file_system,
        process,
        signals
    }
};
