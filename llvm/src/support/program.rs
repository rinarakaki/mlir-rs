/*!
This file declares the llvm::sys::Program class.

This file implements the operating system Program concept.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/Program.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/Program.cpp>
*/

use crate::llvm::{
    adt::{
        array_ref,
        string_ref
    },
    config::llvm_config,
    support::{
        error_or,
        file_system,
        raw_ostream
    }
};
