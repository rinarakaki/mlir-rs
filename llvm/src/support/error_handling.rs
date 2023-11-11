/*!
This file defines an API used to indicate fatal error conditions.  Non-fatal
errors (most of them) should be handled through LLVMContext.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/ErrorHandling.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/ErrorHandling.cpp>
*/
  
// #include "llvm-c/ErrorHandling.h"

use crate::llvm::{
    adt::{
        small_vector,
        twine
    },
    config::config,
    support::{
        debug,
        compiler,
        errc,
        error,
        process,
        raw_ostream,
        signals,
        threading,
        windows_error
    }
};
