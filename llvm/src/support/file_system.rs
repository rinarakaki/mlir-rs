/*!
This file declares the llvm::sys::fs namespace. It is designed after
TR2/boost filesystem (v3), but modified to remove exception handling and the
path class.

All functions return an error_code and their actual work via the last out
argument. The out argument is defined if and only if errc::success is
returned. A function may return any error code in the generic or system
category. However, they shall be equivalent to any error conditions listed
in each functions respective documentation if the condition applies. [ note:
this does not guarantee that error_code will be in the set of explicitly
listed codes, but it does guarantee that if any of the explicitly listed
errors occur, the correct error_code will be used ]. All functions may
return errc::not_enough_memory if there is not enough memory to complete the
operation.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/FileSystem.h>
*/

pub mod unique_id;

use crate::llvm::{
    adt::{
        small_string,
        string_ref,
        twine
    },
    config::llvm_config,
    support::{
        chrono,
        error,
        error_handling,
        error_or,
        md5
    }
};
