/*!
Provides a library for accessing information about this process and other
processes on the operating system. Also provides means of spawning
subprocess for commands. The design of this library is modeled after the
proposed design of the Boost.Process library, and is design specifically to
follow the style of standard libraries and potentially become a proposal
for a standard library.

This file declares the llvm::sys::Process class which contains a collection
of legacy static interfaces for extracting various information about the
current process. The goal is to migrate users of this API over to the new
interfaces.

This file implements the operating system Process concept.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/Process.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/Process.cpp>
*/

use crate::llvm::{
    adt::{
        stl_extras,
        string_extras
    },
    config::{
        config,
        llvm_config
    },
    support::{
        chrono,
        crash_recovery_context,
        data_types,
        error,
        file_system,
        path,
        program
    }
};

/// A collection of legacy interfaces for querying information about the
/// current executing process.
pub struct Process {
}
