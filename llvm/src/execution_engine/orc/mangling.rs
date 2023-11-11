/*!
Name Mangling Utilities for ORC

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ExecutionEngine/Orc/Mangling.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/ExecutionEngine/Orc/Mangling.cpp>
*/

use std::default::Default;

/**
Mangles symbol names then uniques them in the context of an ExecutionSession.
*/
pub struct MangleAndInterner {
}

/**
Maps IR global values to their linker symbol names / flags.

This utility can be used when adding new IR globals in the JIT.
*/
pub struct IRSymbolMapper {
}

#[derive(Default)]
pub struct ManglingOptions {
    emulated_tls: bool
}
