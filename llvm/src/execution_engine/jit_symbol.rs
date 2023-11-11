/*!
# JIT symbol abstraction

Abstraction for target process addresses.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ExecutionEngine/JITSymbol.h>
*/

/// Represents an address in the target process's address space.
pub type JITTargetAddress = u64;


/// Flags for symbols in the JIT.
pub struct JITSymbolFlags {
}

/**
ARM-specific JIT symbol flags.
FIXME: This should be moved into a target-specific header.
*/
pub struct ARMJITSymbolFlags {

}

/// Represents a symbol that has been evaluated to an address already.
pub struct JITEvaluatedSymbol {

}

/// Represents a symbol in the JIT.
pub struct JITSymbol {
}

/**
Symbol resolution interface.

Allows symbol flags and addresses to be looked up by name.
Symbol queries are done in bulk (i.e. you request resolution of a set of symbols, rather than a single one) to reduce IPC overhead in the case of remote JITing, and expose opportunities for parallel compilation.
*/
pub struct JITSymbolResolver {

}
