/*!
# C++ Class to Represent a VM Module

Module.h This file contains the declarations for the Module class.

This file implements the Module class for the IR library.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/Module.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/IR/Module.cpp>
*/

/**
A Module instance is used to store all the information related to an LLVM module. Modules are the top level container of all other LLVM Intermediate Representation (IR) objects. Each module directly contains a list of globals variables, a list of functions, a list of libraries (or other modules) this module depends on, a symbol table, and various data about the target's characteristics.

A module maintains a GlobalList object that is used to hold all constant references to global variables in the module.  When a global variable is destroyed, it should have no entries in the GlobalList.
The main container class for the LLVM Intermediate Representation.
*/
// LLVM_EXTERNAL_VISIBILITY
pub struct Module {
    // LLVMContext &Context;           ///< The LLVMContext from which types and
    // ///< constants are allocated.
    // GlobalListType GlobalList;      ///< The Global Variables in the module
    // FunctionListType FunctionList;  ///< The Functions in the module
    // AliasListType AliasList;        ///< The Aliases in the module
    // IFuncListType IFuncList;        ///< The IFuncs in the module
    // NamedMDListType NamedMDList;    ///< The named metadata in the module
    // std::string GlobalScopeAsm;     ///< Inline Asm at global scope.
    // std::unique_ptr<ValueSymbolTable> ValSymTab; ///< Symbol table for values
    // ComdatSymTabType ComdatSymTab;  ///< Symbol table for COMDATs
    // std::unique_ptr<MemoryBuffer>
    // OwnedMemoryBuffer;              ///< Memory buffer directly owned by this
    //     ///< module, for legacy clients only.
    // std::unique_ptr<GVMaterializer>
    // Materializer;                   ///< Used to materialize GlobalValues
    // std::string ModuleID;           ///< Human readable identifier for the module
    // std::string SourceFileName;     ///< Original source file name for module,
    //     ///< recorded in bitcode.
    // std::string TargetTriple;       ///< Platform target triple Module compiled on
    //     ///< Format: (arch)(sub)-(vendor)-(sys0-(abi)
    // NamedMDSymTabType NamedMDSymTab;  ///< NamedMDNode names.
    // DataLayout DL;                  ///< DataLayout associated with the module
    // StringMap<unsigned>
    // CurrentIntrinsicIds; ///< Keep track of the current unique id count for
    // ///< the specified intrinsic basename.
    // DenseMap<std::pair<Intrinsic::ID, const FunctionType *>, unsigned>
    // UniquedIntrinsicNames; ///< Keep track of uniqued names of intrinsics
    // ///< based on unnamed types. The combination of
    // ///< ID and FunctionType maps to the extension that
    // ///< is used to make the intrinsic name unique.
}

pub enum ModFlagBehaviour {
    /**
    Emits an error if two values disagree, otherwise the resulting value is that of the operands.
    */
    Error = 1,

    /**
    Emits a warning if two values disagree. The result value will be the operand for the flag from the first module being linked.
    */
    Warning = 2,

    /**
    Adds a requirement that another module flag be present and have a specified value after linking is performed. The value must be a metadata pair, where the first element of the pair is the ID of the module flag to be restricted, and the second element of the pair is the value the module flag should be restricted to. This behaviour can be used to restrict the allowable results (via triggering of an error) of linking IDs with the **Override** behaviour.
    */
    Require = 3,

    /**
    Uses the specified value, regardless of the behaviour or value of the other module. If both modules specify **Override**, but the values differ, an error will be emitted.
    */
    Override = 4,

    /// Appends the two values, which are required to be metadata nodes.
    Append = 5,

    /**
    Appends the two values, which are required to be metadata
    nodes. However, duplicate entries in the second list are dropped
    during the append operation.
    */
    AppendUnique = 6,

    /// Takes the max of the two values, which are required to be integers.
    Max = 7,

    /// Takes the min of the two values, which are required to be integers.
    Min = 8,

    // // Markers:
    // ModFlagBehaviorFirstVal = Error,
    // ModFlagBehaviorLastVal = Min
}

pub struct ModuleFlagEntry {
    // ModFlagBehavior Behavior;
    // MDString *Key;
    // Metadata *Val;
}
