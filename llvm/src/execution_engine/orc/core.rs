/*!
Core ORC APIs (Layer, JITDylib, etc.)

Contains core ORC APIs.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ExecutionEngine/Orc/Core.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/ExecutionEngine/Orc/Core.cpp>
*/

use std::{
    cell::Cell,
    sync::Mutex
};

use crate::llvm::{
    adt::{
        bitmask_enum,
        dense_map::DenseMap,
        dense_set,
        function_extras,
        intrusive_ref_cnt_ptr
    },
    execution_engine::{
        jit_link,
        jit_symbol::{JITSymbolFlags, JITTargetAddress},
        orc::{
            executor_process_control,
            shared::wrapper_function_utils,
            task_dispatch
        }        
    },
    support::{
        debug,
        extensible_rtti
    }
};

/// API to remove / transfer ownership of JIT resources.
pub struct ResourceTracker {
    
}

/// Listens for ResourceTracker operations.
pub struct ResourceManager {

}

/**
Lookup flags that apply to each dylib in the search order for a lookup.

If MatchHiddenSymbolsOnly is used (the default) for a given dylib, then only symbols in that Dylib's interface will be searched. If MatchHiddenSymbols is used then symbols with hidden visibility will matcah as well.
*/
pub enum JITDylibLookupFlags {
    MatchExportedSymbolsOnly,
    MatchAllSymbols
}

/**
Lookup flags that apply to each symbol in a lookup.

If RequiredSymbol is used (the default) for a given symbol then that symbol must be found during the lookup or the lookup will fail returning a SymbolNotFound error. If WeaklyReferencedSymbol is used and the given symbol is not found then the query will continue, and no result for the missing symbol will be present in the result (assuming the rest of the lookup succeeds).
*/
pub enum SymbolLookupFlags {
    RequiredSymbol,
    WeaklyReferencedSymbol
}

/**
Describes the kind of lookup being performed. The lookup kind is passed to symbol generators (if they're invoked) to help them determine what definitions to generate.

Static -- Lookup is being performed as-if at static link time (e.g.
          generators representing static archives should pull in new
          definitions).

DLSym -- Lookup is being performed as-if at runtime (e.g. generators
         representing static archives should not pull in new definitions).
*/
pub enum LookupKind {
    Static,
    DLSym
}


/**
A set of symbols to look up, each associated with a SymbolLookupFlags value.

This class is backed by a vector and optimized for fast insertion, deletion and iteration. It does not guarantee a stable order between operations, and will not automatically detect duplicate elements (they can be manually checked by calling the validate method).
*/
pub struct SymbolLookupSet {

}

pub struct SymbolAliasMapEntry {

}

/**
A MaterialisationUnit represents a set of symbol definitions that can be materialised as a group, or individually discarded (when overriding definitions are encountered).

MaterialisationUnits are used when providing lazy definitions of symbols to JITDylibs. The JITDylib will call materialize when the address of a symbol is requested via the lookup method. The JITDylib will call discard if a stronger definition is added or already present.
*/
pub struct MaterialisationUnit {

}

/// Represents the state that a symbol has reached during materialisation.
#[repr(align(8))]
pub enum SymbolState {
    /// No symbol should be in this state.
    Invalid,
    /// Added to the symbol table, never queried.
    NeverSearched,
    /// Queried, materialisation begun.
    Materializing,
    /// Assigned address, still materialising.
    Resolved,
    /// Emitted to memory, but waiting on transitive dependencies.
    Emitted,
    /// Ready and safe for clients to access.
    Ready = 0x3f
}

/**
A symbol query that returns results via a callback when results are ready.

makes a callback when all symbols are available.
*/
pub struct AsynchronousSymbolQuery {

}

/**
Wraps state for a lookup-in-progress.
DefinitionGenerators can optionally take ownership of a LookupState object to suspend a lookup-in-progress while they search for definitions.
*/
pub struct LookupState {

}

/**
Definition generators can be attached to JITDylibs to generate new definitions for otherwise unresolved symbols during lookup.
*/
pub struct DefinitionGenerator {

}

/**
Represents a JIT'd dynamic library.

This class aims to mimic the behaviour of a regular dylib or shared object, but without requiring the contained program representations to be compiled up-front. The JITDylib's content is defined by adding MaterializationUnits, and contained MaterializationUnits will typically rely on the JITDylib's links-against order to resolve external references (similar to a regular dylib).

The JITDylib object is a thin wrapper that references state held by the ExecutionSession. JITDylibs can be removed, clearing this underlying state and leaving the JITDylib object in a defunct state. In this state the JITDylib's name is guaranteed to remain accessible. If the ExecutionSession is still alive then other operations are callable but will return an Error or null result (depending on the API). It is illegal to call any operation other than getName on a JITDylib after the ExecutionSession has been torn down.

JITDylibs cannot be moved or copied. Their address is stable, and useful as a key in some JIT data structures.
*/
pub struct JITDylib {

}

struct SymbolTableEntry {
    address: JITTargetAddress, // = 0;
    flags: JITSymbolFlags,
    state: u8, // : 6;
    materialiser_attached: u8,  // : 1;
    pending_removal: u8  // : 1;
}

/**
Platforms set up standard symbols and mediate interactions between dynamic initialisers (e.g. C++ static constructors) and ExecutionSession state.
Note that Platforms do not automatically run initializers: clients are still responsible for doing this.
*/
pub trait Platform {

}

/// A materialisation task.
pub struct MaterialisationTask {

}

/// An ExecutionSession represents a running JIT program.
pub struct ExecutionSession<P: Platform> {
    session_mutex: Cell<std::recursive_mutex>,
    session_open: bool,  // = true;
    epc: Box<ExecutorProcessControl>,
    platform: Box<P>,
    report_error: ErrorReporter,  // = logErrorsToStdErr,
    dispatch_task: DispatchTaskFunction,  // = runOnCurrentThread,
  
    resource_managers: Vec<Option<ResourceManager>>,
  
    jds: Vec<JITDylibSP>,
  
    // FIXME: Remove this (and run_outstanding_mus) once the linking layer works
    // with callbacks from asynchronous queries.
    outstanding_mus_mutex: Cell<std::recursive_mutex>,
    outstanding_mus:
        Vec<(Box<MaterialisationUnit>, Box<MaterialisationResponsibility>)>,
  
    jit_dispatch_handlers: Cell<Mutex<
        DenseMap<JITTargetAddress, std::shared_ptr<JITDispatchHandlerFunction>>
    >>
}
