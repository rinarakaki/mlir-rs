//! MLIR Global Context Class 
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/MLIRContext.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/MLIRContext.cpp>

#[allow(non_camel_case_types)]
type unsigned = u16;

use std::{
    any::TypeId,
    hash::{Hash, Hasher},
    ptr::null
};
use llvm::{
    adt::{
        dense_map::DenseMap,
        dense_set,
        small_string,
        small_vector::SmallVector,
        string_map::StringMap,
        string_set,
        twine
    },
    support::{
        allocator::BumpPtrAllocator,
        command_line,
        debug,
        mutex,
        rw_mutex,
        thread_pool::ThreadPool,
        raw_ostream
    }
};
use crate::{
    ir::{
        affine_expr,
        affine_map,
        attribute,
        builtins::dialect,
        diagnostics::DiagnosticEngine,
        dialect::{
            Dialect,
            registry::DialectRegistry,
        },
        extensible_dialect::DynamicDialect,
        integer_set,
        location,
        mlir_context,
        operation::{
            implementation,
            support::{OperationName, RegisteredOperationName},
        },
        r#type
    },
    support::{
        debug_action::DebugActionManager,
        llvm,
        storage_uniquer::StorageUniquer,
    }
};

/// MLIRContext is the top-level object for a collection of MLIR operations. It holds immortal uniqued objects like types, and the tables used to unique them.
///
/// MLIRContext gets a redundant "MLIR" prefix because otherwise it ends up with a very generic name ("Context") and because it is uncommon for clients to interact with it.
///
/// The context wrap some multi-threading facilities, and in particular by default it will implicitly create a thread pool.
/// This can be undesirable if multiple context exists at the same time or if a process will be long-lived and create and destroy contexts.
/// To control better thread spawning, an externally owned ThreadPool can be injected in the context. For example:
///
/// ```cpp
/// ThreadPool myThreadPool;
/// while (auto *request = nextCompilationRequests()) {
///    MLIRContext ctx(registry, MLIRContext::Threading::DISABLED);
///    ctx.set_thread_pool(myThreadPool);
///    processRequest(request, cxt);
/// }
/// ```
pub struct MLIRContext {
    /// In most cases, creating operation in unregistered dialect is not desired and indicate a misconfiguration of the compiler. This option enables to detect such use cases
    allow_unregistered_dialects: bool,

    /// Enable support for multi-threading within MLIR.
    threading_is_enabled: bool,

//     /**
//     Track if we are currently executing in a threaded execution environment (like the pass-manager): this is only a debugging feature to help reducing the chances of data races one some context APIs.
//     */
// #ifndef NDEBUG
//     multi_threaded_execution_context: std::atomic<int>{0},
// #endif

    /// If the operation should be attached to diagnostics printed via the Operation::emit methods.
    print_op_on_diagnostic: bool,

    /// If the current stack trace should be attached when emitting diagnostics.
    print_stack_trace_on_diagnostic: bool,

    /* MLIRContextImpl */

    /* Other */

    /// This points to the ThreadPool used when processing MLIR tasks in parallel.
    /// It can't be nullptr when multi-threading is enabled. Otherwise if multi-threading is disabled, and the threadpool wasn't externally provided using `set_thread_pool`, this will be nullptr.
    thread_pool: Option<ThreadPool>, // = nullptr;

    /// In case where the thread pool is owned by the context, this ensures destruction with the context.
    owned_thread_pool: Box<ThreadPool>,

    /// This is a list of dialects that are created referring to this context.
    /// The MLIRContext owns the objects.
    loaded_dialects: DenseMap<&'static str, Box<dyn Dialect>>,
    dialects_registry: DialectRegistry,

    /// An allocator used for AbstractAttribute and AbstractType objects.
    abstract_dialect_symbol_allocator: BumpPtrAllocator,

    /// This is a mapping from operation name to the operation info describing it.
    operations: StringMap<OperationName>,

    /// A vector of operation info specifically for registered operations.
    registered_operations: StringMap<RegisteredOperationName>,

    /// This is a sorted container of registered operations for a deterministic and efficient `getRegisteredOperations` implementation.
    sorted_registered_operations: SmallVector<[RegisteredOperationName; 0]>,

    /// A mutex used when accessing operation information.
    operation_info_mutex: SmartRWMutex<true>,
}

// impl<'a> Default for MLIRContext<'a> {
//     fn default() -> Self {
//         Self {
//             allow_unregistered_dialects: false,
//             threading_is_enabled: true,
//             print_op_on_diagnostic: true,
//             print_stack_trace_on_diagnostic: false,

//         }
//     }
// }

impl MLIRContext {
    /// Return information about all IR dialects loaded in the context.
    pub fn loaded_dialects(&self) -> Vec<*mut Dialect> {
        let mut output = Vec::new();
        output.reserve(self.loaded_dialects.len());
        for dialect in self.loaded_dialects {
            output.push(dialect.second.get());
        }
        array_pod_sort(
            output.begin(), output.end(),
            |lhs: &Dialect, rhs: &Dialect| lhs.namespace() < rhs.namespace()
        );
        output
    }

    /// Return the dialect registry associated with this context.
    pub fn dialect_registry(&self) -> &DialectRegistry {
        &self.dialects_registry
    }

    /// Append the contents of the given dialect registry to the registry associated with this context.
    pub fn append_dialect_registry(&mut self, registry: &DialectRegistry) {
        if registry.is_subset_of(&self.dialects_registry) {
            return;
        }

        assert!(
            self.multi_threaded_execution_context == 0,
            "Appending to the MLIRContext dialect registry while in a \
            multi-threaded execution context.");
        registry.append_to(&self.dialects_registry);

        // For the already loaded dialects, apply any possible extensions immediately.
        registry.apply_extensions(self);
    }

    /// Return information about all available dialects in the registry in this
    /// context.
    pub fn available_dialects(&self) -> Vec<&str> {
        let mut output = Vec::new();
        for dialect in self.dialects_registry.dialect_names() {
            output.push(dialect);
        }
        output
    }

    /// Get a registered IR dialect with the given namespace. If an exact match is not found, then return nullptr.
    pub fn loaded_dialect(&self, name: &str) -> *mut Dialect {
        // Dialects are sorted by name, so we can use binary search for lookup.
        match self.loaded_dialects.find(name) {
            None => None,
            Some(loaded_dialect) => loaded_dialect.second.get()
        }
    }

    /// Get (or create) a dialect for the given derived dialect type. The derived type must provide a static `dialect_namespace` method.
    pub fn get_or_load_dialect<T>(&self) -> *mut T {
        self.get_or_load_dialect___(
            T::dialect_namespace(),
            TypeId::of::<T>(),
            || Box::new(T::new(self))
        ).into()
    }
    
     /// Load a dialect in the context.
    pub fn load_dialect<Dialect>(&self) {
        // Do not load the dialect if it is currently loading. This can happen if a dialect initialiser triggers loading the same dialect recursively.
        if !self.is_dialect_loading(Dialect::dialect_namespace()) {
            self.get_or_load_dialect::<Dialect>();
        }
    }

    // /// Load a list dialects in the context.
    // pub fn load_dialect<Dialect, OtherDialect, ...MoreDialects>(&self) {
    //     load_dialect<Dialect>();
    //     load_dialect<OtherDialect, MoreDialects...>();
    // }

    /// Get (or create) a dynamic dialect for the given name.
    pub fn get_or_load_dynamic_dialect(
        dialect_namespace: &str,
        ctor: fn(Option<DynamicDialect>)  //function_ref<void>
    ) -> Option<DynamicDialect>
    {

    }

    /// Load all dialects available in the registry in this context.
    pub fn load_all_available_dialects(&self) {
        for name in self.available_dialects().iter() {
            self.get_or_load_dialect_(name);
        }
    }

    /// Get (or create) a dialect for the given derived dialect name.
    /// The dialect will be loaded from the registry if no dialect is found.
    /// If no dialect is loaded for this name and none is available in the registry, returns nullptr.
    pub fn get_or_load_dialect_(&self, name: &str) -> *mut Dialect {
        let dialect = self.loaded_dialect(name);
        if !dialect.is_null() {
            return dialect;
        }
            
        let allocator
            = self.dialects_registry.dialect_allocator(name);
        if allocator.is_null() {
            null()
        }
        allocator(self)
    }

    /// Return true if we allow to create operation for unregistered dialects.
    pub fn allows_unregistered_dialects(&self) -> bool {
        self.allow_unregistered_dialects
    }

    /// Enables creating operations in unregistered dialects.
    pub fn allow_unregistered_dialects(&self, allow: bool /* true */) {
        assert!(
            self.multi_threaded_execution_context == 0,
            "Changing MLIRContext `allow-unregistered-dialects` configuration \
            while in a multi-threaded execution context");
        self.allow_unregistered_dialects = allow;
    }

    /// Return true if multi-threading is enabled by the context.
    pub fn is_multithreading_enabled(&self) -> bool {
        self.threading_is_enabled && llvm::llvm_is_multithreaded()
    }

    /// Set the flag specifying if multi-threading is disabled by the context.
    /// The command line debugging flag `--mlir-disable-threading` is overriding
    /// this call and making it a no-op!
    pub fn disable_multithreading(&self, disable: bool /* true */) {

    }

    pub fn enable_multithreading(&self, enable: bool /* true */) {
        self.disable_multithreading(!enable);
    }

    /// Set a new thread pool to be used in this context. This method requires that multithreading is disabled for this context prior to the call. This allows to share a thread pool across multiple contexts, as well as decoupling the lifetime of the threads from the contexts. The thread pool must outlive the context. Multi-threading will be enabled as part of this method.
    /// The command line debugging flag `--mlir-disable-threading` will still prevent threading from being enabled and threading won't be enabled after this call in this case.
    pub fn set_thread_pool(&mut self, thread_pool: &ThreadPool) {
        assert!(
            !self.is_multithreading_enabled(),
            "Expected multi-threading to be disabled when setting a ThreadPool.");
        self.thread_pool = Some(thread_pool);
        self.owned_thread_pool.reset();
        self.enable_multithreading(true);
    }

    /// Return the number of threads used by the thread pool in this context. The number of computed hardware threads can change over the lifetime of a process based on affinity changes, so users should use the number of threads actually in the thread pool for dispatching work. Returns 1 if multithreading is disabled.
    pub fn num_threads(&self) -> usize {
        if self.is_multithreading_enabled() {
            assert!(self.thread_pool,
                   "Multi-threading is enabled but threadpool not set.");
            self.thread_pool.thread_count()
        }
        // No multithreading or active thread pool. Return 1 thread.
        1
    }

    /// Return the thread pool used by this context. This method requires that multithreading be enabled within the context, and should generally not be used directly. Users should instead prefer the threading utilities within Threading.h.
    pub fn thread_pool(&self) -> &ThreadPool {
        assert!(self.is_multithreading_enabled(),
                "Expected multi-threading to be enabled within the context.");
        assert!(self.thread_pool,
                "Multi-threading is enabled but threadpool not set.");
        *self.thread_pool
    }

    /// Return true if we should attach the operation to diagnostics emitted via Operation::emit.
    pub fn should_print_op_on_diagnostic(&self) -> bool {
        self.print_op_on_diagnostic
    }

    /// Set the flag specifying if we should attach the operation to diagnostics emitted via Operation::emit.
    pub fn print_op_on_diagnostic(&self, enable: bool) {
        assert!(
            self.multi_threaded_execution_context == 0,
            "Changing MLIRContext `print-op-on-diagnostic` configuration while in a multi-threaded execution context");
        self.print_op_on_diagnostic = enable;
    }

    /// Return true if we should attach the current stacktrace to diagnostics when emitted.
    pub fn should_print_stack_trace_on_diagnostic(&self) -> bool {
        self.print_stack_trace_on_diagnostic
    }

    /// Set the flag specifying if we should attach the current stacktrace when emitting diagnostics.
    pub fn print_stack_trace_on_diagnostic(&self, enable: bool) {
        assert!(
            self.multi_threaded_execution_context == 0,
            "Changing MLIRContext `print-stacktrace-on-diagnostic` configuration while in a multi-threaded execution context");
        self.print_stack_trace_on_diagnostic = enable;
    }

    /// Return a sorted array containing the information about all registered operations.
    pub fn registered_operations(&self) -> &[RegisteredOperationName] {
        self.sorted_registered_operations.into()
    }

    /// Return true if this operation name is registered in this context.
    pub fn is_operation_registered(&self, name: &str) -> bool {
        RegisteredOperationName::lookup(name, self).has_value()
    }

    /// Returns the diagnostic engine for this context.
    pub fn diag_engine(&self) -> &DiagnosticEngine {
        self.diag_engine
    }

    /// Returns the storage uniquer used for creating affine constructs.
    pub fn affine_uniquer(&self) -> &StorageUniquer {
        self.affine_uniquer
    }

    /// Returns the storage uniquer used for constructing type storage instances.
    /// This should not be used directly.
    pub fn type_uniquer(&self) -> &StorageUniquer {
        self.type_uniquer
    }

    /// Returns the storage uniquer used for constructing attribute storage instances. This should not be used directly.
    pub fn attribute_uniquer(&self) -> &StorageUniquer {
        self.attribute_uniquer
    }

    /// Returns the manager of debug actions within the context.
    pub fn debug_action_manager(&self) -> &DebugActionManager {
        self.debug_action_manager
    }

    // /**
    // These APIs are tracking whether the context will be used in a multithreading environment: this has no effect other than enabling assertions on misuses of some APIs.
    // */
    // pub fn enter_multi_threaded_execution(&self);
    // pub fn exit_multi_threaded_execution(&self);

    /// Get a dialect for the provided namespace and TypeID: abort the program if a dialect exist for this namespace with different TypeID. If a dialect has not been loaded for this namespace/TypeID yet, use the provided ctor to create one on the fly and load it. Returns a pointer to the dialect owned by the context.
    /// The use of this method is in general discouraged in favor of `get_or_load_dialect<DialectClass>()`.
    pub fn get_or_load_dialect___(
        &self,
        dialect_namespace: &str,
        dialect_id: TypeId,
        ctor: fn() -> Box<Dialect>  // function_ref
    ) -> *mut Dialect {

    }

    // /**
    // Returns a hash of the registry of the context that may be used to give a rough indicator of if the state of the context registry has changed. The context registry correlates to loaded dialects and their entities (attributes, operations, types, etc.).
    // */
    // pub fn get_registry_hash(&self) -> llvm::hash_code {
    //     let mut hash = llvm::hash_code(0);
    //     // Factor in number of loaded dialects, attributes, operations, types.
    //     hash = llvm::hash_combine(hash, self.loaded_dialects.len());
    //     hash = llvm::hash_combine(hash, self.registered_attributes.len());
    //     hash = llvm::hash_combine(hash, self.registered_operations.len());
    //     hash = llvm::hash_combine(hash, self.registered_types.len());
    //     hash
    // }

    /// Return true if the given dialect is currently loading.
    fn is_dialect_loading(&self, dialect_namespace: &str) -> bool {
        match self.loaded_dialects.find(dialect_namespace) {
            None => false,
            // None indicates that the dialect is currently being loaded.
            Some(loaded_dialect) => loaded_dialect.second.is_none()
        }
    }
}

impl<'a> Hash for MLIRContext<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Factor in number of loaded dialects, attributes, operations, types.
        state.write_usize(self.loaded_dialects.len());
        state.write_usize(self.registered_attributes.len());
        state.write_usize(self.registered_operations.len());
        state.write_usize(self.registered_types.len());
    }
}

enum Threading {
    Disabled,
    Enabled
}
