/*!
# MLIR Execution Engine and Utils

This file provides a JIT-backed execution engine for MLIR modules.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/ExecutionEngine/ExecutionEngine.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/ExecutionEngine/ExecutionEngine.cpp>
*/

use core::{
    default::Default,
    ptr::null
};

use crate::{
    mlir::ir::operation::Operation,
    llvm::{
        adt::string_map::StringMap,
        execution_engine::{
            jit_event_listener::JITEventListener,
            orc::{
                core::SymbolMap,
                ll_jit::LLJIT,
                mangling::MangleAndInterner
            },
            section_memory_manager::MemoryMapper
        },
        ir::{
            llvm_context::LLVMContext,
            module::Module
        },
        mc::target_registry::TargetRegistry,
        support::{
            code_gen::CodeGenOptLevel,
            error::Error,
            memory_buffer::MemoryBuffer
        },
        sys::{default_target_triple, host_cpu_name, host_cpu_features},
        target::target_machine::TargetMachine
    }
};

/// A simple object cache following Lang's LLJITWithObjectCache example.
pub struct SimpleObjectCache {
    cached_objects: StringMap<Box<MemoryBuffer>>
}

impl SimpleObjectCache {
    // override
    pub fn notify_object_compiled(
        &self,
        module: *const llvm::Module,
        obj_buffer: llvm::MemoryBufferRef);

    // override
    pub fn get_object(
        &self,
        module: *const llvm::Module) -> Box<MemoryBuffer>;

    /// Dump cached object to output file `filename`.
    pub fn dump_to_object_file(&self, filename: &str);

    /// Returns `true` if cache hasn't been populated yet.
    pub fn is_empty(&self) -> bool;
}

pub struct ExecutionEngineOptions {
    /**
    If `llvm_module_builder` is provided, it will be used to create an LLVM module from the given MLIR IR. Otherwise, a default `translateModuleToLLVMIR` function will be used to translate to LLVM IR.
    */
    // function_ref
    llvm_module_builder: fn(*mut Operation, &LLVMContext) -> Box<Module>,

    /**
    If `transformer` is provided, it will be called on the LLVM module during JIT-compilation and can be used, e.g., for reporting or optimisation.
    */
    // function_ref
    transformer: fn(*mut Module) -> Error,

    /**
    `jit_code_gen_opt_level`, when provided, is used as the optimization level for target code generation.
    */
    jit_code_gen_opt_level: Option<CodeGenOptLevel>,

    /**
    If `shared_lib_paths` are provided, the underlying JIT-compilation will open and link the shared libraries for symbol resolution.
    */
    shared_lib_paths: &'static [&'static str],

    /**
    Specifies an existing `section_memory_mapper` to be associated with the
    compiled code. If none is provided, a default memory mapper that directly
    calls into the operating system is used.
    */
    section_memory_mapper: Option<MemoryMapper>,

    /**
    If `enableObjectCache` is set, the JIT compiler will create one to store the object generated for the given module. The contents of the cache can be dumped to a file via the `dump_to_object_file` method.
    */
    enable_object_dump: bool,

    /**
    If enable `enable_gdb_notification_listener` is set, the JIT compiler will notify the llvm's global GDB notification listener.
    */
    enable_gdb_notification_listener: bool,

    /**
    If `enable_perf_notification_listener` is set, the JIT compiler will notify
    the llvm's global Perf notification listener.
    */
    enable_perf_notification_listener: bool
}

impl Default for ExecutionEngineOptions {
    fn default() -> Self {
        Self {
            llvm_module_builder: None,
            transformer: {},  // TODO ?
            jit_code_gen_opt_level: None,
            shared_lib_paths: {}, // TODO ?
            section_memory_mapper: None,
            enable_object_dump: false,
            enable_gdb_notification_listener: true,
            enable_perf_notification_listener: true
        }
    }
}

/**
JIT-backed execution engine for MLIR. Assumes the IR can be converted to LLVM IR. For each function, creates a wrapper function with the fixed interface

```cpp
void _mlir_funcName(*mut ()*)
```

where the only argument is interpreted as a list of pointers to the actual arguments of the function, followed by a pointer to the result. This allows the engine to provide the caller with a generic function pointer that can be used to invoke the JIT-compiled function.
*/
pub struct ExecutionEngine {
    /**
    Ordering of llvmContext and jit is important for destruction purposes: the jit must be destroyed before the context.
    */
    llvm_context: LLVMContext,

    /// Underlying LLJIT.
    jit: Box<LLJIT>,

    /// Underlying cache.
    cache: Box<SimpleObjectCache>,

    /// Names of functions that may be looked up.
    function_names: Vec<String>,

    /// GDB notification listener.
    gdb_listener: *mut JITEventListener,

    /// Perf notification listener.
    perf_listener: *mut JITEventListener,
}

impl ExecutionEngine {
    pub fn new(
        enable_object_dump: bool,
        enable_gdb_notification_listener: bool,
        enable_perf_notification_listener: bool)
    -> Self
    {
        Self {
            cache: if enable_object_dump {
                SimpleObjectCache::new()
            } else {
                null()
            },
            function_names: Vec::new(),
            gdb_listener: if enable_gdb_notification_listener {
                JITEventListener::create_gdb_registration_listener()
            } else {
                null()
            },
            perf_listener: null()
        }

        if enable_perf_notification_listener {
            if listener = JITEventListener::create_perf_jit_event_listener() {
                perf_listener = listener;
            }
            else if listener =
                         JITEventListener::create_intel_jit_event_listener() {
                perf_listener = listener;
            }
        }
    }

    /// Creates an execution engine for the given MLIR IR.
    // static
    pub fn create(
        operation: *mut Operation,
        options: &ExecutionEngineOptions /* {} */) -> llvm::Expected<Box<ExecutionEngine>>;

    /**
    Looks up a packed-argument function wrapping the function with the given name and returns a pointer to it. Propagates errors in case of failure.
    */
    pub const fn lookup_packed(&self, name: &str) -> llvm::Expected<void (*)(*mut ()*)> {
        let result = self.lookup(make_packed_function_name(name));
        if !result {
            return result.take_error();
        }
        reinterpret_cast<void (*)(*mut ()*)>(result.get())
    }

    /**
    Looks up the original function with the given name and returns a pointer to it. This is not necesarily a packed function. Propagates errors in case of failure.
    */
    pub const fn lookup(&self, name: &str) -> llvm::Expected<*mut ()> {
        let expected_symbol = self.jit.lookup(name);

        /*
        JIT lookup may return an Error referring to strings stored internally by the JIT. If the Error outlives the ExecutionEngine, it would want have a dangling reference, which is currently caught by an assertion inside JIT thanks to hand-rolled reference counting. Rewrap the error message into a string before returning. Alternatively, ORC JIT should consider copying
        the string into the error message.
        */
        if !expected_symbol {
            let mut error_message = String::new();
            let os = llvm::raw_string_ostream(error_message);
            llvm::handle_all_errors(
                expected_symbol.take_error(),
                |ei: &llvm::ErrorInfoBase| ei.log(os));
            return make_string_error(os.str());
        }

        let fptr = expected_symbol.to_ptr<*mut ()>();
        if !fptr.is_null()  {
            return fptr;
        }
        make_string_error("looked up function is null")
    }

    /**
    Invokes the function with the given name passing it the list of opaque pointers to the actual arguments.
    */
    pub fn invoke_packed(
        &self,
        name: &str,
        args: &mut [*mut ()] /* nullopt */) -> llvm::Error
    {
        let expected_f_ptr = self.lookup_packed(name);
        if !expected_f_ptr {
            return expected_f_ptr.take_error();
        }
        let fptr = *expected_f_ptr;

        (*fptr)(args.data());

        Error::success()
    }


    /**
    Helper function to wrap an output operand when using ExecutionEngine::invoke.
    */
    // static
    pub fn result<T>(&self, t: &T) -> Result<T> {
        return Result<T>(t);
    }

    /**
    Invokes the function with the given name passing it the list of arguments by value. Function result can be obtain through output parameter using the `Result` wrapper defined above. For example:

    ```mlir
    func @foo(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface }
    ```

    can be invoked:

    ```cpp
    int32_t result = 0;
    llvm::Error error = jit.invoke("foo", 42,
                      result(result));
    ```
    */
    template <typename... Args>
    pub fn invoke(&self, func_name: &str, args: Args...) -> llvm::Error {
        let adapter_name = format!("_mlir_ciface_{}", func_name);
        let mut args_array = SmallVector::<[*mut ()]>::new();
        /*
        Pack every arguments in an array of pointers. Delegate the packing to a trait so that it can be overridden per argument type.
        */
        (Argument<Args>::pack(args_array, args), ...);
        self.invoke_packed(adapter_name, args_array)
    }

    /**
    Set the target triple on the module. This is implicitly done when creating the engine.
    */
    pub fn setup_target_triple(&self, llvm_module: *mut llvm::Module) -> bool {
        // Setup the machine properties from the current architecture.
        let target_triple = default_target_triple();
        let mut error_message = String::new();
        let target = TargetRegistry::lookup_target(target_triple, error_message);
        if !target {
            errs("NO target: {}\n", error_message);
            return true;
        }

        let cpu = String::new(host_cpu_name());
        let features: llvm::SubtargetFeatures;
        let host_features = StringMap::<bool>;

        if host_cpu_features(host_features) {
            for (feature, is_enabled) in host_features {
                features.AddFeature(feature, is_enabled);
            }
        }
        let machine = Box<TargetMachine>(target.create_target_machine(
            target_triple, cpu, features.get_string(), {}, {}));
        if !machine {
            errs("Unable to create target machine\n");
            return true;
        }
        llvm_module.set_data_layout(machine.create_data_layout());
        llvm_module.set_target_triple(target_triple);
        false
    }

    /// Dump object code to output file `filename`.
    pub fn dump_to_object_file(&self, filename: &str) {
        if cache.is_null() {
            llvm::errs(
                "cannot dump ExecutionEngine object code to file: \
                object cache is disabled\n");
            return;
        }
        /*
        Compilation is lazy and it doesn't populate object cache unless requested.
        In case object dump is requested before cache is populated, we need to force compilation manually. 
        */
        if cache.is_empty() {
            for function_name in function_names {
                let result = self.lookup_packed(function_name);
                if !result {
                    llvm::errs(
                        "Could not compile {}:\n  {}\n", function_name,
                            result.take_error());
                    return;
                }
            }
        }
        cache.dump_to_object_file(filename);
    }

    /// Register symbols with this ExecutionEngine.
    pub fn register_symbols(
        &mut self,
        symbol_map: fn(MangleAndInterner) -> SymbolMap)
    {
        let main_jit_dylib = self.jit.main_jit_dylib();
        cant_fail(main_jit_dylib.define(
            absolute_symbols(symbol_map(MangleAndInterner(
                main_jit_dylib.execution_session(), jit.data_layout())))));
    }
}

/**
Trait that defines how a given type is passed to the JIT code. This defaults to passing the address but can be specialised.
*/
struct Argument<T> {
    static void pack(SmallVectorImpl<*mut ()> &args, T &val) {
      args.push(&val);
    }
}

/// Tag to wrap an output parameter when invoking a jitted function.
struct Result<T> {
    Result(T &result) : value(result) {}
    T &value;
}

// Specialization for output parameter: their address is forwarded directly to
// the native code.
template <typename T>
struct Argument<Result<T>> {
    static void pack(SmallVectorImpl<*mut ()> &args, Result<T> &result) {
      args.push(&result.value);
    }
}
