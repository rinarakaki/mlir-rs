/*!
# MLIR CPU Execution Driver Library

This is a library that provides a shared implementation for command line utilities that execute an MLIR file on the CPU by translating MLIR to LLVM IR before JIT-compiling and executing the latter.

The translation can be customized by providing an MLIR to MLIR transformation.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/ExecutionEngine/JitRunner.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/ExecutionEngine/JitRunner.cpp>
*/

use clap::Parser;

use crate::{
    mlir::ir::{
        mlir_context::MLIRContext,
        operation::Operation
    },
    llvm::execution_engine::orc::{
        core::SymbolMap,
        mangling::MangleAndInterner
    }
};

/// JitRunner command line options used by JitRunnerConfig methods
pub struct JitRunnerOptions {
    /// The name of the main function
    main_func_name: &'static str,
    /// The type of the main function (as string, from cmd-line)
    main_func_type: &'static str
}
  
/// Configuration to override functionality of the JitRunner
pub struct JitRunnerConfig {
    /**
    MLIR transformer applied after parsing the input into MLIR IR and before
    passing the MLIR IR to the ExecutionEngine.
    */
    fn<LogicalResult(mlir::*mut Operation,
                                     JitRunnerOptions &options)>
        mlir_transformer = nullptr;
  
    /**
    A custom function that is passed to ExecutionEngine. It processes MLIR and
    creates an LLVM IR module.
    */
    fn<std::unique_ptr<llvm::Module>(*mut Operation,
                                                     llvm::LLVMContext &)>
        llvm_module_builder = nullptr;
  
    /// A callback to register symbols with ExecutionEngine at runtime.
    fn<SymbolMap(MangleAndInterner)>
        runtimesymbol_map = nullptr;
}
  
/**
Entry point for all CPU runners. Expects the common argc/argv arguments for standard C++ main functions. The supplied dialect registry is expected to contain any registers that appear in the input IR, they will be loaded on-demand by the parser.
*/
pub fn jit_runner_main(
    argc: int,
    argv: char **,
    registry: &DialectRegistry,
    config: JitRunnerConfig = {}
) -> int {
    // Create the options struct containing the command line options for the
    // runner. This must come before the command line options are parsed.
    let options = Options;
    llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR CPU execution driver\n");

    if options.host_supports_jit {
        let j = llvm::orc::LLJITBuilder().create();
        if (j) {
            llvm::outs() << "true\n";
        } else {
            llvm::consume_error(j.take_error());
            llvm::outs() << "false\n";
        }
        return 0;
    }

    Optional<usize> opt_level = command_line_opt_level(options);
    SmallVector<std::reference_wrapper<llvm::cl::opt<bool>>, 4> opt_flags{
        options.opt_o0, options.opt_o1, options.opt_o2, options.opt_o3};

    let context = MLIRContext::new(registry);

    let m = parse_mlir_input(options.input_filename, !options.no_implicit_module,
                            &context);
    if !m {
        llvm::errs("could not parse the input IR\n");
        return 1;
    }

    let runner_options = JitRunnerOptions {options.main_func_name, options.main_func_type};
    if config.mlir_transformer {
        if (failed(config.mlir_transformer(m.get(), runner_options)))
        return EXIT_FAILURE;
    }

    let tm_builder_or_error = llvm::orc::JITTargetMachineBuilder::detect_host();
    if !tm_builder_or_error {
        llvm::errs( "Failed to create a JITTargetMachineBuilder for the host\n");
        return EXIT_FAILURE;
    }
    let tm_or_error = tm_builder_or_error.create_target_machine();
    if !tm_or_error {
        llvm::errs("Failed to create a TargetMachine for the host\n");
        return EXIT_FAILURE;
    }

    CompileAndExecuteConfig compile_and_execute_config;
    if opt_level {
        compile_and_execute_config.transformer = mlir::make_optimizing_transformer(
            *opt_level, /*sizeLevel=*/0, /*targetMachine=*/tm_or_error.get());
    }
    compile_and_execute_config.llvm_module_builder = config.llvm_module_builder;
    compile_and_execute_config.runtime_symbol_map = config.runtimesymbol_map;

    // Get the function used to compile and execute the module.
    using CompileAndExecuteFnT =
        Error (*)(Options &, *mut Operation, StringRef, CompileAndExecuteConfig);
    let compile_and_execute_fn =
        StringSwitch<CompileAndExecuteFnT>(options.main_func_type.value())
            .Case("i32", compile_and_execute_single_return_function<i32>)
            .Case("i64", compile_and_execute_single_return_function<i64>)
            .Case("f32", compile_and_execute_single_return_function<f32>)
            .Case("void",
            pub fn compile_and_execute_void_function)
            .Default(nullptr);

    Error error = compile_and_execute_fn
                        ? compile_and_execute_fn(options, m.get(),
                                            options.main_func_name.value(),
                                            compile_and_execute_config)
                        : make_string_error("unsupported function type");

    int exit_code = EXIT_SUCCESS;
    llvm::handle_all_errors(
        std::move(error),
        |info: &llvm::ErrorInfoBase| {
        llvm::errs() << "Error: ";
        info.log(llvm::errs());
        llvm::errs() << '\n';
        exit_code = EXIT_FAILURE;
        });

    exit_code
}


/**
This options struct prevents the need for global static initialisers, and is only initialised if the JITRunner is invoked.
*/
#[derive(Debug, Parser)]
struct Options {
    /// <input file>
    // {llvm::cl::init("-")}
    input_filename: String,
    /// Function to be called.
    // llvm::cl::value_desc("<function name>"), llvm::cl::init("main")}
    #[arg(short = 'e', long)]
    main_func_name: String,
    {
    /// Textual description of the function type to be called
    // {
    //     "entry-point-result",
    //     llvm::cl::value_desc("f32 | i32 | i64 | void"), llvm::cl::init("f32")};
    main_func_type: String,

    opt_flags: llvm::cl::OptionCategory{"opt-like flags"},

    // CLI variables for -On options.

    /// Run opt passes and codegen at O0.
    // {"O0", llvm::cl::cat(opt_flags)};
    opt_o0: bool,
    /// Run opt passes and codegen at O1.
    // {"O1", llvm::cl::cat(opt_flags)};
    opt_o1: bool,
    /// Run opt passes and codegen at O2.
    opt_o2: bool,
    // {"O2", llvm::cl::cat(opt_flags)};
    /// Run opt passes and codegen at O3.
    opt_o3: bool,
    // {"O3", llvm::cl::cat(opt_flags)};

    cl_options_category: llvm::cl::OptionCategory {"linking options"};
    /// Libraries to link dynamically.
    cl_shared_libs: String, {
        "shared-libs",
        llvm::cl::MiscFlags::CommaSeparated, llvm::cl::cat(cl_options_category)};

    // CLI variables for debugging.

    /// Dump JITted-compiled object to file specified with -object-filename (<input file>.o by default).
    #[arg(long)]
    dump_object_file: bool,

    /// Dump JITted-compiled object to file <input file>.o.
    #[arg(long)]
    object_filename: String,

    /// Report host JIT support.
    // llvm::cl::Hidden
    #[arg(long)]
    host_supports_jit: bool,

    /// Disable implicit addition of a top-level module op during parsing.
    #[arg(long)]
    no_implicit_module: bool,
    {
        "no-implicit-module",
        llvm::cl::init(false)};
}

struct CompileAndExecuteConfig {
    /// LLVM module transformer that is passed to ExecutionEngine.
    transformer: std::function<llvm::Error(llvm::Module *)>,

    /**
    A custom function that is passed to ExecutionEngine. It processes MLIR module and creates LLVM IR module.
    */
    llvm_module_builder: fn<std::unique_ptr<llvm::Module>(*mut Operation,
                                                    llvm::LLVMContext &)>,

    /**
    A custom function that is passed to ExecutinEngine to register symbols at runtime.
    */
    runtime_symbol_map: fn(MangleAndInterner) -> SymbolMap
}

} // namespace

// static
pub fn parse_mlir_input(
    StringRef input_filename,
    bool insert_implicit_module,
    MLIRContext *context)
-> OwningOpRef<*mut Operation>
{
    // Set up the input file.
    String error_message;
    let file = open_input_file(input_filename, &error_message);
    if !file {
        llvm::errs("{}\n", error_message);
        return nullptr;
    }

    let source_manager = std::make_shared<llvm::SourceManager>();
    source_manager.AddNewSourceBuffer(std::move(file), SMLoc());
    let module =
        parse_source_file_for_tool(source_manager, context, insert_implicit_module);
    if (!module) {
        return nullptr;
    }
    if !module.get().has_trait<OpTrait::SymbolTable>() {
        llvm::errs("Error: top-level op must be a symbol table.\n");
        return nullptr;
    }
    return module;
}

static
#[inline]
pub fn make_string_error(message: &Twine) -> Error {
    llvm::make_error<llvm::StringError>(
        message.str(),
        llvm::inconvertible_error_code())
}

pub fn command_line_opt_level(options: &Options) -> Optional<usize> {
    Optional<usize> opt_level;
    SmallVector<std::reference_wrapper<llvm::cl::opt<bool>>, 4> opt_flags{
        options.opt_o0, options.opt_o1, options.opt_o2, options.opt_o3};

    // Determine if there is an optimization flag present.
    for j in 0..4 {
        let &flag = opt_flags[j].get();
        if (flag) {
            opt_level = j;
            break;
        }
    }
    opt_level
}

// JIT-compile the given module and run "entry_point" with "args" as arguments.
// static
pub fn compile_and_execute(
    options: &Options,
    module: *mut Operation,
    entry_point: &str,
    config: CompileAndExecuteConfig,
    args: *mut ()*)
-> Error
{
    Optional<llvm::CodeGenOpt::Level> jit_code_gen_opt_level;
    if (let cl_opt_level = command_line_opt_level(options)) {
        jit_code_gen_opt_level = static_cast<llvm::CodeGenOpt::Level>(*cl_opt_level);
    }

    // If shared library implements custom mlir-runner library init and destroy
    // functions, we'll use them to register the library with the execution
    // engine. Otherwise we'll pass library directly to the execution engine.
    SmallVector<SmallString<256>, 4> lib_paths;

    // Use absolute library path so that gdb can find the symbol table.
    transform(
        options.cl_shared_libs, std::back_inserter(lib_paths),
        |lib_path: String| {
            SmallString<256> abs_path(lib_path.begin(), lib_path.end());
            cant_fail(llvm::error_code_to_error(llvm::sys::fs::make_absolute(abs_path)));
            return abs_path;
    });

    // Libraries that we'll pass to the ExecutionEngine for loading.
    SmallVector<StringRef, 4> execution_engine_libs;

    using MlirRunnerInitFn = void (*)(llvm::StringMap<*mut ()> &);
    using MlirRunnerDestroyFn = void (*)();

    llvm::StringMap<*mut ()> export_symbols;
    SmallVector<MlirRunnerDestroyFn> destroy_fns;

    // Handle libraries that do support mlir-runner init/destroy callbacks.
    for lib_path in lib_paths {
        let lib = llvm::sys::DynamicLibrary::get_permanent_library(lib_path.c_str());
        *mut ()init_sym = lib.get_address_of_symbol("__mlir_runner_init");
        *mut ()destroy_sim = lib.get_address_of_symbol("__mlir_runner_destroy");

        // Library does not support mlir runner, load it with ExecutionEngine.
        if !init_sym || !destroy_sim {
            execution_engine_libs.push(lib_path);
            continue;
        }

        let init_fn = reinterpret_cast<MlirRunnerInitFn>(init_sym);
        init_fn(export_symbols);

        let destroy_fn = reinterpret_cast<MlirRunnerDestroyFn>(destroy_sim);
        destroy_fns.push(destroy_fn);
    }

    // Build a runtime symbol map from the config and exported symbols.
    let runtime_symbol_map = |interner: MangleAndInterner| {
        let symbol_map = config.runtime_symbol_map ? config.runtime_symbol_map(interner)
                                                : SymbolMap();
        for export_symbol in export_symbols {
            symbol_map[interner(export_symbol.getKey())] =
            l   lvm::JITEvaluatedSymbol::from_pointer(export_symbol.value());
        }
        return symbol_map;
    };

    mlir::ExecutionEngineOptions engine_options;
    engine_options.llvm_module_builder = config.llvm_module_builder;
    if config.transformer {
        engine_options.transformer = config.transformer;
    }
    engine_options.jit_code_gen_opt_level = jit_code_gen_opt_level;
    engine_options.shared_lib_paths = execution_engine_libs;
    engine_options.enable_object_dump = true;
    let expected_engine = mlir::ExecutionEngine::create(module, engine_options);
    if !expected_engine {
        return expected_engine.take_error();
    }

    let engine = std::move(*expected_engine);
    engine.register_symbols(runtime_symbol_map);

    let expected_f_ptr = engine.lookup_packed(entry_point);
    if !expected_f_ptr {
        return expected_f_ptr.take_error();
    }

    if options.dump_object_file {
        engine.dump_to_object_file(options.object_filename.is_empty()
                                    ? options.input_filename + ".o"
                                    : options.object_filename);
    }

    void (*fptr)(*mut ()*) = *expected_f_ptr;
    (*fptr)(args);

    // Run all dynamic library destroy callbacks to prepare for the shutdown.
    for destroy in destroy_fns {
        destroy();
    }

    Error::success()
}

static
pub fn compile_and_execute_void_function(
    options: &Options,
    module: *mut Operation,
    entry_point: &str,
    config: CompileAndExecuteConfig)
-> Error
{
    let main_function = dyn_cast_or_null<LLVM::LLVMFuncOp>(
        SymbolTable::lookup_symbol_in(module, entry_point));
    if !main_function || main_function.is_empty() {
        return make_string_error("entry point not found");
    }
    *mut ()empty = null();
    compile_and_execute(options, module, entry_point, std::move(config),
                            &empty);
}

template <typename Type>
Error check_compatible_return_type(LLVM::LLVMFuncOp main_function);
template <>
Error check_compatible_return_type<i32>(LLVM::LLVMFuncOp main_function) {
    let result_type = main_function.get_function_type()
                            .cast<LLVM::LLVMFunctionType>()
                            .get_return_type()
                            .dyn_cast<IntegerType>();
    if (!result_type || result_type.get_width() != 32)
        return make_string_error("only single i32 function result supported");
    return Error::success();
}
template <>
Error check_compatible_return_type<i64>(LLVM::LLVMFuncOp main_function) {
    let result_type = main_function.get_function_type()
                            .cast<LLVM::LLVMFunctionType>()
                            .get_return_type()
                            .dyn_cast<IntegerType>();
    if (!result_type || result_type.get_width() != 64)
        return make_string_error("only single i64 function result supported");
    return Error::success();
}
template <>
Error check_compatible_return_type<f32>(LLVM::LLVMFuncOp main_function) {
    if (!main_function.get_function_type()
            .cast<LLVM::LLVMFunctionType>()
            .get_return_type()
            .isa<Float32Type>())
        return make_string_error("only single f32 function result supported");
    return Error::success();
}
template <typename Type>
Error compile_and_execute_single_return_function(
    Options &options,
    *mut Operationmodule,
    StringRef entry_point,
    CompileAndExecuteConfig config) {
    let main_function = dyn_cast_or_null<LLVM::LLVMFuncOp>(
        SymbolTable::lookup_symbol_in(module, entry_point));
    if (!main_function || main_function.isExternal()){
        return make_string_error("entry point not found");}

    if (main_function.get_function_type()
            .cast<LLVM::LLVMFunctionType>()
            .getNumParams() != 0){
        return make_string_error("function inputs not supported");}

    if (Error error = check_compatible_return_type<Type>(main_function)){
        return error;}

    Type res;
    struct {
        *mut ()data;
    } data;
    data.data = &res;
    if (let error = 
        pub fn (options, module, entry_point,
                                        std::move(config), (*mut ()*)&data))
        return error;

    // Intentional printing of the output so we can test.
    llvm::outs() << res << '\n';

    return Error::success();
}
