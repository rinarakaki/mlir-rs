/*!
# MLIR Optimiser Driver Main

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Tools/mlir-opt/mlir_opt_main.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Tools/mlir-opt/mlir_opt_main.cpp>
*/

use clap::{ArgAction::SetTrue, Args, Parser, Subcommand, ValueEnum};

use llvm::{
    mlir::{
        bytecode::writer::{BytecodeWriterConfig, write_bytecode_to_file},
        ir::{
            mlir_context::MLIRContext,
            dialect::registry::DialectRegistry,
            asm_state::{ParserConfig, FallbackAsmResourceMap, AsmState}
        },
        pass::pass_manager::PassManager,
        support::timing,
    },
    llvm::support::source_manager::SourceManager
};

/**
Perform the core processing behind `mlir-opt`:

- `output_stream` is the stream where the resulting IR is printed.
- `buffer` is the in-memory file to parser and process.
- `pass_pipeline` is the specification of the pipeline that will be applied.
- `registry` should contain all the dialects that can be parsed in the source.
- `split_input_file` will look for a "-----" marker in the input file, and load
each chunk in an individual ModuleOp processed separately.
- `verify_diagnostics` enables a verification mode where comments starting with
`expected-(error|note|remark|warning)` are parsed in the input and matched against emitted diagnostics.
- `verify_passes` enables the IR verifier in-between each pass in the pipeline.
- `allow_unregistered_dialects` allows to parse and create operation without
registering the Dialect in the MLIRContext.
- `preload_dialects_in_context` will trigger the upfront loading of all dialects from the global registry in the MLIRContext. This option is deprecated and will be removed soon.
- `emit_bytecode` will generate bytecode output instead of text.
- `implicit_module` will enable implicit addition of a top-level `builtin.module` if one doesn't already exist.
- `dump_pass_pipeline` will dump the pipeline being run to stderr
*/

pub fn mlir_opt_main(
    output_stream: &raw_ostream,
    buffer: std::unique_ptr<MemoryBuffer>,
    pass_pipeline: &PassPipelineCLParser,
    registry: &mut DialectRegistry,
    split_input_file: bool,
    verify_diagnostics: bool,
    verify_passes: bool,
    allow_unregistered_dialects: bool,
    preload_dialects_in_context: bool /* false */,
    emit_bytecode: bool /* false */,
    implicit_module: bool /* false */,
    dump_pass_pipeline: bool /* false */
) -> LogicalResult
{

}

/**
Support a callback to setup the pass manager.

- pass_manager_setup_fn is the callback invoked to setup the pass manager to
  apply on the loaded IR.
*/
pub fn mlir_opt_main(
    output_stream: &mut raw_ostream,
    buffer: std::unique_ptr<MemoryBuffer>,
    pass_manager_setup_fn: PassPipelineFn,
    registry: &mut DialectRegistry,
    split_input_file: bool,
    verify_diagnostics: bool,
    verify_passes: bool,
    allow_unregistered_dialects: bool,
    preload_dialects_in_context: bool,  //  = false,
    emit_bytecode: bool,  // = false,
    implicit_module: bool  // = false
) -> LogicalResult
{

}

/**
Implementation for tools like `mlir-opt`.

- tool_name is used for the header displayed by `--help`.
- registry should contain all the dialects that can be parsed in the source.
- preload_dialects_in_context will trigger the upfront loading of all
  dialects from the global registry in the MLIRContext. This option is
  deprecated and will be removed soon.
*/
pub fn mlir_opt_main(
    int argc,
    char **argv,
    tool_name: &str,
    registry: &mut DialectRegistry,
    preload_dialects_in_context: bool  // = false
) -> LogicalResult
{
    InitLLVM y(argc, argv);

    // Register any command line options.
    registerAsmPrinterCLOptions();
    registerMLIRContextCLOptions();
    registerPassManagerCLOptions();
    registerDefaultTimingManagerCLOptions();
    DebugCounter::registerCLOptions();
    PassPipelineCLParser pass_pipeline("", "Compiler passes to run", "p");
  
    // Build the list of dialects as a header for the --help message.
    let help_header = (tool_name + "\nAvailable Dialects: ").str();
    {
      raw_string_ostream os(help_header);
      interleaveComma(registry.get_dialect_names(), os,
                      |name| { os << name; });
    }
    // Parse pass names in main to ensure static initialization completed.
    cl::ParseCommandLineOptions(argc, argv, help_header);

    let mlir_opt = MLIROpt::parse();
  
    if mlir_opt.show_dialects {
        outs() << "Available Dialects:\n";
        interleave(
            registry.get_dialect_names(), outs(),
            |name| { outs() << name; }, "\n");
        return Ok(());
    }
  
    // Set up the input file.
    let mut error_message = String::new();
    let file = openInputFile(mlir_opt.input_filename, &error_message);
    if !file {
        errs() << error_message << "\n";
        return Err(());
    }
  
    let output = openOutputFile(mlir_opt.output_filename, &error_message);
    if (!output) {
        errs() << error_message << "\n";
        return Err(());
    }
  
    if failed(mlir_opt_main(
        output.os(),
        std::move(file),
        pass_pipeline, registry,
        split_input_file,
        verify_diagnostics,
        verify_passes,
        allow_unregistered_dialects,
        preload_dialects_in_context,
        emit_bytecode,
        /*implicit_module=*/!no_implicit_module,
        dump_pass_pipeline)
    ) {
        return Err(());
    }
  
    // Keep the output file if the invocation of mlir_opt_main was successful.
    output.keep();
    Ok(())
}

/**
Helper wrapper to return the result of `mlir_opt_main` directly from main.

# Example

```cpp
int main(int argc, char **argv) {
    // ...
    return mlir::as_main_return_code(mlir::mlir_opt_main(
        argc, argv, /* ... */);
}
```
*/
inline int as_main_return_code(LogicalResult r) {
  return r.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}


////////////////////////////////////////////////////////

/**
Perform the actions on the input file indicated by the command line flags within the specified context.

This typically parses the main source file, runs zero or more optimization passes, then prints the output.

*/
pub fn perform_actions(
    os: &raw_ostream,
    verify_diagnostics: bool,
    verify_passes: bool,
    source_manager: &std::shared_ptr<SourceManager>,
    context: *mut MLIRContext,
    pass_manager_setup_fn: PassPipelineFn,
    emit_bytecode: bool,
    implicit_module: bool
) -> LogicalResult
{
    DefaultTimingManager tm;
    applyDefaultTimingManagerCLOptions(tm);
    let timing = tm.getRootScope();

    // Disable multi-threading when parsing the input file. This removes the
    // unnecessary/costly context synchronization when parsing.
    let was_threading_enabled = context.is_multithreading_enabled();
    context.disableMultithreading();

    // Prepare the parser config, and attach any useful/necessary resource
    // handlers. Unhandled external resources are treated as passthrough, i.e.
    // they are not processed and will be emitted directly to the output
    // untouched.
    PassReproducerOptions repro_options;
    FallbackAsmResourceMap fallback_resource_map;
    ParserConfig config(context, /*verifyAfterParse=*/true, &fallback_resource_map);
    repro_options.attachResourceParser(config);

    // Parse the input file and reset the context threading state.
    let parser_timing = timing.nest("Parser");
    let op =
        parseSourceFileForTool(source_manager, config, implicit_module);
    context.enable_multithreading(was_threading_enabled);
    if (!op) {
        return Err(());
    }
    parser_timing.stop();

    // Prepare the pass manager, applying command-line and reproducer options.
    PassManager pm(context, OpPassManager::Nesting::Implicit,
                    op.get().getName().get&str());
    pm.enable_verifier(verify_passes);
    applyPassManagerCLOptions(pm);
    pm.enableTiming(timing);
    if (failed(repro_options.apply(pm)) || failed(pass_manager_setup_fn(pm)))
        return Err(());

    // Run the pipeline.
    if (failed(pm.run(*op)))
        return Err(());

    // Print the output.
    let output_timing = timing.nest("Output");
    if (emit_bytecode) {
        BytecodeWriterConfig writer_config(fallback_resource_map);
        write_bytecode_to_file(op.get(), os, writer_config);
    } else {
        AsmState asmState(op.get(), OpPrintingFlags(), /*locationMap=*/nullptr,
                        &fallback_resource_map);
        op.get().print(os, asmState);
        os << '\n';
    }
    return Ok(());
}

/**
Parses the memory buffer.  If successfully, run a series of passes against
it and print the result.
*/
pub fn process_buffer(
    os: &raw_ostream,
    owned_buffer: std::unique_ptr<MemoryBuffer>,
    verify_diagnostics: bool,
    verify_passes: bool,
    allow_unregistered_dialects: bool,
    preload_dialects_in_context: bool,
    emit_bytecode: bool,
    implicit_module: bool,
    pass_manager_setup_fn: PassPipelineFn,
    registry: &mut DialectRegistry,
    thread_pool: *mut ThreadPool
) -> LogicalResult
{
    // Tell source_manager about this buffer, which is what the parser will pick up.
    let source_manager = std::make_shared<SourceManager>();
    source_manager.add_new_source_buffer(std::move(owned_buffer), SMLoc());

    // Create a context just for the current buffer. Disable threading on creation
    // since we'll inject the thread-pool separately.
    let context = MLIRContext::new(registry, MLIRContext::Threading::DISABLED);
    if (thread_pool) {
        context.set_thread_pool(*thread_pool);
    }

    // Parse the input file.
    if (preload_dialects_in_context) {
        context.load_all_available_dialects();
    }
    context.allow_unregistered_dialects(allow_unregistered_dialects);
    if (verify_diagnostics) {
        context.print_op_on_diagnostic(false);
    }
    context.get_debug_action_manager().register_action_handler<DebugCounter>();

    // If we are in verify diagnostics mode then we have a lot of work to do,
    // otherwise just perform the actions without worrying about it.
    if (!verify_diagnostics) {
        SourceManagerDiagnosticHandler SourceManagerHandler(*source_manager, &context);
        return perform_actions(
            os, verify_diagnostics, verify_passes, source_manager,
            &context, pass_manager_setup_fn, emit_bytecode,
            implicit_module);
    }

    SourceManagerDiagnosticVerifierHandler SourceManagerHandler(*source_manager, &context);

    // Do any processing requested by command line flags.  We don't care whether
    // these actions succeed or fail, we only care what diagnostics they produce
    // and whether they match our expectations.
    (void)perform_actions(os, verify_diagnostics, verify_passes, source_manager, &context,
                        pass_manager_setup_fn, emit_bytecode, implicit_module);

    // Verify the diagnostic handler to make sure that each of the diagnostics
    // matched.
    return SourceManagerHandler.verify();
}

LogicalResult
mlir::mlir_opt_main(
    output_stream: &mut raw_ostream,
    buffer: std::unique_ptr<MemoryBuffer>,
    pass_manager_setup_fn: PassPipelineFn,
    registry: &mut DialectRegistry,
    split_input_file: bool,
    verify_diagnostics: bool,
    verify_passes: bool,
    allow_unregistered_dialects: bool,
    preload_dialects_in_context: bool,
    emit_bytecode: bool,
    implicit_module: bool
) {
    // The split-input-file mode is a very specific mode that slices the file
    // up into small pieces and checks each independently.
    // We use an explicit threadpool to avoid creating and joining/destroying
    // threads for each of the split.
    ThreadPool *thread_pool = nullptr;

    // Create a temporary context for the sake of checking if
    // --mlir-disable-threading was passed on the command line.
    // We use the thread-pool this context is creating, and avoid
    // creating any thread when disabled.
    MLIRContext threadPoolCtx;
    if (threadPoolCtx.is_multithreading_enabled())
        thread_pool = &threadPoolCtx.getThreadPool();

    let chunkFn
        = |chunk_buffer: std::unique_ptr<MemoryBuffer>, os: &raw_ostream|
            process_buffer(
                os, std::move(chunk_buffer), verify_diagnostics,
                verify_passes, allow_unregistered_dialects,
                preload_dialects_in_context, emit_bytecode, implicit_module,
                pass_manager_setup_fn, registry, thread_pool);
    };
    split_and_process_buffer(
        std::move(buffer), chunkFn, output_stream,
        split_input_file, /*insertMarkerInOutput=*/true)
}

LogicalResult mlir::mlir_opt_main(
    output_stream: &mut raw_ostream,
    buffer: std::unique_ptr<MemoryBuffer>,
    pass_pipeline: &PassPipelineCLParser,
    registry: &mut DialectRegistry,
    split_input_file: bool,
    verify_diagnostics: bool,
    verify_passes: bool,
    allow_unregistered_dialects: bool,
    preload_dialects_in_context: bool,
    emit_bytecode: bool,
    implicit_module: bool,
    dump_pass_pipeline: bool
) {
    let pass_manager_setup_fn = [&](PassManager &pm) {
        let errorHandler = [&](const Twine &msg) {
        emitError(UnknownLoc::get(pm.getContext())) << msg;
        return Err(());
        };
        if (failed(pass_pipeline.addToPipeline(pm, errorHandler)))
        return Err(());
        if (dump_pass_pipeline) {
        pm.dump();
        errs() << "\n";
        }
        return Ok(());
    };
    return mlir_opt_main(
        output_stream,
        std::move(buffer),
        pass_manager_setup_fn,
        registry, split_input_file, verify_diagnostics, verify_passes,
        allow_unregistered_dialects, preload_dialects_in_context,
        emit_bytecode, implicit_module);
}


#[derive(Debug, Parser)]
pub struct MLIROpt {
    /// <input file>
    #[arg(default_value_t = "-")]
    input_filename: String,
    // cl::value_desc("filename"),
    /// Output filename.
    #[arg(short = 'o', long, default_value_t = "-")]
    output_filename: String,
    /// Split the input file into pieces and process each chunk independently.
    #[arg(long, default_value_t = false)]
    split_input_file: bool,
    /// Check that emitted diagnostics match expected-* lines on the corresponding line.
    #[arg(default_value_t = false)]
    verify_diagnostics: bool,
    /// Run the verifier after each transformation pass.
    #[arg(long = "verify-each", default_value_t = true)]
    verify_passes: bool,
    /// Allow operation with no registered dialects.
    #[arg(long, default_value_t = false)]
    allow_unregistered_dialects: bool,
    /// Print the list of registered dialects.
    #[arg(long, default_value_t = false)]
    show_dialects: bool,
    /// Emit bytecode when generating output.
    #[arg(long, default_value_t = false)]
    emit_bytecode: bool,
    /// Disable implicit addition of a top-level module op during parsing.
    #[arg(long, default_value_t = false)]
    no_implicit_module: bool,
    /// Print the pipeline that will be run.
    #[arg(long, default_value_t = false)]
    dump_pass_pipeline: bool
}
