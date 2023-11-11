/*!
An ORC-based JIT for compiling LLVM IR.

- include <https://github.com/llvm/llvm-project/tree/main/llvm/include/llvm/ExecutionEngine/Orc/LLJIT.h>
- lib <https://github.com/llvm/llvm-project/tree/main/llvm/lib/ExecutionEngine/Orc/LLJIT.cpp>
*/

use crate::llvm::{
    execution_engine::orc::{
        compile_on_demand_layer,
        compile_utils,
        core::{ExecutionSession, JITDylib},
        execution_utils,
        ir_compile_layer::IRCompileLayer,
        ir_transform_layer::IRTransformLayer,
        jit_target_machine_builder,
        layer::ObjectLayer,
        object_transform_layer::ObjectTransformLayer,
        thread_safe_module
    },
    ir::data_layout::DataLayout,
    support::thread_pool::ThreadPool,
    target_parser::triple::Triple
};

/**
A pre-fabricated ORC JIT stack that can serve as an alternative to MCJIT.

Create instances using LLJITBuilder.
*/
pub struct LLJIT {
    es: Box<ExecutionSession>,
    ps: Box<PlatformSupport>,

    main: Option<JITDylib>,  // * = nullptr;

    dl: DataLayout,
    tt: Triple,
    compile_threads: Box<ThreadPool>,

    obj_linking_layer: Box<ObjectLayer>,
    obj_transform_layer: Box<ObjectTransformLayer>,
    compile_layer: Box<IRCompileLayer>,
    transform_layer: Box<IRTransformLayer>,
    init_helper_transform_layer: Box<IRTransformLayer>
}

/// Initialiser support for LLJIT.
pub struct PlatformSupport {
}

/**
An extended version of LLJIT that supports lazy function-at-a-time compilation of LLVM IR.
*/
pub struct LLLazyJIT {

}

pub struct LLJITBuilderState {
}
