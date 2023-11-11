/*!
# Layer Interfaces

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ExecutionEngine/Orc/Layer.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/ExecutionEngine/Orc/Layer.cpp>
*/

use crate::llvm::execution_engine::orc::{
    core::ExecutionSession,
    mangling::ManglingOptions
};

/**
IRMaterialisationUnit is a convenient base class for MaterializationUnits wrapping LLVM IR. Represents materialization responsibility for all symbols in the given module. If symbols are overridden by other definitions, then their linkage is changed to available-externally.
*/
pub struct IRMaterialisationUnit {

}

/// Interface for layers that accept LLVM IR.
pub struct IRLayer<'a> {
    clone_to_new_context_on_emit: bool,  // = false;
    execution_session: &'a ExecutionSession,
    mangling_options: Option<&'a ManglingOptions>,
}

/**
MaterialisationUnit that materializes modules by calling the 'emit' method
on the given IRLayer.
*/
pub struct BasicIRLayerMaterialisationUnit {

}

/// Interface for Layers that accept object files.
pub struct ObjectLayer {

}
