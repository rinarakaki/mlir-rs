/*!
- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Pass/Pass.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Pass/Pass.cpp>
*/

use crate::{
    mlir::{
        ir::{
            diagnostics,
            dialect,
            operation::{
                Operation,
                definition
            },
            threading,
            verifier
        },
        pass::{
            analysis_manager::{AnalysisManager, PreservedAnalyses},
            pass_detail,
            pass_registry
        },
        support::file_utilities
    },
    llvm::{
        adt::{
            pointer_int_pair::PointerIntPair,
            statistic,
            stl_extras,
            scope_exit
        },
        support::{
            command_line,
            crash_recovery_context,
            mutex,
            signals,
            // threading,
            tool_output_file
        }
    }
};

/// The state for a single execution of a pass. This provides a unified
/// interface for accessing and initializing necessary state for pass execution.
pub struct PassExecutionState {
    /// The current operation being transformed and a bool for if the pass
    /// signaled a failure.
    ir_and_pass_failed: PointerIntPair<Option<Operation>, 1, bool>,

    /// The analysis manager for the operation.
    analysis_manager: AnalysisManager,

    /// The set of preserved analyses for the current execution.
    preserved_analyses: PreservedAnalyses,

    /// This is a callback in the PassManager that allows to schedule dynamic
    /// pipelines that will be rooted at the provided operation.
    // function_ref
    pipeline_executor: fn((&OpPassManager, Option<Operation>)) -> Result<(), Box<dyn std::error::Error>>
}

/// The abstract base pass class. This class contains information describing the
/// derived pass object, e.g its kind and abstract TypeID.
pub struct Pass {
}

struct OpPassManager {
}
