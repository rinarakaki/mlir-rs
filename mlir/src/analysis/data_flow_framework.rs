//! This file defines a generic framework for writing data-flow analysis in MLIR.
//! The framework consists of a solver, which runs the fixed-point iteration and manages analysis dependencies, and a data-flow analysis class used to implement specific analyses.
//! 
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Analysis/DataFlowFramework.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Analysis/DataFlowFramework.cpp>

use core::{
    collections::VecDeque,
    cmp::PartialEq,
    ops::{BitAnd, BitOr}
};
use llvm::{
    adt::{
        dense_map::DenseMap,
        set_vector,
    },
    support::{
        debug,
        type_name
    }
};
use crate::{
    ir::operation,
    support::{
        storage_uniquer::StorageUniquer,
        type_id::TypeId
    }
};

/// A result type used to indicate if a change happened. Boolean operations on
/// ChangeResult behave as though `Change` is truthy.
pub enum ChangeResult {
    NoChange,
    Change,
}

/// Abstract class for generic program points. In classical data-flow analysis, programs points represent positions in a program to which lattice elements are attached. In sparse data-flow analysis, these can be SSA values, and in dense data-flow analysis, these are the program points before and after every operation.
/// 
/// In the general MLIR data-flow analysis framework, program points are an extensible concept. Program points are uniquely identifiable objects to which analysis states can be attached. The semantics of program points are defined by the analyses that specify their transfer functions.
/// 
/// Program points are implemented using MLIR's storage uniquer framework and type ID system to provide RTTI.
pub struct GenericProgramPoint {
    /// The type identifier of the program point.
    type_id: TypeId
}

impl GenericProgramPoint {
    /// Get the abstract program point's type identifier.
    pub const fn type_id(&self) -> TypeId {
        self.type_id
    }
}

/// Base class for generic program points based on a concrete program point type and a content key. This class defines the common methods required for operability with the storage uniquer framework.
/// 
/// The provided key type uniquely identifies the concrete program point instance and are the data members of the class.
/// 
/// Two program points are equal if their values are equal.
#[derive(PartialEq)]
pub struct GenericProgramPointBase<Value> {
    /// The program point value.
    value: Value,
}

// impl<Concrete, Value> GenericProgramPointBase<Value> {
//     /// Provide LLVM-style RTTI using type IDs.
//     pub fn classof(point: GenericProgramPoint) -> bool {
//         point.type_id() == TypeId::of::<Concrete>()
//     }

//     /// Get the contents of the program point.
//     pub const fn value(&self) -> &Value {
//         &self.value
//     }
// }

/// Fundamental IR components are supported as first-class program points.
pub struct ProgramPoint {
}

/**
The general data-flow analysis solver. This class is responsible for orchestrating child data-flow analyses, running the fixed-point iteration algorithm, managing analysis state and program point memory, and tracking dependencies beteen analyses, program points, and analysis states.

Steps to run a data-flow analysis:

1. Load and initialise children analyses. Children analyses are instantiated
   in the solver and initialised, building their dependency relations.
2. Configure and run the analysis. The solver invokes the children analyses
   according to their dependency relations until a fixed point is reached.
3. Query analysis state results from the solver.

TODO: Optimize the internal implementation of the solver.
*/
pub struct DataFlowSolver<'a> {
    /**
    The solver's work queue. Work items can be inserted to the front of the queue to be processed greedily, speeding up computations that otherwise quickly degenerate to quadratic due to propagation of state updates.
    */
    worklist: VecDeque<WorkItem<'a>>,

    /// Type-erased instances of the children analyses.
    child_analyses: Vec<Box<DataFlowAnalysis<'a>>>,

    /// The storage uniquer instance that owns the memory of the allocated program
    /// points.
    uniquer: StorageUniquer,

    /// A type-erased map of program points to associated analysis states for
    /// first-class program points.
    analysis_states: DenseMap<(ProgramPoint, TypeId), Box<AnalysisState>>
}

type WorkItem<'a> = (ProgramPoint, Option<DataFlowAnalysis<'a>>);

/**
Base class for generic analysis states. Analysis states contain data-flow information that are attached to program points and which evolve as the analysis iterates.

This class places no restrictions on the semantics of analysis states beyond these requirements.

1. Querying the state of a program point prior to visiting that point results in uninitialised state. Analyses must be aware of unintialised states.
2. Analysis states can reach fixpoints, where subsequent updates will never trigger a change in the state.
3. Analysis states that are uninitialised can be forcefully initialised to a default value.
*/
pub struct AnalysisState {
}

/**
Base class for all data-flow analyses. A child analysis is expected to build an initial dependency graph (and optionally provide an initial state) when
initialised and define transfer functions when visiting program points.

In classical data-flow analysis, the dependency graph is fixed and analyses define explicit transfer functions between input states and output states.
In this framework, however, the dependency graph can change during the analysis, and transfer functions are opaque such that the solver doesn't know what states calling `visit` on an analysis will be updated. This allows multiple analyses to plug in and provide values for the same state.

Generally, when an analysis queries an uninitialised state, it is expected to "bail out", i.e., not provide any updates. When the value is initialised, the solver will re-invoke the analysis. If the solver exhausts its worklist, however, and there are still uninitialised states, the solver "nudges" the analyses by default-initialising those states.
*/
pub struct DataFlowAnalysis<'a> {
    /// The parent data-flow solver.
    solver: &'a DataFlowSolver<'a>
}
