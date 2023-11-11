//! This header file defines utilities and analyses for performing alias queries and related memory queries in MLIR.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Analysis/AliasAnalysis.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Analysis/AliasAnalysis.cpp>

pub mod local_alias_analysis;

use core::{
    cmp::PartialEq,
    fmt::Display
};
use llvm::{
    adt::small_vector::SmallVector,
    support::raw_ostream::RawOStream
};
use crate::ir::{
    operation::Operation,
    value::Value
};

/// The possible results of an alias query.
#[derive(PartialEq)]
pub enum AliasResult {
    /// The two locations do not alias at all.
    /// 
    /// This value is arranged to convert to false, while all other values convert to true. This allows a boolean context to convert the result to a binary flag indicating whether there is the possibility of aliasing.
    NoAlias = 0,
    /// The two locations may or may not alias. This is the least precise
    /// result.
    MayAlias,
    /// The two locations alias, but only due to a partial overlap.
    PartialAlias,
    /// The two locations precisely alias each other.
    MustAlias,
}

/**
The possible results of whether a memory access modifies or references a memory location. The possible results are: no access at all, a modification, a reference, or both a modification and a reference.

Note: This is a simplified version of the ModRefResult in `llvm/Analysis/AliasAnalysis.h`, and namely removes the `Must` concept. If this becomes useful/necessary we should add it here.
*/
#[must_use]
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub enum ModRefResult {
    /// The access neither references nor modifies the value stored in memory.
    NoModRef = 0,
    /// The access may reference the value stored in memory.
    Ref = 1,
    /// The access may modify the value stored in memory.
    Mod = 2,
    // /// The access may reference and may modify the value stored in memory.
    ModRef = 1 | 2
}

impl ModRefResult {
    // /// Print this ModRef result to the provided output stream.
    // pub const fn print(&self, os: &RawOStream) {
    //     write!(os, "{}", self);
    // }

    /// Merge this ModRef result with `other` and return the result.
    pub fn merge(&self, other: &Self) -> Self {
        self.max(other).clone()
    }

    /// Intersect this ModRef result with `other` and return the result.
    pub fn intersect(&self, other: &Self) -> Self {
        self.min(other).clone()
    }
}

impl Display for ModRefResult {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// This class contains various internal trait classes used by the main
/// AliasAnalysis class below.
struct AliasAnalysisTraits {
}

/// This class represents the `Concept` of an alias analysis implementation.
/// It is the abstract base class used by the AliasAnalysis class for querying into derived analysis implementations.
struct Concept {
}

/// This class represents the `Model` of an alias analysis implementation `ImplT`. A model is instantiated for each alias analysis implementation to implement the `Concept` without the need for the derived implementation to inherit from the `Concept` class.
struct Model<Impl> {
    _unimplemented: std::marker::PhantomData<Impl>
}

/// This class represents the main alias analysis interface in MLIR. It functions as an aggregate of various different alias analysis implementations. This aggregation allows for utilizing the strengths of different alias analysis implementations that either target or have access to different aliasing information. This is especially important for MLIR given the scope of different types of memory models and aliasing behaviors.
/// For users of this analysis that want to perform aliasing queries, see the `Alias Queries` section below for the available methods. For users of this analysis that want to add a new alias analysis implementation to the aggregate, see the `Alias Implementations` section below.
pub struct AliasAnalysis {
    /// A set of internal alias analysis implementations.
    impls: SmallVector<Box<Concept>>  // 4
}

impl AliasAnalysis {
    /**
    Add a new alias analysis implementation `AnalysisT` to this analysis aggregate. This allows for users to access this implementation when performing alias queries. Implementations added here must provide the following:
    
    - AnalysisT(AnalysisT &&)
    - AliasResult alias(Value lhs, Value rhs)
      - This method returns an `AliasResult` that corresponds to the
        aliasing behaviour between `lhs` and `rhs`. The conservative "I don't
        know" result of this method should be MayAlias.
    - ModRefResult getModRef(Operation *op, Value location)
      - This method returns a `ModRefResult` that corresponds to the
        modify-reference behaviour of `op` on the given `location`. The
        conservative "I don't know" result of this method should be ModRef.
    */
    pub fn add_analysis_implementation<Analysis>(
        &mut self, analysis: &&Analysis
    ) {
        // self.impls.push(
        //     std::make_unique<Model<Analysis>>(std::forward<Analysis>(analysis)));
    }

    // /// Given two values, return their aliasing behaviour.
    // pub fn alias(&self, lhs: Value, rhs: Value) -> AliasResult {
    //     // Check each of the alias analysis implemenations for an alias result.
    //     for r#impl in self.impls.iter() {
    //         let result = r#impl.alias(lhs, rhs);
    //         if !result.is_may() {
    //             return result;
    //         }
    //     }
    //     AliasResult::MayAlias
    // }

    // /// Return the modify-reference behaviour of `op` on `location`.
    // pub fn get_mod_ref(
    //     &self, operation: Option<Operation>, location: Value
    // ) -> ModRefResult
    // {
    //     /*
    //     Compute the mod-ref behaviour by refining a top `ModRef` result with each of the alias analysis implementations. We early exit at the point where we refine down to a `NoModRef`.
    //     */
    //     let result = ModRefResult::ModRef;
    //     for r#impl in self.impls.iter() {
    //         let result = result.intersect(
    //             r#impl.get_mod_ref(operation, location));
    //         if result == ModRefResult::NoModRef {
    //             return result;
    //         }
    //     }
    //     result
    // }
}
