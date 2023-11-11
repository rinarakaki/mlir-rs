/*!
This file contains the implementation of a local stateless alias analysis.
This analysis walks from the values being compared to determine their
potential for aliasing.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Analysis/AliasAnalysis/LocalAliasAnalysis.cpp>
*/

use crate::{
    mlir::{
        analysis::alias_analysis,
        ir::{
            matchers,
            function::interfaces
        },
        interfaces::{
            control_flow_interfaces,
            side_effect_interfaces,
            view_like_interface
        }
    }
};

/**
This class implements a local form of alias analysis that tries to identify
the underlying values addressed by each value and performs a few basic
checks to see if they alias.
*/
pub struct LocalAliasAnalysis {
}
