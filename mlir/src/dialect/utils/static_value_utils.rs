/*!
This header file defines utilities for dealing with static values, e.g.,
converting back and forth between Value and FoldResult. Such functionality
is used in multiple dialects.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Utils/StaticValueUtils.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Utils/StaticValueUtils.cpp>
*/

use crate::{
    mlir::{
        dialect::arith::utils,
        ir::{
            matchers,
            operation::definition
        },
        support::llvm
    },
    llvm::adt::{
        aps_int,
        small_vector
    }
};
