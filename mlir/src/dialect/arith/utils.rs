/*!
This header file defines prototypes for various transformation utilities for
the Arith dialect. These are not passes by themselves but are used
either by passes, optimization sequences, or in turn by other transformation
utilities.

This file implements utilities for the Linalg dialect.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Arith/Utils/Utils.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Arith/Utils/Utils.cpp>
*/

use crate::{
    mlir::{
        dialect::arith::ir::arith,
        ir::{
            matchers,
            pattern_match,
            value
        }
    },
    llvm::adt::small_bit_vector
};
