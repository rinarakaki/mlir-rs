//! This file defines various utilies for multithreaded processing within MLIR.
//! These utilities automatically handle many of the necessary threading
//! conditions, such as properly ordering diagnostics, observing if threading is
//! disabled, etc. These utilities should be used over other threading utilities
//! whenever feasible.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Threading.h>

use crate::{
    mlir::ir::{
        diagnostics,
        mlir_context::MLIRContext
    },
    llvm::{
        adt::sequence,
        support::thread_pool
    }
};
