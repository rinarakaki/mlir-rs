//! This file forward declares and imports various common LLVM datatypes that
//! MLIR wants to use unqualified.
//!
//! Note that most of these are forward declared and then imported into the MLIR
//! namespace with using decls, rather than being #included.  This is because we
//! want clients to explicitly #include the files they need.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Support/LLVM.h>

use llvm::{
    adt::none,
    support::casting
};
