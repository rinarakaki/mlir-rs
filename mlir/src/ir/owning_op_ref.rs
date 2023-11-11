//! MLIR OwningOpRef
//!
//! This file provides a base class for owning op refs.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OwningOpRef.h>

use core::ops::Drop;
use rustc_data_structures::owning_ref::OwningRef;

/// This class acts as an owning reference to an op, and will automatically destroy the held op on destruction if the held op is valid.
///
/// Note that OpBuilder and related functionality should be highly preferred instead, and this should only be used in situations where existing solutions are not viable.
pub struct OwningOpRef<OpTy> {
    operation: OpTy
}

impl Drop for OwningOpRef<OpTy> {
    fn drop(&mut self) {
        
    }
}
