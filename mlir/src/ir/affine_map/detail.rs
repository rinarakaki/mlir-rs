/*!
# MLIR Affine Map Details Class

This holds implementation details of AffineMap.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/AffineMapDetail.h>
*/

use crate::{
    mlir::{
        ir::{
            affine_expr,
            affine_map
        },
        support::storage_uniquer
    },
    llvm::{
        adt::array_ref,
        support::trailing_objects
    }
};
