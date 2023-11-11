/*!
- include <https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Utils/ReshapeOpsUtils.cpp>
*/

use crate::{
    mlir::{
        dialect::utils::static_value_utils,
        ir::{
            affine_map,
            builders,
            operation::implementation,
            pattern_match
        },
        support::llvm
    },
    llvm::adt::string_ref
};
