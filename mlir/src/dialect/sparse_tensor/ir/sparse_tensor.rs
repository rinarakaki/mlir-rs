/*!
# Sparse tensor dialect

- include <https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Dialect/SparseTensor/IR/SparseTensor.h>
*/

use crate::mlir::{
    dialect::sparse_tensor::ir::enums,
    ir::{
        dialect,
        operation::{
            definition,
            implementation
        },
        tensor_encoding
    },
    interfaces::{
        infer_type_op_interface,
        side_effect_interfaces
    }
};
