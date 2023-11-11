//! - include <https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Interfaces>
//! - lib <https://github.com/llvm/llvm-project/tree/main/mlir/lib/Interfaces>

pub mod cast_interfaces;
pub mod call_interfaces;
pub mod control_flow_interfaces;
pub mod copy_op_interface;
pub mod data_layout_interfaces;
pub mod destination_style_op_interface;
pub mod fold_interfaces;
pub mod infer_int_range_interface;
pub mod infer_type_op_interface;
pub mod parallel_combining_op_interface;
pub mod shaped_op_interfaces;
pub mod side_effect_interface_base;
pub mod side_effect_interfaces;
pub mod tiling_interface;
pub mod vector_interfaces;
pub mod view_like_interface;
