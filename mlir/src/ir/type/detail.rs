/*!
# MLIR Type Storage Details

This holds implementation details of Type.

- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/TypeDetail.h>
*/

use crate::{
    mlir::ir::{
        affine_map,
        builtins::types::Signedness,
        mlir_context,
        operation::support,
        type_range,
        r#type::Type
    },
    llvm::{
        adt::bit,
        support::trailing_objects
    }
};

/// Integer Type Storage and Uniquing.
pub struct IntegerTypeStorage {
    width: usize, // : 30;
    signedness: Signedness //: 2;
}

impl IntegerTypeStorage {
    fn new(width: usize, signedness: Signedness) -> Self {
        Self { width, signedness }
    }
}

/// Function Type Storage and Uniquing.
pub struct FunctionTypeStorage {
    num_inputs: usize,
    num_outputs: usize,
    inputs_and_outputs: Option<dyn Type>
}

/// A type representing a collection of other types.
pub struct TupleTypeStorage {

}
