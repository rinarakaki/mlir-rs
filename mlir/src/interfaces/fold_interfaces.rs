/*!
# Folding Interfaces

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/FoldInterfaces.h>
*/

use crate::{
    mlir::{
        ir::{
            attribute::Attribute,
            dialect::interface,
            operation::{
                Operation,
                definition::FoldResult,
            },
            region::Region
        },
        support::logical_result::LogicalResult
    },
    llvm::adt::small_vector::SmallVector
};

/**
Define a fold interface to allow for dialects to control specific aspects of the folding behaviour for operations they define.
*/
pub trait DialectFoldInterface {
    /**
    Registered fallback fold for the dialect. Like the fold hook of each operation, it attempts to fold the operation with the specified constant operand values - the elements in `inputs` will correspond directly to the inputs of the operation, but may be null if non-constant. If folding is successful, this fills in the `outputs` vector. If not, this returns failure and `outputs` is unspecified.
    */
    fn fold(
        operation: *mut Operation,
        inputs: &[dyn Attribute],
        outputs: &SmallVector<[FoldResult]>
    ) -> LogicalResult;

    /**
    Registered hook to check if the given region, which is attached to an operation that is *not* isolated from above, should be used when materialising constants. The folder will generally materialize constants into the top-level isolated region, this allows for materializing into a lower level ancestor region if it is more profitable/correct.
    */
    fn should_materialise_into(region: *mut Region) -> bool;
}

