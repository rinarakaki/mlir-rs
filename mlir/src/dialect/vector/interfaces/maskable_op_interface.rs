/*!
# MaskableOpInterface Defs

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Vector/Interfaces/MaskableOpInterface.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Vector/Interfaces/MaskableOpInterface.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Vector/Interfaces/MaskableOpInterface.cpp>
*/

use super::masking_op_interface::MaskingOpInterface;

/**
The 'MaskableOpInterface' defines an operation that can be masked using a MaskingOpInterface (e.g., `vector.mask`) and provides information about its masking constraints and semantics.
*/
pub trait MaskableOpInterface: OpInterface {
    /**
    Returns true if the operation is masked by a MaskingOpInterface.
    */
    fn is_masked(&self) -> bool {
        let parent = self.parent();
        !parent.is_null() &&
               isa::<MaskingOpInterface>(parent);
    }

    /// Returns the MaskingOpInterface masking this operation.
    fn masking_op(&self) -> MaskingOpInterface {
        cast<MaskingOpInterface>(self.parent())
    }

    /**
    Returns true if the operation can have a passthru argument when masked.
    */
    fn supports_passthru(&self) -> bool {
        false
    }

    /**
    Returns the mask type expected by this operation. Mostly used for verification purposes. It requires the operation to be vectorised.
    */
    fn expected_mask_type(&self) -> Type;
}
