/*!
# Asm Interfaces for Operation

This file contains interfaces and other utilities for interacting with the AsmParser and AsmPrinter.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpAsmInterface.td>
*/

use crate::mlir::{
    ir::{
        attr_type_base::AttrOrTypeParameter,
        operation::{
            base,
            definition::OperationInterface
        }
    },
//    table_gen::attr_or_type_def::AttrOrTypeParameter
};

/// This interface provides hooks to interact with the AsmPrinter and AsmParser classes.
pub trait OpAsmOpInterface: OperationInterface {
    /**
    Get a special name to use when printing the results of this operation.
    The given callback is invoked with a specific result value that starts a result "pack", and the name to give this result pack. To signal that a result pack should use the default naming scheme, a None can be passed in instead of the name.

    For example, if you have an operation that has four results and you want to split these into three distinct groups you could do the following:

    ```cpp
    setNameFn(getResult(0), "first_result");
    setNameFn(getResult(1), "middle_results");
    setNameFn(getResult(3), ""); // use the default numbering.
    ```

    This would print the operation as follows:

    ```mlir
    %first_result, %middle_results:2, %0 = "my.op" ...
    ```
    */
    fn asm_output_names(&self, set_name: OpAsmSetValueNameFn);

    /**
    Get a special name to use when printing the block arguments for a region immediately nested under this operation.
    */
    fn asm_input_names(
        &self,
        region: &Region,
        set_name: OpAsmSetValueNameFn) {
    }

    /**
    Get the name to use for a given block inside a region attached to this operation.

    For example if this operation has multiple blocks:

        ```mlir
          some.op() ({
            ^bb0:
              ...
            ^bb1:
              ...
          })
        ```

        the method will be invoked on each of the blocks allowing the op to
        print:

        ```mlir
          some.op() ({
            ^custom_foo_name:
              ...
            ^custom_bar_name:
              ...
          })
        ```
    */
    fn asm_block_names(
        &self, set_name: OpAsmSetBlockNameFn) {
    }

    /**
    Return the default dialect used when printing/parsing operations in
      regions nested under this operation. This allows for eliding the dialect
      prefix from the operation name, for example it would be possible to omit
      the `spirv.` prefix from all operations within a SpirV module if this method
      returned `spv`. The default implementation returns an empty string which
      is ignored.
    */
    fn default_dialect() -> &'static str {
    }
}

/// This parameter represents a handle to a resource that is encoded into the `dialect_resources` section of the assembly format. This parameter expects a C++ `handle_type` that derives from `AsmDialectResourceHandleBase` and implements a derived handle to the desired resource type.
pub trait ResourceHandleParameter: AttrOrTypeParameter {
}
