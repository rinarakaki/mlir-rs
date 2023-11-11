/*!
This header defines various interfaces and utilities necessary for dialects
to hook into bytecode serialisation.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Bytecode/BytecodeImplementation.h>
*/

use crate::{
    mlir::{
        ir::{
            attribute::Attribute,
            diagnostics,
            dialect::{
                self,
                interface
            },
            operation::implementation
        }
    },
    llvm::adt::twine::Twine
};

/**
This class defines a virtual interface for reading a bytecode stream, providing hooks into the bytecode reader. As such, this class should only be derived and defined by the main bytecode reader, users (i.e. dialects) should generally only interact with this class via the BytecodeDialectInterface below.
*/
pub struct DialectBytecodeReader {
}

/**
This class defines a virtual interface for writing to a bytecode stream, providing hooks into the bytecode writer. As such, this class should only be derived and defined by the main bytecode writer, users (i.e. dialects) should generally only interact with this class via the BytecodeDialectInterface below.
*/
pub struct DialectBytecodeWriter {
}
