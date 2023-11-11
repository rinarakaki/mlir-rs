/*!
This header defines enum values describing the structure of MLIR bytecode
files.

- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Bytecode/Encoding.h>
*/

#[repr(u8)]
enum ID {
    /// This section contains strings referenced within the bytecode.
    String,

    /// This section contains the dialects referenced within an IR module.
    Dialect,

    /// This section contains the attributes and types referenced within an IR
    /// module.
    AttrType,

    /// This section contains the offsets for the attribute and types within the
    /// AttrType section.
    AttrTypeOffset,

    /// This section contains the list of operations serialized into the bytecode,
    /// and their nested regions/operations.
    IR,

    /// This section contains the resources of the bytecode.
    Resource,

    /// This section contains the offsets of resources within the Resource
    /// section.
    ResourceOffset,

    /// The total number of section types.
    NumSections,
}
