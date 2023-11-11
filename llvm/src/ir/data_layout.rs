/*!
# Data Size & Alignment Info

This file defines layout properties related to datatype size/offset/alignment information.  It uses lazy annotations to cache information about how structure types are laid out and used.

This structure should be created once, filled in if the defaults are not correct and then passed around by const&.  None of the members functions require modification to the object.

This file defines layout properties related to datatype size/offset/alignment information.

This structure should be created once, filled in if the defaults are not correct and then passed around by const&.  None of the members functions require modification to the object.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/DataLayout.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/IR/DataLayout.cpp>
*/

/// Enum used to categorize the alignment types stored by LayoutAlignElem
pub enum AlignType {
    Invalid = 0,
    Integer = 'I',
    Vector = 'V',
    Float = 'F',
    Aggregate = 'A'
}

/**
Layout alignment element.

Stores the alignment data associated with a given alignment type (integer, vector, float) and type bit width.

NOTE: The unusual order of elements in the structure attempts to reduce padding and make the structure slightly more cache friendly.
*/
pub struct LayoutAlignElem {

}

/**
Layout pointer alignment element.

Stores the alignment data associated with a given pointer and address space.

NOTE: The unusual order of elements in the structure attempts to reduce padding and make the structure slightly more cache friendly.
*/
pub struct PointerAlignElem {

}

/**
A parsed version of the target data layout string in and methods for querying it.

The target data layout string is specified *by the target* - a frontend generating LLVM IR is required to generate the right target data for the target being codegen'd to.
*/
pub struct DataLayout {
}

enum ManglingMode {
    None,
    ELF,
    MachO,
    WinCOFF,
    WinCOFFX86,
    GOFF,
    Mips,
    XCOFF
}

/**
Used to lazily calculate structure layout information for a target machine, based on the DataLayout structure.
*/
pub struct StructLayout {

}
