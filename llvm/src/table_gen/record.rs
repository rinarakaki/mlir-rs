/*!
- include <https://github.com/llvm/llvm-project/tree/main/llvm/include/llvm/TableGen/Record.h>
- lib <https://github.com/llvm/llvm-project/tree/main/llvm/lib/TableGen/Record.cpp>
*/

pub struct RecTy {
}

/// Subclass discriminator (for dyn_cast<> et al.)
enum RecTyKind {
    Bit,
    Bits,
    Int,
    String,
    List,
    Dag,
    Record
}

pub struct Init {
}

/**
Discriminator enum (for isa<>, dyn_cast<>, et al.)

This enum is laid out by a preorder traversal of the inheritance
hierarchy, and does not contain an entry for abstract classes, as per
the recommendation in docs/HowToSetUpLLVMStyleRTTI.rst.

We also explicitly include "first" and "last" values for each
interior node of the inheritance tree, to make it easier to read the
corresponding classof().

We could pack these a bit tighter by not having the FirstXXXInit
and LastXXXInit be their own values, but that would degrade
readability for really no benefit.
*/
#[repr(align(8))]
enum InitKind{
    First, // unused; silence a spurious warning
    FirstTypedInit,
    BitInit,
    BitsInit,
    DagInit,
    DefInit,
    FieldInit,
    IntInit,
    ListInit,
    FirstOpInit,
    BinOpInit,
    TernOpInit,
    UnOpInit,
    LastOpInit,
    CondOpInit,
    FoldOpInit,
    IsAOpInit,
    ExistsOpInit,
    AnonymousNameInit,
    StringInit,
    VarInit,
    VarListElementInit,
    VarBitInit,
    VarDefInit,
    LastTypedInit,
    UnsetInit
}

/// This is the common superclass of types that have a specific,
/// explicit type, stored in ValueTy.
pub struct TypedInit {

}

/// This class represents a field in a record, including its name, type,
/// value, and source location.
pub struct RecordVal {

}

pub struct Record {
}
