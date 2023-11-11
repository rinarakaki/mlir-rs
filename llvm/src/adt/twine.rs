/*!
# Fast Temporary String Concatenation

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/ADT/Twine.h>
*/

use crate::llvm::{
    adt::small_vector,
    support::error_handling
};

/**
A lightweight data structure for efficiently representing the concatenation of temporary values as strings.

A Twine is a kind of rope, it represents a concatenated string using a binary-tree, where the string is the preorder of the nodes. Since the Twine can be efficiently rendered into a buffer when its result is used, it avoids the cost of generating temporary values for intermediate string results -- particularly in cases when the Twine result is never required. By explicitly tracking the type of leaf nodes, we can also avoid the creation of temporary strings for conversions operations (such as appending an integer to a string).

A Twine is not intended for use directly and should not be stored, its implementation relies on the ability to store pointers to temporary stack objects which may be deallocated at the end of a statement. Twines should only be used accepted as const references in arguments, when an API wishes to accept possibly-concatenated strings.

Twines support a special 'null' value, which always concatenates to form itself, and renders as an empty string. This can be returned from APIs to effectively nullify any concatenations performed on the result.

# Implementation

Given the nature of a Twine, it is not possible for the Twine's concatenation method to construct interior nodes; the result must be represented inside the returned value. For this reason a Twine object actually holds two values, the left- and right-hand sides of a concatenation. We also have nullary Twine objects, which are effectively sentinel values that represent empty strings.

Thus, a Twine can effectively have zero, one, or two children. The \see
isNullary(), \see isUnary(), and \see isBinary() predicates exist for
testing the number of children.

We maintain a number of invariants on Twine objects (FIXME: Why):
 - Nullary twines are always represented with their Kind on the left-hand
   side, and the Empty kind on the right-hand side.
 - Unary twines are always represented with the value on the left-hand
   side, and the Empty kind on the right-hand side.
 - If a Twine has another Twine as a child, that child should always be
   binary (otherwise it could have been folded into the parent).

These invariants are check by \see isValid().

# Efficiency Considerations

The Twine is designed to yield efficient and small code for common situations. For this reason, the concat() method is inlined so that concatenations of leaf nodes can be optimized into stores directly into a single stack allocated object.

In practice, not all compilers can be trusted to optimize concat() fully, so we provide two additional methods (and accompanying operator+ overloads) to guarantee that particularly important cases (cstring plus StringRef) codegen as desired.
*/
pub struct Twine {
    /**
    Prefix in the concatenation, which may be uninitialised for Null or Empty kinds.
    */
    lhs: Child,

    /**
    Suffix in the concatenation, which may be uninitialised for Null or Empty kinds.
    */
    rhs: Child,

    /// NodeKind of the left hand side, \see getLHSKind().
    lhs_kind: NodeKind,  // = EmptyKind;

    /// NodeKind of the right hand side, \see getRHSKind().
    rhs_kind: NodeKind // = EmptyKind;
}

/// NodeKind - Represent the type of an argument.
#[repr(u8)]
enum NodeKind {
    /// An empty string; the result of concatenating anything with it is also
    /// empty.
    NullKind,

    /// The empty string.
    EmptyKind,

    /// A pointer to a Twine instance.
    TwineKind,

    /// A pointer to a C string instance.
    CStringKind,

    /// A pointer to an std::string instance.
    StdStringKind,

    /**
    A Pointer and Length representation. Used for std::string_view, StringRef, and SmallString.  Can't use a StringRef here because they are not trivally constructible.
    */
    PtrAndLengthKind,

    /// A pointer to a formatv_object_base instance.
    FormatvObjectKind,

    /// A char value, to render as a character.
    CharKind,

    /// An unsigned int value, to render as an unsigned decimal integer.
    DecUIKind,

    /// An int value, to render as a signed decimal integer.
    DecIKind,

    /// A pointer to an unsigned long value, to render as an unsigned decimal
    /// integer.
    DecULKind,

    /// A pointer to a long value, to render as a signed decimal integer.
    DecLKind,

    /// A pointer to an unsigned long long value, to render as an unsigned
    /// decimal integer.
    DecULLKind,

    /// A pointer to a long long value, to render as a signed decimal integer.
    DecLLKind,

    /**
    A pointer to a uint64_t value, to render as an unsigned hexadecimal integer.
    */
    UHexKind
}

// union
enum Child {
//     const Twine *twine;
//     const char *cString;
//     const std::string *stdString;
//     struct {
//         const char *ptr;
//         size_t length;
//     } ptrAndLength;
//     const formatv_object_base *formatvObject;
//     char character;
//     unsigned int decUI;
//     int decI;
//     const unsigned long *decUL;
//     const long *decL;
//     const unsigned long long *decULL;
//     const long long *decLL;
//     const uint64_t *uHex;
}
