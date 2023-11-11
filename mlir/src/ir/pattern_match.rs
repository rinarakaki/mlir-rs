//! PatternMatcher Classes
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/PatternMatch.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/PatternMatch.cpp>

use crate::{
    mlir::ir::{
        block_and_value_mapping,
        builders,
        builtins::operations,
        mlir_context::MLIRContext,
        operation::support::OperationName
    },
    llvm::{
        adt::{
            function_extras,
            pointer_int_pair::PointerIntPair,
            small_vector::SmallVector
        },
        support::type_name
    }
};

/// This class represents the benefit of a pattern match in a unitless scheme that ranges from 0 (very little benefit) to 65K.  The most common unit to use here is the "number of operations matched" by the pattern.
///
/// This also has a sentinel representation that can be used for patterns that fail to match.
pub struct PatternBenefit;

/// This class contains all of the data related to a pattern, but does not contain any methods or logic for the actual matching. This class is solely used to interface with the metadata of a pattern, such as the benefit or root operation.
pub struct Pattern {
    /// The value used to match the root operation of the pattern.
    root_value: *const (),
    root_kind: RootKind,

    /// The expected benefit of matching this pattern.
    benefit: PatternBenefit,

    /**
    The context this pattern was created from, and a boolean flag indicating whether this pattern has bounded recursion or not.
    */
    context_and_has_bounded_recursion: PointerIntPair<*mut MLIRContext, 1, bool>,

    /// A list of the potential operations that may be generated when rewriting an op with this pattern.
    generated_operations: SmallVector<[OperationName; 2]>,

    /// A readable name for this pattern. May be empty.
    debug_name: &'static str,

    /// The set of debug labels attached to this pattern.
    debug_labels: SmallVector<[&'static str; 0]>
}

impl Pattern {
    /// Return a list of operations that may be generated when rewriting an
    /// operation instance with this pattern.
    &[OperationName] get_generated_ops() const { return generatedOps; }

    /// Return the root node that this pattern matches. Patterns that can match
    /// multiple root types return std::nullopt.
    Option<OperationName> get_root_kind() const {
        if (rootKind == RootKind::OperationName)
        return OperationName::getFromOpaquePointer(rootValue);
        return std::nullopt;
    }

    /// Return the interface ID used to match the root operation of this pattern.
    /// If the pattern does not use an interface ID for deciding the root match,
    /// this returns std::nullopt.
    Option<TypeId> get_root_interface_id() const {
        if (rootKind == RootKind::InterfaceID)
        return TypeID::getFromOpaquePointer(rootValue);
        return std::nullopt;
    }

    /// Return the trait ID used to match the root operation of this pattern.
    /// If the pattern does not use a trait ID for deciding the root match, this
    /// returns std::nullopt.
    Option<TypeId> get_root_trait_id() const {
        if (rootKind == RootKind::TraitID)
        return TypeID::getFromOpaquePointer(rootValue);
        return std::nullopt;
    }

    /// Return the benefit (the inverse of "cost") of matching this pattern.  The
    /// benefit of a Pattern is always static - rewrites that may have dynamic
    /// benefit can be instantiated multiple times (different Pattern instances)
    /// for each benefit that they may return, and be guarded by different match
    /// condition predicates.
    PatternBenefit get_benefit() const { return benefit; }

    /// Returns true if this pattern is known to result in recursive application,
    /// i.e. this pattern may generate IR that also matches this pattern, but is
    /// known to bound the recursion. This signals to a rewrite driver that it is
    /// safe to apply this pattern recursively to generated IR.
    bool has_bounded_rewrite_recursion() const {
        return contextAndHasBoundedRecursion.getInt();
    }

    /// Return the MLIRContext used to create this pattern.
    MLIRContext *get_context() const {
        return contextAndHasBoundedRecursion.getPointer();
    }

    /// Return a readable name for this pattern. This name should only be used for
    /// debugging purposes, and may be empty.
    &'static str get_debug_name() const { return debug_name; }

    /// Set the human readable debug name used for this pattern. This name will
    /// only be used for debugging purposes.
    void set_debug_name(name: &'static str) { debug_name = name; }

    /// Return the set of debug labels attached to this pattern.
    &'static [str] get_debug_labels() const { return debug_labels; }

    /// Add the provided debug labels to this pattern.
    void add_debug_labels(&[&'static str] labels) {
        debug_labels.append(labels.begin(), labels.end());
    }
    void add_debug_labels(label: &'static str) { debug_labels.push_back(label); }
}

/// This enum represents the kind of value used to select the root operations that match this pattern.
pub enum RootKind {
    /// The pattern root matches "any" operation.
    Any,
    /// The pattern root is matched using a concrete operation name.
    OperationName,
    /// The pattern root is matched using an interface id.
    InterfaceId,
    /// The patter root is matched using a trait id.
    TraitId
}

/// RewritePattern is the common base class for all DAG to DAG replacements.
/// There are two possible usages of this class:
///
/// - Multi-step RewritePattern with "match" and `rewrite`
///    - By overloading the "match" and `rewrite` functions, the user can
///       separate the concerns of matching and rewriting.
/// - Single-step RewritePattern with `matchAndRewrite`
///    - By overloading the `matchAndRewrite` function, the user can perform
///       the rewrite in the same call as the match.
pub trait RewritePattern {

}

/// This class coordinates the application of a rewrite on a set of IR, providing a way for clients to track mutations and create new operations.
/// This class serves as a common API for IR mutation between pattern rewrites and non-pattern rewrites, and facilitates the development of shared IR transformation utilities.
pub trait Rewriter {
}

/// This class coordinates rewriting a piece of IR outside of a pattern rewrite, providing a way to keep track of the mutations made to the IR. This class should only be used in situations where another `RewriterBase` instance, such as a `PatternRewriter`, is not available.
pub trait IRRewriter: Rewriter {
}

/// A special type of `RewriterBase` that coordinates the application of a rewrite pattern on the current IR being matched, providing a way to keep track of any mutations made. This class should be used to perform all necessary IR mutations within a rewrite pattern, as the pattern driver may be tracking various state that would be invalidated when a mutation takes place.
pub trait PatternRewriter: Rewriter {
}

/// Storage type of byte-code interpreter values. These are passed to constraint functions as arguments.
pub struct PDLValue {
}

/// The underlying kind of a PDL value.
pub enum PDLValueKind {
    Attribute,
    Operation,
    Type,
    TypeRange,
    Value,
    ValueRange
}

/// The class represents a list of PDL results, returned by a native rewrite method. It provides the mechanism with which to pass PDLValues back to the PDL bytecode.
pub struct PDLResultList;

/// An individual configuration for a pattern, which can be accessed by native functions via the PDLPatternConfigSet. This allows for injecting additional configuration into PDL patterns that is specific to certain compilation flows.
pub struct PDLPatternConfig;

/**
This struct provides a convenient way to determine how to process a given type as either a PDL parameter, or a result value. This allows for supporting complex types in constraint and rewrite functions, without requiring the user to hand-write the necessary glue code themselves.
Specialisations of this class should implement the following methods to enable support as a PDL argument or result type:

```cpp
static LogicalResult verifyAsArg(
function_ref<LogicalResult(const Twine &)> errorFn, PDLValue pdlValue,
size_t argIdx);
```

This method verifies that the given PDLValue is valid for use as a
    value of `T`.

```cpp
static T processAsArg(PDLValue pdlValue);
```

This method processes the given PDLValue as a value of `T`.

```cpp
static void processAsResult(PatternRewriter &, PDLResultList &results,
                            const T &value);
```

This method processes the given value of `T` as the result of a
    function invocation. The method should package the value into an
    appropriate form and append it to the given result list.


If the type `T` is based on a higher order value, consider using `ProcessPDLValueBasedOn` as a base class of the specialisation to simplify the implementation.
*/
pub struct ProcessPDLValue<T, Enabled = ()>;
