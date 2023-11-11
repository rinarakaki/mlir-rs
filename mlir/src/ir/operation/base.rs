/*!
# Base Operation Definition File

This is the base operation definition file.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td>
*/

use crate::mlir::ir::{
    builtins::types::VectorType,
    dialect::base
};

// ----------------------------------------------------------------------===//
// Predicate Definitions
// ----------------------------------------------------------------------===//

/**
Base class for logical predicates.

Predicates are used to compose constraints (see next section for details).
There are two categories of predicates:

1. CPred: the primitive leaf predicate.
2. Compound predicate: a predicate composed from child predicates using
   predicate combiners ("conjunction", "disjunction", "negation" or
   "substitution").
*/
pub enum Pred {

}

/**
A logical predicate wrapping any C expression.

This is the basis for composing more complex predicates. It is the "atom"
predicate from the perspective of TableGen and the "interface" between
TableGen and C++. What is inside is already C++ code, which will be treated
as opaque strings with special placeholders to be substituted.

## Special placeholders

Special placeholders can be used to refer to entities in the context where
this predicate is used. They serve as "hooks" to the enclosing environment.
The following special placeholders are supported in constraints for an op:

* `$_builder` will be replaced by a mlir::Builder instance.
* `$_op` will be replaced by the current operation.
* `$_self` will be replaced with the entity this predicate is attached to.
  E.g., `BoolAttr` is an attribute constraint that wraps a
  `CPred<"$_self.isa<BoolAttr>()">` (see the following sections for details).
  Then for `F32:$attr`,`$_self` will be replaced by `$attr`.
  For type constraints, it's a little bit special since we want the
  constraints on each type definition reads naturally and we want to attach
  type constraints directly to an operand/result, $_self will be replaced
  by the operand/result's type. E.g., for `F32` in `F32:$operand`, its
  `$_self` will be expanded as `getOperand(...).getType()`.

One thing to be noticed, while using these placeholders in the C expression,
the type of placeholder is only guaranteed to be the base type. For example,
if you have a predicate in the form `CPred<"CheckType($_self)">, the argument
type of the function `CheckType` should be `mlir::Type`.
*/
class CPred<code pred> : Pred {
    code predExpr = "(" # pred # ")";
}

// Predicate combiners

/**
A predicate that holds if all of its children hold.  Always holds for zero children.
*/
pub struct And(Pred, Pred);

// <list<Pred> children> : CombinedPred<PredCombinerAnd, children>;

/**
A predicate that holds if any of its children hold.  Never holds for zero children.
*/
pub struct Or(Pred, Pred);

// <list<Pred> children> : CombinedPred<PredCombinerOr, children>;

// A predicate that holds if its child does not.
pub struct Not(Pred);

// : CombinedPred<PredCombinerNot, [child]>;

/**
A predicate that substitutes "pat" with "repl" in predicate calls of the
leaves of the predicate tree (i.e., not CombinedPred).

This is plain string substitution without regular expressions or captures.
New predicates with more complex logical can be introduced should the need
arise.
*/
pub SubstLeaves<string pat, string repl, Pred child>
    : CombinedPred<PredCombinerSubstLeaves, [child]> {
    string pattern = pat;
    string replacement = repl;
}

/**
A predicate that prepends `pre` and appends `suf` to the final predicate string composed from `child`. This is plain string concatenation and there will be no substitution happening for `pre` and `suf`.
*/
class Concat<string pre, Pred child, string suf> :
    CombinedPred<PredCombinerConcat, [child]> {
    string prefix = pre;
    string suffix = suf;
}

// ----------------------------------------------------------------------
// Constraint Definitions
// ----------------------------------------------------------------------

// TODO: Merge Constraints into Pred.

/**
Base class for named constraints.

An op's operands/attributes/results can have various requirements, e.g., having certain types, having values inside a certain range, and so on.
Besides, for a graph rewrite rule, the source pattern used to match against the existing graph has conditions, like the op's operand must be of a more
constrained subtype, the attribute must have a certain value, and so on.

These requirements and conditions are modeled using this class. Records of
this class are used to generate verification code in op verifier, and
matching code in pattern matcher.

Constraints are predicates with descriptive names, to facilitate inspection,
provide nice error messages, etc.
*/
pub trait Constraint {
    /// The predicates that this constraint requires.
    const predicate: Pred;
}
  
/*
Subclasses used to differentiate different constraint kinds. These are used as markers for the TableGen backend to handle different constraint kinds differently if needed. Constraints not deriving from the following subclasses are considered as uncategorized constraints.
*/

/// Subclass for constraints on a type.
pub trait TypeConstraint: Constraint {
    /// The name of the C++ Type class if known, or Type if not.
    const cppClassName: String; // Type
}

// ----------------------------------------------------------------------
// Type Definitions
// ----------------------------------------------------------------------

// A type, carries type constraints.
pub trait Type: TypeConstraint {
    const description: String;
    const builderCall: String;
}

// L333
/// Any type at all.
pub struct AnyType = Type<CPred<"true">>;

// None type
pub type NoneType
    = Type<CPred<"NoneType>">,
    //   BuildableType<"$_builder.getType<::mlir::NoneType>()">;

// Any type from the given list
pub type AnyTypeOf<list<Type> allowedTypes>
    = Type<
    // Satisfy any of the allowed types' conditions.
    Or<!foreach(allowedtype, allowedTypes, allowedtype.predicate)>;

// // A type that satisfies the constraints of all given types.
// class AllOfType<list<Type> allowedTypes, string summary = "",
//                 string cppClassName = Type> : Type<
//     // Satisfy all of the allowedf types' conditions.
//     And<!foreach(allowedType, allowedTypes, allowedType.predicate)>,
//     !if(!eq(summary, ""),
//         !interleave(!foreach(t, allowedTypes, t.summary), " and "),
//         summary),
//     cppClassName>;

// // A type that satisfies additional predicates.
// class ConfinedType<r#type: Type, list<Pred> predicates, string summary = "",
//                    string cppClassName = type.cppClassName> : Type<
//     And<!listconcat([type.predicate], !foreach(pred, predicates, pred))>,
//     summary, cppClassName>;

// ----------------------------------------------------------------------
// Vector Types
// ----------------------------------------------------------------------

pub trait ShapedContainerType<list<Type> allowedTypes,
                          Pred containerPred> :
    Type<And<[containerPred,
              Concat<"[](::mlir::Type elementType) { return ",
                SubstLeaves<"$_self", "elementType", AnyTypeOf<allowedTypes>.predicate>,
                "; }($_self.cast<::mlir::ShapedType>().getElementType())">]>,
         >;

// L559
pub type VectorOf<list<Type> allowedTypes>
    = ShapedContainerType<allowedTypes, IsVectorTypePred, "vector",
                      VectorType>;

// Temporary vector type clone that allows gradual transition to 0-D vectors.
// TODO: Remove this when all ops support 0-D vectors.
pub type VectorOfAnyRankOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsVectorOfAnyRankTypePred, "vector",
                      VectorType>;

// L702
pub type AnyVector = VectorOf<[AnyType]>;

// Temporary vector type clone that allows gradual transition to 0-D vectors.
def AnyVectorOfAnyRank = VectorOfAnyRankOf<[AnyType]>;

// ----------------------------------------------------------------------
// Tensor Types
// ----------------------------------------------------------------------

// L720
/**
Unranked tensor type whose element type is from the given `allowedTypes` list.
*/
pub trait UnrankedTensorOf {

}

// L729
/**
Any tensor type whose element type is from the given `allowedTypes` list, and which additionally satisfies an optional list of predicates.

TODO: use `Constraint` instead of `Pred`, so we can generate a better default summary (a la `ConfinedAttr`).
*/
pub type Tensor<Type> = ();

pub type RankedTensor<Type> = Tensor<Type>;

// L757
pub type AnyRankedTensor =  RankedTensor<[AnyType]>;

// L1532
/// Attributes containing symbol references.
pub struct SymbolRefAttr {}

/**
OpInterfaceTrait corresponds to a specific `OpInterface` class defined in C++. The purpose to wrap around C++ symbol string with this class is to make interfaces specified for ops in TableGen less alien and more integrated.
*/
pub struct OpInterfaceTrait;

/// OpInterface represents an interface registered to an operation.
pub trait OpInterface {  //: Interface, OpInterfaceTrait {

}

// L2070
/// TypeInterface represents an interface registered to a type.
pub trait TypeInterface {  // : Interface, InterfaceTrait {

}
