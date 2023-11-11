//! Various Common Matchers
//!
//! This file provides a simple and efficient mechanism for performing general tree-based pattern matching over MLIR. This mechanism is inspired by LLVM's include/llvm/IR/PatternMatch.h.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Matchers.h>

use crate::{
    mlir::ir::{
        attribute::Attribute,
        builtins::{
            attributes::FloatAttribute,
            types
        },
        operation::{
            Operation,
            definition::{ConstantLike, FoldResult},
        }
    },
    llvm::adt::small_vector::SmallVector
};

use super::{builtins::attributes::IntegerAttribute, value::Value};

/// The matcher that matches a certain kind of Attribute and binds the value inside the Attribute.
pub struct AttributeValueBinder<A: Attribute> {
    bind_value: *mut A::ValueType
}

impl<T: Attribute> AttributeValueBinder<T> {
    /// Creates a matcher instance that binds the value to bv if match succeeds.
    pub fn new(bind_value: *mut A::ValueType) -> Self {
        Self { bind_value }
    }

    pub fn r#match(&self, attribute: dyn Attribute) -> bool {
        match attribute.downcast_ref::<T>() {
            None => false,
            Some(int_attr) => {
                unsafe {
                    *self.bind_value = int_attr.value();
                }
                return true;
            }
        }
    }
}

/// Check to see if the specified operation is ConstantLike.  This includes some quick filters to avoid a semi-expensive test in the common case.
pub const fn is_constant_like(operation: *mut Operation) -> bool {
       operation.num_inputs() == 0
    && operation.num_outputs() == 1
    && operation.has_trait::<ConstantLike>()
}

/// The matcher that matches operations that have the `ConstantLike` trait.
pub struct ConstantOperationMatcher;

impl ConstantOperationMatcher {
    fn r#match(&self, operation: *mut Operation) -> bool {
        is_constant_like(operation)
    }
}

/// The matcher that matches operations that have the `ConstantLike` trait, and binds the folded attribute value.
pub struct ConstantOperationBinder<T> {
    bind_value: *mut T
}

impl<T> ConstantOperationBinder::<T> {
    /// Creates a matcher instance that binds the constant attribute value to `bind_value` if match succeeds.
    fn new(bind_value: *mut T) -> Self {
        Self { bind_value }
    }

    fn r#match(&self, operation: *mut Operation) -> bool {
        if !is_constant_like(operation) {
            return false;
        }
    
        // Fold the constant to an attribute.
        let outputs = SmallVector::<[FoldResult; 1]>::new();
        operation.fold(null(), outputs).expect(
            "Expected ConstantLike operation to be foldable."
        );
    
        if let Ok(attr) = outputs[0].get::<Attribute>().dyn_cast::<T>() {
            if !self.bind_value.is_null() {
                unsafe { *self.bind_value = attr; }
            }
            return true;
        }
        return false;
    }
}

/// The matcher that matches a constant scalar / vector splat / tensor splat float operation and binds the constant float value.
pub struct ConstantFloatOperationBinder {
    bind_value: *mut FloatAttribute::ValueType
}

impl ConstantFloatOperationBinder {
    /// Creates a matcher instance that binds the value to bv if match succeeds.
    pub fn new(bind_value: *mut FloatAttribute::ValueType) -> Self {
        Self { bind_value }
    }

    pub fn r#match(&self, operation: *mut Operation) -> bool {
        let mut attr = Attribute;
        if !ConstantOperationBinder::<Attribute>(&attr).r#match(operation) {
            return false;
        }
        let r#type = operation.output(0).r#type();
    
        if r#type.isa<FloatType>() {
            return AttributeValueBinder<FloatAttr>(bind_value).r#match(attr);
        }
        if r#type.isa::<VectorType, RankedTensorType>() {
            if let splat_attr = attr.dyn_cast<SplatElementsAttr>() {
                return AttributeValueBinder<FloatAttr>(bind_value)
                    .r#match(splat_attr.get_splat_value<Attribute>());
            }
        }
        false
    }
}

/// The matcher that matches a given target constant scalar / vector splat / tensor splat float value that fulfills a predicate.
pub struct ConstantFloatPredicateMatcher {
    predicate: fn(&APFloat) -> bool
}

impl ConstantFloatPredicateMatcher {
    pub fn r#match(&self, operation: *mut Operation) -> bool {
        let value = APFloat(APFloat::Bogus());
        ConstantFloatOperationBinder(&value).r#match(operation) && predicate(value)
    }
}

/// The matcher that matches a constant scalar / vector splat / tensor splat integer operation and binds the constant integer value.
struct ConstantIntOperationBinder {
    bind_value: *mut IntegerAttribute::ValueType
}

impl ConstantIntOperationBinder {
    /// Creates a matcher instance that binds the value to bv if match succeeds.
    pub fn new(bind_value: *mut IntegerAttribute::ValueType) -> Self {
        Self { bind_value }
    }

    pub fn r#match(&self, operation: *mut Operation) -> bool {
        let attr = Attribute;
        if !ConstantOperationBinder::<Attribute>(&attr).r#match(operation) {
            return false;
        }
        let r#type = operation.output(0).r#type();

        if r#type.downcast_ref::<IntegerType, IndexType>() {
            return AttributeValueBinder<IntegerAttr>(bind_value).r#match(attr);
        }
        if r#type.downcast_ref::<VectorType, RankedTensorType>() {
            if let splat_attr = attr.dyn_cast<SplatElementsAttr>() {
                return AttributeValueBinder<IntegerAttr>(bind_value)
                    .r#match(splat_attr.get_splat_value<Attribute>());
            }
        }
        false
    }
}

/// The matcher that matches a given target constant scalar / vector splat / tensor splat integer value that fulfills a predicate.
pub struct ConstantIntPredicateMatcher {
    pub predicate: fn(&APInt) -> bool
}

impl ConstantIntPredicateMatcher {
    pub fn r#match(&self, operation: *mut Operation) -> bool {
        let mut value = APInt;
        ConstantIntOperationBinder::new(&mut value).r#match(operation)
        && self.predicate(value)
    }
}

/// The matcher that matches a certain kind of operation.
pub struct OperationMatcher<OpClass> {
    _marker: core::marker::PhantomData<OpClass>
}

impl<OpClass> OperationMatcher<OpClass> {
    pub fn r#match(&self, operation: *mut Operation) -> bool { return isa<OpClass>(operation); }
}

// /// Statically switch to a Value matcher.
// std::enable_if_t<
//     llvm::is_detected<has_operation_or_value_matcher_t, MatcherClass, Value>::value,
//                  bool>
// pub fn match_operand_or_value_at_index<MatcherClass>(
//     operation: *mut Operation,
//     index: usize,
//     matcher: &MatcherClass)
// -> bool
// {
//     matcher.r#match(operation.input(index))
// }

// /// Statically switch to an Operation matcher.
// std::enable_if_t<llvm::is_detected<has_operation_or_value_matcher_t,
//                                    MatcherClass, Operation *>::value,
//                  bool>
// pub fn match_operand_or_value_at_index<MatcherClass>(
//     operation: *mut Operation,
//     index: usize,
//     matcher: &MatcherClass)
// -> bool
// {
//     let defining_operation = operation.input(index).defining_operation();
//     if !defining_operation.is_null() {
//         return matcher.r#match(defining_operation);
//     }
//     false
// }

/// Terminal matcher, always returns true.
pub struct AnyValueMatcher;

impl AnyValueMatcher {
    pub const fn r#match(&self, _: Value) -> bool {
        true
    }
}

/// Terminal matcher, always returns true.
pub struct AnyCapturedValueMatcher {
    what: *mut Value
}

impl AnyCapturedValueMatcher {
    pub fn new(what: *mut Value) -> Self {
        Self { what }
    }

    pub const fn r#match(&self, value: Value) -> bool {
        unsafe { *self.what = value; }
        true
    }
}

/// Binds to a specific value and matches it.
pub struct PatternMatcherValue {
    value: Value
}

impl PatternMatcherValue {
    pub fn new(value: Value) -> Self {
        Self { value }
    }

    pub const fn r#match(&self, value: Value) -> bool {
        value == self.value
    }
}

/// RecursivePatternMatcher that composes.
pub struct RecursivePatternMatcher<OpType> {
    operand_matchers: Vec<OperandMatchers>
}

impl RecursivePatternMatcher {
    pub fn new(operand_matchers: Vec<OperandMatchers>) -> Self {
        Self { operand_matchers }
    }

    pub fn r#match(&self, operation: *mut Operation) -> bool {
        if !isa<OpType>(operation)
        || operation.num_operands() != sizeof...(OperandMatchers)
        {
            return false;
        }
        let output = true;
        enumerate(operand_matchers, |index, matcher| {
            output &= match_operand_or_value_at_index(operation, index, matcher);
        });
        output
    }
}

/// Matches a constant foldable operation.
#[inline]
pub fn m_Constant() -> ConstantOperationMatcher {
    ConstantOperationMatcher::new()
}

/// Matches a value from a constant foldable operation and writes the value to `bind_value`.
#[inline]
pub fn m_Constant<T>(bind_value: *mut T) -> ConstantOperationBinder::<T> {
    ConstantOperationBinder::new(bind_value)
}

/// Matches a constant scalar / vector splat / tensor splat float (both positive and negative) zero.
#[inline]
pub fn m_AnyZeroFloat() -> ConstantFloatPredicateMatcher {
    return {|value: &APFloat| { return value.isZero(); }};
}

/// Matches a constant scalar / vector splat / tensor splat float positive zero.
#[inline]
pub fn m_PosZeroFloat() -> ConstantFloatPredicateMatcher {
    return {|value: &APFloat| { return value.isPosZero(); }};
}

/// Matches a constant scalar / vector splat / tensor splat float negative zero.
#[inline]
pub fn m_NegZeroFloat() -> ConstantFloatPredicateMatcher {
    return {|value: &APFloat| { return value.isNegZero(); }};
}

/// Matches a constant scalar / vector splat / tensor splat float ones.
#[inline] 
pub fn m_OneFloat() -> ConstantFloatPredicateMatcher {
    return {|value: &APFloat| {
        return APFloat(value.getSemantics(), 1) == value;
    }};
}

/// Matches a constant scalar / vector splat / tensor splat float positive infinity.
#[inline]
pub fn m_PosInfFloat() -> ConstantFloatPredicateMatcher {
    return {|value: &APFloat| {
        return !value.isNegative() && value.isInfinity();
    }};
}

/// Matches a constant scalar / vector splat / tensor splat float negative infinity.
#[inline]
pub fn m_NegInfFloat() -> ConstantFloatPredicateMatcher {
    return {|value: &APFloat| {
        return value.isNegative() && value.isInfinity();
    }};
}

/// Matches a constant scalar / vector splat / tensor splat integer zero.
#[inline]
pub fn m_Zero() -> ConstantIntPredicateMatcher {
    return {|value: &APInt| value == 0 };
}

/// Matches a constant scalar / vector splat / tensor splat integer that is any non-zero value.
#[inline]
pub fn m_NonZero() -> ConstantIntPredicateMatcher {
    return {|value: &APInt| value != 0 };
}

/// Matches a constant scalar / vector splat / tensor splat integer one.
#[inline]
pub fn m_One() -> ConstantIntPredicateMatcher {
    return {|value: &APInt| value == 1};
}

/// Matches the given OpClass.
#[inline]
pub fn m_Op<OpClass>() -> OperationMatcher<OpClass> {
    OperationMatcher::<OpClass>
}


/// Entry point for matching a pattern over a Value.
#[inline]
pub fn match_pattern<Pattern>(value: Value, pattern: &Pattern) -> bool {
    // TODO: handle other cases
    let operation = value.defining_operation();
    if !operation.is_null() {
        return pattern.r#match(operation);
    }
    false
}

/// Entry point for matching a pattern over an Operation.
#[inline]
pub fn match_pattern<Pattern>(operation: *mut Operation, pattern: &Pattern) -> bool {
    pattern.r#match(operation)
}

/// Matches a constant holding a scalar/vector/tensor float (splat) and writes the float value to bind_value.
#[inline]
pub fn m_ConstantFloat(bind_value: *mut FloatAttr::ValueType) -> ConstantFloatOperationBinder {
    return ConstantFloatOperationBinder(bind_value);
}

/// Matches a constant holding a scalar/vector/tensor integer (splat) and writes the integer value to bind_value.
#[inline]
pub fn m_ConstantInt(bind_value: *mut IntegerAttr::ValueType)
-> ConstantIntOperationBinder
{
    ConstantIntOperationBinder::new(bind_value)
}

pub fn m_Op<OpType>(matchers: Vec<Matchers>) -> {
    RecursivePatternMatcher<OpType>(matchers)
}

#[inline]
pub fn m_Any() -> _ { AnyValueMatcher }

#[inline]
pub fn m_Any(val: *mut Value) -> _ { AnyCapturedValueMatcher::new(val) }

#[inline]
pub fn m_Val(v: Value) -> _ { PatternMatcherValue::new(v) }
