/*!
# Classes for Defining Concrete Operation Types

This file implements helper classes for implementing the `Op` types. This includes the `Op` type, which is the base class for `Op` class definitions, as well as number of traits in the `OpTrait` namespace that provide a declarative way to specify properties of `Op`s.

The purpose of these types are to allow light-weight implementation of concrete operations (like DimOp) with very little boilerplate.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpDefinition.h>
*/

use core::{
    any::Any,
    cmp::PartialEq,
    fmt::Display
};

use crate::{
    mlir::{
        ir::{
            attribute::Attribute,
            builtins::types::{FloatType, Tensor, VectorType},
            dialect,
            operation::Operation,
            r#type::Type,
            type_utilities::{
                element_type_or_self, verify_compatible_shapes
            }
        },
        support::interface_support::Interface
    },
    llvm::{
        adt::{
            pointer_union::PointerUnion,
            small_vector::SmallVector
        },
        support::pointer_like_type_traits
    }
};

type LogicalResult = Result<(), Box<dyn std::error::Error>>;

/**
This class implements `Optional` functionality for LogicalResult. We don't directly use Optional here, because it provides an implicit conversion to 'bool' which we want to avoid. This class is used to implement tri-state 'parseOptional' functions that may have a failure mode when parsing that shouldn't be attributed to "not present".
*/
pub struct OptionalParseResult {
}

/**
This is the concrete base class that holds the operation pointer and has non-generic methods that only depend on State (to avoid having them instantiated on template types that don't affect them.

This also has the fallback implementations of customization hooks for when they aren't customised.
*/
#[derive(PartialEq)]
pub struct State {
    state: *mut Operation
}

impl State {
    
}

/// This class represents a single result from folding an operation.
pub struct FoldResult {

}

// ----------------------------------------------------------------------
// Operation Trait Types
// ----------------------------------------------------------------------

pub fn fold_idempotent(operation: *mut Operation) -> FoldResult {
    if operation.num_inputs() == 1 {
        if
            let Some(argument_op) = operation.input(0).defining_operation()
            && operation.name() == argument_op.name()
        {
            // Replace the outer operation output with the inner operation.
            return operation.input(0);
        }
    } else if operation.input(0) == operation.input(1) {
        return operation.input(0);
    }
    
    return {};
}

pub fn fold_involution(operation: *mut Operation) -> FoldResult {
    let argument_op = operation.input(0).defining_operation();
    if
        let Some(argument_op) = operation.input(0).defining_operation()
        && operation.name() == argument_op.name()
    {
        // Replace the outer involutions output with inner's input.
        return argument_op.input(0);
    }

    return {};
}

pub fn verify_zero_operands(operation: *mut Operation) -> LogicalResult {
    if operation.num_inputs() != 0 {
        operation.emit_op_error("requires zero operands").emit();
        Err(())
    }
    Ok(())
}

pub fn verify_one_operand(operation: *mut Operation) -> LogicalResult {
    if operation.num_inputs() != 1 {
        operation.emit_op_error("requires a single operand");
        Err(())
    }
    Ok(())
}

pub fn verify_n_operands(operation: *mut Operation, num_inputs: usize) -> LogicalResult {
    if operation.num_inputs() != num_inputs {
        let message = format!(
            "expected {} operands, but found {}",
            num_inputs,
            operation.num_inputs()
        );
        operation.emit_op_error(message).emit();
        Err(())
    }
    Ok(())
}

/**
If this is a vector type, or a tensor type, return the scalar element type that it is built around, otherwise return the type unmodified.
*/
// TODO
pub fn get_tensor_or_vector_element_type(r#type: &dyn Any) -> impl Type {
    if r#type.downcast_ref::<VectorType>() {
        return r#type.element_type();
    }
  
    // Look through tensor<vector<...>> to find the underlying element type.
    if r#type.downcast_ref::<Tensor>() {
        return get_tensor_or_vector_element_type(r#type.element_type());
    }
    return r#type;
}

// pub fn verify_is_idempotent(operation: *mut Operation) -> LogicalResult {}

// pub fn verify_is_involution(operation: *mut Operation) -> LogicalResult {}

pub fn verify_at_least_n_operands(operation: *mut Operation, num_inputs: usize) -> LogicalResult {
    if operation.num_inputs() < num_inputs {
        let message = format!(
            "expected {} or more operands, but found {}",
            num_inputs,
            operation.num_inputs()
        );
        operation.emit_op_error(message).emit();
        Err(())
    }
    Ok(())
}

pub fn verify_operands_are_float_like(operation: *mut Operation) -> LogicalResult {
    for r#type in operation.input_types() {
        let r#type = get_tensor_or_vector_element_type(r#type);
        if !matches!(r#type, FloatType) {
            operation.emit_op_error("requires a float type").emit();
            return Err(())
        }
    }
    Ok(())
}

// pub fn verify_operands_are_signless_integer_like(operation: *mut Operation) -> LogicalResult {}

// pub fn verify_same_type_operands(operation: *mut Operation) -> LogicalResult {
//     // Zero or one operand always have the "same" type.
//     unsigned nOperands = operation.num_inputs();
//     if (nOperands < 2)
//         Ok(())

//     auto type = operation.input(0).r#type();
//     for (auto op_type : llvm::drop_begin(operation.input_types(), 1))
//         if (op_type != type)
//         return operation.emit_op_error() << "requires all operands to have the same type";
//     Ok(())
// }

pub fn verify_zero_regions(operation: *mut Operation) -> LogicalResult {
    if operation.num_regions() != 0 {
        operation.emit_op_error("requires zero regions").emit();
        return Err(());
    }
    Ok(())
}

pub fn verify_one_region(operation: *mut Operation) -> LogicalResult {
    if operation.num_regions() != 1 {
        operation.emit_op_error("requires one region").emit();
        return Err(());
    }
    Ok(())
}

pub fn verify_n_regions(operation: *mut Operation, num_regions: usize) -> LogicalResult {
    if operation.num_regions() != num_regions {
        operation.emit_op_error(
            format!("expected {} regions", num_regions)).emit();
        return Err(());
    }
    Ok(())
}

pub fn verify_at_least_n_regions(operation: *mut Operation, num_regions: usize) -> LogicalResult {
    if operation.num_regions() < num_regions {
        operation.emit_op_error(
            format!("expected {} or more regions", num_regions));
        return Err(());
    }
    Ok(())
}

pub fn verify_zero_results(operation: *mut Operation) -> LogicalResult {
    if operation.num_outputs() != 0 {
        operation.emit_op_error("requires zero results").emit();
        return Err(())
    }
    Ok(())
}

pub fn verify_one_result(operation: *mut Operation) -> LogicalResult {
    if operation.num_outputs() != 1 {
        operation.emit_op_error("requires one result").emit();
        return Err(());
    }
    Ok(())
}

pub fn verify_n_results(operation: *mut Operation, num_inputs: usize) -> LogicalResult {
    if operation.num_outputs() != num_inputs {
        operation.emit_op_error(
            format!("expected {} results", num_inputs)).emit();
        return Err(())
    }
    Ok(())
}

pub fn verify_at_least_n_results(operation: *mut Operation, num_inputs: usize) -> LogicalResult {
    if operation.num_outputs() < num_inputs {
        operation.emit_op_error(
            format!("expected {} or more results", num_inputs)).emit();
        Err(())
    }
    Ok(())
}

pub fn verify_same_operands_shape(operation: *mut Operation) -> LogicalResult {
    if verify_at_least_n_operands(operation, 1).is_err() {
        return Err(());
    }

    if verify_compatible_shapes(operation.input_types()).is_err() {
        operation.emit_op_error("requires the same shape for all operands").emit();
        return Err(());
    }
    Ok(())
}

pub fn verify_same_operands_and_result_shape(operation: *mut Operation) -> LogicalResult {
    if verify_at_least_n_operands(operation, 1).is_err()
    || verify_at_least_n_results(operation, 1).is_err()
    {
        return Err(());
    }
    let types = SmallVector::<[Type; 8]>::new(operation.input_types());
    // to_vector<4>
    types.append(SmallVector::<[Type; 4]>::new(operation.output_types()));

    if verify_compatible_shapes(types).is_err() {
        operation.emit_op_error("requires the same shape for all operands and results").emit();
        Err(())
    }

    Ok(())
}

pub fn verify_same_operands_element_type(operation: *mut Operation) -> LogicalResult {
    if verify_at_least_n_operands(operation, 1).is_err() {
        return Err(());
    }
    let element_type = element_type_or_self(operation.input(0));

    for operand in operation.inputs().skip(1) {
        if element_type_or_self(operand) != element_type {
            operation.emit_op_error("requires the same element type for all operands").emit();
            return Err(())
        }
    }

    Ok(())
}

pub fn verify_same_operands_and_result_element_type(operation: *mut Operation) -> LogicalResult {
    if verify_at_least_n_operands(operation, 1).is_err()
    || verify_at_least_n_results(operation, 1).is_err() {
        return Err(());
    }

    let element_type = element_type_or_self(operation.output(0));

    // Verify result element type matches first result's element type.
    for output in operation.outputs().skip(1) {
        if element_type_or_self(output) != element_type {
            operation.emit_op_error(
                "requires the same element type for all operands and results");
            return Err(());
        }
    }

    // Verify operand's element type matches first result's element type.
    for input in operation.inputs() {
        if element_type_or_self(input) != element_type {
            operation.emit_op_error(
                "requires the same element type for all operands and results");
            return Err(());
        }
    }

    Ok(())
}

// pub fn verify_same_operands_and_result_type(operation: *mut Operation) -> LogicalResult {
//     if (failed(verify_at_least_n_operands(operation, 1)) ||
//         failed(verify_at_least_n_results(operation, 1)))
//         return Err(());

//     auto type = operation.output(0).r#type();
//     auto element_type = element_type_or_self(type);
//     Attribute encoding = nullptr;
//     if (auto ranked_type = dyn_cast<RankedTensorType>(type))
//         encoding = ranked_type.encoding();
//     for (auto resultType : llvm::drop_begin(operation.output_types())) {
//         if (element_type_or_self(resultType) != element_type ||
//             failed(verify_compatible_shape(resultType, type)))
//         return operation.emit_op_error()
//                 << "requires the same type for all operands and results";
//         if (encoding)
//         if (auto ranked_type = dyn_cast<RankedTensorType>(resultType);
//             encoding != ranked_type.encoding())
//             return operation.emit_op_error()
//                 << "requires the same encoding for all operands and results";
//     }
//     for (auto op_type : operation.input_types()) {
//         if (element_type_or_self(op_type) != element_type ||
//             failed(verify_compatible_shape(op_type, type)))
//         return operation.emit_op_error()
//                 << "requires the same type for all operands and results";
//         if (encoding)
//         if (auto ranked_type = dyn_cast<RankedTensorType>(op_type);
//             encoding != ranked_type.encoding())
//             return operation.emit_op_error()
//                 << "requires the same encoding for all operands and results";
//     }
//     Ok(())
// }

// pub fn verify_results_are_bool_like(operation: *mut Operation) -> LogicalResult {}

// pub fn verify_results_are_float_like(operation: *mut Operation) -> LogicalResult {}

// pub fn verify_results_are_signless_integer_like(operation: *mut Operation) -> LogicalResult {}

// pub fn verify_is_terminator(operation: *mut Operation) -> LogicalResult {
//     Block *block = operation.block();
//     // Verify that the operation is at the end of the respective parent block.
//     if (!block || &block.back() != operation)
//         return operation.emit_op_error("must be the last operation in the parent block");
//     Ok(())
// }

// pub fn verify_zero_successors(operation: *mut Operation) -> LogicalResult {
//     if (operation.num_successors() != 0) {
//         return operation.emit_op_error("requires 0 successors but found ")
//                << operation.num_successors();
//       }
//       Ok(())
// }

// pub fn verify_one_successor(operation: *mut Operation) -> LogicalResult {
//     if (operation.num_successors() != 1) {
//         return operation.emit_op_error("requires 1 successor but found ")
//                << operation.num_successors();
//       }
//       return verify_terminator_successors(operation);
// }

// pub fn verify_n_successors(operation: *mut Operation, num_successors: usize) -> LogicalResult {
//     if (operation.num_successors() != num_successors) {
//         return operation.emit_op_error("requires ")
//                << num_successors << " successors but found "
//                << operation.num_successors();
//       }
//       return verify_terminator_successors(operation);
// }

// pub fn verify_at_least_n_successors(operation: *mut Operation, num_successors: usize) -> LogicalResult {
//     if operation.num_successors() < num_successors {
//         return operation.emit_op_error("requires at least ")
//                << num_successors << " successors but found "
//                << operation.num_successors();
//     }
//     verify_terminator_successors(operation)
// }

// pub fn verify_value_size_attr(
//     operation: *mut Operation,
//     attr_name: &str,
//     value_group_name: &str,
//     expected_count: size_t
// ) -> LogicalResult
// {
//     let size_attr = operation.attr_of_type::<DenseI32ArrayAttr>(attr_name);
//     if (!size_attr)
//       return operation.emit_op_error("requires dense i32 array attribute '")
//              << attr_name << "'";
  
//     ArrayRef<int32_t> sizes = size_attr.as_array_ref();
//     if (llvm::any_of(sizes, [](int32_t element) { return element < 0; })) {
//         return operation.emit_op_error("'")
//              << attr_name << "' attribute cannot have negative elements";
//     }
  
//     size_t total_count =
//         std::accumulate(sizes.begin(), sizes.end(), 0,
//                         [](unsigned all, int32_t one) { return all + one; });
  
//     if (total_count != expectedCount) {
//         return operation.emit_op_error()
//                 << valueGroupName << " count (" << expectedCount
//                 << ") does not match with the total size (" << total_count
//                 << ") specified in attribute '" << attr_name << "'";
//     }
//     Ok(())
// }

// pub fn verify_operand_size_attr(operation: *mut Operation, size_attr_name: &str) -> LogicalResult {
//     verify_value_size_attr(
//         operation, attr_name, "operand", operation.num_inputs())
// }

// pub fn verify_result_size_attr(operation: *mut Operation, size_attr_name: &str) -> LogicalResult {
//     verify_value_size_attr(
//         operation, attr_name, "result", operation.num_outputs())
// }

// pub fn verify_no_region_arguments(operation: *mut Operation) -> LogicalResult {
//     for region in operation.regions() {
//         if region.is_empty() {
//             continue;
//         }
    
//         if region.num_inputs() != 0 {
//             if operation.num_regions() > 1 {
//                 return operation.emit_op_error("region #")
//                     << region.region_number() << " should have no arguments";
//             }
//             return operation.emit_op_error("region should have no arguments");
//         }
//     }
//     Ok(())
// }

// pub fn verify_elementwise(operation: *mut Operation) -> LogicalResult {
//     auto is_mappable_type = |r#type: Type| {
//         return r#type.isa<VectorType, Tensor>();
//     };
//     auto result_mappable_types = llvm::to_vector<1>(
//         llvm::make_filter_range(operation.output_types(), is_mappable_type));
//     auto operand_mappable_types = llvm::to_vector<2>(
//         llvm::make_filter_range(operation.input_types(), is_mappable_type));

//     /*
//     If the operation only has scalar operand/result types, then we have nothing to check.
//     */
//     if (result_mappable_types.is_empty() && operand_mappable_types.is_empty())
//     Ok(())

//     if (!result_mappable_types.is_empty() && operand_mappable_types.is_empty())
//     return operation.emit_op_error("if a result is non-scalar, then at least one "
//                             "operand must be non-scalar");

//     assert(!operand_mappable_types.is_empty());

//     if (result_mappable_types.is_empty())
//     return operation.emit_op_error("if an operand is non-scalar, then there must be at "
//                             "least one non-scalar result");

//     if (result_mappable_types.size() != operation.num_outputs())
//     return operation.emit_op_error(
//         "if an operand is non-scalar, then all results must be non-scalar");

//     SmallVector<Type, 4> types = llvm::to_vector<2>(
//         llvm::concat<Type>(operand_mappable_types, result_mappable_types));
//     TypeId expected_base_ty = types.front().getTypeID();
//     if (!llvm::all_of(types,
//                     [&](Type t) { return t.getTypeID() == expected_base_ty; }) ||
//         failed(verify_compatible_shapes(types))) {
//     return operation.emit_op_error() << "all non-scalar operands/results must have the "
//                                 "same shape and base type";
//     }

//     Ok(())
// }

// pub fn verify_is_isolated_from_above(operation: *mut Operation) -> LogicalResult {
//     assert!(
//         operation.has_trait::<OpTrait::IsIsolatedFromAbove>(),
//         "Intended to check IsolatedFromAbove ops.");

//     /*
//     List of regions to analyze. Each region is processed independently, with respect to the common `limit` region, so we can look at them in any order.
//     Therefore, use a simple vector and push/pop back the current region.
//     */
//     let mut pending_regions = SmallVector::<*mut Region, 8>::new();
//     for region in operation.regions() {
//         pending_regions.push(&region);

//         // Traverse all operations in the region.
//         while !pending_regions.is_empty() {
//             for o : pending_regions.pop_back_val().ops() {
//                 for operand in o.operands() {
//                     /*
//                     Check that any value that is used by an operation is defined in the same region as either an operation result.
//                     */
//                     let operand_region = operand.parent_region();
//                     if operand_region.is_none() {
//                         return o.emit_error("operation's operand is unlinked");
//                     }
//                     if !region.is_ancestor(operand_region) {
//                         return o.emit_op_error("using value defined outside the region")
//                                 .attach_note(operation.location())
//                             << "required by region isolation constraints";
//                     }
//                 }

//                 /*
//                 Schedule any regions in the operation for further checking.  Don't recurse into other IsolatedFromAbove ops, because they will check themselves.
//                 */
//                 if o.num_regions() &&
//                     !operatio.has_trait::<OpTrait::IsIsolatedFromAbove>()
//                 {
//                     for sub_region in o.regions() {
//                         pending_regions.push(&sub_region);
//                     }
//                 }
//             }
//         }
//     }

//     Ok(())
// }

/**
Helper class for implementing traits.  Clients are not expected to interact with this directly, so its members are all protected.
*/
// template <typename Self, template <typename> class TraitType>
pub trait Trait {

}

/**
Utility trait base that provides accessors for derived traits that have multiple operands.
*/
// template <typename Self, template <typename> class TraitType>
pub trait MultiOperand: Trait {

}

/**
`verifyInvariantsImpl` verifies the invariants like the types, attrs, .etc.
It should be run after core traits and before any other user defined traits.
In order to run it in the correct order, wrap it with OpInvariants trait so that tblgen will be able to put it in the right order.
*/
pub trait Invariants: Trait {

}

/// This class provides the API for ops that are known to have no SSA operand.
pub trait ZeroOperands: Trait {

}

/// This class provides the API for ops that are known to have exactly one SSA operand.
pub trait OneOperand: Trait {

}

/**
This class provides the API for ops that are known to have a specified number of operands.  This is used as a trait like this:

```cpp
class FooOp : public Op<FooOp, OpTrait::NOperands<2>::Impl> {
```
*/
pub struct NOperands<const N: usize> {

}

/**
This class provides the API for ops that are known to have a at least a specified number of operands.  This is used as a trait like this:

```cpp
class FooOp : public Op<FooOp, OpTrait::AtLeastNOperands<2>::Impl> {
```
*/
pub struct AtLeastNOperands<const N: usize> {
}

/// This class provides the API for ops which have an unknown number of SSA operands.
pub struct VariadicOperands {

}

/// This class provides verification for ops that are known to have zero regions.
pub trait ZeroRegions: Trait {

}

/// Utility trait base that provides accessors for derived traits that have multiple regions.
pub trait MultiRegion: Trait {

}

/// This class provides APIs for ops that are known to have a single region.
pub trait OneRegion: Trait {

}

/// This class provides the API for ops that are known to have a specified number of regions.
pub struct NRegions<const N: usize> {

}

/// This class provides APIs for ops that are known to have at least a specified number of regions.
pub struct AtLeastNRegions<const N: usize> {

}

/// This class provides the API for ops which have an unknown number of regions.
pub trait VariadicRegions: MultiRegion {

}

/// This class provides return value APIs for ops that are known to have zero results.
pub trait ZeroResults: Trait {

}

/// Utility trait base that provides accessors for derived traits that have multiple results.
// template <typename Self, template <typename> class TraitType>
pub trait MultiResult: Trait {

}

/// This class provides return value APIs for ops that are known to have a single result.  ResultType is the concrete type returned by r#type().
pub trait OneResult: Trait {

}

/**
This trait is used for return value APIs for ops that are known to have a specific type other than `Type`.  This allows the "r#type()" member to be more specific for an operation.  This should be used in conjunction with OneResult, and occur in the trait list before OneResult.
*/
pub struct OneTypedResult<ResultT> {

}

/**
This class provides the API for ops that are known to have a specified number of results. This is used as a trait like this:

```cpp
class FooOp : public Op<FooOp, OpTrait::NResults<2>::Impl> {
```
*/
pub struct NResults<const N: usize> {

}

/**
This class provides the API for ops that are known to have at least a specified number of results. This is used as a trait like this:

```cpp
class FooOp : public Op<FooOp, OpTrait::AtLeastNResults<2>::Impl> {
```
*/
pub struct AtLeastNResults<const N: usize> {

}

/// This class provides the API for ops which have an unknown number of results.
pub trait VariadicResults: MultiResult {

}


/// This class indicates that the regions associated with this operation don't have terminators.
pub trait NoTerminator<Concrete>: Trait {

}

/// This class provides the API for ops that are known to be terminators.
pub trait IsTerminator<C>: Trait {

}

/// This class provides verification for ops that are known to have zero successors.
pub trait ZeroSuccessors<C>: Trait {

}

/// Utility trait base that provides accessors for derived traits that have multiple successors.
// template <typename Self, template <typename> class TraitType>
pub trait MultiSuccessor: Trait {

}

/// This class provides APIs for ops that are known to have a single successor.
pub trait OneSuccessor<C>: Trait {

}

/// This class provides the API for ops that are known to have a specified number of successors.
pub struct NSuccessors<const N: usize> {

}

/// This class provides APIs for ops that are known to have at least a specified number of successors.
pub struct AtLeastNSuccessors<const N: usize> {

}

/// This class provides the API for ops which have an unknown number of successors.
pub trait VariadicSuccessors<C>: MultiSuccessor {

}


/// This class provides APIs and verifiers for ops with regions having a single block.
pub trait SingleBlock<C>: Trait {

}

/**
This class provides APIs and verifiers for ops with regions having a single block that must terminate with `TerminatorOpType`.
*/
pub struct SingleBlockImplicitTerminator<TerminatorOpType> {

}

pub trait SameOperandsShape: Trait {

}

pub trait SameOperandsAndResultShape: Trait {
    
}

pub trait SameOperandsElementType: Trait {
    
}

pub trait SameOperandsAndResultElementType: Trait {
    
}

pub trait SameOperandsAndResultType: Trait {
    
}

pub trait ResultsAreBoolLike: Trait {
    
}

pub trait ResultsAreFloatLike: Trait {
    
}

pub trait ResultsAreSignlessIntegerLike: Trait {
    
}

pub trait IsCommutative: Trait {
    
}

pub trait IsInvolution: Trait {
    
}

pub trait IsIdempotent: Trait {
    
}

pub trait OperandsAreFloatLike: Trait {
    
}

pub trait OperandsAreSignlessIntegerLike: Trait {
    
}

pub trait SameTypeOperands: Trait {
    
}

/**
This class provides the API for a sub-set of ops that are known to be constant-like. These are non-side effecting operations with one result and zero operands that can always be folded to a specific attribute value.
*/
pub trait ConstantLike<T>: Trait {

}

/**
This class provides the API for ops that are known to be isolated from above.
*/
pub trait IsIsolatedFromAbove<T>: Trait {

}

/**
A trait of region holding operations that defines a new scope for polyhedral optimisation purposes. Any SSA values of 'index' type that either dominate such an operation or are used at the top-level of such an operation automatically become valid symbols for the polyhedral scope defined by that operation. For more details, see `Traits.md#AffineScope`.
*/
pub trait AffineScope<T>: Trait {

}

/**
A trait of region holding operations that define a new scope for automatic allocations, i.e., allocations that are freed when control is transferred back from the operation's region. Any operations performing such allocations (for eg. memref.alloca) will have their allocations automatically freed at their closest enclosing operation with this trait.
*/
pub trait AutomaticAllocationScope<T>: Trait {

}

/**
This class provides a verifier for ops that are expecting their parent to be one of the given parent ops
*/
pub struct HasParent {

}


/**
A trait for operations that have an attribute specifying operand segments.

Certain operations can have multiple variadic operands and their size relationship is not always known statically. For such cases, we need a per-operation-instance specification to divide the operands into logical groups or segments. This can be modeled by attributes. The attribute will be named as `operand_segment_sizes`.

This trait verifies the attribute for specifying operand segments has the correct type (1D vector) and values (non-negative), etc.
*/
pub trait AttrSizedOperandSegments<T>: Trait {

}

/// Similar to `AttrSizedOperandSegments` but used for results.
pub trait AttrSizedResultSegments<T>: Trait {
    
}

/**
This trait provides a verifier for ops that are expecting their regions to not have any arguments
*/
pub trait NoRegionArguments<T>: Trait {
    
}

/**
This trait is used to flag operations that consume or produce values of `MemRef` type where those references can be 'normalised'.
TODO: Right now, the operands of an operation are either all normalisable, or not. In the future, we may want to allow some of the operands to be
normalisable.
*/
pub trait MemRefsNormalisable<T>: Trait {
    
}

/**
This trait tags element-wise ops on vectors or tensors.

NOTE: Not all ops that are "elementwise" in some abstract sense satisfy this trait. In particular, broadcasting behaviour is not allowed.

An `Elementwise` operation must satisfy the following properties:

1. If any result is a vector/tensor then at least one operand must also be a
   vector/tensor.
2. If any operand is a vector/tensor then there must be at least one result
   and all results must be vectors/tensors.
3. All operand and result vector/tensor types must be of the same shape. The
   shape may be dynamic in which case the operation's behaviour is undefined for
   non-matching shapes.
4. The operation must be elementwise on its vector/tensor operands and
   results. When applied to single-element vectors/tensors, the result must
   be the same per elememnt.

TODO: Avoid hardcoding vector/tensor, and generalize this trait to a new interface `ElementwiseTypeInterface` that describes the container types for
which the operation is elementwise.

# Rationale

- 1. and 2. guarantee a well-defined iteration space and exclude the cases
  of 0 non-scalar operands or 0 non-scalar results, which complicate a
  generic definition of the iteration space.
- 3. guarantees that folding can be done across scalars/vectors/tensors with
  the same pattern, as otherwise lots of special handling for type
  mismatches would be needed.
- 4. guarantees that no error handling is needed. Higher-level dialects
  should reify any needed guards or error handling code before lowering to
  an `Elementwise` operation.
*/
pub trait Elementwise<T>: Trait {

}

/**
This trait tags `Elementwise` operatons that can be systematically scalarised. All vector/tensor operands and results are then replaced by scalars of the respective element type. Semantically, this is the operation on a single element of the vector/tensor.

# Rationale

Allow to define the vector/tensor semantics of elementwise operations based on the same operation's behaviour on scalars. This provides a constructive procedure for IR transformations to, e.g., create scalar loop bodies from tensor ops.

# Example

```
%tensor_select = "arith.select"(%pred_tensor, %true_val, %false_val)
                     : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>)
                     -> tensor<?xf32>
```

can be scalarized to

```
%scalar_select = "arith.select"(%pred, %true_val_scalar, %false_val_scalar)
                     : (i1, f32, f32) -> f32
```
*/
pub trait Scalarisable<T>: Trait {

}

/**
This trait tags `Elementwise` operatons that can be systematically vectorised. All scalar operands and results are then replaced by vectors with the respective element type. Semantically, this is the operation on multiple elements simultaneously. See also `Tensorizable`.

# Rationale

Provide the reverse to `Scalarisable` which, when chained together, allows reasoning about the relationship between the tensor and vector case.
Additionally, it permits reasoning about promoting scalars to vectors via broadcasting in cases like `%select_scalar_pred` below.
*/
pub trait Vectorisable<T>: Trait {

}

/**
This trait tags `Elementwise` operatons that can be systematically tensorised. All scalar operands and results are then replaced by tensors with the respective element type. Semantically, this is the operation on multiple elements simultaneously. See also `Vectorizable`.

# Rationale

Provide the reverse to `Scalarisable` which, when chained together, allows reasoning about the relationship between the tensor and vector case.
Additionally, it permits reasoning about promoting scalars to tensors via broadcasting in cases like `%select_scalar_pred` below.

Examples:

```mlir
%scalar = "arith.addf"(%a, %b) : (f32, f32) -> f32
```

can be tensorised to

```mlir
%tensor = "arith.addf"(%a, %b) : (tensor<?xf32>, tensor<?xf32>)
              -> tensor<?xf32>
```

```mlir
%scalar_pred = "arith.select"(%pred, %true_val, %false_val)
                   : (i1, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
```

can be tensorised to

```mlir
%tensor_pred = "arith.select"(%pred, %true_val, %false_val)
                   : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>)
                   -> tensor<?xf32>
```
*/
pub trait Tensorisable<T>: Trait {
    
}

// ----------------------------------------------------------------------
// Operation Definition Classes
// ----------------------------------------------------------------------

/**
This provides public APIs that all operations should have. The template argument `Self` should be the concrete type by CRTP and the others are base classes by the policy pattern.
*/
pub trait Op {  // State
    // /**
    // A hook for static assertion that the external interface model T is targeting the concrete type of this op. The model can also be a fallback model that works for every op.
    // */
    // fn check_interface_target<T>() {
    //     assert!(std::is_same<InterfaceTargetOrOpT<T>::type,
    //                             Self>::value,
    //                 "Attaching an interface to the wrong op kind");
    // }

    // /**
    // Returns an interface map containing the interfaces registered to this
    // operation.
    // */
    // fn interface_map() -> InterfaceMap {
    //     InterfaceMap::get::<Traits<Self>>();
    // }

    // /**
    // Return the internal implementations of each of the OperationName hooks.
    // Implementation of `FoldHookFn` OperationName hook.
    // */
    // fn fold_hook_fn() -> OperationName::FoldHookFn {
    //     // If the operation is single result and defines a `fold` method.
    //     if constexpr (llvm::is_one_of<OpTrait::OneResult<Self>,
    //                                 Traits<Self>>::value &&
    //                 detect_has_single_result_fold<Self>::value){
    //     return [](operation: *mut Operation, operands: &[impl Attribute],
    //                 results: &SmallVector<[FoldResult]>) {
    //         return fold_single_result_hook<Self>(op, operands, results);
    //     };}

    //     // The operation is not single result and defines a `fold` method.
    //     if detect_has_fold<Self>::value {
    //         return
    //             |operation: *mut Operation,
    //              operands: &[impl Attribute],
    //              results: &SmallVector<[FoldResult]>|
    //                  fold_hook::<Self>(op, operands, results)
    //         ;
    //     }

    //     // The operation does not define a `fold` method.
    //     return |operation: *mut Operation, operands: &[impl Attribute],
    //             results: &SmallVector<[FoldResult]>| {
    //         // In this case, we only need to fold the traits of the operation.
    //         return op_definition_impl::fold_traits<Traits<Self>>(
    //             op, operands, results);
    //     };
    // }
    // /**
    // Return the result of folding a single result operation that defines a `fold` method.
    // */
    // template <typename ConcreteOpT>
    
    // fn fold_single_result_hook(
    //     operation: *mut Operation, operands: &[impl Attribute],
    //     results: &SmallVector<[FoldResult]>
    // ) -> LogicalResult
    // {
    //     let result = cast<ConcreteOpT>(operation).fold(operands);

    //     // If the fold failed or was in-place, try to fold the traits of the
    //     // operation.
    //     if (!result || result.template dyn_cast<Value>() == operation.output(0)) {
    //     if (succeeded(op_definition_impl::fold_traits<Traits<Self>...>(
    //         operation, operands, results)))
    //         return Ok(());
    //     return success(static_cast<bool>(result));
    //     }
    //     results.push(result);
    //     return Ok(());
    // }
    // /// Return the result of folding an operation that defines a `fold` method.
    // template <typename ConcreteOpT>
    // pub fn fold_hook(
    //     operation: *mut Operation,
    //     operands: &[impl Attribute],
    //     results: &SmallVector<[FoldResult]>
    // ) -> LogicalResult
    // {
    //     let result = cast<ConcreteOpT>(op).fold(operands, results);

    //     // If the fold failed or was in-place, try to fold the traits of the
    //     // operation.
    //     if (failed(result) || results.is_empty()) {
    //         if (succeeded(op_definition_impl::fold_traits<Traits<Self>...>(
    //             op, operands, results))) {
    //             return Ok(());
    //         }
    //     }
    //     return result;
    // }

    // /// Implementation of `GetCanonicalisationPatternsFn` OperationName hook.
    // pub fn get_get_canonicalisation_patterns_fn()
    // -> OperationName::GetCanonicalisationPatternsFn {
    //     &Self::get_canonicalisation_patterns
    // }

    // /// Implementation of `GetHasTraitFn`
    // pub fn get_has_trait_fn() -> OperationName::HasTraitFn {
    //     return
    //         |type_id: TypeId| op_definition_impl::has_trait<Traits>(id);
    // }
    // /// Implementation of `ParseAssemblyFn` OperationName hook.
    // pub fn get_parse_assembly_fn() -> OperationName::ParseAssemblyFn {
    //     &Self::parse
    // }

    // /// Implementation of `PrintAssemblyFn` OperationName hook.
    // pub fn get_print_assembly_fn() -> OperationName::PrintAssemblyFn {
    //     if (detect_has_print::<Self>::value) {
    //         return |operation: *mut Operation, p: &OpAsmPrinter, &str default_dialect| {
    //             OpState::print_op_name(op, p, default_dialect);
    //             return cast<Self>(op).print(p);
    //     };}
    //     return |operation: *mut Operation, printer: &OpAsmPrinter, default_dialect: &str|
    //      OpState::print(op, printer, default_dialect)
    //     ;
    // }

    // /// Implementation of `PopulateDefaultAttrsFn` OperationName hook.
    // pub fn get_populate_default_attrs_fn() -> OperationName::PopulateDefaultAttrsFn {
    //     Self::populateDefaultAttrs
    // }

    // /// Implementation of `VerifyInvariantsFn` OperationName hook.
    // pub fn verify_invariants(operation: *mut Operation) -> LogicalResult {
    //     assert!(has_no_data_members(),
    //             "Op class shouldn't define new data members");
    //     return failure(
    //         failed(op_definition_impl::verifyTraits<Traits<Self>...>(op)) ||
    //         failed(cast<Self>(op).verify()));
    // }

    // static OperationName::VerifyInvariantsFn get_verify_invariants_fn() {
    //     return static_cast<LogicalResult (*)(Operation *)>(&verify_invariants);
    // }

    // /// Implementation of `VerifyRegionInvariantsFn` OperationName hook.
    // static LogicalResult verify_region_invariants(operation: *mut Operation) {
    //     assert!(has_no_data_members(),
    //                 "Op class shouldn't define new data members");
    //     return failure(
    //         failed(op_definition_impl::verifyRegionTraits<Traits<Self>...>(
    //             op)) ||
    //         failed(cast<Self>(op).verifyRegions()));
    // }

    // static OperationName::VerifyRegionInvariantsFn get_verify_region_invariants_fn() {
    //     return static_cast<LogicalResult (*)(Operation *)>(&verify_region_invariants);
    // }

    // static constexpr bool has_no_data_members() {
    //     // Checking that the derived class does not define any member by comparing
    //     // its size to an ad-hoc EmptyOp.
    //     class EmptyOp : public Op<EmptyOp, Traits...> {};
    //     return sizeof(Self) == sizeof(EmptyOp);
    // }
}

/**
This class represents the base of an operation interface. See the definition of `detail::Interface` for requirements on the `Traits` type.
*/
pub trait OpInterface: Interface {

}
