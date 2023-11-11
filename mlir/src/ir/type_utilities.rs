//! Helper Function for Type Queries
//!
//! This file defines generic type utilities.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/TypeUtilities.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/TypeUtilities.cpp>

use crate::{
    mlir::{
        ir::{
            builtins::{
                type_interfaces::ShapedType,
                types::TupleType
            },
            r#type::Type,
            type_range::TypeRange
        },
        support::logical_result::LogicalResult
    },
    llvm::adt::small_vector::SmallVector
};

/*
Utility Functions
*/

/// Return the element type or return the type itself.
pub fn element_type_or_self(r#type: dyn Type) -> impl Type {
    if r#type.downcast_ref::<ShapedType>() {
        return r#type.element_type();
    }
    r#type
}

// /// Return the element type or return the type itself.
// Type element_type_or_self(Attribute attr);
// Type element_type_or_self(Value val);

/// Get the types within a nested Tuple. A helper for the class method that handles storage concerns, which is tricky to do in tablegen.
pub fn get_flattened_types(t: TupleType) -> SmallVector<[Type; 10]> {
   
}

/// Return true if the specified type is an opaque type with the specified dialect and typeData.
pub fn is_opaque_type_with_name(
    r#type: dyn Type, dialect: &str, type_data: &str
) -> bool {
    if let Some(opaque) = r#type.downcast_ref::<OpaqueType>() {
        return opaque.dialect_namespace() == dialect
            && opaque.type_data() == type_data;
    }
    false
}

/// Returns success if the given two shapes are compatible. That is, they have the same size and each pair of the elements are equal or one of them is dynamic.
pub fn verify_compatible_shape(shape1: &[i64], shape2: &[i64])
-> LogicalResult {
     if shape1.len() != shape2.len() {
         return Err(());
     }
    for (dim1, dim2) in shape1.zip(shape2) {
        if !ShapedType::is_dynamic(dim1) && !ShapedType::is_dynamic(dim2) &&
            dim1 != dim2 {
            return Err(());
        }
    }
    Ok(())
}

// /// Returns success if the given two types have compatible shape. That is,
// /// they are both scalars (not shaped), or they are both shaped types and at
// /// least one is unranked or they have compatible dimensions. Dimensions are
// /// compatible if at least one is dynamic or both are equal. The element type
// /// does not matter.
// LogicalResult verify_compatible_shape(Type type1, Type type2);

// /// Returns success if the given two arrays have the same number of elements and
// /// each pair wise entries have compatible shape.
// LogicalResult verify_compatible_shapes(TypeRange types1, TypeRange types2);

/// Returns success if all given types have compatible shapes. That is, they are all scalars (not shaped), or they are all shaped types and any ranked shapes have compatible dimensions. The element type does not matter.
pub fn verify_compatible_shapes(types: TypeRange) -> LogicalResult {
    // let shaped_types = llvm::to_vector<8>(
    //     types.map(|r#type: impl Type| r#type.template dyn_cast::<ShapedType>())
    // );
    Ok(())
}

/// Dimensions are compatible if all non-dynamic dims are equal.
pub fn verify_compatible_dims(dims: &[i64]) -> LogicalResult {
    if dims.is_empty() {
        return Ok(());
    }
    let static_dim = std::accumulate(
        dims, dims.front(), |fold, dim|
            if ShapedType::is_dynamic(dim) { fold } else { dim }
    );
    success(dims.all(|dim| ShapedType::is_dynamic(dim) || dim == static_dim))
}
