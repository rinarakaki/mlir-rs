//! # Builtin Type Interfaces
//!
//! This file contains definitions for type interfaces that closely interact with attributes, types, and operations in the builtin dialect.
//!
//! - include
//!   - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BuiltinTypeInterfaces.h>
//!   - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BuiltinTypeInterfaces.td>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/BuiltinTypeInterfaces.cpp>

use llvm::adt::sequence;
use crate::ir::{
    builtins::types::ComplexType,
    diagnostics,
    operation::base::TypeInterface,
    r#type::Type
};

// ----------------------------------------------------------------------
// MemRefElementTypeInterface
// ----------------------------------------------------------------------

/// Indication that this type can be used as element in memref types.
///
/// Implementing this interface establishes a contract between this type and the memref type indicating that this type can be used as element of ranked or unranked memrefs. The type is expected to:
///
/// - model an entity stored in memory;
/// - have non-zero size.
///
/// For example, scalar values such as integers can implement this interface, but indicator types such as `void` or `unit` should not.
///
/// The interface currently has no methods and is used by types to opt into being memref elements. This may change in the future, in particular to require types to provide their size or alignment given a data layout.
pub trait MemRefElementTypeInterface: TypeInterface {

}

// ----------------------------------------------------------------------
// ShapedType
// ----------------------------------------------------------------------

const DYNAMIC: i64 = i64::MIN;

/// This interface provides a common API for interacting with multi-dimensional container types. These types contain a shape and an element type.
///
/// A shape is a list of sizes corresponding to the dimensions of the container.
/// If the number of dimensions in the shape is unknown, the shape is 'unranked'.
/// If the number of dimensions is known, the shape 'ranked'. The sizes of the dimensions of the shape must be positive, or kDynamic (in which case the size of the dimension is dynamic, or not statically known).
pub trait ShapedType: TypeInterface {
    // const ELEMENT: Type;

    /// Returns a clone of this type with the given shape and element type. If a shape is not provided, the current shape of the type is used.
    fn clone_with(&self, shape: Option<&[i64]>, element_type: Type) -> Self;

    /// Returns the element type of this shaped type.
    fn element_type(&self) -> Type;

    /// Returns if this type is ranked, i.e. it has a known number of dimensions.
    fn has_rank(&self) -> bool;

    /// Returns the shape of this type if it is ranked, otherwise asserts.
    fn shape(&self) -> &[i64];

    /// Whether the given dimension size indicates a dynamic dimension.
    fn is_dynamic(dValue: i64) -> bool{
        dValue == DYNAMIC
    }
        
    /// Whether the given shape has any size that indicates a dynamic dimension.
    fn is_dynamic_shape(&self, dSizes: &[i64]) -> bool {
        dSizes.any(|dSize: i64| { self.is_dynamic(dSize) })
    }

    /// Returns the number of elements present in the given shape.
    fn num_elements(shape: &[i64]) -> i64 {
        let mut num = 1;
        for dim in shape {
            num *= dim;
            assert!(
                num >= 0,
                "Integer overflow in element count computation.");
        }
        num
    }

    /// Returns the total amount of bits occupied by a value of this type. This does not take into account any memory layout or widening constraints, e.g. a vector<3xi57> may report to occupy 3x57=171 bit, even though in practice it will likely be stored as in a 4xi64 vector register. Fails with an assertion if the size cannot be computed statically, e.g. if the type has a dynamic shape or if its elemental type does not have a known bit width.
    fn size_in_bits(&self) -> i64 {
        assert!(
            self.has_static_shape(),
            "Cannot get the bit size of an aggregate with a dynamic shape");

        let mut element_type = self.element_type();
        if element_type.is_int_or_float() {
            return element_type.get_int_or_float_bit_width() * self.num_elements();
        }

        if let complex_type = element_type.dyn_cast::<ComplexType>() {
            element_type = complex_type.element_type();
            return element_type.get_int_or_float_bit_width() * self.num_elements() * 2;
        }
        self.num_elements() * <element_type as ShapedType>.size_in_bits()
    }

    /// Return a clone of this type with the given new shape and element type.
    fn clone(&self, shape: &[i64], element_type: Type) -> Self {
        self.clone_with(shape, element_type)
    }

    // /// Return a clone of this type with the given new shape.
    // fn clone(&self, shape: &[i64]) -> Self {
    //     self.clone_with(shape, self.element_type())
    // }

    // /// Return a clone of this type with the given new element type.
    // fn clone(&self, element_type: impl Type) -> Self {
    //     self.clone_with(None, element_type)
    // }
    
    /// If an element type is an integer or a float, return its width. Otherwise, abort.
    fn element_type_bit_width(&self) -> usize {
        return self.element_type().get_int_or_float_bit_width();
    }
    
    /// If this is a ranked type, return the rank. Otherwise, abort.
    fn rank(&self) -> i64 {
        assert!(
            self.has_rank(),
            "Cannot query rank of unranked shaped type.");
        self.shape().size();
    }
    
    /// If it has static shape, return the number of elements. Otherwise, abort.
    fn num_elements(&self) -> i64 {
        assert!(
            self.has_static_shape(),
            "Cannot get element count of dynamic shaped type");
        Self::num_elements(self.shape())
    }
    
    /// Returns true if this dimension has a dynamic size (for ranked types); aborts for unranked types.
    fn is_dynamic_dim(&self, index: usize) -> bool {
        assert!(index < self.rank(), "Invalid index for shaped type");
        Self::is_dynamic(self.shape()[index])
    }
    
    /// Returns if this type has a static shape, i.e. if the type is ranked and all dimensions have known size (>= 0).
    fn has_static_shape(&self) -> bool {
        self.has_rank() && !Self::is_dynamic_shape(self.shape())
    }
    
    // /// Returns if this type has a static shape and the shape is equal to `shape` return true.
    // fn has_static_shape(&self, shape: &[i64]) -> bool {
    //     self.has_static_shape() && self.shape() == shape
    // }
    
    /// If this is a ranked type, return the number of dimensions with dynamic size. Otherwise, abort.
    fn num_dynamic_dims(&self) -> i64 {
        llvm::count_if(self.shape(), Self::is_dynamic)
    }
    
    /// If this is ranked type, return the size of the specified dimension. Otherwise, abort.
    fn dim_size(&self, index: usize) -> i64 {
        assert!(index < self.rank(), "Invalid index for shaped type");
        self.shape()[index]
    }
    
    /// Returns the position of the dynamic dimension relative to just the dynamic dimensions, given its `index` within the shape.
    fn dynamic_dim_index(&self, index: usize) -> usize {
        assert!(index < self.rank(), "Invalid index");
        assert!(Self::is_dynamic(self.dim_size(index)), "invalid index");
        return llvm::count_if(self.shape().take_front(index),
                                Self::is_dynamic);
    }
}
