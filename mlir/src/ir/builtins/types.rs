//! # Builtin Type Definitions
//!
//! Defines the set of builtin MLIR types, or the set of types necessary for the validity of and defining the IR.
//!
//! - include
//!   - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BuiltinTypes.h>
//!   - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BuiltinTypes.td>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/BuiltinTypes.cpp>

use core::{
    any::Any,
    iter::IntoIterator,
    slice::Iter
};

use llvm::adt::{
    ap_float::*,
    bit_vector::BitVector,
    dense_set::SmallDenseSet,
    sequence,
    small_vector::SmallVector,
    twine,
    type_switch,
};
use crate::ir::{
    affine_expr::AffineExpr,
    affine_map,
    attribute::Attribute,
    builtins::{
        attribute_interfaces,
        attributes::StringAttribute,
        // dialect,
        type_interfaces::ShapedType
    },
    diagnostics,
    dialect,
    function::interfaces,
    mlir_context::MLIRContext,
    operation::implementation,
    sub_element_interfaces,
    tensor_encoding,
    r#type::{
        Type,
        detail
    },
    type_range::TypeRange
};

pub enum TypeKind {
    ComplexType(ComplexType),
    IndexType(IndexType),
    IntegerType(IntegerType)
}

// ----------------------------------------------------------------------
// ComplexType
// ----------------------------------------------------------------------

/// Complex number with a parameterized element type.
///
/// # Syntax
///
/// ```text
/// complex-type ::= `complex` `<` type `>`
/// ```
///
/// The value of `complex` type represents a complex number with a parameterised element type, which is composed of a real and imaginary value of that element type. The element must be a floating point or integer scalar type.
///
/// # Examples
///
/// ```mlir
/// complex<f32>
/// complex<i32>
/// ```
pub struct ComplexType {
    element_type: Box<dyn Type>
}

// ----------------------------------------------------------------------
// IndexType
// ----------------------------------------------------------------------

/// Integer-like type with unknown platform-dependent bit width.
///
/// Syntax:
///
/// ```text
/// // Target word-sized integer.
/// index-type ::= `index`
/// ```
///
/// The index type is a signless integer whose size is equal to the natural machine word of the target ( [rationale](../../Rationale/Rationale/#integer-signedness-semantics) ) and is used by the affine constructs in MLIR.
///
/// # Rationale
///
/// Integers of platform-specific bit widths are practical to express sizes, dimensionalities and subscripts.
pub struct IndexType {

}

impl IndexType {
    pub fn new(context: *mut MLIRContext) -> Self;

    // /// Storage bit width used for IndexType by internal compiler data
    // /// structures.
    // const usize kInternalStorageBitWidth = 64;
}

// ----------------------------------------------------------------------
// IntegerType
// ----------------------------------------------------------------------

/// Integer type with arbitrary precision up to a fixed limit.
///
/// # Syntax
///
/// ```text
/// // Sized integers like i1, i4, i8, i16, i32.
/// signed-integer-type ::= `si` [1-9][0-9]*
/// usize-integer-type ::= `ui` [1-9][0-9]*
/// signless-integer-type ::= `i` [1-9][0-9]*
/// integer-type ::= signed-integer-type |
///                     usize-integer-type |
///                     signless-integer-type
/// ```
///
/// Integer types have a designated bit width and may optionally have signedness semantics.
///
/// # Rationale
///
/// Low precision integers (like `i2`, `i4` etc) are useful for low-precision inference chips, and arbitrary precision integers are useful for hardware synthesis (where a 13 bit multiplier is a lot cheaper/smaller than a 16 bit one).
pub struct IntegerType {
    width: usize,
    signedness: Signedness
}

impl IntegerType {
    /// Integer representation maximal bitwidth.
    /// Note: This is aligned with the maximum width of IntegerType.
    const MAX_WIDTH: usize = (1 << 24) - 1;

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn signedness(&self) -> Signedness {
        self.signedness
    }

    /// Return true if this is a signless integer type.
    pub const fn is_signless(&self) -> bool {
        self.signedness == Signedness::Signless
    }

    /// Return true if this is a signed integer type.
    pub const fn is_signed(&self) -> bool {
        self.signedness == Signedness::Signed
    }

    /// Return true if this is an usize integer type.
    pub const fn is_unsigned(&self) -> bool {
        self.signedness == Signedness::Unsigned
    }

    /// Get or create a new IntegerType with the same signedness as `this` and a bitwidth scaled by `scale`.
    /// Return null if the scaled element type cannot be represented.
    pub fn scale_element_bitwidth(&self, scale: usize) -> Self {
        if scale == 0 {
            return IntegerType::new();
        }
        IntegerType::get(self.context(), scale * self.width, self.signedness)
    }
}

/// Signedness semantics.
#[repr(align(32))]
pub enum Signedness {
    /// No signedness semantics
    Signless,
    /// Signed integer
    Signed,
    /// Unsigned integer
    Unsigned,
}

// ----------------------------------------------------------------------
// FloatType
// ----------------------------------------------------------------------

pub struct FloatType;

impl FloatType {
    // Convenience factories.
    // static FloatType get_bf16(MLIRContext *ctx);
    // static FloatType get_f16(MLIRContext *ctx);
    // static FloatType get_f32(MLIRContext *ctx);
    // static FloatType get_f64(MLIRContext *ctx);
    // static FloatType get_f80(MLIRContext *ctx);
    // static FloatType get_f128(MLIRContext *ctx);
    // static FloatType get_float8e5m2(MLIRContext *ctx);
    // static FloatType get_float8e4m3fn(MLIRContext *ctx);

    /// Return the bitwidth of this float type.
    pub fn width(&self) -> usize {

    }

    // /// Return the width of the mantissa of this type.
    // // fp_mantissa_width
    // pub const fn precision(&self) -> usize {
    //     match self {
    //         Float8E5M2Type => Float8E5M2,
    //         Float8E4M3FNType => Float8E4M3FN,
    //         Float16Type => BFloat,
    //         BFloat16Type => IEEEhalf,
    //         Float32Type => IEEEsingle::PRECISION,
    //         Float64Type => IEEEdouble,
    //         Float80Type => x87DoubleExtended,
    //         Float128Type => IEEEquad
    //     }
    // }

    // /**
    // Get or create a new FloatType with bitwidth scaled by `scale`.
    // Return null if the scaled element type cannot be represented.
    // */
    // pub fn scale_element_bitwidth(&self, scale: usize) -> Self {
    //     if scale == 0 {
    //         return FloatType::new();
    //     }
    //     let ctx = self.context();
    //     if (isF16() || isBF16()) {
    //         if (scale == 2){
    //         return FloatType::getF32(ctx);}
    //         if (scale == 4){
    //         return FloatType::getF64(ctx);}
    //     }
    //     if (isF32()) {
    //         if (scale == 2) {
    //             return FloatType::getF64(ctx);
    //         }
    //     }
    //     return FloatType::new();
    // }

    // /// Return the floating semantics of this float type.
    // pub const fn float_semantics(&self) -> &impl Semantics {
    //     match self {
    //         Float8E5M2Type => Float8E5M2,
    //         Float8E4M3FNType => Float8E4M3FN,
    //         Float16Type => BFloat,
    //         BFloat16Type => IEEEhalf,
    //         Float32Type => IEEEsingle,
    //         Float64Type => IEEEdouble,
    //         Float80Type => x87DoubleExtended,
    //         Float128Type => IEEEquad
    //     }
    // }
}

pub enum FloatTypeKind {
    Float8E5M2Type,
    Float8E4M3FNType,
    Float16Type,
    BFloat16Type,
    Float32Type,
    Float64Type,
    Float80Type,
    Float128Type
}

// ----------------------------------------------------------------------
// FunctionType
// ----------------------------------------------------------------------

/// Map from a list of inputs to a list of results.
///
/// # Syntax
///
/// ```text
/// // Function types may have multiple results.
/// function-result-type ::= type-list-parens | non-function-type
/// function-type ::= type-list-parens `->` function-result-type
/// ```
///
/// The function type can be thought of as a function signature. It consists of a list of formal parameter types and a list of formal result types.
pub struct FunctionType {
    inputs: &'static [dyn Type],
    outputs: &'static [dyn Type]
}

impl FunctionType {
    // /// Input types.
    // pub const fn num_inputs(&self) -> usize;

    pub const fn input(&self, index: usize) -> impl Type {
        self.inputs[index]
    }

    // /// Output types.
    // pub const fn num_outputs(&self) -> usize;

    pub const fn output(&self, index: usize) -> impl Type {
        self.outputs[index]
    }

    // /// Returns a clone of this function type with the given argument and result types.
    // pub const fn clone(&self, inputs: TypeRange, outputs: TypeRange) -> Self;

//     /// Returns a new function type with the specified arguments and results inserted.
//     pub fn get_with_args_and_results(
//         input_indices: &[usize],
//         input_types: TypeRange,
//         output_indices: &[usize],
//         output_types: TypeRange
//     ) -> Self
//     {
//         SmallVector<Type> argStorage, resultStorage;
//   TypeRange newArgTypes = function_interface_impl::insert_types_into(
//       getInputs(), input_indices, input_types, argStorage);
//   TypeRange newResultTypes = function_interface_impl::insert_types_into(
//       getResults(), resultIndices, resultTypes, resultStorage);
//   return clone(newArgTypes, newResultTypes);
//     }

//     /// Returns a new function type without the specified arguments and results.
//     pub fn get_without_args_and_results(
//         input_indices: &BitVector,
//         output_indices: &BitVector
//     ) -> Self {
//     }
}

// ----------------------------------------------------------------------
// OpaqueType
// ----------------------------------------------------------------------

pub struct OpaqueType {
    dialect_namespace: StringAttribute,
    type_data: &'static str
}

// ----------------------------------------------------------------------
// Vector
// ----------------------------------------------------------------------

/// Multi-dimensional SIMD vector type.
///
/// Syntax:
///
/// ```text
/// vector-type ::= `vector` `<` vector-dim-list vector-element-type `>`
/// vector-element-type ::= float-type | integer-type | index-type
/// vector-dim-list := (static-dim-list `x`)? (`[` static-dim-list `]` `x`)?
/// static-dim-list ::= decimal-literal (`x` decimal-literal)*
/// ```
///
/// The vector type represents a SIMD style vector used by target-specific operation sets like AVX or SVE. While the most common use is for 1D vectors (e.g. `vector<16xf32>`) we also support multidimensional registers on targets that support them (like TPUs). The dimensions of a vector type can be fixed-length, scalable, or a combination of the two. The scalable dimensions in a vector are indicated between square brackets ([ ]), and all fixed-length dimensions, if present, must precede the set of scalable dimensions. That is, a `vector<2x[4]xf32>` is valid, but `vector<[4]x2xf32>` is not.
///
/// Vector shapes must be positive decimal integers. 0D vectors are allowed by omitting the dimension: `vector<f32>`.
///
/// Note: hexadecimal integer literals are not allowed in vector type declarations, `vector<0x42xi32>` is invalid because it is interpreted as a 2D vector with shape `(0, 42)` and zero shapes are not allowed.
///
/// Examples:
///
/// ```mlir
/// // A 2D fixed-length vector of 3x42 i32 elements.
/// vector<3x42xi32>
///
/// // A 1D scalable-length vector that contains a multiple of 4 f32 elements.
/// vector<[4]xf32>
///
/// // A 2D scalable-length vector that contains a multiple of 2x8 i8 elements.
/// vector<[2x8]xf32>
///
/// // A 2D mixed fixed/scalable vector that contains 4 scalable vectors of 4 f32 elements.
/// vector<4x[4]xf32>
/// ```
///
/// //
/// // Builder
/// //
/// This is a builder type that keeps local references to arguments. Arguments that are passed into the builder must outlive the builder.
pub struct Vector<T: Type, const N: usize> {
    pub shape: [i64; N],
    // storage: SmallVector<[i64]>,
    pub element_type: T,
    num_scalable_dims: Box<dyn Attribute>
}

// D : {n, m : N} -> (Fin n) -> (Fin m);
// V A D 0 = (D 0) -> A;
// V A D (S n) = V ((D S n) -> A) D n;

pub enum VectorElementType {
    IntegerType(IntegerType),
    IndexType(IndexType),
    FloatType(FloatType)
}

impl Vector {
    // /// This is a builder type that keeps local references to arguments.
    // /// Arguments that are passed into the builder must outlive the builder.
    // class Builder;

    /// Returns true if the vector contains scalable dimensions.
    pub const fn is_scalable(&self) -> bool {
        self.num_scalable_dims > 0
    }

    /// Get or create a new Vector with the same shape as `this` and an element type of bitwidth scaled by `scale`.
    /// Return null if the scaled element type cannot be represented.
    pub fn scale_element_bitwidth(&self, scale: usize) -> Self {
        if scale == 0 {
            return Self::new();
        }
        let element_type: &dyn Any = &self.element_type;
        if element_type.downcast_ref::<IntegerType>().is_some()
        || element_type.downcast_ref::<FloatType>().is_some()
        {
            let scaled_et = self.element_type.scale_element_bitwidth(scale);
            if scaled_et {
                return Self::get(self.shape, scaled_et, self.num_scalable_dims);
            }
        }
        Self::new()
    }
}

impl const ShapedType for Vector {
    /// Clone this vector type with the given shape and element type. If the provided shape is `None`, the current shape of the type is used.
    fn clone_with(
        &self,
        shape: Option<&[i64]>,
        element_type: impl Type
    ) -> Self
    {
        Self::new(shape.value_or(self.shape), element_type,
            self.num_scalable_dims)
    }

    /// Returns if this type is ranked (always true).
    fn has_rank(&self) -> bool {
        true
    }
}

// ----------------------------------------------------------------------
// Tensor
// ----------------------------------------------------------------------

/// Multi-dimensional array with a fixed number of dimensions.
///
/// # Syntax
///
/// ```text
/// tensor-type ::= `tensor` `<` dimension-list type (`,` encoding)? `>`
/// dimension-list ::= (dimension `x`)*
/// dimension ::= `?` | decimal-literal
/// encoding ::= attribute-value
/// ```
///
/// Values with tensor type represents aggregate N-dimensional data values, and have a known element type and a fixed rank with a list of dimensions. Each dimension may be a static non-negative decimal constant or be dynamically determined (indicated by `?`).
///
/// The runtime representation of the MLIR tensor type is intentionally abstracted - you cannot control layout or get a pointer to the data. For low level buffer access, MLIR has a [`memref` type](#memref-type). This abstracted runtime representation holds both the tensor data values as well as information about the (potentially dynamic) shape of the tensor. The [`dim` operation](MemRef.md/#memrefdim-mlirmemrefdimop) returns the size of a dimension from a value of tensor type.
///
/// The `encoding` attribute provides additional information on the tensor.
/// An empty attribute denotes a straightforward tensor without any specific structure. But particular properties, like sparsity or other specific characteristics of the data of the tensor can be encoded through this attribute. The semantics are defined by a type and attribute interface and must be respected by all passes that operate on tensor types.
/// TODO: provide this interface, and document it further.
///
/// Note: hexadecimal integer literals are not allowed in tensor type declarations to avoid confusion between `0xf32` and `0 x f32`. Zero sizes are allowed in tensors and treated as other sizes, e.g., `tensor<0 x 1 x i32>` and `tensor<1 x 0 x i32>` are different types. Since zero sizes are not allowed in some other types, such tensors should be optimized away before lowering tensors to vectors.
///
/// # Examples
///
/// Known rank but unknown dimensions:
///
/// ```mlir
/// tensor<? x ? x ? x ? x f32>
/// ```
///
/// Partially known dimensions:
///
/// ```mlir
/// tensor<? x ? x 13 x ? x f32>
/// ```
///
/// Full static shape:
///
/// ```mlir
/// tensor<17 x 4 x 13 x 4 x f32>
/// ```
///
/// Tensor with rank zero. Represents a scalar:
///
/// ```mlir
/// tensor<f32>
/// ```
///
/// Zero-element dimensions are allowed:
///
/// ```mlir
/// tensor<0 x 42 x f32>
/// ```
///
/// Zero-element tensor of f32 type (hexadecimal literals not allowed here):
///
/// ```mlir
/// tensor<0xf32>
/// ```
///
/// Tensor with an encoding attribute (where #ENCODING is a named alias):
///
/// ```mlir
/// tensor<?x?xf64, #ENCODING>
/// ```
///
/// Tensor types represent multi-dimensional arrays, and have two variants: `RankedTensorType` and `UnrankedTensorType`.
/// Note: This class attaches the ShapedType trait to act as a mixin to provide many useful utility functions. This inheritance has no effect on derived tensor types.
pub struct Tensor<T, const N: usize> {
    shape: [i64; N],
    element_type: T,
    encoding: Box<dyn Attribute>
}

// impl Type for Tensor {

// }

impl const ShapedType for Tensor {
    /// Clone this type with the given shape and element type. If the provided shape is `None`, the current shape of the type is used.
    fn clone_with(
        &self,
        shape: Option<&[i64]>,
        element_type: Type
    ) -> Self
    {
        match self {
            UnrankedTensorType => {
                match shape {
                    None => UnrankedTensorType::get(element_type),
                    Some(shape) => RankedTensorType::get(shape, element_type)
                }
            },
            RankedTensorType => {
                match shape {
                    None => RankedTensorType::get(
                        self.shape(), element_type, self.encoding()),
                    Some(shape) => RankedTensorType::get(
                        shape.value_or(self.shape()),
                        element_type,
                        self.encoding())
                }
            }
        }
    }

    // /// Returns the element type of this tensor type.
    // fn element_type(&self) -> Type {
    //     return TypeSwitch<Tensor, Type>(*self)
    //     .Case<RankedTensorType, UnrankedTensorType>(
    //         |r#type: auto| { r#type.element_type(); });
    // }

    /// Returns if this type is ranked, i.e. it has a known number of dimensions.
    fn has_rank(&self) -> bool {
        match self {
            UnrankedTensorType => false,
            RankedTensorType => true
        }
    }

    // /// Returns the shape of this tensor type.
    // fn shape(&self) -> &[i64] {
    //     cast::<RankedTensorType>().shape()
    // }
}

//
// By me.
//

pub enum TensorElementType {
    ComplexType,
    FloatType,
    IntegerType,
    OpaqueType,
    Vector,
    IndexType
}

// ----------------------------------------------------------------------
// RankedTensorType
// ----------------------------------------------------------------------

/// This is a builder type that keeps local references to arguments. Arguments that are passed into the builder must outlive the builder.
pub struct RankedTensorType {
    shape: &'static [i64],
    // storage: SmallVector<[i64]>,
    element_type: Box<dyn Type>,
    encoding: Box<dyn Attribute>
}

// ----------------------------------------------------------------------
// UnrankedTensorType
// ----------------------------------------------------------------------

pub struct UnrankedTensorType {
}

// ----------------------------------------------------------------------
// MemRef
// ----------------------------------------------------------------------

/// Shaped reference to a region of memory.
///
/// =====================================================================
/// BaseMemRefType
/// =====================================================================
///
/// This class provides a shared interface for ranked and unranked memref types.
/// Note: This class attaches the ShapedType trait to act as a mixin to provide many useful utility functions. This inheritance has no effect on derived memref types.
///
/// =====================================================================
/// MemRef
/// =====================================================================
///
/// # Syntax
///
/// ```text
/// memref-type ::= `memref` `<` dimension-list-ranked type
///                 (`,` layout-specification)? (`,` memory-space)? `>`
/// layout-specification ::= attribute-value
/// memory-space ::= attribute-value
/// ```
///
/// A `memref` type is a reference to a region of memory (similar to a buffer pointer, but more powerful). The buffer pointed to by a memref can be allocated, aliased and deallocated. A memref can be used to read and write data from/to the memory region which it references. Memref types use the same shape specifier as tensor types. Note that `memref<f32>`, `memref<0 x f32>`, `memref<1 x 0 x f32>`, and `memref<0 x 1 x f32>` are all different types.
///
/// A `memref` is allowed to have an unknown rank (e.g. `memref<*xf32>`). The purpose of unranked memrefs is to allow external library functions to receive memref arguments of any rank without versioning the functions based on the rank. Other uses of this type are disallowed or will have undefined behaviour.
///
/// Are accepted as elements:
///
/// - built-in integer types;
/// - built-in index type;
/// - built-in floating point types;
/// - built-in vector types with elements of the above types;
/// - another memref type;
/// - any other type implementing `MemRefElementTypeInterface`.
///
/// # Layout
///
/// A memref may optionally have a layout that indicates how indices are transformed from the multi-dimensional form into a linear address. The layout must avoid internal aliasing, i.e., two distinct tuples of _in-bounds_ indices must be pointing to different elements in memory. The layout is an attribute that implements `MemRefLayoutAttrInterface`. The bulitin dialect offers two kinds of layouts: strided and affine map, each of which is available as an attribute. Other attributes may be used to represent the layout as long as they can be converted to a [semi-affine map](Affine.md/#semi-affine-maps) and implement the required interface. Users of memref are expected to fallback to the affine representation when handling unknown memref layouts. Multi-dimensional affine forms are interpreted in _row-major_ fashion.
///
/// In absence of an explicit layout, a memref is considered to have a multi-dimensional identity affine map layout.  Identity layout maps do not contribute to the MemRef type identification and are discarded on construction. That is, a type with an explicit identity map is `memref<?x?xf32, (i,j)->(i,j)>` is strictly the same as the one without a layout, `memref<?x?xf32>`.
///
/// ## Affine Map Layout
///
/// The layout may be represented directly as an affine map from the index space to the storage space. For example, the following figure shows an index map which maps a 2-dimensional index from a 2x2 index space to a 3x3 index space, using symbols `S0` and `S1` as offsets.
///
/// ![Index Map Example](/includes/img/index-map.svg)
///
/// Semi-affine maps are sufficiently flexible to represent a wide variety of dense storage layouts, including row- and column-major and tiled:
///
/// ```mlir
/// // MxN matrix stored in row major layout in memory:
/// #layout_map_row_major = (i, j) -> (i, j)
///
/// // MxN matrix stored in column major layout in memory:
/// #layout_map_col_major = (i, j) -> (j, i)
///
/// // MxN matrix stored in a 2-d blocked/tiled layout with 64x64 tiles.
/// #layout_tiled = (i, j) -> (i floordiv 64, j floordiv 64, i mod 64, j mod 64)
/// ```
///
/// ## Strided Layout
///
/// Memref layout can be expressed using strides to encode the distance, in number of elements, in (linear) memory between successive entries along a particular dimension. For example, a row-major strided layout for `memref<2x3x4xf32>` is `strided<[12, 4, 1]>`, where the last dimension is
/// contiguous as indicated by the unit stride and the remaining strides are
/// products of the sizes of faster-variying dimensions. Strided layout can also
/// express non-contiguity, e.g., `memref<2x3, strided<[6, 2]>>` only accesses
/// even elements of the dense consecutive storage along the innermost
/// dimension.
///
/// The strided layout supports an optional _offset_ that indicates the
/// distance, in the number of elements, between the beginning of the memref
/// and the first accessed element. When omitted, the offset is considered to
/// be zero. That is, `memref<2, strided<[2], offset: 0>>` and
/// `memref<2, strided<[2]>>` are strictly the same type.
///
/// Both offsets and strides may be _dynamic_, that is, unknown at compile time.
/// This is represented by using a question mark (`?`) instead of the value in
/// the textual form of the IR.
///
/// The strided layout converts into the following canonical one-dimensional
/// affine form through explicit linearization:
///
/// ```mlir
/// affine_map<(d0, ... dN)[offset, stride0, ... strideN] ->
///             (offset + d0 * stride0 + ... dN * strideN)>
/// ```
///
/// Therefore, it is never subject to the implicit row-major layout
/// interpretation.
///
/// # Codegen of Unranked Memref
///
/// Using unranked memref in codegen besides the case mentioned above is highly discouraged. Codegen is concerned with generating loop nests and specialised instructions for high-performance, unranked memref is concerned with hiding the rank and thus, the number of enclosing loops required to iterate over the data. However, if there is a need to code-gen unranked memref, one possible path is to cast into a static ranked type based on the dynamic rank. Another possible path is to emit a single while loop conditioned on a linear index and perform delinearization of the linear index to a dynamic array containing the (unranked) indices. While this is possible, it is expected to not be a good idea to perform this during codegen as the cost of the translations is expected to be prohibitive and optimizations at this level are not expected to be worthwhile. If expressiveness is the main concern, irrespective of performance, passing unranked memrefs to an external C++ library and implementing rank-agnostic logic there is expected to be significantly simpler.
///
/// Unranked memrefs may provide expressiveness gains in the future and help bridge the gap with unranked tensors. Unranked memrefs will not be expected to be exposed to codegen but one may query the rank of an unranked memref (a special op will be needed for this purpose) and perform a switch and cast to a ranked memref as a prerequisite to codegen.
///
/// Example:
///
/// ```mlir
/// // With static ranks, we need a function for each possible argument type
/// %A = alloc() : memref<16x32xf32>
/// %B = alloc() : memref<16x32x64xf32>
/// call @helper_2D(%A) : (memref<16x32xf32>)->()
/// call @helper_3D(%B) : (memref<16x32x64xf32>)->()
///
/// // With unknown rank, the functions can be unified under one unranked type
/// %A = alloc() : memref<16x32xf32>
/// %B = alloc() : memref<16x32x64xf32>
/// // Remove rank info
/// %A_u = memref_cast %A : memref<16x32xf32> -> memref<*xf32>
/// %B_u = memref_cast %B : memref<16x32x64xf32> -> memref<*xf32>
/// // call same function with dynamic ranks
/// call @helper(%A_u) : (memref<*xf32>)->()
/// call @helper(%B_u) : (memref<*xf32>)->()
/// ```
///
/// The core syntax and representation of a layout specification is a [semi-affine map](Affine.md/#semi-affine-maps). Additionally, syntactic sugar is supported to make certain layout specifications more intuitive to read. For the moment, a `memref` supports parsing a strided form which is converted to a semi-affine map automatically.
///
/// The memory space of a memref is specified by a target-specific attribute.
/// It might be an integer value, string, dictionary or custom dialect attribute.
/// The empty memory space (attribute is None) is target specific.
///
/// The notionally dynamic value of a memref value includes the address of the buffer allocated, as well as the symbols referred to by the shape, layout map, and index maps.
///
/// Examples of memref static type
///
/// ```mlir
/// // Identity index/layout map
/// #identity = affine_map<(d0, d1) -> (d0, d1)>
///
/// // Column major layout.
/// #col_major = affine_map<(d0, d1, d2) -> (d2, d1, d0)>
///
/// // A 2-d tiled layout with tiles of size 128 x 256.
/// #tiled_2d_128x256 = affine_map<(d0, d1) -> (d0 div 128, d1 div 256, d0 mod 128, d1 mod 256)>
///
/// // A tiled data layout with non-constant tile sizes.
/// #tiled_dynamic = affine_map<(d0, d1)[s0, s1] -> (d0 floordiv s0, d1 floordiv s1,
///                                 d0 mod s0, d1 mod s1)>
///
/// // A layout that yields a padding on two at either end of the minor dimension.
/// #padded = affine_map<(d0, d1) -> (d0, (d1 + 2) floordiv 2, (d1 + 2) mod 2)>
///
///
/// // The dimension list "16x32" defines the following 2D index space:
/// //
/// //   { (i, j) : 0 <= i < 16, 0 <= j < 32 }
/// //
/// memref<16x32xf32, #identity>
///
/// // The dimension list "16x4x?" defines the following 3D index space:
/// //
/// //   { (i, j, k) : 0 <= i < 16, 0 <= j < 4, 0 <= k < N }
/// //
/// // where N is a symbol which represents the runtime value of the size of
/// // the third dimension.
/// //
/// // %N here binds to the size of the third dimension.
/// %A = alloc(%N) : memref<16x4x?xf32, #col_major>
///
/// // A 2-d dynamic shaped memref that also has a dynamically sized tiled
/// // layout. The memref index space is of size %M x %N, while %B1 and %B2
/// // bind to the symbols s0, s1 respectively of the layout map #tiled_dynamic.
/// // Data tiles of size %B1 x %B2 in the logical space will be stored
/// // contiguously in memory. The allocation size will be
/// // (%M ceildiv %B1) * %B1 * (%N ceildiv %B2) * %B2 f32 elements.
/// %T = alloc(%M, %N) [%B1, %B2] : memref<?x?xf32, #tiled_dynamic>
///
/// // A memref that has a two-element padding at either end. The allocation
/// // size will fit 16 * 64 float elements of data.
/// %P = alloc() : memref<16x64xf32, #padded>
///
/// // Affine map with symbol 's0' used as offset for the first dimension.
/// #imapS = affine_map<(d0, d1) [s0] -> (d0 + s0, d1)>
/// // Allocate memref and bind the following symbols:
/// // '%n' is bound to the dynamic second dimension of the memref type.
/// // '%o' is bound to the symbol 's0' in the affine map of the memref type.
/// %n = ...
/// %o = ...
/// %A = alloc (%n)[%o] : <16x?xf32, #imapS>
/// ```
///
/// // 
/// // Builder
/// //
///
/// This is a builder type that keeps local references to arguments. Arguments that are passed into the builder must outlive the builder.
pub struct MemRef<T: Type, const N: usize> {
    shape: &'static [i64; N],
    pub element_type: T,
    pub layout: MemRefLayoutAttrInterface,
    pub memory_space: Box<dyn Attribute>
}

impl MemRef {
    /**
    Returns the memory space in which data referred to by this memref resides.
    */
    pub const fn memory_space(&self) -> Attribute {

    }
}

impl const ShapedType for MemRef {
    /// Clone this type with the given shape and element type. If the provided shape is `None`, the current shape of the type is used.
    fn clone_with(&self, shape: Option<&[i64]>, element_type: Type) -> Self {

    }

    /// Returns the element type of this memref type.
    fn element_type(&self) -> Type {

    }

    /// Returns if this type is ranked, i.e. it has a known number of dimensions.
    fn has_rank(&self) -> bool {

    }

    /// Returns the shape of this memref type.
    fn shape(&self) -> &[i64] {

    }

    // /// Return true if the specified element type is ok in a memref.
    // static bool is_valid_element_type(r#type: Type);
}

/// Given an `original_shape` and a `reduced_shape` assumed to be a subset of `original_shape` with some `1` entries erased, return the set of indices that specifies which of the entries of `original_shape` are dropped to obtain `reduced_shape`. The returned mask can be applied as a projection to `original_shape` to obtain the `reduced_shape`. This mask is useful to track which dimensions must be kept when e.g. compute MemRef strides under rank-reducing operations. Return std::nullopt if reduced_shape cannot be obtained by dropping only `1` entries in `original_shape`.
pub fn compute_rank_reduction_mask(
    original_shape: &[i64],
    reduced_shape: &[i64]
) -> Option<SmallDenseSet<usize>>
{

}

/// Enum that captures information related to verifier error conditions on slice insert/extract type of ops.
pub enum SliceVerificationResult {
    Success,
    RankTooLarge,
    SizeMismatch,
    ElemTypeMismatch,
    // Error codes to ops with a memory space and a layout annotation.
    MemSpaceMismatch,
    LayoutMismatch
}

/// Check if `original_type` can be rank reduced to `candidate_reduced_type` type
/// by dropping some dimensions with static size `1`.
/// Return `SliceVerificationResult::Success` on success or an appropriate error
/// code.
pub fn is_rank_reduced_type(
    original_type: ShapedType,
    candidate_reduced_type: ShapedType
) -> SliceVerificationResult
{
    if (original_type == candidate_reduced_type){
    return SliceVerificationResult::Success;}

    let original_shaped_type = original_type.cast::<ShapedType>();
    let candidate_reduced_shaped_type =
        candidate_reduced_type.cast::<ShapedType>();

    // Rank and size logic is valid for all ShapedTypes.
    let original_shape = original_shaped_type.shape();
    let candidate_reduced_shape = candidate_reduced_shaped_type.shape();
    let original_rank = original_shape.size();
    let candidate_reduced_rank = candidate_reduced_shape.size();
    if candidate_reduced_rank > original_rank {
        return SliceVerificationResult::RankTooLarge;
    }

    let optional_unused_dims_mask =
        compute_rank_reduction_mask(original_shape, candidate_reduced_shape);

    // Sizes cannot be matched in case empty vector is returned.
    if !optional_unused_dims_mask {
        return SliceVerificationResult::SizeMismatch;
    }
    if original_shaped_type.get_element_type() !=
        candidate_reduced_shaped_type.get_element_type() {
        return SliceVerificationResult::ElemTypeMismatch;
    }

    return SliceVerificationResult::Success;
}

// ----------------------------------------------------------------------
// UnrankedMemRefType
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// UnitType
// ----------------------------------------------------------------------

/// A unit type.
///
/// UnitType is a unit type, i.e. a type with exactly one possible value, where its value does not have a defined dynamic representation.
pub struct UnitType;

// ----------------------------------------------------------------------
// TupleType
// ----------------------------------------------------------------------

pub struct TupleType {
    types: &'static [dyn Type]
}

impl TupleType {
    /**
    Accumulate the types contained in this tuple and tuples nested within it. Note that this only flattens nested tuples, not any other container type, e.g. a tuple<i32, tensor<i32>, tuple<f32, tuple<i64>>> is flattened to (i32, tensor<i32>, f32, i64)
    */
    pub fn flattened_types(&self, types: &SmallVector<[impl Type]>) {
        for  r#type in self.types {
            if let nested_tuple = r#type.dyn_cast::<TupleType>() {
                nested_tuple.flattened_types(types);
            } else {
                types.push(r#type);
            }
        }
    }

    /// Return the number of held types.
    pub const fn len(&self) -> usize {
        1  // TODO
    }

    /// Return the element type at index `index`.
    pub const fn r#type(&self, index: usize) -> impl Type {
        assert!(index < self.len(), "Invalid index for tuple type");
        self.types[index]
    }
}

impl<'a> IntoIterator for TupleType {
    type Item = impl Type;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.types.into_iter()
    }
}

// ----------------------------------------------------------------------
// Type Utilities
// ----------------------------------------------------------------------

// /**
// Returns the strides of the MemRef if the layout map is in strided form.
// MemRefs with a layout map in strided form include:

// 1. empty or identity layout map, in which case the stride information is
//     the canonical form computed from sizes;
// 2. a StridedLayoutAttr layout;
// 3. any other layout that be converted into a single affine map layout of
//     the form `K + k0 * d0 + ... kn * dn`, where K and ki's are constants or
//     symbols.

// A stride specification is a list of integer values that are either static or dynamic (encoded with ShapedType::kDynamic). Strides encode the distance in the number of elements between successive entries along a particular dimension.
// */
// pub fn get_strides_and_offset(
//     t: MemRef,
//     strides: &SmallVector<[i64]>,
//     offset: &i64
// ) -> Result<(), Box<dyn std::error::Error>>
// {
//     Err(())
// }

// /**
// Wrapper around get_strides_and_offset(MemRef, SmallVectorImpl<i64>,
// i64) that will assert if the logical result is not succeeded.
// */
// pub fn get_strides_and_offset(t: MemRef) -> (SmallVector<[i64]>, i64) {

// }

// /**
// Return a version of `t` with identity layout if it can be determined statically that the layout is the canonical contiguous strided layout.
// Otherwise pass `t`'s layout into `simplifyAffineMap` and return a copy of `t` with simplified layout.
// */
// pub fn canonicalise_strided_layout(t: MemRef) -> MemRef {

// }

// /**
// Given MemRef `sizes` that are either static or dynamic, returns the canonical 'contiguous' strides AffineExpr. Strides are multiplicative and once a dynamic dimension is encountered, all canonical strides become dynamic and need to be encoded with a different symbol.
// For canonical strides expressions, the offset is always 0 and and fastest varying stride is always `1`.

// Examples:

//   - memref<3x4x5xf32> has canonical stride expression
//         `20*exprs[0] + 5*exprs[1] + exprs[2]`.
//   - memref<3x?x5xf32> has canonical stride expression
//         `s0*exprs[0] + 5*exprs[1] + exprs[2]`.
//   - memref<3x4x?xf32> has canonical stride expression
//         `s1*exprs[0] + s0*exprs[1] + exprs[2]`.
// */
// pub fn make_canonical_strided_layout_expr(
//     sizes: &[i64],
//     exprs: &[AffineExpr],
//     context: *mut MLIRContext
// ) -> AffineExpr
// {

// }

// /**
// Return the result of makeCanonicalStrudedLayoutExpr for the common case where `exprs` is {d0, d1, .., d_(sizes.size()-1)}
// */
// pub fn make_canonical_strided_layout_expr(
//     sizes: &[i64],
//     context: *mut MLIRContext
// ) -> AffineExpr
// {

// }

// /// Return true if the layout for `t` is compatible with strided semantics.
// pub fn is_strided(t: MemRef) -> bool {

// }
