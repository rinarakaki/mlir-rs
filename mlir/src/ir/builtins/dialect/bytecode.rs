/*!
# Builtin Bytecode Implementation

- lib
  - <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/BuiltinDialectBytecode.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/BuiltinDialectBytecode.cpp>
*/

use crate::{
    mlir::{
        bytecode::implementation,
        ir::{
            builtins::{
                dialect,
                types
            },
            diagnostics,
            dialect_resource_blob_manager
        },
    },
    llvm::adt::type_switch
};

/// This enum contains marker codes used to indicate which attribute is currently being decoded, and how it should be decoded. The order of these codes should generally be unchanged, as any changes will inevitably break compatibility with older bytecode.
pub enum AttributeCode {
  ///   ArrayAttr {
  ///     elements: Attribute[]
  ///   }
  ///
    ArrayAttr = 0,

  ///   DictionaryAttr {
  ///     attrs: <StringAttribute, Attribute>[]
  ///   }
    DictionaryAttr = 1,

  ///   StringAttribute {
  ///     value: string
  ///   }
    StringAttribute = 2,

  ///   StringAttrWithType {
  ///     value: string,
  ///     type: Type
  ///   }
  /// A variant of StringAttribute with a type.
    StringAttrWithType = 3,

  ///   FlatSymbolRefAttr {
  ///     rootReference: StringAttribute
  ///   }
  /// A variant of SymbolRefAttr with no leaf references.
    FlatSymbolRefAttr = 4,

  ///   SymbolRefAttr {
  ///     rootReference: StringAttribute,
  ///     leafReferences: FlatSymbolRefAttr[]
  ///   }
    SymbolRefAttr = 5,

  ///   TypeAttr {
  ///     value: Type
  ///   }
    TypeAttr = 6,

  ///   UnitAttr {
  ///   }
    UnitAttr = 7,

  ///   IntegerAttr {
  ///     type: Type
  ///     value: APInt,
  ///   }
    IntegerAttr = 8,

  ///   FloatAttr {
  ///     type: FloatType
  ///     value: APFloat
  ///   }
    FloatAttr = 9,

  ///   CallSiteLoc {
  ///    callee: LocationAttribute,
  ///    caller: LocationAttribute
  ///   }
    CallSiteLoc = 10,

  ///   FileLineColLoc {
  ///     file: StringAttribute,
  ///     line: varint,
  ///     column: varint
  ///   }
    FileLineColLoc = 11,

  ///   FusedLoc {
  ///     locations: LocationAttribute[]
  ///   }
    FusedLoc = 12,

  ///   FusedLocWithMetadata {
  ///     locations: LocationAttribute[],
  ///     metadata: Attribute
  ///   }
  /// A variant of FusedLoc with metadata.
    FusedLocWithMetadata = 13,

  ///   NameLoc {
  ///     name: StringAttribute,
  ///     childLoc: LocationAttribute
  ///   }
    NameLoc = 14,

  ///   UnknownLoc {
  ///   }
    UnknownLoc = 15,

  ///   DenseResourceElementsAttr {
  ///     type: Type,
  ///     handle: ResourceHandle
  ///   }
    DenseResourceElementsAttr = 16,

  ///   DenseArrayAttr {
  ///     type: RankedTensorType,
  ///     data: blob
  ///   }
    DenseArrayAttr = 17,

  ///   DenseIntOrFPElementsAttr {
  ///     type: ShapedType,
  ///     data: blob
  ///   }
    DenseIntOrFPElementsAttr = 18,

  ///   DenseStringElementsAttr {
  ///     type: ShapedType,
  ///     isSplat: varint,
  ///     data: string[]
  ///   }
    DenseStringElementsAttr = 19,

  ///   SparseElementsAttr {
  ///     type: ShapedType,
  ///     indices: DenseIntElementsAttr,
  ///     values: DenseElementsAttr
  ///   }
    SparseElementsAttr = 20,
}

/// This enum contains marker codes used to indicate which type is currently
/// being decoded, and how it should be decoded. The order of these codes should
/// generally be unchanged, as any changes will inevitably break compatibility
/// with older bytecode.
enum TypeCode {
  ///   IntegerType {
  ///     widthAndSignedness: varint // (width << 2) | (signedness)
  ///   }
  ///
    IntegerType = 0,

  ///   IndexType {
  ///   }
  ///
    IndexType = 1,

  ///   FunctionType {
  ///     inputs: Type[],
  ///     results: Type[]
  ///   }
  ///
    FunctionType = 2,

  ///   BFloat16Type {
  ///   }
  ///
    BFloat16Type = 3,

  ///   Float16Type {
  ///   }
  ///
    Float16Type = 4,

  ///   Float32Type {
  ///   }
  ///
    Float32Type = 5,

  ///   Float64Type {
  ///   }
  ///
    Float64Type = 6,

  ///   Float80Type {
  ///   }
  ///
    Float80Type = 7,

  ///   Float128Type {
  ///   }
  ///
    Float128Type = 8,

  ///   ComplexType {
  ///     elementType: Type
  ///   }
  ///
    ComplexType = 9,

  ///   MemRef {
  ///     shape: svarint[],
  ///     elementType: Type,
  ///     layout: Attribute
  ///   }
  ///
    MemRef = 10,

  ///   MemRefTypeWithMemSpace {
  ///     memorySpace: Attribute,
  ///     shape: svarint[],
  ///     elementType: Type,
  ///     layout: Attribute
  ///   }
  /// Variant of MemRef with non-default memory space.
    MemRefTypeWithMemSpace = 11,

  ///   NoneType {
  ///   }
  ///
    NoneType = 12,

  ///   RankedTensorType {
  ///     shape: svarint[],
  ///     elementType: Type,
  ///   }
  ///
    RankedTensorType = 13,

  ///   RankedTensorTypeWithEncoding {
  ///     encoding: Attribute,
  ///     shape: svarint[],
  ///     elementType: Type
  ///   }
  /// Variant of RankedTensorType with an encoding.
    RankedTensorTypeWithEncoding = 14,

  ///   TupleType {
  ///     elementTypes: Type[]
  ///   }
    TupleType = 15,

  ///   UnrankedMemRefType {
  ///     shape: svarint[]
  ///   }
  ///
    UnrankedMemRefType = 16,

  ///   UnrankedMemRefTypeWithMemSpace {
  ///     memorySpace: Attribute,
  ///     shape: svarint[]
  ///   }
  /// Variant of UnrankedMemRefType with non-default memory space.
    UnrankedMemRefTypeWithMemSpace = 17,

  ///   UnrankedTensorType {
  ///     elementType: Type
  ///   }
  ///
    UnrankedTensorType = 18,

  ///   VectorType {
  ///     shape: svarint[],
  ///     elementType: Type
  ///   }
  ///
    VectorType = 19,

  ///   VectorTypeWithScalableDims {
  ///     numScalableDims: varint,
  ///     shape: svarint[],
  ///     elementType: Type
  ///   }
  /// Variant of VectorType with scalable dimensions.
    VectorTypeWithScalableDims = 20,
}
