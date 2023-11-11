/*!
# MLIR Builtin Attribute Classes

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BuiltinAttributes.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BuiltinAttributes.td>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/BuiltinAttributes.cpp>
*/

use crate::{
    mlir::ir::{
        affine_map::AffineMap,
        attribute::{
            Attribute, NamedAttribute,
            detail
        },
        builtins::{
            attribute_interfaces,
            dialect
        },
        dialect::Dialect,
        dialect_resource_blob_manager,
        integer_set::IntegerSet,
        mlir_context::MLIRContext,
        operation::{
            self,
            implementation
        },
        sub_element_interfaces,
        symbol_table,
        r#type::Type
    },
    llvm::{
        adt::{
            ap_float,
            aps_int,
            sequence,
            type_switch
        },
        support::endian
    }
};

// ----------------------------------------------------------------------
// AffineMapAttribute
// ----------------------------------------------------------------------

/**
An Attribute containing an AffineMap object.

# Syntax

```text
affine-map-attribute ::= `affine_map` `<` affine-map `>`
```

# Examples

```mlir
affine_map<(d0) -> (d0)>
affine_map<(d0, d1, d2) -> (d0, d1)>
```
*/
pub struct AffineMapAttribute {
    value: AffineMap
}

impl Attribute for AffineMapAttribute {
    type ValueType = AffineMap;
}

// ----------------------------------------------------------------------
// ArrayAttribute
// ----------------------------------------------------------------------

/**
A collection of other Attribute values.

# Syntax

```text
array-attribute ::= `[` (attribute-value (`,` attribute-value)*)? `]`
```

An array attribute is an attribute that represents a collection of attribute values.

# Examples

```mlir
[]
[10, i32]
[affine_map<(d0, d1, d2) -> (d0, d1)>, i32, "string attribute"]
```
*/
pub struct ArrayAttribute {
    value: &'static [dyn Attribute]
}

impl Attribute for ArrayAttribute {
    type ValueType = &'static [dyn Attribute];
}

// ----------------------------------------------------------------------
// DenseArrayAttribute
// ----------------------------------------------------------------------

/**
A dense array of integer or floating point elements.

A dense array attribute is an attribute that represents a dense array of primitive element types. Contrary to DenseIntOrFPElementsAttr this is a flat unidimensional array which does not have a storage optimisation for splat. This allows to expose the raw array through a C++ API as `&[T]` for compatible types. The element type must be bool or an integer or float whose bitwidth is a multiple of 8. Bool elements are stored as bytes.

This is the base class attribute. Access to C++ types is intended to be managed through the subclasses `DenseI8ArrayAttr`, `DenseI16ArrayAttr`, `DenseI32ArrayAttr`, `DenseI64ArrayAttr`, `DenseF32ArrayAttr`, and `DenseF64ArrayAttr`.

# Syntax

```text
dense-array-attribute ::= `array` `<` (integer-type | float-type)
                                          (`:` tensor-literal)? `>`
```

# Examples

```mlir
array<i8>
array<i32: 10, 42>
array<f64: 42., 12.>
```

When a specific subclass is used as argument of an operation, the declarative assembly will omit the type and print directly:

```mlir
[1, 2, 3]
```
*/
pub struct DenseArrayAttribute {
    element_type: Box<dyn Type>,
    size: i64
}

// ----------------------------------------------------------------------
// DenseIntOrFPElementsAttribute
// ----------------------------------------------------------------------

/**

*/
pub struct DenseIntOrFPElementsAttribute {
}

// ----------------------------------------------------------------------
// DenseStringElementsAttribute
// ----------------------------------------------------------------------

/**
An Attribute containing a dense multi-dimensional array of strings.

# Syntax

```text
dense-string-elements-attribute ::= `dense` `<` attribute-value `>` `:`
                                        ( tensor-type | vector-type )
```

A dense string elements attribute is an elements attribute containing a densely packed vector or tensor of string values. There are no restrictions placed on the element type of this attribute, enabling the use of dialect specific string types.

# Examples

```mlir
// A splat tensor of strings.
dense<"example"> : tensor<2x!foo.string>
// A tensor of 2 string elements.
dense<["example1", "example2"]> : tensor<2x!foo.string>
```
*/
pub struct DenseStringAttrinute {
    r#type: AttributeSelfTypeParameter<"", "ShapedType">,
    value: &'static [&'static str]
}

// ----------------------------------------------------------------------
// DenseResourceElementsAttribute
// ----------------------------------------------------------------------

/**
An Attribute containing a dense multi-dimensional array backed by a resource.

# Syntax

```text
dense-resource-elements-attribute ::=
      `dense_resource` `<` resource-handle `>` `:` shaped-type
```

A dense resource elements attribute is an elements attribute backed by a handle to a builtin dialect resource containing a densely packed array of values. This class provides the low-level attribute, which should only be interacted with in very generic terms, actual access to the underlying resource data is intended to be managed through one of the subclasses, such as; `DenseBoolResourceElementsAttr`, `DenseUI64ResourceElementsAttr`, `DenseI32ResourceElementsAttr`, `DenseF32ResourceElementsAttr`, `DenseF64ResourceElementsAttr`, etc.

# Examples

```mlir
// A tensor referencing a builtin dialect resource, `resource_1`, with two
// unsigned i32 elements.
dense_resource<resource_1> : tensor<2xui32>
```
*/
pub struct DenseResourceElementsAttribute;

// ----------------------------------------------------------------------
// DictionaryAttribute
// ----------------------------------------------------------------------

/**
A dictionary of named Attribute values.

# Syntax

```
dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
```

A dictionary attribute is an attribute that represents a sorted collection of named attribute values. The elements are sorted by name, and each name must be unique within the collection.

# Examples

```mlir
{}
{attr_name = "string attribute"}
{int_attr = 10, "string attr name" = "string attribute"}
```
*/
pub struct DictionaryAttribute {
    value: &'static [NamedAttribute]
}

impl Attribute for DictionaryAttribute {
    type ValueType = &'static [NamedAttribute];
}

// ----------------------------------------------------------------------
// FloatAttribute
// ----------------------------------------------------------------------

/**
An Attribute containing a floating-point value.

# Syntax

```text
float-attribute ::= (float-literal (`:` float-type)?)
                      | (hexadecimal-literal `:` float-type)
```

A float attribute is a literal attribute that represents a floating point value of the specified [float type](#floating-point-types). It can be represented in the hexadecimal form where the hexadecimal value is interpreted as bits of the underlying binary representation. This form is useful for representing infinity and NaN floating point values. To avoid confusion with integer attributes, hexadecimal literals _must_ be followed by a float type to define a float attribute.

# Examples

```mlir
42.0         // float attribute defaults to f64 type
42.0 : f32   // float attribute of f32 type
0x7C00 : f16 // positive infinity
0x7CFF : f16 // NaN (one of possible values)
42 : f32     // Error: expected integer type
```
*/
pub struct FloatAttribute {
    r#type: AttributeSelfTypeParameter<"">,
    value: APFloatParameter<"">
}

impl Attribute for FloatAttribute {
    type ValueType = APFloat;
}

// ----------------------------------------------------------------------
// IntegerAttribute
// ----------------------------------------------------------------------

/**
An Attribute containing a integer value.

# Syntax

    ```text
    integer-attribute ::= (integer-literal ( `:` (index-type | integer-type) )?)
                          | `true` | `false`
    ```

    An integer attribute is a literal attribute that represents an integral
    value of the specified integer or index type. `i1` integer attributes are
    treated as `boolean` attributes, and use a unique assembly format of either
    `true` or `false` depending on the value. The default type for non-boolean
    integer attributes, if a type is not specified, is signless 64-bit integer.

    Examples:

    ```mlir
    10 : i32
    10    // : i64 is implied here.
    true  // A bool, i.e. i1, value.
    false // A bool, i.e. i1, value.
    ```
*/
pub struct IntegerAttribute {
    r#type: AttributeSelfTypeParameter<"">,
    value: APInt
}

impl Attribute for IntegerAttribute {
    type ValueType = APInt;
}

// ----------------------------------------------------------------------
// IntegerSetAttribute
// ----------------------------------------------------------------------

/**

An Attribute containing an IntegerSet object.

Syntax:

    ```
    integer-set-attribute ::= `affine_set` `<` integer-set `>`
    ```

    Examples:

    ```mlir
    affine_set<(d0) : (d0 - 2 >= 0)>
    ```

*/
pub struct IntegerSetAttribute {
    value: IntegerSet
}

impl Attribute for IntegerSetAttribute {
    type ValueType = IntegerSet;
}

// ----------------------------------------------------------------------
// OpaqueAttribute
// ----------------------------------------------------------------------

/**
An opaque representation of another Attribute.

Syntax:

    ```
    opaque-attribute ::= dialect-namespace `<` attr-data `>`
    ```

    Opaque attributes represent attributes of non-registered dialects. These are
    attribute represented in their raw string form, and can only usefully be
    tested for attribute equality.

    Examples:

    ```mlir
    #dialect<"opaque attribute data">
    ```

*/
pub struct OpaqueAttribute {
    dialect_namespace: StringAttribute,
    attr_data: StringRefParameter<"">,
    r#type: AttributeSelfTypeParameter<"">
}

// ----------------------------------------------------------------------
// SparseElementsAttribute
// ----------------------------------------------------------------------

/**
An opaque representation of a multi-dimensional array.

# Syntax

```text
sparse-elements-attribute ::= `sparse` `<` attribute-value `,`
                                  attribute-value `>` `:`
                                  ( tensor-type | vector-type )
```

A sparse elements attribute is an elements attribute that represents a sparse vector or tensor object. This is where very few of the elements are non-zero.

The attribute uses COO (coordinate list) encoding to represent the sparse elements of the elements attribute. The indices are stored via a 2-D tensor of 64-bit integer elements with shape [N, ndims], which specifies the indices of the elements in the sparse tensor that contains non-zero values.
The element values are stored via a 1-D tensor with shape [N], that supplies the corresponding values for the indices.

# Example

```mlir
sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>

// This represents the following tensor:
///  [[1, 0, 0, 0],
///   [0, 0, 5, 0],
///   [0, 0, 0, 0]]
```
*/
pub struct SparseElementsAttribute {
    r#type: AttributeSelfTypeParameter<"", ShapedType>,
    indices: DenseIntElementsAttribute,
    values: DenseElementsAttribute
}

// ----------------------------------------------------------------------
// StridedLayoutAttribute
// ----------------------------------------------------------------------

/**
An Attribute representing a strided layout of a shaped type.

# Syntax

```text
strided-layout-attribute ::= `strided` `<` `[` stride-list `]`
                                 (`,` `offset` `:` dimension)? `>`
stride-list ::= /*empty*/
                  | dimension (`,` dimension)*
dimension ::= decimal-literal | `?`
```

A strided layout attribute captures layout information of the memref type in the canonical form. Specifically, it contains a list of _strides_, one for each dimension. A stride is the number of elements in the linear storage one must step over to reflect an increment in the given dimension. For example, a `MxN` row-major contiguous shaped type would have the strides `[N, 1]`. The layout attribute also contains the _offset_ from the base pointer of the shaped type to the first effectively accessed element,
    expressed in terms of the number of contiguously stored elements.

Strides must be positive and the offset must be non-negative. Both the strides and the offset may be _dynamic_, i.e. their value may not be known at compile time. This is expressed as a `?` in the assembly syntax and as
    `ShapedType::kDynamic` in the code. Stride and offset values
    must satisfy the constraints above at runtime, the behaviour is undefined
    otherwise.

See [Dialects/Builtin.md#memreftype](MemRef type) for more information.
*/
pub struct StridedLayoutAttribute {
    offset: i64,
    strides: ArrayRefParameter<i64>
}

// ----------------------------------------------------------------------
// StringAttribute
// ----------------------------------------------------------------------

/**
An Attribute containing a string.

# Syntax

```text
string-attribute ::= string-literal (`:` type)?
```

A string attribute is an attribute that represents a string literal value.

# Examples

```mlir
"An important string"
"string with a type" : !dialect.string
```
*/
pub struct StringAttribute {
    value: &'static str,
    r#type: AttributeSelfTypeParameter<"">
}

impl Attribute for StringAttribute {
    type ValueType = &'static str;
}

impl StringAttribute {
    /**
    If the value of this string is prefixed with a dialect namespace,
    returns the dialect corresponding to that namespace if it is loaded,
    nullptr otherwise. For example, the string `llvm.fastmathflags` would
    return the LLVM dialect, assuming it is loaded in the context.
    */
    pub const fn referenced_dialect(&self) -> impl Dialect {
        
    }

    /// Return a pointer to the start of the string data.
    pub const fn data(&self) -> &char { self.value.data() }

    /// Return the number of bytes in this string.
    pub const fn len(&self) -> usize {
        self.value.len()
    }

    // /// Iterate over the underlying string data.
    // StringRef::iterator begin() const { return getValue().begin(); }
    // StringRef::iterator end() const { return getValue().end(); }

    /// Compare the underlying string value to the one in `rhs`.
    pub const fn compare(&self, rhs: Self) -> int {
        if self == rhs {
            return 0;
        }
        self.value.compare(rhs.value())
    }

    /**
    Return an empty StringAttribute with NoneType type. This is a special variant of the `get` method that is used by the MLIRContext to cache the instance.
    */
    fn empty_string_attr_unchecked(context: *mut MLIRContext) -> StringAttribute {

    }
}

/// Enable conversion to StringRef.
impl From<StringAttribute> for &str {
    fn from(value: StringAttribute) -> Self {
        value.value
    }
}

// impl IntoIterator for StringAttribute {
//     type Item = char or u8;
//     type IntoIter = ;
    
//     fn into_iter(self) -> Self::IntoIter {
        
//     }
// }

// ----------------------------------------------------------------------
// SymbolRefAttribute
// ----------------------------------------------------------------------

/**
An Attribute containing a symbolic reference to an Operation.

# Syntax:

```text
symbol-ref-attribute ::= symbol-ref-id (`::` symbol-ref-id)*
```

A symbol reference attribute is a literal attribute that represents a named reference to an operation that is nested within an operation with the `OpTrait::SymbolTable` trait. As such, this reference is given meaning by the nearest parent operation containing the `OpTrait::SymbolTable` trait. It may optionally contain a set of nested references that further resolve to a symbol nested within a different symbol table.

This attribute can only be held internally by
[array attributes](#array-attribute),
[dictionary attributes](#dictionary-attribute)(including the top-level
operation attribute dictionary) as well as attributes exposing it via
the `SubElementAttrInterface` interface. Symbol reference attributes
nested in types are currently not supported.

# Rationale

Identifying accesses to global data is critical to enabling efficient multi-threaded compilation. Restricting global data access to occur through symbols and limiting the places that can legally hold a symbol reference simplifies reasoning about these data accesses.

See [`Symbols And SymbolTables`](../SymbolsAndSymbolTables.md) for more information.

# Examples

```mlir
@flat_reference
@parent_reference::@nested_reference
```
*/
pub struct SymbolRefAttribute {
    root_reference: StringAttribute,
    nested_references: &'static [FlatSymbolRefAttribute]
}

// ----------------------------------------------------------------------
// TypeAttribute
// ----------------------------------------------------------------------

/**
An Attribute containing a Type.

# Syntax

```
type-attribute ::= type
```

A type attribute is an attribute that represents a [type object](#type-system).

Examples:

```mlir
i32
!dialect.type
```
 */
pub struct TypeAttribute {
    value: Box<dyn Type>
}

impl Attribute for TypeAttribute {
    type ValueType = dyn Type;
}

// ----------------------------------------------------------------------
// UnitAttribute
// ----------------------------------------------------------------------

/**
An Attribute value of `unit` type.

Syntax:

```text
unit-attribute ::= `unit`
```

A unit attribute is an attribute that represents a value of `unit` type. The `unit` type allows only one value forming a singleton set. This attribute value is used to represent attributes that only have meaning from their existence.

One example of such an attribute could be the `swift.self` attribute. This attribute indicates that a function parameter is the self/context parameter.
It could be represented as a [boolean attribute](#boolean-attribute)(true or false), but a value of false doesn't really bring any value. The parameter either is the self/context or it isn't.


# Examples

```mlir
// A unit attribute defined with the `unit` value specifier.
func.func @verbose_form() attributes {dialectName.unitAttr = unit}

// A unit attribute in an attribute dictionary can also be defined without
// the value specifier.
func.func @simple_form() attributes {dialectName.unitAttr}
```
*/
pub struct UnitAttribute;

impl UnitAttribute {
    fn new(context: *mut MLIRContext) -> Self {
        
    }
}
