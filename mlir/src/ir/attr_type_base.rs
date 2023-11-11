//! Base Attribute/Type Definition File
//!
//! This file contains the base set of constructs for defining Attribute and Type classes.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/AttrTypeBase.td>

use crate::mlir::ir::operation::base;

// -------------------------------------------------------------------------
// AttrTrait definitions
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// TypeTrait definitions
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// Builders
// -------------------------------------------------------------------------

//===----------------------------------------------------------------------
// Builders
//===----------------------------------------------------------------------

/**
Class for defining a custom getter.

TableGen generates several generic getter methods for each attribute and type by default, corresponding to the specified dag parameters. If the default generated ones cannot cover some use case, custom getters can be defined using instances of this class.

The signature of the `get` is always either:

```cpp
static <ClassName> get(MLIRContext *context, <other-parameters>...) {
  <body>...
}
```

or:

```cpp
static <ClassName> get(MLIRContext *context, <parameters>...);
```

To define a custom getter, the parameter list and body should be passed in as separate template arguments to this class. The parameter list is a TableGen DAG with `ins` operation with named arguments, which has either:

- string initialisers ("Type":$name) to represent a typed parameter, or
- CArg-typed initialisers (CArg<"Type", "default">:$name) to represent a typed parameter that may have a default value.

The type string is used verbatim to produce code and, therefore, must be a valid C++ type. It is used inside the C++ namespace of the parent Type's dialect; explicit namespace qualification like `::mlir` may be necessary if Types are not placed inside the `mlir` namespace. The default value string is used verbatim to produce code and must be a valid C++ initializer the given type. For example, the following signature specification

```text
AttrOrTypeBuilder<(ins "int":$integerArg, CArg<"float", "3.0f">:$floatArg)>
```

has an integer parameter and a float parameter with a default value.

If an empty string is passed in for `body`, then *only* the builder declaration will be generated; this provides a way to define complicated builders entirely in C++. If a `body` string is provided, the `Base::get` method should be invoked using `$_get`, e.g.:

```text
AttrOrTypeBuilder<(ins "int":$integerArg, CArg<"float", "3.0f">:$floatArg), [{
  return $_get($_ctxt, integerArg, floatArg);
}]>
```

This is necessary because the `body` is also used to generate `getChecked` methods, which have a different underlying `Base::get*` call.

*/
pub trait AttrOrTypeBuilder {
}

/*
Definitions
*/

/// Define a new attribute or type, named `name`, that inherits from the given C++ base class.
pub trait AttrOrTypeDefinition {
}

/*
Parameters
*/

/// 'Parameters' should be subclasses of this or simple strings (which is a shorthand for AttrOrTypeParameter<"C++Type">).
pub trait AttrOrTypeParameter {
}

pub trait AttributeParameter {
  
}
