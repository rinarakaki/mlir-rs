//! Enum Attributes
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/EnumAttr.td>

use crate::mlir::ir::attr_type_base::AttributeParameter;

/// A C++ enum as an attribute parameter. The parameter implements a parser and printer for the enum by dispatching calls to `stringToSymbol` and `symbolToString`.
pub struct EnumParameter;

/**
An attribute backed by a C++ enum. The attribute contains a single parameter `value` whose type is the C++ enum class.

# Example

```text
def MyEnum : I32EnumAttr<"MyEnum", "a simple enum", [
                           I32EnumAttrCase<"First", 0, "first">,
                           I32EnumAttrCase<"Second", 1, "second>]> {
  let genSpecializedAttr = 0;
}

type MyEnumEnumAttribute = EnumAttribute<MyDialect, MyEnum, "enum">;
```

By default, the assembly format of the attribute works best with operation assembly formats. For example:

```text
def MyOp : Op<MyDialect, "my_op"> {
  let arguments = (enum: MyEnumAttr);
  let assembly_format = "$enum attr-dict";
}
```

The op will appear in the IR as `my_dialect.my_op first`. However, the generic format of the attribute will be `#my_dialect<"enum first">`. Override the attribute's assembly format as required.
*/
pub struct EnumAttribute {
    /// The enum attribute has one parameter: the C++ enum value.
    value: EnumParameter  // <enumInfo>
}
