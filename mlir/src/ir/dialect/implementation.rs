/*!
This file contains utilities classes for implementing dialect attributes and types.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/DialectImplementation.h>
*/

use crate::mlir::ir::{
    attribute::Attribute,
    operation::implementation::{AsmParser, AsmPrinter}
};

/**
This is a pure-virtual base class that exposes the asmprinter hooks necessary to implement a custom printAttribute/printType() method on a dialect.
*/
pub trait DialectAsmPrinter: AsmPrinter {

}


/**
The DialectAsmParser has methods for interacting with the asm parser when parsing attributes and types.
*/
pub trait DialectAsmParser: AsmParser {

}

/**
Provide a template class that can be specialised by users to dispatch to parsers. Auto-generated parsers generate calls to `FieldParser<T>::parse`, where `T` is the parameter storage type, to parse custom types. Parse an attribute.
*/
pub trait FieldParser {
    fn parse(parser: &impl AsmParser) -> Result<Self>;
}

// Parse an attribute.
impl<T: Attribute> FieldParser for T {
  fn parse(parser: &impl AsmParser) -> Result<Self> {
        let value = Attribute::new();
        if parser.parse_custom_attribute_with_fallback(value) {
            Err(())
        }
        Ok(value)
    }
}

// Parse an attribute.
impl<T: Attribute> FieldParser for T {
    fn parse(parser: &impl AsmParser) -> Result<Self> {
        let value = Attribute::new();
        if parser.parse_custom_attribute_with_fallback(value) {
            Err(())
        }
        Ok(value)
    }
}
