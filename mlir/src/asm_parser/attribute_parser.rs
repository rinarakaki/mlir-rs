/*!
This file implements the parser for the MLIR Types.

- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/AsmParser/AttributeParser.cpp>
*/

use crate::{
    mlir::{
        asm_parser::{
            asm_parser_impl,
            asm_parser_state,
            parser::Parser,
            token::{Token, TokenKind}
        },
        ir::{
            affine_map,
            attribute::Attribute,
            builtins::{
                attributes,
                dialect,
                type_interfaces::ShapedType,
                types
            },
            dialect::implementation,
            dialect_resource_blob_manager,
            integer_set,
            operation::{
                definition::OptionalParseResult,
                support::NamedAttrList
            },
            r#type::Type,
        },
        support::logical_result::LogicalResult
    },
    llvm::{
        adt::string_extras,
        support::endian
    }
};

// impl<'a> Parser<'a> {
//     /**
//     Parse an arbitrary attribute.

//     attribute-value ::= `unit`
//                        | bool-literal
//                        | integer-literal (`:` (index-type | integer-type))?
//                        | float-literal (`:` float-type)?
//                        | string-literal (`:` type)?
//                        | type
//                        | `[` `:` (integer-type | float-type) tensor-literal `]`
//                        | `[` (attribute-value (`,` attribute-value)*)? `]`
//                        | `{` (attribute-entry (`,` attribute-entry)*)? `}`
//                        | symbol-ref-id (`::` symbol-ref-id)*
//                        | `dense` `<` tensor-literal `>` `:`
//                          (tensor-type | vector-type)
//                        | `sparse` `<` attribute-value `,` attribute-value `>`
//                          `:` (tensor-type | vector-type)
//                        | `strided` `<` `[` comma-separated-int-or-question `]`
//                          (`,` `offset` `:` integer-literal)? `>`
//                        | extended-attribute

//     */
//     pub fn parse_attribute(&self, r#type: Type) -> impl Attribute {
//     }

//     // T: Attribute | ArrayAttr | StringAttribute | SymbolRefAttr
//     /// Parse an optional attribute with the provided type.
//     pub fn parse_optional_attribute<T>(&self, attribute: &T, r#type: Type) -> OptionalParseResult {

//     }

//     /// Parse an optional attribute that is demarcated by a specific token.
//     pub fn parse_optional_attribute_with_token<T>(
//         &self, kind: TokenKind, attribute: &mut T, r#type: Type
//     ) -> OptionalParseResult {
//         // if getToken().isNot(kind) {
//         //     return None
//         // }

//         // match self.parse_attribute(r#type) {
//         //     Ok(parsed_attr) => {
//         //         *attr = parsed_attr.cast<AttributeT>();
//         //         return success()
//         //     },
//         //     Err(_) => return Err(())
//         // }
//     }

//     /// Parse an attribute dictionary.
//     pub fn parse_attribute_dict(&self, attributes: &NamedAttrList) -> LogicalResult {

//     }

//     /// Parse an extended attribute.
//     pub fn parse_extended_attr(&self, r#type: Type) -> impl Attribute {

//     }

//     /// Parse a float attribute.
//     pub fn parse_float_attr(&self, r#type: Type, is_negative: bool) -> impl Attribute {

//     }

//     /// Parse a decimal or a hexadecimal literal, which can be either an integer
//     /// or a float attribute.
//     pub fn parse_dec_or_hex_attr(&self,r#type: Type, is_negative: bool) -> impl Attribute {

//     }

//     /// Parse a dense elements attribute.
//     pub fn parse_dense_elements_attr(&self, r#type: Type) -> impl Attribute {

//     }
//     pub fn parse_elements_literal_type(&self, r#type: Type) -> ShapedType {

//     }

//     /// Parse a dense resource elements attribute.
//     pub fn parse_dense_resource_elements_attr(&self, r#type: Type) -> Attribute {

//     }

//     /// Parse a DenseArrayAttr.
//     pub fn parse_dense_array_attr(&self, r#type: Type) -> impl Attribute {

//     }

//     /// Parse a sparse elements attribute.
//     pub fn parse_sparse_elements_attr(&self, r#type: Type) -> impl Attribute {

//     }

//     /// Parse a strided layout attribute.
//     pub fn parse_strided_layout_attr(&self) -> impl Attribute {

//     }
// }
