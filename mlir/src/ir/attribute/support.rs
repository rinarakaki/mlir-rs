/*!
This file defines support types for registering dialect extended attributes.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/AttributeSupport.h>
*/

use crate::{
    mlir::{
        ir::{
            dialect::Dialect,
            mlir_context,
            storage_uniquer_support,
            r#type::{Type, HasTraitFn}
        },
        support::interface_support::InterfaceMap,
    },
    llvm::adt::{
        pointer_int_pair,
        twine
    }
};

/**
Base storage class appearing in an attribute. Derived storage classes should only be constructed within the context of the AttributeUniquer.
*/
#[repr(align(8))]
pub struct AttributeStorage {

}

/**
A utility class to get, or create, unique instances of attributes within an MLIRContext. This class manages all creation and uniquing of attributes.
*/
pub struct AttributeUniquer {
    
}
