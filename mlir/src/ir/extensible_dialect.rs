//! Extensible Dialect
//!
//! This file defines the DynamicOpDefinition class, the DynamicTypeDefinition class, and the DynamicAttrDefinition class, which represent respectively operations, types, and attributes that can be defined at runtime. They can be registered at runtime to an extensible dialect, using the ExtensibleDialect class defined in this file.
//!
//! For a more complete documentation, see <https://mlir.llvm.org/docs/ExtensibleDialects/>.
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/ExtensibleDialect.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/IR/ExtensibleDialect.cpp>

use core;

/// The definition of a dynamic attribute. A dynamic attribute is an attribute that is defined at runtime, and that can be registered at runtime by an extensible dialect (a dialect inheriting ExtensibleDialect). This class stores the parser, the printer, and the verifier of the attribute. Each dynamic attribute definition refers to one instance of this class.
pub struct DynamicAttrDefinition {
    /// The name should be prefixed with the dialect name followed by '.'.
    name: String,

    /// Dialect in which this attribute is defined.
    dialect: *mut ExtensibleDialect,

    /// The attribute verifier. It checks that the attribute parameters satisfy the invariants.
    verifier: Verifier,

    /// The attribute parameters parser. It parses only the parameters, and expects the attribute name to have already been parsed.
    parser: Parser,

    /// The attribute parameters printer. It prints only the parameters, and expects the attribute name to have already been printed.
    printer: Printer,

    /// Context in which the concrete attributes are uniqued.
    context: *mut MLIRContext
}

/// A dynamic attribute instance. This is an attribute whose definition is defined at runtime.
/// It is possible to check if an attribute is a dynamic attribute using `my_attr.isa<DynamicAttr>()`, and getting the attribute definition of a dynamic attribute using the `DynamicAttr::getAttrDef` method.
/// All dynamic attributes have the same storage, which is an array of attributes.
pub struct DynamicAttr {

}

/// The definition of a dynamic type. A dynamic type is a type that is defined at runtime, and that can be registered at runtime by an extensible dialect (a dialect inheriting ExtensibleDialect). This class stores the parser, the printer, and the verifier of the type. Each dynamic type definition refers to one instance of this class.
pub struct DynamicTypeDefinition {

}

/// A dynamic type instance. This is a type whose definition is defined at runtime.
/// It is possible to check if a type is a dynamic type using `my_type.isa<DynamicType>()`, and getting the type definition of a dynamic type using the `DynamicType::getTypeDef` method.
/// All dynamic types have the same storage, which is an array of attributes.
pub struct DynamicType {

}

/// The definition of a dynamic op. A dynamic op is an op that is defined at runtime, and that can be registered at runtime by an extensible dialect (a dialect inheriting ExtensibleDialect). This class stores the functions that are in the OperationName class, and in addition defines the TypeID of the op that will be defined.
/// Each dynamic operation definition refers to one instance of this class.
pub struct DynamicOpDefinition {

}

/// A dialect that can be extended with new operations/types/attributes at runtime.
pub struct ExtensibleDialect {
    /// The set of all dynamic types registered.
    dyn_types: DenseMap<TypeId, std::unique_ptr<DynamicTypeDefinition>>,

    /// This structure allows to get in O(1) a dynamic type given its name.
    name_to_dyn_types: StringMap<*mut DynamicTypeDefinition>,

    /// The set of all dynamic attributes registered.
    dyn_attrs: DenseMap<TypeId, std::unique_ptr<DynamicAttrDefinition>>,

    /// This structure allows to get in O(1) a dynamic attribute given its name.
    name_to_dyn_attrs: StringMap<*mut DynamicAttrDefinition>
}

/// A dialect that can be defined at runtime. It can be extended with new operations, types, and attributes at runtime.
pub struct DynamicDialect {

}
