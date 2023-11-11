//! # Classes for Implementing Op Types
//!
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpImplementation.h>

use core::{
    any::TypeId,
    cmp::PartialEq,
    hash::{Hash, Hasher},
    io::Write
};

use crate::{
    mlir::{
        ir::{
            attr_type_base::AttrOrType,
            attribute::Attribute,
            builtins::types,
            dialect::{
                Dialect,
                interface
            },
            operation::definition,
            r#type::Type
        },
        support::{
            logical_result::LogicalResult,
        }
    },
    llvm::{
        adt::{
            stl_extras::{interleave_comma, has_single_element},
            twine
        },
        support::sm_loc::SMLoc
    }
};

/// This class represents an opaque handle to a dialect resource entry.
pub struct AsmDialectResourceHandle {
    /// The opaque handle to the dialect resource.
    resource: Option<()>,  // = nullptr;
    /// The type of the resource referenced.
    opaque_id: TypeId,
    /// The dialect owning the given resource.
    dialect: Option<dyn Dialect>
}

impl PartialEq for AsmDialectResourceHandle {
    fn eq(&self, other: &Self) -> bool {
        self.resource == other.resource
    }
}

impl AsmDialectResourceHandle {
    /// Return an opaque pointer to the referenced resource.
    pub const fn resource(&self) -> *mut () {
        self.resource
    }

    /// Return the type ID of the resource.
    pub const fn type_id(&self) -> TypeId {
        self.opaque_id
    }

    /// Return the dialect that owns the resource.
    pub const fn dialect(&self) -> Option<Dialect> {
        self.dialect
    }
}

impl Hash for AsmDialectResourceHandle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u8(self.resource());
    }
}


/// This base class exposes generic asm printer hooks, usable across the various derived printers.
pub trait AsmPrinter: Write {
    // 
    // AsmPrinterImpl
    //

    /// The output stream for the printer.
    raw_ostream &os;

    /// An underlying assembly printer state.
    AsmStateImpl &state;

    /// A set of flags to control the printer's behaviour.
    OpPrintingFlags printerFlags;

    /// A tracker for the number of new lines emitted during printing.
    NewLineCounter newLine;

//   /// Return the raw output stream used by this printer.
//   virtual raw_ostream &stream() const;

    /**
    Print the given floating point value in a stabilized form that can be roundtripped through the IR. This is the companion to the `parse_float` hook on the AsmParser.
    */
    fn print_float(&self, value: &APFloat);

    fn print_type(&self, r#type: impl Type);

    fn print_attribute(&self, attribute: impl Attribute);

    // /// Trait to check if `AttrType` provides a `print` method.
    // template <typename AttrOrType>
    // using has_print_method =
    //     decltype(std::declval<AttrOrType>().print(std::declval<AsmPrinter &>()));
    // template <typename AttrOrType>
    // using detect_has_print_method =
    //     llvm::is_detected<has_print_method, AttrOrType>;

    /**
    Print the provided attribute in the context of an operation custom printer/parser: this will invoke directly the print method on the attribute class and skip the `#dialect.mnemonic` prefix in most cases.
    */
    // template <
    //     std::enable_if_t<detect_has_print_method<AttrOrType>::value>
    //         *sfinae = nullptr>
    fn print_stripped_attr_or_type<AttrOrType>(
        &self, attr_or_type: AttrOrType
    ) {
        match print_alias(attr_or_type) {
            Ok(_) => {},
            Err(_) => attr_or_type.print(*self)
        };
    }

    /**
    Print the provided array of attributes or types in the context of an operation custom printer/parser: this will invoke directly the print method on the attribute class and skip the `#dialect.mnemonic` prefix in most cases.
    */
    // template <
    //     std::enable_if_t<detect_has_print_method<AttrOrType>::value>
    //         *sfinae = nullptr>
    fn print_stripped_attr_or_type<ArrayOrType: IntoIterator>
        (&self, attr_or_types: &[AttrOrType])
    {
        interleave_comma(
            attr_or_types,
            self.stream(),
            |attr_or_type: AttrOrType| {
                self.print_stripped_attr_or_type(attr_or_type);
            }
        );
    }

    /**
    SFINAE for printing the provided attribute in the context of an operation custom printer in the case where the attribute does not define a print method.
    */
    // template <
    //     std::enable_if_t<!detect_has_print_method<AttrOrType>::value>
    //         *sfinae = nullptr>
    fn print_stripped_attr_or_type<AttrOrType>(
        &self, attr_or_type: AttrOrType
    ) {
        *self << attr_or_type;
    }

    /**
    Print the given attribute without its type. The corresponding parser must provide a valid type for the attribute.
    */
    fn print_attribute_without_type(&self, attribute: Attribute);

    /**
    Print the given string as a keyword, or a quoted and escaped string if it
    has any special or non-printable characters in it.
    */
    fn print_keyword_or_string(&self, keyword: &str);

    /**
    Print the given string as a symbol reference, i.e. a form representable by a SymbolRefAttr. A symbol reference is represented as a string prefixed with '@'. The reference is surrounded with `""`'s and escaped if it has any special or non-printable characters in it.
    */
    fn print_symbol_name(&self, symbol_ref: &Self);

    /// Print a handle to the given dialect resource.
    fn print_resource_handle(&self, resource: &AsmDialectResourceHandle);

    /// Print an optional arrow followed by a type list.
    fn print_optional_arrow_type_list<I: Iterator>(&self, types: &I) {
        if types.count() > 0 {
            self.print_arrow_type_list(types);
        }
    }
    
    fn print_arrow_type_list<I: Iterator>(&self, types: &I) {
        let os = self.stream() << " -> ";

        let wrapped = !has_single_element(types) ||
                    matches!(types.next(), FunctionType);
        if wrapped {
            os << '(';
        }
        interleave_comma(types, *this);
        if wrapped {
            os << ')';
        }
    }

    /// Print the two given type ranges in a functional form.
    fn print_functional_type<
        I: Iterator,
        O: Iterator
    >(&self, inputs: &I, outputs: &O) {
        let os = self.stream();
        os << '(';
        interleave_comma(inputs, *self);
        os << ')';
        self.print_arrow_type_list(outputs);
    }

    // private

    // /**
    // Print the alias for the given attribute, return failure if no alias could be printed.
    // */
    // virtual Result<(), Box<dyn std::error::Error>> print_alias(attribute: Attribute);

    // /**
    // Print the alias for the given type, return failure if no alias could be printed.
    // */
    // virtual Result<(), Box<dyn std::error::Error>> print_alias(r#type: Type);
}

/// This enum describes the different kinds of elision for the type of an attribute when printing it.
pub enum AttrTypeElision {
    /// The type must not be elided,
    Never,
    /// The type may be elided when it matches the default used in the parser (for example i64 is the default for integer attributes).
    May,
    /// The type must be elided.
    Must
}

/// This enum is used to represent the binding strength of the enclosing
/// context that an AffineExprStorage is being printed in, so we can
/// intelligently produce parens.
enum BindingStrength {
    Weak,   // + and -
    Strong, // All other binary operators.
}

/// This class represents a specific instance of a symbol Alias.
pub struct SymbolAlias {
    /// The main name of the alias.
    name: StringRef,
    /// The suffix index of the alias.
    suffix_index: u32,  // : 30;
    /// A flag indicating whether this alias is for a type.
    is_type: bool,  // : 1;
    /// A flag indicating whether this alias may be deferred or not.
    is_deferrable: bool  // : 1;
}

/// This class represents a utility that initializes the set of attribute and
/// type aliases, without the need to store the extra information within the
/// main AliasState class or pass it around via function arguments.
pub struct AliasInitialiser {
    /// The set of asm interfaces within the context.
    DialectInterfaceCollection<OpAsmDialectInterface> &interfaces;

    /// An allocator used for alias names.
    llvm::BumpPtrAllocator &alias_allocator;

    /// The set of built aliases.
    llvm::MapVector<const void *, InProgressAliasInfo> aliases;

    /// Storage and stream used when generating an alias.
    SmallString<32> alias_buffer;
    llvm::raw_svector_ostream alias_os;
}

/// This class manages the state for type and attribute aliases.
pub struct AliasState {
      /// Mapping between attribute/type and alias.
  llvm::MapVector<const void *, SymbolAlias> attr_type_to_alias;

  /// An allocator used for alias names.
  llvm::BumpPtrAllocator alias_allocator;
}

/**
This is a pure-virtual base class that exposes the asmprinter hooks necessary to implement a custom print() method.
*/
pub trait OpAsmPrinter: AsmPrinter {

}

// L456
/// This base class exposes generic asm parser hooks, usable across the various derived parsers.
pub trait AsmParser {
}

/// These are the supported delimiters around operand lists and region argument lists, used by parseOperandList.
enum Delimiter {
    /// Zero or more operands with no delimiters.
    None,
    /// Parens surrounding zero or more operands.
    Paren,
    /// Square brackets surrounding zero or more operands.
    Square,
    /// <> brackets surrounding zero or more operands.
    LessGreater,
    /// {} brackets surrounding zero or more operands.
    Braces,
    /// Parens supporting zero or more operands, or nothing.
    OptionalParen,
    /// Square brackets supporting zero or more ops, or nothing.
    OptionalSquare,
    /// <> brackets supporting zero or more ops, or nothing.
    OptionalLessGreater,
    /// {} brackets surrounding zero or more operands, or nothing.
    OptionalBraces,
}

/**
This class represents a StringSwitch like class that is useful for parsing expected keywords. On construction, it invokes `parse_keyword` and processes each of the provided cases statements until a match is hit. The provided `ResultT` must be assignable from `failure()`.
*/
struct KeywordSwitch<'a, ResultT = LogicalResult> {
    /// The parser used to construct this switch.
    parser: &'a dyn AsmParser,

    /// The location of the keyword, used to emit errors as necessary.
    loc: SMLoc,

    /// The parsed keyword itself.
    keyword: &'static str,

    /// The result of the switch statement or none if currently unknown.
    result: Option<ResultT>
}

// L1258
/**
The OpAsmParser has methods for interacting with the asm parser: parsing things from it, emitting errors etc. It has an intentionally high-level API that is designed to reduce/constrain syntax innovation in individual operations.

For example, consider an op like this:

```mlir
%x = load %p[%1, %2] : memref<...>
```

The `%x = load` tokens are already parsed and therefore invisible to the custom op parser.  This can be supported by calling `parse_operand_list` to parse the %p, then calling `parse_operand_list` with a `square_delimiter` to parse the indices, then calling `parseColonTypeList` to parse the result type.
*/
pub struct OpAsmParser {

}

// L1546
pub struct OpAsmDialectInterface {

}

/// Holds the result of `getAlias` hook call.
enum AliasResult {
    /**
    The object (type or attribute) is not supported by the hook and an alias was not provided.
    */
    NoAlias,
    /// An alias was provided, but it might be overriden by other hook.
    OverridableAlias,
    /**
    An alias was provided and it should be used (no other hooks will be checked).
    */
    FinalAlias
}
