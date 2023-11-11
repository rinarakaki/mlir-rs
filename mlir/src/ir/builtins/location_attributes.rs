/*!
# Builtin Locations

Defines the set of builtin MLIR location attributes.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BuiltinLocationAttributes.td>
*/

use core::{
    any::TypeId,
    convert::From
};

use crate::mlir::ir::{
    attribute::Attribute,
    builtins::attributes::StringAttribute,
    location::Location,
    mlir_context::MLIRContext
};

/// Base class for Builtin dialect location attributes.
pub struct BuiltinLocationAttr {

}

/**
A callsite source location.

# Syntax

```text
callsite-location ::= `callsite` `(` location `at` location `)`
```

An instance of this location allows for representing a directed stack of location usages. This connects a location of a `callee` with the location of a `caller`.

# Example

```mlir
loc(callsite("foo" at "mysource.cc":10:8))
```
*/
pub struct CallSiteLocation {
    callee: Location,
    caller: Location
}

/// A file:line:column source location.
///
/// # Syntax
///
/// ```text
/// filelinecol-location ::= string-literal `:` integer-literal `:`
///                           integer-literal
/// ```
///
/// An instance of this location represents a tuple of file, line number, and column number. This is similar to the type of location that you get from most source languages.
///
/// # Example
///
/// ```mlir
/// loc("mysource.cc":10:8)
/// ```
pub struct FileLineColLocation {
    filename: StringAttribute,
    line: usize,
    column: usize
}

/// A tuple of other source locations.
///
/// # Syntax
///
/// ```text
/// fused-location ::= `fused` fusion-metadata? `[` location (location `,`)* `]`
/// fusion-metadata ::= `<` attribute-value `>`
/// ```
///
/// An instance of a `fused` location represents a grouping of several other source locations, with optional metadata that describes the context of the fusion. There are many places within a compiler in which several constructs may be fused together, e.g. pattern rewriting, that normally result partial or even total loss of location information. With `fused` locations, this is a non-issue.
///
/// # Example
///
/// ```mlir
/// loc(fused["mysource.cc":10:8, "mysource.cc":22:8)
/// loc(fused<"CSE">["mysource.cc":10:8, "mysource.cc":22:8])
/// ```
pub struct FusedLocation<M: Attribute> {
    locations: &'static [Location],
    metadata: M
}

/// A named source location.
///
/// # Syntax
///
/// ```text
/// name-location ::= string-literal (`(` location `)`)?
/// ```
///
/// An instance of this location allows for attaching a name to a child location.This can be useful for representing the locations of variable, or node, definitions.
///
/// # Example
///
/// ```mlir
/// loc("CSE"("mysource.cc":10:8))
/// ```
pub struct NameLocation {
    name: StringAttribute,
    child_location: Location
}

/// An opaque source location.
///
/// An instance of this location essentially contains a pointer to some data structure that is external to MLIR and an optional location that can be used if the first one is not suitable. Since it contains an external structure, only the optional location is used during serialisation.
pub struct OpaqueLocation {
    underlying_location: usize,
    underlying_type_id: TypeId,
    fallback_location: Location
}

impl OpaqueLocation {
    // /**
    // Returns an instance of opaque location which contains a given pointer to
    // an object. The corresponding MLIR location is set to UnknownLoc.
    // */
    // template <typename T>
    // static Self get(T underlying_location, MLIRContext *context);

    // /// Returns an instance of opaque location which contains a given pointer to
    // /// an object and an additional MLIR location.
    // template <typename T>
    // static Self get(T underlying_location, Location fallback_location) {
    //   return get(reinterpret_cast<usize>(underlying_location),
    //              TypeId::get<T>(), fallback_location);
    // }

    // /// Returns a pointer to some data structure that opaque location stores.
    // template <typename T> static T get_underlying_location(Location location) {
    //   assert(isa<T>(location));
    //   return reinterpret_cast<T>(
    //       location.cast<Self>().get_underlying_location());
    // }

    // /**
    // Returns a pointer to some data structure that opaque location stores.
    // Returns nullptr if provided location is not opaque location or if it contains a pointer of different type.
    // */
    // template <typename T>
    // static T get_underlying_location_or_null(Location location) {
    //   return isa<T>(location)
    //              ? reinterpret_cast<T>(
    //                    location.cast<Self>().get_underlying_location())
    //              : T(nullptr);
    // }

    // /**
    // Checks whether provided location is opaque location and contains a pointer to an object of particular type.
    // */
    // template <typename T> static bool isa(Location location) {
    //     auto opaque_loc = location.dyn_cast<OpaqueLocation>();
    //     return opaque_loc && opaque_loc.get_underlying_type_id() == TypeId::get<T>();
    // }
}

/// An unspecified source location.
///
/// # Syntax
///
/// ```text
/// unknown-location ::= `?`
/// ```
///
/// Source location information is an extremely integral part of the MLIR infrastructure. As such, location information is always present in the IR, and must explicitly be set to unknown. Thus, an instance of the `unknown` location represents an unspecified source location.
///
/// # Example
///
/// ```mlir
/// loc(?)
/// ```
pub struct UnknownLocation;

impl UnknownLocation {
    pub fn get(context: Option<MLIRContext>) -> Self;
}
